"""只读网关。

独立于 ``masskb serve``。不修改检索、健康检查、入库、ask 或 Python 客户端的默认契约。
页面仍是包内的查询页。上游只访问写死的 ``GET /health`` 和 ``POST /search``。

公开检索假定专用样例库，不把混有私密文档的集合当成安全边界。
每次只按包内样例预先算好的 ``doc_id`` 请求上游，让过滤发生在检索之前。
返回的命中必须同时对上该分块的 ``doc_id``、``chunk_id`` 和正文；对不上的 payload 整条丢掉。
因此同名 ``source`` 或伪造正文都不会被显示。样例点被别的入库覆盖后，这里也不会把旧正文补回去。

对外健康检查只有 ``ok`` 和固定为 false 的 ``openai_key_configured``。
成功的检索 JSON 只有 ``query`` 和 ``hits`` 里的 ``source``、``chunk_id``、``text``。
两条纯 dense 查询按命中自己的有限数值 ``score`` 降序合并；分数相同则保留先出现的那条。
没有分数，或分数是布尔、NaN、无穷，这条命中直接丢掉，不补一个名次。
目录里没有被上游返回的分块也不会被填回去。

``client_wait_seconds`` 只决定何时给浏览器回「检索超时」。
到点不取消上游读取，名额也要等这次读取结束才释放。
因此名额是网关尚未结束的上游调用数，不是 ``serve`` 进程锁或线程池的并发上限。
每个样例 ``doc_id`` 只请求一次，失败不重试。
进程关闭会关掉自己的连接并不再无限等待；``serve`` 里已经开始的调用仍可能做完。
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.parse import urlsplit

import httpx

from masskb.ingest import chunk_text

PUBLIC_SAMPLE_SOURCES = frozenset({"lanheliu.md", "qingpingguo.md"})
DEFAULT_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_UPSTREAM = "http://127.0.0.1:8080"
CLIENT_WAIT_SECONDS = 20.0
UPSTREAM_TIMEOUT_SECONDS = 120.0
ADMISSION_LIMIT = 2
MAX_BODY_BYTES = 4096
MAX_QUERY_CHARS = 200
MAX_UPSTREAM_BYTES = 256 * 1024
SHUTDOWN_JOIN_SECONDS = 2.0

_SEARCH_KEYS = frozenset(
    {
        "query",
        "top_k",
        "doc_id",
        "hybrid",
        "answer",
        "rerank",
        "rewrite",
        "graph_hops",
        "aliases",
    }
)
_FALSE_FLAGS = ("hybrid", "answer", "rerank", "rewrite", "graph_hops", "aliases")
_ROUTES = frozenset({"/", "/health", "/search"})

_PAGE_HEADERS = (
    ("Content-Security-Policy", (
        "default-src 'none'; "
        "script-src 'unsafe-inline'; "
        "style-src 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'none'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "form-action 'self'; "
        "frame-ancestors 'none'"
    )),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Cache-Control", "no-store"),
)

_DETAIL_FORBIDDEN = "不允许"
_DETAIL_INVALID = "请求无效"
_DETAIL_TOO_LARGE = "请求过大"
_DETAIL_BUSY = "服务正忙"
_DETAIL_TIMEOUT = "检索超时"
_DETAIL_SEARCH_FAILED = "检索失败"


class GateError(ValueError):
    """网关配置不能用。"""


@dataclass(frozen=True)
class SampleChunk:
    """包内样例的一条分块。``doc_id`` 与 ``chunk_id`` 用和入库相同的算法。"""

    source: str
    doc_id: str
    chunk_id: str
    text: str


@dataclass(frozen=True)
class GateConfig:
    """一次网关进程的边界。超时和名额有默认值，测试可以收短。"""

    upstream: str = DEFAULT_UPSTREAM
    allowed_hosts: frozenset[str] = DEFAULT_ALLOWED_HOSTS
    admission_limit: int = ADMISSION_LIMIT
    client_wait_seconds: float = CLIENT_WAIT_SECONDS
    upstream_timeout_seconds: float = UPSTREAM_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "upstream", normalize_upstream(self.upstream))
        hosts = frozenset(normalize_allowed_host(host) for host in self.allowed_hosts)
        if not hosts:
            raise GateError("至少要允许一个 Host")
        object.__setattr__(self, "allowed_hosts", hosts)
        if self.admission_limit < 1:
            raise GateError("名额至少为 1")
        if self.client_wait_seconds <= 0 or self.upstream_timeout_seconds <= 0:
            raise GateError("超时必须大于 0")


@dataclass(frozen=True)
class _Outcome:
    kind: str
    reply: tuple[int, Any] | None = None


class Admission:
    """尚未结束的上游调用计数。释放只发生在那次读取的 finally 里。"""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._inflight = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._inflight >= self.limit:
                return False
            self._inflight += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._inflight < 1:
                raise RuntimeError("名额释放多于获取")
            self._inflight -= 1

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight


def normalize_upstream(value: str) -> str:
    """只保留协议、主机和端口。路径、查询、用户信息都拒绝。"""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"}:
        raise GateError("upstream 只接受 http 或 https")
    if parsed.username is not None or parsed.password is not None:
        raise GateError("upstream 不能带用户信息")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise GateError("upstream 不能带路径、查询或片段")
    if parsed.hostname is None or parsed.port is None:
        raise GateError("upstream 必须写明主机和端口")
    return f"{parsed.scheme}://{parsed.netloc}"


def normalize_allowed_host(value: str) -> str:
    """额外允许的主机名。不接受通配、端口、路径或用户信息。``::1`` 是唯一保留的冒号形式。"""
    host = value.strip().lower().rstrip(".")
    if host == "::1":
        return host
    if host == "" or any(char in host for char in " /\\@:*[]"):
        raise GateError("allow-host 必须是主机名或 IP")
    return host


def hostname_from_host_header(value: str | None) -> str | None:
    """从 Host 头取出主机名。端口忽略，畸形值拒绝。"""
    if value is None:
        return None
    raw = value.strip()
    if raw == "" or any(char in raw for char in " \t\r\n/@\\"):
        return None
    if raw.startswith("["):
        end = raw.find("]")
        if end <= 1:
            return None
        host = raw[1:end]
        rest = raw[end + 1 :]
        if rest and not re.fullmatch(r":\d{1,5}", rest):
            return None
        if ":" in rest and not (1 <= int(rest[1:]) <= 65535):
            return None
        return host.lower()
    if raw.count(":") > 1:
        return None
    if ":" in raw:
        host, port = raw.rsplit(":", 1)
        if host == "" or not port.isdigit() or not (1 <= int(port) <= 65535):
            return None
        return host.lower()
    return raw.lower()


def classify_request(method: str, target: str) -> str | None:
    """只认出三条精确路由。编码、查询串、绝对 URL 和点段都不是路由。"""
    if "\\" in target or "\x00" in target or target.startswith("//"):
        return None
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        return None
    path = parsed.path
    if "%" in path or ".." in path or path not in _ROUTES:
        return None
    if method == "GET" and path == "/":
        return "page"
    if method == "GET" and path == "/health":
        return "health"
    if method == "POST" and path == "/search":
        return "search"
    return "method"


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """拒绝重复键。标准库 ``json.loads`` 会默默留下后一个值。"""
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate key")
        payload[key] = value
    return payload


def parse_search_body(raw: bytes) -> tuple[dict[str, Any] | None, str | None]:
    """只接受查询页现在发出的那一组字段，并拒绝打开任何开关。"""
    if len(raw) > MAX_BODY_BYTES:
        return None, "size"
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None, "invalid"
    if not isinstance(payload, dict) or set(payload) != _SEARCH_KEYS:
        return None, "invalid"
    query = payload["query"]
    if not isinstance(query, str) or query.strip() == "" or len(query) > MAX_QUERY_CHARS:
        return None, "invalid"
    if type(payload["top_k"]) is not int or payload["top_k"] != 5:
        return None, "invalid"
    if payload["doc_id"] is not None:
        return None, "invalid"
    for key in _FALSE_FLAGS:
        if payload[key] is True:
            return None, "forbidden"
        if payload[key] is not False:
            return None, "invalid"
    return {
        "query": query,
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "answer": False,
        "rerank": False,
        "rewrite": False,
        "graph_hops": False,
        "aliases": False,
    }, None


def load_sample_catalog() -> tuple[SampleChunk, ...]:
    """用和 ``ingest --root`` 相同的切分，钉死包内样例的 id 和正文。"""
    chunks: list[SampleChunk] = []
    for name in sorted(PUBLIC_SAMPLE_SOURCES):
        raw = files("masskb").joinpath("samples", name).read_text(encoding="utf-8")
        document = chunk_text(raw, source=name)
        for chunk in document.chunks:
            chunks.append(
                SampleChunk(
                    source=name,
                    doc_id=document.doc_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                )
            )
    if len(chunks) != 4:
        raise GateError("公开样例必须是 4 条分块")
    if len({item.chunk_id for item in chunks}) != 4:
        raise GateError("公开样例分块 id 不能重复")
    return tuple(chunks)


def sample_doc_ids(catalog: tuple[SampleChunk, ...]) -> tuple[str, ...]:
    """按样例文件名排序后的文档 id，每个只出现一次。"""
    ordered: list[str] = []
    for item in catalog:
        if item.doc_id not in ordered:
            ordered.append(item.doc_id)
    return tuple(ordered)


@dataclass(frozen=True)
class TrustedHit:
    """一条已经对上样例正文的命中。``score`` 只用于排序，不写进公开 JSON。"""

    source: str
    chunk_id: str
    text: str
    score: float


def finite_score(value: object) -> float | None:
    """只接受有限的整数或浮点。布尔、NaN、无穷和非数字都没有名次。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def project_sample_hits(
    body: object,
    catalog: tuple[SampleChunk, ...],
) -> list[TrustedHit] | None:
    """按上游顺序收下和目录逐字一致、且分数可用的命中。"""
    if not isinstance(body, dict):
        return None
    hits = body.get("hits")
    if not isinstance(hits, list):
        return None
    by_id = {item.chunk_id: item for item in catalog}
    accepted: list[TrustedHit] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        chunk_id = hit.get("chunk_id")
        pinned = by_id.get(chunk_id) if isinstance(chunk_id, str) else None
        if pinned is None:
            continue
        if hit.get("doc_id") != pinned.doc_id or hit.get("source") != pinned.source:
            continue
        if hit.get("text") != pinned.text:
            continue
        score = finite_score(hit.get("score"))
        if score is None:
            continue
        accepted.append(
            TrustedHit(
                source=pinned.source,
                chunk_id=pinned.chunk_id,
                text=pinned.text,
                score=score,
            )
        )
    return accepted


def rank_trusted_hits(groups: list[list[TrustedHit]]) -> list[dict[str, str]]:
    """按分数降序合并。分数相同保留先出现的；同一分块只留更高的分数。"""
    best: dict[str, tuple[TrustedHit, int]] = {}
    encounter = 0
    for group in groups:
        for hit in group:
            previous = best.get(hit.chunk_id)
            if previous is None or hit.score > previous[0].score:
                best[hit.chunk_id] = (hit, encounter)
            encounter += 1
    ordered = sorted(best.values(), key=lambda item: (-item[0].score, item[1]))
    return [
        {"source": hit.source, "chunk_id": hit.chunk_id, "text": hit.text}
        for hit, _index in ordered
    ]


def public_health_ok(status: int, body: object) -> bool:
    return status == 200 and isinstance(body, dict) and body.get("ok") is True


def query_page_bytes() -> bytes:
    text = files("masskb").joinpath("static", "index.html").read_text(encoding="utf-8")
    return text.encode("utf-8")


def run_limited(
    admission: Admission,
    client_wait_seconds: float,
    call: Callable[[], Any],
    *,
    workers: list[threading.Thread] | None = None,
    workers_lock: threading.Lock | None = None,
) -> _Outcome:
    """占用一个名额调用上游。等候超时只返回，不取消调用，也不释放名额。"""
    if not admission.try_acquire():
        return _Outcome("busy")
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["reply"] = call()
        except Exception as exc:  # noqa: BLE001 — 失败变成安全错误，不把异常文本送出去
            box["error"] = exc
        finally:
            admission.release()

    thread = threading.Thread(target=worker, name="masskb-gate-upstream", daemon=True)
    if workers is not None and workers_lock is not None:
        with workers_lock:
            alive = [item for item in workers if item.is_alive()]
            workers[:] = alive
            workers.append(thread)
    thread.start()
    thread.join(client_wait_seconds)
    if thread.is_alive():
        return _Outcome("timeout")
    if "error" in box:
        return _Outcome("error")
    return _Outcome("ok", box.get("reply"))


class GateServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: GateConfig) -> None:
        self.config = config
        self.admission = Admission(config.admission_limit)
        self.workers: list[threading.Thread] = []
        self.workers_lock = threading.Lock()
        self.catalog = load_sample_catalog()
        self.http = httpx.Client(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(config.upstream_timeout_seconds),
            headers={"user-agent": "masskb-gate"},
        )
        super().__init__(address, GateHandler)

    def fetch(self, method: str, route: str, content: bytes | None = None) -> tuple[int, Any]:
        if route not in {"/health", "/search"}:
            raise GateError("上游路由不在白名单")
        headers = {"accept": "application/json"}
        if content is not None:
            headers["content-type"] = "application/json"
        with self.http.stream(
            method,
            f"{self.config.upstream}{route}",
            content=content,
            headers=headers,
        ) as response:
            return _fetch_reply(response)

    def finish(self) -> None:
        """停止接受连接并关掉客户端。不等待上游的 120 秒上限。"""
        self.shutdown()
        self.server_close()
        self.http.close()
        self._join_workers(SHUTDOWN_JOIN_SECONDS)

    def _join_workers(self, limit: float) -> None:
        deadline = time.monotonic() + limit
        with self.workers_lock:
            workers = list(self.workers)
        for thread in workers:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(remaining)


class GateHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: GateServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)
        self.close_connection = True

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._reject_other("PUT")

    def do_PATCH(self) -> None:
        self._reject_other("PATCH")

    def do_DELETE(self) -> None:
        self._reject_other("DELETE")

    def do_HEAD(self) -> None:
        self._reject_other("HEAD")

    def do_OPTIONS(self) -> None:
        self._reject_other("OPTIONS")

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _reject_other(self, method: str) -> None:
        kind = classify_request(method, self.path)
        if kind == "method":
            self._send_json(405, {"detail": _DETAIL_FORBIDDEN})
            return
        self._send_json(404, {"detail": _DETAIL_FORBIDDEN})

    def _dispatch(self, method: str) -> None:
        kind = classify_request(method, self.path)
        if kind is None:
            self._send_json(404, {"detail": _DETAIL_FORBIDDEN})
            return
        if kind == "method":
            self._send_json(405, {"detail": _DETAIL_FORBIDDEN})
            return
        host_values = self.headers.get_all("Host") or []
        if len(host_values) != 1:
            self._send_json(403, {"detail": _DETAIL_FORBIDDEN})
            return
        host = hostname_from_host_header(host_values[0])
        if host is None or host not in self.server.config.allowed_hosts:
            self._send_json(403, {"detail": _DETAIL_FORBIDDEN})
            return
        if kind == "page":
            self._send_page()
            return
        if kind == "health":
            self._proxy_health()
            return
        self._proxy_search()

    def _send_page(self) -> None:
        self._send_bytes(200, query_page_bytes(), "text/html; charset=utf-8", _PAGE_HEADERS)

    def _proxy_health(self) -> None:
        outcome = run_limited(
            self.server.admission,
            self.server.config.client_wait_seconds,
            lambda: self.server.fetch("GET", "/health"),
            workers=self.server.workers,
            workers_lock=self.server.workers_lock,
        )
        ok = False
        if outcome.kind == "ok" and outcome.reply is not None:
            status, body = outcome.reply
            ok = public_health_ok(status, body)
        if outcome.kind == "busy":
            self._send_json(429, {"detail": _DETAIL_BUSY})
            return
        self._send_json(200, {"ok": ok, "openai_key_configured": False})

    def _proxy_search(self) -> None:
        raw, error = self._read_body()
        if error == "size":
            self._send_json(413, {"detail": _DETAIL_TOO_LARGE})
            return
        if error is not None or raw is None:
            self._send_json(422, {"detail": _DETAIL_INVALID})
            return
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            self._send_json(422, {"detail": _DETAIL_INVALID})
            return
        parsed, problem = parse_search_body(raw)
        if problem == "size":
            self._send_json(413, {"detail": _DETAIL_TOO_LARGE})
            return
        if problem == "forbidden":
            self._send_json(403, {"detail": _DETAIL_FORBIDDEN})
            return
        if problem is not None or parsed is None:
            self._send_json(422, {"detail": _DETAIL_INVALID})
            return
        outcome = run_limited(
            self.server.admission,
            self.server.config.client_wait_seconds,
            lambda: search_pinned(self.server, parsed["query"]),
            workers=self.server.workers,
            workers_lock=self.server.workers_lock,
        )
        if outcome.kind == "busy":
            self._send_json(429, {"detail": _DETAIL_BUSY})
            return
        if outcome.kind == "timeout":
            self._send_json(504, {"detail": _DETAIL_TIMEOUT})
            return
        if outcome.kind != "ok" or not isinstance(outcome.reply, list):
            self._send_json(503, {"detail": _DETAIL_SEARCH_FAILED})
            return
        self._send_json(200, {"query": parsed["query"], "hits": outcome.reply})

    def _read_body(self) -> tuple[bytes | None, str | None]:
        if self.headers.get_all("Transfer-Encoding"):
            return None, "invalid"
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1:
            return None, "invalid"
        try:
            declared = int(lengths[0].strip())
        except ValueError:
            return None, "invalid"
        if declared < 0:
            return None, "invalid"
        if declared > MAX_BODY_BYTES:
            return None, "size"
        got = bytearray()
        while len(got) < declared:
            piece = self.rfile.read(min(1024, declared - len(got)))
            if not piece:
                return None, "invalid"
            got.extend(piece)
            if len(got) > MAX_BODY_BYTES:
                return None, "size"
        return bytes(got), None

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, raw, "application/json; charset=utf-8", ())

    def _send_bytes(
        self,
        status: int,
        raw: bytes,
        content_type: str,
        extra: tuple[tuple[str, str], ...],
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.send_header("X-Content-Type-Options", "nosniff")
            if not any(name.lower() == "cache-control" for name, _value in extra):
                self.send_header("Cache-Control", "no-store")
            for name, value in extra:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionError, TimeoutError):
            return


def search_pinned(server: GateServer, query: str) -> list[dict[str, str]]:
    """每个预置 ``doc_id`` 只查一次。失败即停止，不重试，也不再查下一个。"""
    catalog = server.catalog
    groups: list[list[TrustedHit]] = []
    for doc_id in sample_doc_ids(catalog):
        payload = json.dumps(
            {
                "query": query,
                "top_k": 5,
                "doc_id": doc_id,
                "hybrid": False,
                "answer": False,
                "rerank": False,
                "rewrite": False,
                "graph_hops": False,
                "aliases": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        status, body = server.fetch("POST", "/search", payload)
        if status != 200:
            raise GateError("upstream search failed")
        projected = project_sample_hits(body, catalog)
        if projected is None:
            raise GateError("upstream search shape")
        groups.append(projected)
    return rank_trusted_hits(groups)


def _fetch_reply(response: httpx.Response) -> tuple[int, Any]:
    """按块累计上游正文，超过上限就关闭连接，不再把剩余内容读进内存。"""
    parts: list[bytes] = []
    total = 0
    for block in response.iter_bytes():
        if not block:
            continue
        total += len(block)
        if total > MAX_UPSTREAM_BYTES:
            response.close()
            raise GateError("upstream body too large")
        parts.append(block)
    content = b"".join(parts)
    if content.strip() == b"":
        return response.status_code, None
    return response.status_code, json.loads(content.decode("utf-8"))


def bind_gate(config: GateConfig, host: str, port: int) -> GateServer:
    """在后台线程监听，供测试关闭。命令行用 ``serve_gate`` 占住前台。"""
    server = GateServer((host, port), config)
    thread = threading.Thread(target=server.serve_forever, name="masskb-gate", daemon=True)
    thread.start()
    return server


def serve_gate(config: GateConfig, host: str, port: int) -> None:
    """前台运行，直到进程收到中断。关闭时不再等待上游的 120 秒上限。"""
    server = GateServer((host, port), config)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.http.close()
        server._join_workers(SHUTDOWN_JOIN_SECONDS)
