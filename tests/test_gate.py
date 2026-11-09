"""只读网关。上游是本机假 HTTP，不连接 Qdrant、Neo4j 或模型。"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from masskb.__main__ import main
from masskb.api import _PAGE_HEADERS, create_app, query_page_html
from masskb.client import MassKB, MassKBParseError
from masskb.gate import (
    PUBLIC_SAMPLE_SOURCES,
    GateConfig,
    GateError,
    bind_gate,
    normalize_upstream,
    public_search_hits,
)

ROOT = Path(__file__).resolve().parent.parent
_CHUNK = "ab" * 32
_PRIVATE = "机密正文不应出现"
_GRAPH_ERROR = "ServiceUnavailable: bolt://secret.internal:7687 /var/lib/masskb/private.md"


class _Upstream:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.hold = threading.Event()
        self.hold.set()
        self.health_status = 200
        self.search_status = 200
        self.health_body: dict[str, Any] = {
            "ok": True,
            "qdrant": {
                "ok": True,
                "url": "http://qdrant.internal:6333",
                "collections": ["private_docs"],
            },
            "neo4j": {
                "ok": True,
                "uri": "bolt://neo4j.internal:7687",
                "ping": 1,
                "error": "masskb-dev-pass",
            },
            "openai_key_configured": True,
        }
        self.search_body: dict[str, Any] = {
            "query": "被改写的查询",
            "top_k": 5,
            "collection": "masskb_chunks",
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "doc_id": None,
            "hits": [
                {
                    "score": 0.9,
                    "text": _PRIVATE,
                    "source": "/tmp/secret/notes.md",
                    "chunk_id": _CHUNK,
                    "doc_id": "private-doc",
                },
                {
                    "score": 0.2,
                    "text": "青苹果长在虚构果园的东边。",
                    "source": "qingpingguo.md",
                    "chunk_id": _CHUNK,
                    "doc_id": "sample-doc",
                    "prev_chunk_id": "prev",
                    "next_chunk_id": "next",
                },
            ],
            "graph": {"ok": False, "skipped": False, "error": _GRAPH_ERROR},
            "mode": "dense",
            "score_kind": "cosine",
            "hybrid": {"ok": False, "skipped": False, "error": "bm25 exploded"},
            "answer": {
                "ok": False,
                "skipped": True,
                "text": None,
                "citations": [],
                "model": None,
                "error": None,
            },
            "rerank": {"ok": True, "skipped": True, "error": None, "model": "gpt-4o-mini"},
            "rewrite": {
                "ok": False,
                "skipped": False,
                "text": "改写泄漏",
                "model": "gpt-4o-mini",
                "error": "rewrite failed at /opt/data",
            },
            "graph_hops": {
                "ok": True,
                "skipped": True,
                "error": None,
                "found": None,
                "expanded": None,
            },
        }

    def record(self, method: str, path: str, headers: dict[str, str], body: bytes) -> None:
        with self.lock:
            self.requests.append(
                {"method": method, "path": path, "headers": headers, "body": body}
            )

    def count(self, path: str) -> int:
        with self.lock:
            return sum(1 for item in self.requests if item["path"] == path)


def _handler(state: _Upstream) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            self._take("GET", b"")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self._take("POST", self.rfile.read(length))

        def _take(self, method: str, body: bytes) -> None:
            headers = {key.lower(): value for key, value in self.headers.items()}
            state.record(method, self.path, headers, body)
            if not state.hold.wait(5):
                self._send(504, {"detail": "held"})
                return
            if method == "GET" and self.path == "/health":
                self._send(state.health_status, state.health_body)
                return
            if method == "POST" and self.path == "/search":
                self._send(state.search_status, state.search_body)
                return
            self._send(404, {"error": f"unexpected {method} {self.path}"})

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def _start_upstream(state: _Upstream) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _config(upstream: str, **overrides: object) -> GateConfig:
    values: dict[str, object] = {
        "upstream": upstream,
        "client_wait_seconds": 2,
        "upstream_timeout_seconds": 3,
        "admission_limit": 2,
    }
    values.update(overrides)
    return GateConfig(**values)  # type: ignore[arg-type]


def _page_body(query: str = "青苹果", **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "query": query,
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "answer": False,
        "rerank": False,
        "rewrite": False,
        "graph_hops": False,
        "aliases": False,
    }
    body.update(overrides)
    return body


def _raw(port: int, payload: bytes) -> bytes:
    sock = socket.create_connection(("127.0.0.1", port), timeout=3)
    try:
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            piece = sock.recv(65536)
            if not piece:
                break
            chunks.append(piece)
    finally:
        sock.close()
    return b"".join(chunks)


def _status_and_json(raw: bytes) -> tuple[int, dict[str, Any]]:
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split()[1])
    return status, json.loads(body.decode())


def _request_bytes(
    port: int,
    method: str,
    target: str,
    body: bytes | None = None,
    host: str | None = None,
    extra: str = "",
) -> bytes:
    host_header = host if host is not None else f"127.0.0.1:{port}"
    lines = [
        f"{method} {target} HTTP/1.1",
        f"Host: {host_header}",
        "Accept: application/json",
        "Connection: close",
    ]
    if body is not None:
        lines.append("Content-Type: application/json")
        lines.append(f"Content-Length: {len(body)}")
    if extra:
        lines.append(extra)
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode()
    if body is not None:
        raw += body
    return _raw(port, raw)


def _open_stack(
    state: _Upstream,
    **overrides: object,
) -> tuple[ThreadingHTTPServer, Any, int]:
    upstream = _start_upstream(state)
    host, port = upstream.server_address[:2]
    gate = bind_gate(_config(f"http://{host}:{port}", **overrides), "127.0.0.1", 0)
    return upstream, gate, gate.server_address[1]


@pytest.fixture
def stack():
    state = _Upstream()
    upstream, gate, gate_port = _open_stack(state)
    try:
        yield state, gate_port
    finally:
        state.hold.set()
        gate.finish()
        upstream.shutdown()
        upstream.server_close()


def test_sample_files_are_the_allowlist_and_have_no_secrets() -> None:
    directory = ROOT / "examples" / "public"
    names = {path.name for path in directory.iterdir() if path.is_file()}
    assert names == set(PUBLIC_SAMPLE_SOURCES)
    for path in directory.iterdir():
        text = path.read_text(encoding="utf-8")
        for marker in ("sk-", "masskb-dev-pass", "BEGIN ", "AKIA", "/tmp/", "password"):
            assert marker not in text


def test_upstream_rejects_userinfo_path_and_missing_port() -> None:
    with pytest.raises(GateError):
        normalize_upstream("http://user:secret@127.0.0.1:8080")
    with pytest.raises(GateError):
        normalize_upstream("http://127.0.0.1:8080/search")
    with pytest.raises(GateError):
        normalize_upstream("http://127.0.0.1/health")
    with pytest.raises(GateError):
        GateConfig(allowed_hosts=frozenset({"*.example.com"}))


def test_serve_routes_stay_intact() -> None:
    paths = {getattr(route, "path", "") for route in create_app().routes}
    assert {"/", "/health", "/search", "/ask", "/ingest"} <= paths


def test_gate_help_does_not_change_serve_defaults(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["gate", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--allow-host" in text
    assert "--upstream" in text
    assert "NEO4J_DEV_PASSWORD" not in text
    assert main(["gate", "--allow-host", "bad host"]) == 1


def test_public_page_matches_serve_page(stack) -> None:
    _state, port = stack
    raw = _request_bytes(port, "GET", "/")
    head, _, body = raw.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200")
    folded = head.decode().lower()
    assert "content-security-policy" in folded
    assert "connect-src 'self'" in folded
    assert body.decode() == query_page_html()
    assert _PAGE_HEADERS["Content-Security-Policy"] in head.decode()


def test_health_and_search_are_whitelisted(stack) -> None:
    state, port = stack
    status, health = _status_and_json(_request_bytes(port, "GET", "/health"))
    assert status == 200
    assert health == {"ok": True, "openai_key_configured": False}
    rendered_health = json.dumps(health, ensure_ascii=False)
    for secret in ("qdrant.internal", "neo4j.internal", "private_docs", "masskb-dev-pass"):
        assert secret not in rendered_health

    body = json.dumps(_page_body(), ensure_ascii=False).encode()
    status, found = _status_and_json(
        _request_bytes(
            port,
            "POST",
            "/search",
            body,
            extra="Authorization: Bearer super-secret-token",
        )
    )
    assert status == 200
    assert set(found) == {"query", "hits"}
    assert found["query"] == "青苹果"
    assert found["hits"] == [
        {
            "source": "qingpingguo.md",
            "chunk_id": _CHUNK,
            "text": "青苹果长在虚构果园的东边。",
        }
    ]
    encoded = json.dumps(found, ensure_ascii=False)
    for secret in (
        _PRIVATE,
        _GRAPH_ERROR,
        "bm25 exploded",
        "改写泄漏",
        "/tmp/secret",
        "gpt-4o-mini",
        "super-secret-token",
    ):
        assert secret not in encoded
    assert state.count("/search") == 1
    recorded = state.requests[-1]
    assert recorded["path"] == "/search"
    forwarded = json.loads(recorded["body"].decode())
    assert forwarded["answer"] is False
    assert forwarded["query"] == "青苹果"
    headers = recorded["headers"]
    assert headers["host"].startswith("127.0.0.1:")
    for banned in ("authorization", "cookie", "x-forwarded-for", "x-api-key"):
        assert banned not in headers


def test_sdk_cannot_treat_gate_health_as_serve_health(stack) -> None:
    _state, port = stack
    client = MassKB(f"http://127.0.0.1:{port}", timeout=2)
    try:
        with pytest.raises(MassKBParseError):
            client.health()
    finally:
        client.close()


def test_private_hit_with_bad_chunk_id_is_dropped() -> None:
    hits = public_search_hits(
        {
            "hits": [
                {"source": "qingpingguo.md", "chunk_id": "not-a-hash", "text": "青苹果"},
                {"source": "other.md", "chunk_id": _CHUNK, "text": _PRIVATE},
            ]
        },
        PUBLIC_SAMPLE_SOURCES,
    )
    assert hits == []


def test_forbidden_routes_and_flags_do_not_reach_upstream(stack) -> None:
    state, port = stack
    probes = [
        ("POST", "/ingest", _page_body()),
        ("POST", "/ask", _page_body()),
        ("GET", "/openapi.json", None),
        ("GET", "/docs", None),
        ("GET", "/redoc", None),
        ("POST", "/v1/embeddings", {"input": "x"}),
        ("GET", "/search/../ingest", None),
        ("POST", "//ingest", _page_body()),
        ("GET", "/search%2f..%2fingest", None),
        ("POST", "/health", _page_body()),
        ("GET", "/search", None),
        ("POST", "http://127.0.0.1/ingest", _page_body()),
    ]
    for method, target, payload in probes:
        body = None if payload is None else json.dumps(payload).encode()
        raw = _request_bytes(port, method, target, body)
        status = int(raw.split()[1])
        assert status in {403, 404, 405}, (method, target, status)
        assert _PRIVATE.encode() not in raw
    answer = json.dumps(_page_body(answer=True)).encode()
    status, payload = _status_and_json(_request_bytes(port, "POST", "/search", answer))
    assert status == 403
    assert payload == {"detail": "不允许"}
    assert state.requests == []


def test_rejected_host_does_not_call_upstream(stack) -> None:
    state, port = stack
    status, payload = _status_and_json(
        _request_bytes(port, "GET", "/health", host="query.example")
    )
    assert status == 403
    assert payload == {"detail": "不允许"}
    assert state.requests == []
    assert int(_request_bytes(port, "GET", "/health", host="127.0.0.1.evil").split()[1]) == 403


def test_allow_host_adds_a_future_name_without_removing_loopback() -> None:
    state = _Upstream()
    upstream, gate, gate_port = _open_stack(
        state,
        allowed_hosts=frozenset({"localhost", "127.0.0.1", "::1", "query.example"}),
    )
    try:
        allowed = _request_bytes(gate_port, "GET", "/health", host="query.example:443")
        assert int(allowed.split()[1]) == 200
        still = _request_bytes(gate_port, "GET", "/health", host="127.0.0.1")
        assert int(still.split()[1]) == 200
        other = _request_bytes(gate_port, "GET", "/health", host="query.example.evil")
        assert int(other.split()[1]) == 403
    finally:
        state.hold.set()
        gate.finish()
        upstream.shutdown()
        upstream.server_close()


def test_oversize_and_long_query_do_not_reach_upstream(stack) -> None:
    state, port = stack
    header = (
        f"POST /search HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        "Content-Type: application/json\r\nContent-Length: 5000\r\nConnection: close\r\n\r\n"
    ).encode()
    status, payload = _status_and_json(_raw(port, header))
    assert status == 413
    assert payload == {"detail": "请求过大"}
    long_query = json.dumps(_page_body("果" * 201)).encode()
    status, payload = _status_and_json(_request_bytes(port, "POST", "/search", long_query))
    assert status == 422
    assert payload == {"detail": "请求无效"}
    assert state.requests == []


def test_timeout_does_not_free_the_slot_or_start_another_upstream() -> None:
    state = _Upstream()
    state.hold.clear()
    upstream, gate, port = _open_stack(
        state,
        admission_limit=1,
        client_wait_seconds=0.3,
        upstream_timeout_seconds=3,
    )
    try:
        first = _request_bytes(port, "POST", "/search", json.dumps(_page_body()).encode())
        status, payload = _status_and_json(first)
        assert status == 504
        assert payload == {"detail": "检索超时"}
        assert state.count("/search") == 1
        blocked = _request_bytes(port, "POST", "/search", json.dumps(_page_body("蓝河流")).encode())
        blocked_status, blocked_body = _status_and_json(blocked)
        assert blocked_status == 429
        assert blocked_body == {"detail": "服务正忙"}
        assert state.count("/search") == 1
        assert gate.admission.inflight == 1
        state.hold.set()
        gate.finish()
        gate = None
        assert state.count("/search") == 1
        again_upstream, again_gate, again_port = _open_stack(state)
        try:
            status, found = _status_and_json(
                _request_bytes(again_port, "POST", "/search", json.dumps(_page_body()).encode())
            )
            assert status == 200
            assert found["hits"][0]["source"] == "qingpingguo.md"
        finally:
            again_gate.finish()
            again_upstream.shutdown()
            again_upstream.server_close()
    finally:
        state.hold.set()
        if gate is not None:
            gate.finish()
        upstream.shutdown()
        upstream.server_close()


def test_closed_client_does_not_free_the_slot() -> None:
    state = _Upstream()
    state.hold.clear()
    upstream, gate, port = _open_stack(
        state,
        admission_limit=1,
        client_wait_seconds=2,
        upstream_timeout_seconds=3,
    )
    try:
        body = json.dumps(_page_body()).encode()
        request = (
            f"POST /search HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode() + body
        sock = socket.create_connection(("127.0.0.1", port), timeout=3)
        sock.sendall(request)
        sock.close()
        deadline = threading.Event()
        for _ in range(50):
            if state.count("/search") == 1:
                deadline.set()
                break
            threading.Event().wait(0.05)
        assert deadline.is_set()
        status, payload = _status_and_json(
            _request_bytes(port, "POST", "/search", json.dumps(_page_body("蓝河流")).encode())
        )
        assert status == 429
        assert payload == {"detail": "服务正忙"}
        assert state.count("/search") == 1
        assert gate.admission.inflight == 1
    finally:
        state.hold.set()
        gate.finish()
        upstream.shutdown()
        upstream.server_close()
