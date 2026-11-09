"""本机 HTTP 的同步客户端。

调用已经启动的 ``masskb serve``。
不导入检索管线，不读取配置，也不启动服务。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx


class MassKBError(Exception):
    """客户端错误的基类。"""


class MassKBConnectionError(MassKBError):
    """连不上服务、超时，或传输层失败。"""


class MassKBResponseError(MassKBError):
    """HTTP 状态不是成功，或正文为空。"""

    def __init__(self, detail: str, *, status_code: int, body: object = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.body = body


class MassKBValidationError(MassKBResponseError):
    """请求被拒绝，HTTP 422。"""


class MassKBServiceError(MassKBResponseError):
    """服务端硬失败，HTTP 503。"""


class MassKBParseError(MassKBError):
    """正文不是约定的 JSON。"""

    def __init__(self, message: str, *, path: str) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class QdrantHealth:
    """健康检查里的 Qdrant 一侧。"""

    ok: bool
    url: str
    collections: tuple[str, ...] | None = None
    error: str | None = None


@dataclass(frozen=True)
class Neo4jHealth:
    """健康检查里的 Neo4j 一侧。"""

    ok: bool
    uri: str
    ping: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class Health:
    """``GET /health`` 的正文。``ok`` 为 false 时调用也不抛错。"""

    ok: bool
    qdrant: QdrantHealth
    neo4j: Neo4jHealth
    openai_key_configured: bool


@dataclass(frozen=True)
class SearchHit:
    """一条命中。字段与服务端 JSON 一致，类型由客户端解析得到。"""

    score: float
    text: str | None
    source: str | None
    chunk_id: str | None
    doc_id: str | None
    index: int | None
    start: int | None
    end: int | None
    prev_chunk_id: str | None
    next_chunk_id: str | None
    dense_score: float | None
    sparse_score: float | None
    dense_rank: int | None
    sparse_rank: int | None
    graph_via: str | None


@dataclass(frozen=True)
class GraphStatus:
    """这次检索有没有补上前后邻。"""

    ok: bool
    skipped: bool
    error: str | None


@dataclass(frozen=True)
class GraphHopStatus:
    """这次有没有沿已有边扩展命中。``found`` 与 ``expanded`` 可以是 null。"""

    ok: bool
    skipped: bool
    error: str | None
    found: int | None
    expanded: int | None
    aliases: bool = False


@dataclass(frozen=True)
class HybridStatus:
    """这次有没有做稀疏腿。"""

    ok: bool
    skipped: bool
    error: str | None
    rrf_k: int | None
    candidate_k: int | None


@dataclass(frozen=True)
class RerankStatus:
    """这次有没有做 listwise 重排。"""

    ok: bool
    skipped: bool
    error: str | None
    model: str | None


@dataclass(frozen=True)
class RewriteStatus:
    """这次有没有改写查询。``text`` 是检索文本，不是用户原文。"""

    ok: bool
    skipped: bool
    text: str | None
    model: str | None
    error: str | None


@dataclass(frozen=True)
class AnswerStatus:
    """一次回答。``citations`` 来自 JSON 数组。"""

    ok: bool
    skipped: bool
    text: str | None
    citations: tuple[str, ...]
    model: str | None
    error: str | None


@dataclass(frozen=True)
class SearchResult:
    """``POST /search`` 的正文。这不是管线里的 dataclass。"""

    query: str
    top_k: int
    collection: str
    embedding_model: str
    doc_id: str | None
    hits: tuple[SearchHit, ...]
    graph: GraphStatus
    mode: str
    score_kind: str
    hybrid: HybridStatus
    answer: AnswerStatus
    rerank: RerankStatus
    rewrite: RewriteStatus
    graph_hops: GraphHopStatus


@dataclass(frozen=True)
class AgentStep:
    """工具循环里的一步。``search`` 与 ``health`` 至少有一个为空。"""

    index: int
    action: str
    ok: bool
    error: str | None
    query: str | None
    search: SearchResult | None
    health: Health | None


@dataclass(frozen=True)
class AskResult:
    """``POST /ask`` 的正文。``query`` 是用户原文。"""

    query: str
    max_steps: int
    model: str | None
    stopped: str
    steps: tuple[AgentStep, ...]
    answer: AnswerStatus
    top_k: int
    doc_id: str | None
    hybrid: bool
    rewrite: bool
    rerank: bool
    graph_hops: bool
    aliases: bool = False


@dataclass(frozen=True)
class IngestChunk:
    """入库响应里的一个分块。字段与切分结果的 JSON 一致。"""

    doc_id: str
    chunk_id: str
    source: str
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class IngestDocument:
    """入库响应里的一个源文件。"""

    source: str
    doc_id: str
    chunks: tuple[IngestChunk, ...]


@dataclass(frozen=True)
class IngestReportView:
    """``report`` 对象。不含 ``file_count`` / ``chunk_count``。"""

    files: tuple[IngestDocument, ...]


@dataclass(frozen=True)
class IngestPreview:
    """``POST /ingest`` 在 ``dry_run`` 为 true 时的正文。"""

    dry_run: bool
    entities: bool
    report: IngestReportView
    aliases: bool = False
    drop_aliases: bool = False


@dataclass(frozen=True)
class IngestWrite:
    """``POST /ingest`` 写入成功的正文。这不是管线里的 dataclass。"""

    report: IngestReportView
    collection: str
    vector_size: int
    points_upserted: int
    graph_documents: int
    graph_chunks: int
    graph_skipped: bool
    entities_skipped: bool
    graph_persons: int
    graph_concepts: int
    graph_mentions: int
    entities_dropped: int
    aliases_skipped: bool = True
    aliases_ok: bool = True
    aliases_error: str | None = None
    graph_aliases: int = 0
    aliases_dropped: int = 0


def _where(path: str) -> str:
    return path or "响应"


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _fail(path: str, message: str) -> MassKBParseError:
    return MassKBParseError(message, path=path)


def _expect_dict(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _fail(path, f"{_where(path)} 必须是对象")
    return value


def _expect_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise _fail(path, f"{path} 必须是数组")
    return value


def _require(data: dict[str, object], key: str, path: str) -> object:
    full = f"{path}.{key}" if path else key
    if key not in data:
        raise _fail(full, f"缺少 {full}")
    return data[key]


def _expect_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(path, f"{path} 必须是布尔值")
    return value


def _expect_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, f"{path} 必须是整数")
    return value


def _expect_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(path, f"{path} 必须是数字")
    return float(value)


def _expect_str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise _fail(path, f"{path} 必须是字符串")
    return value


def _required_bool(data: dict[str, object], key: str, path: str) -> bool:
    full = f"{path}.{key}" if path else key
    return _expect_bool(_require(data, key, path), full)


def _required_int(data: dict[str, object], key: str, path: str) -> int:
    full = f"{path}.{key}" if path else key
    return _expect_int(_require(data, key, path), full)


def _required_str(data: dict[str, object], key: str, path: str) -> str:
    full = f"{path}.{key}" if path else key
    return _expect_str(_require(data, key, path), full)


def _optional_str(data: dict[str, object], key: str, path: str) -> str | None:
    full = f"{path}.{key}" if path else key
    value = _require(data, key, path)
    if value is None:
        return None
    return _expect_str(value, full)


def _optional_int(data: dict[str, object], key: str, path: str) -> int | None:
    full = f"{path}.{key}" if path else key
    value = _require(data, key, path)
    if value is None:
        return None
    return _expect_int(value, full)


def _optional_float(data: dict[str, object], key: str, path: str) -> float | None:
    full = f"{path}.{key}" if path else key
    value = _require(data, key, path)
    if value is None:
        return None
    return _expect_float(value, full)


def _optional_key_str(data: dict[str, object], key: str, path: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    full = f"{path}.{key}" if path else key
    return _expect_str(data[key], full)


def _default_bool(data: dict[str, object], key: str, path: str, default: bool) -> bool:
    """缺字段时用关闭态默认值，便于读取还没有这些字段的旧服务。"""
    if key not in data:
        return default
    full = f"{path}.{key}" if path else key
    return _expect_bool(data[key], full)


def _default_int(data: dict[str, object], key: str, path: str, default: int) -> int:
    if key not in data:
        return default
    full = f"{path}.{key}" if path else key
    return _expect_int(data[key], full)


def _default_optional_str(data: dict[str, object], key: str, path: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    full = f"{path}.{key}" if path else key
    return _expect_str(data[key], full)


def _optional_key_int(data: dict[str, object], key: str, path: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    full = f"{path}.{key}" if path else key
    return _expect_int(data[key], full)


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    items = _expect_list(value, path)
    parsed: list[str] = []
    for index, item in enumerate(items):
        parsed.append(_expect_str(item, f"{path}[{index}]"))
    return tuple(parsed)


def _parse_qdrant(data: dict[str, object], path: str) -> QdrantHealth:
    prefix = _join(path, "qdrant")
    raw = _expect_dict(_require(data, "qdrant", path), prefix)
    collections: tuple[str, ...] | None = None
    if "collections" in raw and raw["collections"] is not None:
        collections = _string_tuple(raw["collections"], f"{prefix}.collections")
    return QdrantHealth(
        ok=_required_bool(raw, "ok", prefix),
        url=_required_str(raw, "url", prefix),
        collections=collections,
        error=_optional_key_str(raw, "error", prefix),
    )


def _parse_neo4j(data: dict[str, object], path: str) -> Neo4jHealth:
    prefix = _join(path, "neo4j")
    raw = _expect_dict(_require(data, "neo4j", path), prefix)
    return Neo4jHealth(
        ok=_required_bool(raw, "ok", prefix),
        uri=_required_str(raw, "uri", prefix),
        ping=_optional_key_int(raw, "ping", prefix),
        error=_optional_key_str(raw, "error", prefix),
    )


def _parse_health(value: object, path: str) -> Health:
    raw = _expect_dict(value, path)
    prefix = path
    configured = _require(raw, "openai_key_configured", prefix)
    configured_path = _join(prefix, "openai_key_configured")
    return Health(
        ok=_required_bool(raw, "ok", prefix),
        qdrant=_parse_qdrant(raw, prefix),
        neo4j=_parse_neo4j(raw, prefix),
        openai_key_configured=_expect_bool(configured, configured_path),
    )


def _parse_graph_hops(data: dict[str, object], path: str) -> GraphHopStatus:
    prefix = _join(path, "graph_hops")
    raw = _expect_dict(_require(data, "graph_hops", path), prefix)
    return GraphHopStatus(
        ok=_required_bool(raw, "ok", prefix),
        skipped=_required_bool(raw, "skipped", prefix),
        error=_optional_str(raw, "error", prefix),
        found=_optional_int(raw, "found", prefix),
        expanded=_optional_int(raw, "expanded", prefix),
        aliases=_default_bool(raw, "aliases", prefix, False),
    )


def _parse_graph(data: dict[str, object], path: str) -> GraphStatus:
    prefix = _join(path, "graph")
    raw = _expect_dict(_require(data, "graph", path), prefix)
    return GraphStatus(
        ok=_required_bool(raw, "ok", prefix),
        skipped=_required_bool(raw, "skipped", prefix),
        error=_optional_str(raw, "error", prefix),
    )


def _parse_hybrid(data: dict[str, object], path: str) -> HybridStatus:
    prefix = _join(path, "hybrid")
    raw = _expect_dict(_require(data, "hybrid", path), prefix)
    return HybridStatus(
        ok=_required_bool(raw, "ok", prefix),
        skipped=_required_bool(raw, "skipped", prefix),
        error=_optional_str(raw, "error", prefix),
        rrf_k=_optional_int(raw, "rrf_k", prefix),
        candidate_k=_optional_int(raw, "candidate_k", prefix),
    )


def _parse_rerank(data: dict[str, object], path: str) -> RerankStatus:
    prefix = _join(path, "rerank")
    raw = _expect_dict(_require(data, "rerank", path), prefix)
    return RerankStatus(
        ok=_required_bool(raw, "ok", prefix),
        skipped=_required_bool(raw, "skipped", prefix),
        error=_optional_str(raw, "error", prefix),
        model=_optional_str(raw, "model", prefix),
    )


def _parse_rewrite(data: dict[str, object], path: str) -> RewriteStatus:
    prefix = _join(path, "rewrite")
    raw = _expect_dict(_require(data, "rewrite", path), prefix)
    return RewriteStatus(
        ok=_required_bool(raw, "ok", prefix),
        skipped=_required_bool(raw, "skipped", prefix),
        text=_optional_str(raw, "text", prefix),
        model=_optional_str(raw, "model", prefix),
        error=_optional_str(raw, "error", prefix),
    )


def _parse_answer(data: dict[str, object], path: str, key: str = "answer") -> AnswerStatus:
    prefix = _join(path, key)
    raw = _expect_dict(_require(data, key, path), prefix)
    return AnswerStatus(
        ok=_required_bool(raw, "ok", prefix),
        skipped=_required_bool(raw, "skipped", prefix),
        text=_optional_str(raw, "text", prefix),
        citations=_string_tuple(_require(raw, "citations", prefix), f"{prefix}.citations"),
        model=_optional_str(raw, "model", prefix),
        error=_optional_str(raw, "error", prefix),
    )


def _parse_hit(value: object, path: str) -> SearchHit:
    raw = _expect_dict(value, path)
    return SearchHit(
        score=_expect_float(_require(raw, "score", path), f"{path}.score"),
        text=_optional_str(raw, "text", path),
        source=_optional_str(raw, "source", path),
        chunk_id=_optional_str(raw, "chunk_id", path),
        doc_id=_optional_str(raw, "doc_id", path),
        index=_optional_int(raw, "index", path),
        start=_optional_int(raw, "start", path),
        end=_optional_int(raw, "end", path),
        prev_chunk_id=_optional_str(raw, "prev_chunk_id", path),
        next_chunk_id=_optional_str(raw, "next_chunk_id", path),
        dense_score=_optional_float(raw, "dense_score", path),
        sparse_score=_optional_float(raw, "sparse_score", path),
        dense_rank=_optional_int(raw, "dense_rank", path),
        sparse_rank=_optional_int(raw, "sparse_rank", path),
        graph_via=_optional_str(raw, "graph_via", path),
    )


def _parse_search(value: object, path: str) -> SearchResult:
    raw = _expect_dict(value, path)
    hits_path = _join(path, "hits")
    hits = _expect_list(_require(raw, "hits", path), hits_path)
    return SearchResult(
        query=_required_str(raw, "query", path),
        top_k=_required_int(raw, "top_k", path),
        collection=_required_str(raw, "collection", path),
        embedding_model=_required_str(raw, "embedding_model", path),
        doc_id=_optional_str(raw, "doc_id", path),
        hits=tuple(_parse_hit(item, f"{hits_path}[{index}]") for index, item in enumerate(hits)),
        graph=_parse_graph(raw, path),
        mode=_required_str(raw, "mode", path),
        score_kind=_required_str(raw, "score_kind", path),
        hybrid=_parse_hybrid(raw, path),
        answer=_parse_answer(raw, path),
        rerank=_parse_rerank(raw, path),
        rewrite=_parse_rewrite(raw, path),
        graph_hops=_parse_graph_hops(raw, path),
    )


def _parse_step(value: object, path: str) -> AgentStep:
    raw = _expect_dict(value, path)
    search = _require(raw, "search", path)
    health = _require(raw, "health", path)
    return AgentStep(
        index=_required_int(raw, "index", path),
        action=_required_str(raw, "action", path),
        ok=_required_bool(raw, "ok", path),
        error=_optional_str(raw, "error", path),
        query=_optional_str(raw, "query", path),
        search=None if search is None else _parse_search(search, f"{path}.search"),
        health=None if health is None else _parse_health(health, f"{path}.health"),
    )


def _parse_ask(value: object) -> AskResult:
    raw = _expect_dict(value, "")
    steps = _expect_list(_require(raw, "steps", ""), "steps")
    return AskResult(
        query=_required_str(raw, "query", ""),
        max_steps=_required_int(raw, "max_steps", ""),
        model=_optional_str(raw, "model", ""),
        stopped=_required_str(raw, "stopped", ""),
        steps=tuple(_parse_step(item, f"steps[{index}]") for index, item in enumerate(steps)),
        answer=_parse_answer(raw, ""),
        top_k=_required_int(raw, "top_k", ""),
        doc_id=_optional_str(raw, "doc_id", ""),
        hybrid=_required_bool(raw, "hybrid", ""),
        rewrite=_required_bool(raw, "rewrite", ""),
        rerank=_required_bool(raw, "rerank", ""),
        graph_hops=_required_bool(raw, "graph_hops", ""),
        aliases=_default_bool(raw, "aliases", "", False),
    )


def _parse_ingest_chunk(value: object, path: str) -> IngestChunk:
    raw = _expect_dict(value, path)
    return IngestChunk(
        doc_id=_required_str(raw, "doc_id", path),
        chunk_id=_required_str(raw, "chunk_id", path),
        source=_required_str(raw, "source", path),
        index=_required_int(raw, "index", path),
        start=_required_int(raw, "start", path),
        end=_required_int(raw, "end", path),
        text=_required_str(raw, "text", path),
    )


def _parse_ingest_report(value: object, path: str) -> IngestReportView:
    raw = _expect_dict(value, path)
    files_path = _join(path, "files")
    files = _expect_list(_require(raw, "files", path), files_path)
    documents: list[IngestDocument] = []
    for index, item in enumerate(files):
        doc_path = f"{files_path}[{index}]"
        doc = _expect_dict(item, doc_path)
        chunks_path = f"{doc_path}.chunks"
        chunks = _expect_list(_require(doc, "chunks", doc_path), chunks_path)
        documents.append(
            IngestDocument(
                source=_required_str(doc, "source", doc_path),
                doc_id=_required_str(doc, "doc_id", doc_path),
                chunks=tuple(
                    _parse_ingest_chunk(chunk, f"{chunks_path}[{chunk_index}]")
                    for chunk_index, chunk in enumerate(chunks)
                ),
            )
        )
    return IngestReportView(files=tuple(documents))


def _parse_ingest_preview(raw: dict[str, object]) -> IngestPreview:
    flag = _require(raw, "dry_run", "")
    if flag is not True:
        raise _fail("dry_run", "dry_run 必须是 true")
    return IngestPreview(
        dry_run=True,
        entities=_required_bool(raw, "entities", ""),
        report=_parse_ingest_report(_require(raw, "report", ""), "report"),
        aliases=_default_bool(raw, "aliases", "", False),
        drop_aliases=_default_bool(raw, "drop_aliases", "", False),
    )


def _parse_ingest_write(raw: dict[str, object]) -> IngestWrite:
    return IngestWrite(
        report=_parse_ingest_report(_require(raw, "report", ""), "report"),
        collection=_required_str(raw, "collection", ""),
        vector_size=_required_int(raw, "vector_size", ""),
        points_upserted=_required_int(raw, "points_upserted", ""),
        graph_documents=_required_int(raw, "graph_documents", ""),
        graph_chunks=_required_int(raw, "graph_chunks", ""),
        graph_skipped=_required_bool(raw, "graph_skipped", ""),
        entities_skipped=_required_bool(raw, "entities_skipped", ""),
        graph_persons=_required_int(raw, "graph_persons", ""),
        graph_concepts=_required_int(raw, "graph_concepts", ""),
        graph_mentions=_required_int(raw, "graph_mentions", ""),
        entities_dropped=_required_int(raw, "entities_dropped", ""),
        aliases_skipped=_default_bool(raw, "aliases_skipped", "", True),
        aliases_ok=_default_bool(raw, "aliases_ok", "", True),
        aliases_error=_default_optional_str(raw, "aliases_error", ""),
        graph_aliases=_default_int(raw, "graph_aliases", "", 0),
        aliases_dropped=_default_int(raw, "aliases_dropped", "", 0),
    )


def _parse_ingest(value: object) -> IngestPreview | IngestWrite:
    raw = _expect_dict(value, "")
    if "dry_run" in raw:
        if raw["dry_run"] is True:
            return _parse_ingest_preview(raw)
        raise _fail("dry_run", "写入结果不能包含 dry_run")
    return _parse_ingest_write(raw)


def _detail_text(body: object) -> str:
    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            parts: list[str] = []
            for item in detail:
                if isinstance(item, dict) and isinstance(item.get("msg"), str):
                    parts.append(item["msg"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            if parts:
                return "; ".join(parts)
            return json.dumps(detail, ensure_ascii=False)
        return json.dumps(detail, ensure_ascii=False)
    if isinstance(body, str):
        return body
    return json.dumps(body, ensure_ascii=False)


def _read_body(response: httpx.Response) -> object:
    if response.content.strip() == b"":
        raise MassKBResponseError("", status_code=response.status_code, body=None)
    try:
        return response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MassKBParseError("响应不是 JSON", path="") from exc


def _raise_for_status(response: httpx.Response, body: object, *, health: bool) -> None:
    code = response.status_code
    if code == 200:
        return
    if health and code == 503 and isinstance(body, dict) and "ok" in body:
        return
    detail = _detail_text(body)
    if code == 422:
        raise MassKBValidationError(detail, status_code=code, body=body)
    if code == 503:
        raise MassKBServiceError(detail, status_code=code, body=body)
    raise MassKBResponseError(detail, status_code=code, body=body)


class MassKB:
    """同步调用 ``GET /health``、``POST /search``、``POST /ask`` 和 ``POST /ingest``。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """``http_client`` 为空时，用 ``base_url`` 与 ``timeout`` 建连。

        自己创建的连接在 ``close()`` 时关闭。传入 ``http_client`` 时直接复用，
        不关闭，也不改它的地址和超时。不跟随重定向。
        """
        if http_client is None:
            self._http = httpx.Client(
                base_url=base_url.rstrip("/"),
                timeout=timeout,
                follow_redirects=False,
            )
            self._owns_client = True
        else:
            self._http = http_client
            self._owns_client = False
        self._closed = False

    def close(self) -> None:
        """关闭自己创建的连接。传入的 ``http_client`` 保持打开。"""
        if not self._owns_client or self._closed:
            return
        self._closed = True
        self._http.close()

    def __enter__(self) -> MassKB:  # noqa: PYI034 - 目标是 Python 3.10，不用 Self
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def health(self) -> Health:
        """``GET /health``。

        库不健康时仍返回对象。只有 ``detail`` 的 503 抛出服务错误。
        """
        response = self._send("GET", "/health", None)
        body = _read_body(response)
        _raise_for_status(response, body, health=True)
        return _parse_health(body, "")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        hybrid: bool = False,
        answer: bool = False,
        rerank: bool = False,
        rewrite: bool = False,
        graph_hops: bool = False,
        aliases: bool = False,
    ) -> SearchResult:
        """``POST /search``。字段与 CLI 相同。

        查询原文不去空格。软失败留在结果里。
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "doc_id": doc_id,
            "hybrid": hybrid,
            "answer": answer,
            "rerank": rerank,
            "rewrite": rewrite,
            "graph_hops": graph_hops,
            "aliases": aliases,
        }
        response = self._send("POST", "/search", payload)
        body = _read_body(response)
        _raise_for_status(response, body, health=False)
        return _parse_search(body, "")

    def ask(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        hybrid: bool = False,
        rewrite: bool = False,
        rerank: bool = False,
        graph_hops: bool = False,
        aliases: bool = False,
        max_steps: int = 3,
    ) -> AskResult:
        """``POST /ask``。不发送 ``answer``。``max_steps`` 默认 3。"""
        payload = {
            "query": query,
            "top_k": top_k,
            "doc_id": doc_id,
            "hybrid": hybrid,
            "rewrite": rewrite,
            "rerank": rerank,
            "graph_hops": graph_hops,
            "aliases": aliases,
            "max_steps": max_steps,
        }
        response = self._send("POST", "/ask", payload)
        body = _read_body(response)
        _raise_for_status(response, body, health=False)
        return _parse_ask(body)

    def ingest(
        self,
        path: str,
        *,
        root: str | None = None,
        chunk_size: int = 800,
        overlap: int = 100,
        dry_run: bool = False,
        entities: bool = False,
        aliases: bool = False,
        drop_aliases: bool = False,
    ) -> IngestPreview | IngestWrite:
        """``POST /ingest``。字段与 CLI 的写入参数相同，不含 ``preview``。

        ``dry_run`` 为真时返回预览，服务端不调用 ``pipeline.ingest``。
        否则返回写入结果。路径原样发送。默认 ``chunk_size`` 为 800，``overlap`` 为 100。
        """
        payload = {
            "path": path,
            "root": root,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "dry_run": dry_run,
            "entities": entities,
            "aliases": aliases,
            "drop_aliases": drop_aliases,
        }
        response = self._send("POST", "/ingest", payload)
        body = _read_body(response)
        _raise_for_status(response, body, health=False)
        return _parse_ingest(body)

    def _send(self, method: str, path: str, payload: dict[str, object] | None) -> httpx.Response:
        try:
            if payload is None:
                return self._http.request(method, path)
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return self._http.request(
                method,
                path,
                content=body,
                headers={"content-type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise MassKBConnectionError(str(exc)) from exc
