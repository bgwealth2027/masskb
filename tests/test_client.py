"""同步客户端：不绑定端口，也不连接 Qdrant、Neo4j 或 OpenAI。"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

import httpx
import pytest
from anyio.from_thread import start_blocking_portal

from masskb.client import (
    IngestPreview,
    IngestWrite,
    MassKB,
    MassKBConnectionError,
    MassKBParseError,
    MassKBResponseError,
    MassKBServiceError,
    MassKBValidationError,
)
from masskb.ingest import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

_Handler = Callable[[httpx.Request], httpx.Response]


def _hit(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "score": 0.5,
        "text": "原文",
        "source": "a.md",
        "chunk_id": "c",
        "doc_id": "d",
        "index": 0,
        "start": 0,
        "end": 2,
        "prev_chunk_id": None,
        "next_chunk_id": "c-next",
        "dense_score": None,
        "sparse_score": None,
        "dense_rank": None,
        "sparse_rank": None,
    }
    payload.update(overrides)
    return payload


def _status_answer(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": True,
        "skipped": True,
        "text": None,
        "citations": [],
        "model": None,
        "error": None,
    }
    payload.update(overrides)
    return payload


def _search_body(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "query": "质量",
        "top_k": 5,
        "collection": "masskb_chunks",
        "embedding_model": "embed-test",
        "doc_id": None,
        "hits": [_hit()],
        "graph": {"ok": True, "skipped": False, "error": None},
        "mode": "dense",
        "score_kind": "cosine",
        "hybrid": {
            "ok": True,
            "skipped": True,
            "error": None,
            "rrf_k": None,
            "candidate_k": None,
        },
        "answer": _status_answer(),
        "rerank": {"ok": True, "skipped": True, "error": None, "model": None},
        "rewrite": {
            "ok": True,
            "skipped": True,
            "text": None,
            "model": None,
            "error": None,
        },
    }
    payload.update(overrides)
    return payload


def _step(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "index": 1,
        "action": "finish",
        "ok": True,
        "error": None,
        "query": None,
        "search": None,
        "health": None,
    }
    payload.update(overrides)
    return payload


def _ask_body(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "query": "质量",
        "max_steps": 3,
        "model": "gpt-test",
        "stopped": "finish",
        "steps": [
            _step(
                action="search",
                query="质量",
                search=_search_body(query="质量"),
            ),
            _step(index=2, action="finish"),
        ],
        "answer": _status_answer(
            ok=True,
            skipped=False,
            text="回答",
            citations=["c"],
            model="gpt-test",
        ),
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "rewrite": False,
        "rerank": False,
    }
    payload.update(overrides)
    return payload


_HEALTH = {
    "ok": True,
    "qdrant": {"ok": True, "url": "http://qdrant.test:6333", "collections": ["docs"]},
    "neo4j": {"ok": True, "uri": "bolt://neo4j.test:7687", "ping": 1},
    "openai_key_configured": False,
}


@contextmanager
def _mock(handler: _Handler) -> Iterator[MassKB]:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    try:
        yield MassKB(http_client=http)
    finally:
        http.close()


def _loads(request: httpx.Request) -> dict[str, object]:
    return json.loads(request.content.decode())


def test_happy_path_requests_and_parsed_results() -> None:
    seen: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        if request.url.path == "/health":
            return httpx.Response(200, json=_HEALTH)
        if request.url.path == "/search":
            return httpx.Response(200, json=_search_body())
        if request.url.path == "/ask":
            return httpx.Response(200, json=_ask_body())
        raise AssertionError(request.url.path)

    with _mock(handler) as kb:
        health = kb.health()
        found = kb.search("质量")
        answered = kb.ask("质量")

    assert seen[0][0:2] == ("GET", "/health")
    assert seen[0][2] == b""
    assert health.ok is True
    assert health.qdrant.collections == ("docs",)
    assert health.neo4j.ping == 1
    assert health.openai_key_configured is False

    search_body = json.loads(seen[1][2].decode())
    assert seen[1][0:2] == ("POST", "/search")
    assert search_body == {
        "query": "质量",
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "answer": False,
        "rerank": False,
        "rewrite": False,
    }
    assert "质量".encode() in seen[1][2]
    assert b"\\u8d28" not in seen[1][2]
    assert found.query == "质量"
    assert found.hits[0].text == "原文"
    assert found.hits[0].next_chunk_id == "c-next"
    assert found.answer.citations == ()
    assert found.rewrite.skipped is True

    ask_body = json.loads(seen[2][2].decode())
    assert seen[2][0:2] == ("POST", "/ask")
    assert ask_body == {
        "query": "质量",
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "rewrite": False,
        "rerank": False,
        "max_steps": 3,
    }
    assert "answer" not in ask_body
    assert answered.stopped == "finish"
    assert answered.answer.text == "回答"
    assert answered.answer.citations == ("c",)
    assert answered.steps[0].search is not None
    assert answered.steps[0].search.hits[0].chunk_id == "c"


def test_search_and_ask_keep_query_and_flags() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_loads(request))
        if request.url.path == "/search":
            return httpx.Response(200, json=_search_body(query=" 质量 ", top_k=3, doc_id="abc"))
        return httpx.Response(200, json=_ask_body(query=" 质量 ", top_k=3, doc_id="abc"))

    with _mock(handler) as kb:
        found = kb.search(
            " 质量 ",
            top_k=3,
            doc_id="abc",
            hybrid=True,
            answer=True,
            rerank=True,
            rewrite=True,
        )
        answered = kb.ask(
            " 质量 ",
            top_k=3,
            doc_id="abc",
            hybrid=True,
            rewrite=True,
            rerank=True,
            max_steps=2,
        )

    assert seen[0] == {
        "query": " 质量 ",
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": True,
        "answer": True,
        "rerank": True,
        "rewrite": True,
    }
    assert found.query == " 质量 "
    assert "answer" not in seen[1]
    assert seen[1] == {
        "query": " 质量 ",
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": True,
        "rewrite": True,
        "rerank": True,
        "max_steps": 2,
    }
    assert answered.query == " 质量 "


def test_unhealthy_payload_is_returned_and_detail_503_raises() -> None:
    degraded = {
        "ok": False,
        "qdrant": {
            "ok": False,
            "url": "http://qdrant.test:6333",
            "error": "ConnectionError: refused",
        },
        "neo4j": {"ok": True, "uri": "bolt://neo4j.test:7687", "ping": 1},
        "openai_key_configured": False,
    }

    def degraded_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json=degraded)

    with _mock(degraded_handler) as kb:
        health = kb.health()

    assert health.ok is False
    assert health.qdrant.error == "ConnectionError: refused"
    assert health.qdrant.collections is None
    assert health.neo4j.ok is True

    def detail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "neo4j down"})

    with _mock(detail_handler) as kb, pytest.raises(MassKBServiceError) as raised:
        kb.health()

    assert raised.value.status_code == 503
    assert raised.value.detail == "neo4j down"
    assert raised.value.body == {"detail": "neo4j down"}


def test_validation_and_service_errors_keep_detail() -> None:
    cases = [
        ("/search", 422, {"detail": "查询不能为空"}, MassKBValidationError),
        (
            "/search",
            422,
            {"detail": [{"msg": "Field required"}, {"msg": "top_k 必须为正整数"}]},
            MassKBValidationError,
        ),
        ("/search", 503, {"detail": "缺少真实 OPENAI_API_KEY"}, MassKBServiceError),
        (
            "/ask",
            422,
            {"detail": "max_steps 必须是 1 到 3 的整数，收到 4"},
            MassKBValidationError,
        ),
        ("/ask", 503, {"detail": "缺少真实 OPENAI_API_KEY"}, MassKBServiceError),
    ]

    for path, status, body, exc_type in cases:
        def handler(
            request: httpx.Request,
            *,
            _status: int = status,
            _body: dict[str, object] = body,
        ) -> httpx.Response:
            return httpx.Response(_status, json=_body)

        with _mock(handler) as kb, pytest.raises(exc_type) as raised:
            if path == "/search":
                kb.search("质量")
            else:
                kb.ask("质量")

        assert isinstance(raised.value, MassKBResponseError)
        assert raised.value.status_code == status
        assert raised.value.body == body
        if isinstance(body["detail"], list):
            assert raised.value.detail == "Field required; top_k 必须为正整数"
        else:
            assert raised.value.detail == body["detail"]


def test_soft_failures_stay_in_the_result() -> None:
    search_body = _search_body(
        hits=[],
        score_kind="cosine",
        graph={"ok": True, "skipped": True, "error": None},
        rewrite={
            "ok": False,
            "skipped": False,
            "text": None,
            "model": "gpt-4o-mini",
            "error": "invalid_json",
        },
    )
    ask_body = _ask_body(
        stopped="max_steps",
        answer=_status_answer(
            ok=False,
            skipped=False,
            text=None,
            citations=[],
            model="gpt-test",
            error="max_steps",
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search":
            return httpx.Response(200, json=search_body)
        return httpx.Response(200, json=ask_body)

    with _mock(handler) as kb:
        found = kb.search("质量")
        answered = kb.ask("质量")

    assert found.hits == ()
    assert found.rewrite.ok is False
    assert found.rewrite.error == "invalid_json"
    assert answered.stopped == "max_steps"
    assert answered.answer.text is None
    assert answered.answer.error == "max_steps"


def test_unknown_keys_are_ignored_and_bad_shape_raises() -> None:
    extra = _search_body()
    extra["future"] = {"keep": True}
    hit = _hit(score=1)
    hit["note"] = "ignore"
    extra["hits"] = [hit]

    def ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=extra)

    with _mock(ok_handler) as kb:
        found = kb.search("质量")

    assert found.hits[0].score == 1.0
    assert not hasattr(found, "future")

    missing = _search_body()
    del missing["hits"]

    def missing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=missing)

    with _mock(missing_handler) as kb, pytest.raises(MassKBParseError) as missing_exc:
        kb.search("质量")

    assert missing_exc.value.path == "hits"

    bad_top_k = _search_body(top_k=True)

    def bool_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bad_top_k)

    with _mock(bool_handler) as kb, pytest.raises(MassKBParseError) as bool_exc:
        kb.search("质量")

    assert bool_exc.value.path == "top_k"


def test_owned_client_closes_and_injected_client_stays_open() -> None:
    kb = MassKB("http://127.0.0.1:9", timeout=0.01)
    with kb:
        pass
    kb.close()
    with pytest.raises(RuntimeError):
        kb.health()

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_HEALTH)

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    injected = MassKB(http_client=http)
    assert injected.health().ok is True
    injected.close()
    assert injected.health().ok is True
    assert calls["n"] == 2
    assert http.is_closed is False
    http.close()


def test_transport_error_is_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with _mock(handler) as kb, pytest.raises(MassKBConnectionError):
        kb.health()


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


class _SyncASGITransport(httpx.BaseTransport):
    """httpx 0.28 的 ASGITransport 只有异步接口。

    同步客户端经这个门户调用它。
    """

    def __init__(self, asgi: httpx.ASGITransport, app: object) -> None:
        self._asgi = asgi
        self._closed = False
        self._portal_cm = start_blocking_portal()
        self._portal = self._portal_cm.__enter__()
        try:
            self._client, self._lifespan = self._portal.call(self._open, app)
        except BaseException:
            self._portal_cm.__exit__(None, None, None)
            raise

    async def _open(self, app: object) -> tuple[httpx.AsyncClient, object]:
        lifespan = app.router.lifespan_context(app)  # type: ignore[attr-defined]
        await lifespan.__aenter__()
        try:
            client = httpx.AsyncClient(transport=self._asgi, base_url="http://test")
            await client.__aenter__()
        except BaseException:
            await lifespan.__aexit__(None, None, None)
            raise
        return client, lifespan

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def once() -> tuple[int, bytes]:
            response = await self._client.request(
                request.method,
                request.url.path,
                content=request.content,
                headers=request.headers,
            )
            return response.status_code, bytes(response.content)

        status, content = self._portal.call(once)
        return httpx.Response(status, content=content)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def shutdown() -> None:
            await self._client.aclose()
            await self._lifespan.__aexit__(None, None, None)  # type: ignore[attr-defined]

        try:
            self._portal.call(shutdown)
        finally:
            self._portal_cm.__exit__(None, None, None)


def _ingest_chunk() -> dict[str, object]:
    return {
        "doc_id": "d",
        "chunk_id": "c",
        "source": "笔记.md",
        "index": 0,
        "start": 0,
        "end": 2,
        "text": "质量",
    }


def _ingest_report() -> dict[str, object]:
    return {"files": [{"source": "笔记.md", "doc_id": "d", "chunks": [_ingest_chunk()]}]}


def _ingest_preview_body() -> dict[str, object]:
    return {"dry_run": True, "entities": True, "report": _ingest_report()}


def _ingest_write_body() -> dict[str, object]:
    return {
        "report": _ingest_report(),
        "collection": "masskb_chunks",
        "vector_size": 1536,
        "points_upserted": 1,
        "graph_documents": 1,
        "graph_chunks": 1,
        "graph_skipped": False,
        "entities_skipped": True,
        "graph_persons": 0,
        "graph_concepts": 0,
        "graph_mentions": 0,
        "entities_dropped": 0,
    }


def test_ingest_request_and_parsed_results() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen.append(body)
        if body["dry_run"] is True:
            assert "笔记".encode() in request.content
            assert b"\\u7b14" not in request.content
            return httpx.Response(200, json=_ingest_preview_body())
        return httpx.Response(200, json=_ingest_write_body())

    with _mock(handler) as kb:
        preview = kb.ingest("笔记.md", root="data", dry_run=True, entities=True)
        written = kb.ingest("a.md")

    keys = {"path", "root", "chunk_size", "overlap", "dry_run", "entities"}
    assert set(seen[0]) == keys
    assert seen[0] == {
        "path": "笔记.md",
        "root": "data",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_CHUNK_OVERLAP,
        "dry_run": True,
        "entities": True,
    }
    assert DEFAULT_CHUNK_SIZE == 800
    assert DEFAULT_CHUNK_OVERLAP == 100
    assert isinstance(preview, IngestPreview)
    assert preview.dry_run is True
    assert preview.entities is True
    assert preview.report.files[0].chunks[0].text == "质量"
    assert set(seen[1]) == keys
    assert seen[1]["dry_run"] is False
    assert seen[1]["root"] is None
    assert seen[1]["entities"] is False
    assert isinstance(written, IngestWrite)
    assert written.collection == "masskb_chunks"
    assert written.points_upserted == 1
    assert written.entities_skipped is True
    assert written.report.files[0].source == "笔记.md"


def test_ingest_parse_errors() -> None:
    missing = _ingest_write_body()
    del missing["collection"]

    def missing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=missing)

    with _mock(missing_handler) as kb, pytest.raises(MassKBParseError) as missing_exc:
        kb.ingest("a.md")

    assert missing_exc.value.path == "collection"

    def false_dry_run(request: httpx.Request) -> httpx.Response:
        body = _ingest_write_body()
        body["dry_run"] = False
        return httpx.Response(200, json=body)

    with _mock(false_dry_run) as kb, pytest.raises(MassKBParseError) as dry_exc:
        kb.ingest("a.md")

    assert dry_exc.value.path == "dry_run"

    def bad_chunk(request: httpx.Request) -> httpx.Response:
        body = _ingest_preview_body()
        report = body["report"]
        assert isinstance(report, dict)
        files = report["files"]
        assert isinstance(files, list)
        first = files[0]
        assert isinstance(first, dict)
        chunks = first["chunks"]
        assert isinstance(chunks, list)
        chunk = chunks[0]
        assert isinstance(chunk, dict)
        chunk["index"] = True
        return httpx.Response(200, json=body)

    with _mock(bad_chunk) as kb, pytest.raises(MassKBParseError) as index_exc:
        kb.ingest("a.md", dry_run=True)

    assert index_exc.value.path == "report.files[0].chunks[0].index"


def test_ingest_validation_and_service_errors() -> None:
    cases = [
        (422, {"detail": "path 不能为空"}, MassKBValidationError),
        (503, {"detail": "缺少真实 OPENAI_API_KEY"}, MassKBServiceError),
    ]
    for status, body, exc_type in cases:

        def handler(
            request: httpx.Request,
            *,
            _status: int = status,
            _body: dict[str, object] = body,
        ) -> httpx.Response:
            return httpx.Response(_status, json=_body)

        with _mock(handler) as kb, pytest.raises(exc_type) as raised:
            kb.ingest("a.md")

        assert raised.value.status_code == status
        assert raised.value.detail == body["detail"]
        assert raised.value.body == body


def test_asgi_round_trip_matches_server_json(tmp_path: Path) -> None:
    from masskb.answer import AnswerStatus
    from masskb.api import create_app
    from masskb.ingest import Chunk, Document, IngestReport, ingest_path
    from masskb.pipeline import (
        AgentStep,
        AskResult,
        GraphStatus,
        IngestWriteResult,
        SearchHit,
        SearchResult,
    )

    hit = SearchHit(
        score=0.5,
        text="原文",
        source="a.md",
        chunk_id="c",
        doc_id="d",
        index=0,
        start=0,
        end=2,
        prev_chunk_id=None,
        next_chunk_id="c-next",
    )
    search = SearchResult(
        query="质量",
        top_k=5,
        collection="masskb_chunks",
        embedding_model="embed-test",
        doc_id=None,
        hits=(hit,),
        graph=GraphStatus(ok=True, skipped=False),
    )
    nested = SearchResult(
        query="质量",
        top_k=3,
        collection="masskb_chunks",
        embedding_model="embed-test",
        doc_id="abc",
        hits=(hit,),
        graph=GraphStatus(ok=True, skipped=False),
    )
    ask = AskResult(
        query="质量",
        max_steps=2,
        model="gpt-test",
        stopped="finish",
        steps=(
            AgentStep(index=1, action="search", ok=True, query="质量", search=nested),
            AgentStep(index=2, action="finish", ok=True),
        ),
        answer=AnswerStatus(
            ok=True,
            skipped=False,
            text="回答",
            citations=("c",),
            model="gpt-test",
        ),
        top_k=3,
        doc_id="abc",
        hybrid=True,
        rewrite=True,
        rerank=True,
    )
    healthy = {
        "ok": True,
        "qdrant": {"ok": True, "url": "http://qdrant.test:6333", "collections": ["docs"]},
        "neo4j": {"ok": True, "uri": "bolt://neo4j.test:7687", "ping": 1},
        "openai_key_configured": False,
    }
    degraded = {
        "ok": False,
        "qdrant": {
            "ok": False,
            "url": "http://qdrant.test:6333",
            "error": "ConnectionError: refused",
        },
        "neo4j": {"ok": True, "uri": "bolt://neo4j.test:7687", "ping": 1},
        "openai_key_configured": False,
    }

    note = tmp_path / "笔记.md"
    note.write_text("质量\n", encoding="utf-8")
    chunk = Chunk(
        doc_id="d",
        chunk_id="c",
        source="笔记.md",
        index=0,
        start=0,
        end=2,
        text="质量",
    )
    written = IngestWriteResult(
        report=IngestReport(files=(Document(source="笔记.md", doc_id="d", chunks=(chunk,)),)),
        collection="masskb_chunks",
        vector_size=1536,
        points_upserted=1,
        graph_documents=1,
        graph_chunks=1,
        graph_skipped=False,
    )

    class Pipeline:
        def __init__(self) -> None:
            self.health_payload = healthy
            self.seen_search: dict[str, object] = {}
            self.seen_ask: dict[str, object] = {}
            self.seen_ingest: dict[str, object] = {}
            self.closed: list[str] = []

        def healthcheck(self) -> dict[str, object]:
            return self.health_payload

        def search(self, query: str, **kwargs: object) -> SearchResult:
            self.seen_search = {"query": query, **kwargs}
            return search

        def ask(self, query: str, **kwargs: object) -> AskResult:
            self.seen_ask = {"query": query, **kwargs}
            return ask

        def ingest(self, path: Path, **kwargs: object) -> IngestWriteResult:
            self.seen_ingest = {"path": path, **kwargs}
            return written

        def close(self) -> None:
            self.closed.append("closed")

    pipeline = Pipeline()
    app = create_app(pipeline)  # type: ignore[arg-type]
    asgi = httpx.ASGITransport(app=app)
    http = httpx.Client(transport=_SyncASGITransport(asgi, app), base_url="http://test")
    try:
        with MassKB(http_client=http) as kb:
            health = kb.health()
            found = kb.search("质量")
            answered = kb.ask(
                "质量",
                top_k=3,
                doc_id="abc",
                hybrid=True,
                rewrite=True,
                rerank=True,
                max_steps=2,
            )
            pipeline.health_payload = degraded
            failed = kb.health()
            ingested = kb.ingest(
                "notes/a.md",
                root="/data",
                chunk_size=40,
                overlap=5,
                entities=True,
            )
            preview = kb.ingest(str(note), root=str(tmp_path), dry_run=True, entities=True)
    finally:
        http.close()

    assert isinstance(asgi, httpx.ASGITransport)
    assert health.ok is True
    assert health.qdrant.collections == ("docs",)
    assert _json_ready(found) == json.loads(json.dumps(asdict(search)))
    assert _json_ready(answered) == json.loads(json.dumps(asdict(ask)))
    assert answered.steps[0].search is not None
    assert answered.steps[0].search.answer.skipped is True
    assert failed.ok is False
    assert failed.qdrant.error == "ConnectionError: refused"
    assert pipeline.seen_search == {
        "query": "质量",
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "answer": False,
        "rerank": False,
        "rewrite": False,
    }
    assert pipeline.seen_ask == {
        "query": "质量",
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": True,
        "rewrite": True,
        "rerank": True,
        "max_steps": 2,
    }
    assert "answer" not in pipeline.seen_ask
    assert isinstance(ingested, IngestWrite)
    assert _json_ready(ingested) == json.loads(json.dumps(asdict(written)))
    assert pipeline.seen_ingest == {
        "path": Path("notes/a.md"),
        "root": Path("/data"),
        "chunk_size": 40,
        "overlap": 5,
        "entities": True,
    }
    preview_report = ingest_path(note, root=tmp_path)
    assert isinstance(preview, IngestPreview)
    assert preview.dry_run is True
    assert preview.entities is True
    assert _json_ready(preview.report) == json.loads(json.dumps(asdict(preview_report)))
    assert pipeline.closed == ["closed"]


def test_import_does_not_load_pipeline() -> None:
    script = """
import sys
from masskb import MassKB, __version__
import masskb.client
assert __version__ == "0.1.0"
assert MassKB is masskb.client.MassKB
forbidden = (
    "masskb.pipeline",
    "masskb.api",
    "masskb.config",
    "masskb.agent",
    "masskb.ingest",
)
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit("loaded " + ",".join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
