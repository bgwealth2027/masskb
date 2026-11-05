"""只读 HTTP：不连接 Qdrant、Neo4j 或 OpenAI，也不绑定端口。"""

from __future__ import annotations

import json
from dataclasses import asdict

from fastapi.testclient import TestClient

from masskb.answer import AnswerStatus
from masskb.api import create_app
from masskb.pipeline import GraphStatus, RerankStatus, RewriteStatus, SearchHit, SearchResult


def _hit() -> SearchHit:
    return SearchHit(
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


def _result(**overrides: object) -> SearchResult:
    payload: dict[str, object] = {
        "query": "质量",
        "top_k": 5,
        "collection": "masskb_chunks",
        "embedding_model": "embed-test",
        "doc_id": None,
        "hits": (_hit(),),
        "graph": GraphStatus(ok=True, skipped=False),
    }
    payload.update(overrides)
    return SearchResult(**payload)  # type: ignore[arg-type]


class _Pipeline:
    def __init__(self, result: SearchResult | None = None) -> None:
        self.result = result or _result()
        self.seen: dict[str, object] = {}
        self.calls: list[str] = []
        self.closed: list[str] = []
        self.health_payload: dict[str, object] = {
            "ok": True,
            "qdrant": {"ok": True, "url": "http://qdrant.test:6333", "collections": ["docs"]},
            "neo4j": {"ok": True, "uri": "bolt://neo4j.test:7687", "ping": 1},
            "openai_key_configured": False,
        }
        self.health_error: Exception | None = None
        self.search_error: Exception | None = None

    def healthcheck(self) -> dict[str, object]:
        self.calls.append("health")
        if self.health_error is not None:
            raise self.health_error
        return self.health_payload

    def search(self, query: str, **kwargs: object) -> SearchResult:
        self.calls.append("search")
        self.seen = {"query": query, **kwargs}
        if self.search_error is not None:
            raise self.search_error
        return self.result

    def close(self) -> None:
        self.closed.append("closed")


def test_post_search_matches_cli_json() -> None:
    result = _result()
    pipeline = _Pipeline(result)
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/search", json={"query": "质量"})

    assert response.status_code == 200
    assert response.content == json.dumps(asdict(result), ensure_ascii=False).encode("utf-8")
    assert "质量".encode() in response.content
    assert b"\\u8d28" not in response.content
    assert pipeline.seen == {
        "query": "质量",
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "answer": False,
        "rerank": False,
        "rewrite": False,
    }


def test_post_search_forwards_flags_without_trimming_query() -> None:
    pipeline = _Pipeline(_result(query=" 质量 ", top_k=3, doc_id="abc"))
    body = {
        "query": " 质量 ",
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": True,
        "answer": True,
        "rerank": True,
        "rewrite": True,
    }
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/search", json=body)

    assert response.status_code == 200
    assert pipeline.seen == {
        "query": " 质量 ",
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": True,
        "answer": True,
        "rerank": True,
        "rewrite": True,
    }
    assert response.json()["query"] == " 质量 "
    assert response.json()["doc_id"] == "abc"


def test_post_search_soft_failures_stay_200() -> None:
    result = _result(
        score_kind="cosine",
        answer=AnswerStatus(
            ok=False,
            skipped=False,
            text=None,
            citations=(),
            model="gpt-4o-mini",
            error="uncited",
        ),
        rerank=RerankStatus(
            ok=False,
            skipped=False,
            error="invalid_json",
            model="gpt-4o-mini",
        ),
        rewrite=RewriteStatus(
            ok=False,
            skipped=False,
            text=None,
            model="gpt-4o-mini",
            error="invalid_json",
        ),
    )
    pipeline = _Pipeline(result)
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/search", json={"query": "质量"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["hits"][0]["chunk_id"] == "c"
    assert payload["hits"][0]["score"] == 0.5
    assert payload["score_kind"] == "cosine"
    assert payload["answer"]["ok"] is False
    assert payload["answer"]["error"] == "uncited"
    assert payload["rerank"]["ok"] is False
    assert payload["rerank"]["error"] == "invalid_json"
    assert payload["rewrite"]["ok"] is False
    assert payload["rewrite"]["error"] == "invalid_json"
    assert payload["rewrite"]["text"] is None


def test_post_search_empty_hits() -> None:
    result = _result(hits=(), graph=GraphStatus(ok=True, skipped=True), doc_id=None)
    pipeline = _Pipeline(result)
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/search", json={"query": "质量", "doc_id": None})

    assert response.status_code == 200
    payload = response.json()
    assert payload["hits"] == []
    assert payload["doc_id"] is None
    assert payload["top_k"] == 5
    assert payload["graph"] == {"ok": True, "skipped": True, "error": None}


def test_search_validation_does_not_call_pipeline() -> None:
    pipeline = _Pipeline()
    bodies = [
        {},
        {"query": "   "},
        {"query": "质量", "top_k": 0},
        {"query": "质量", "top_k": True},
        {"query": "质量", "doc_id": ""},
        {"query": "质量", "doc_id": "   "},
        {"query": "质量", "entities": True},
    ]
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        statuses = [client.post("/search", json=body).status_code for body in bodies]

    assert statuses == [422, 422, 422, 422, 422, 422, 422]
    assert pipeline.calls == []


def test_search_value_error_is_422() -> None:
    pipeline = _Pipeline()
    pipeline.search_error = ValueError("查询不能为空")
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/search", json={"query": "质量"})

    assert response.status_code == 422
    assert response.json() == {"detail": "查询不能为空"}


def test_search_hard_failure_is_503_and_closes_once() -> None:
    pipeline = _Pipeline()
    pipeline.search_error = RuntimeError("缺少真实 OPENAI_API_KEY")
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/search", json={"query": "质量"})
        assert response.status_code == 503
        assert response.json() == {"detail": "缺少真实 OPENAI_API_KEY"}
    assert pipeline.closed == ["closed"]


def test_health_ok_and_failure() -> None:
    pipeline = _Pipeline()
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        ok = client.get("/health")
        assert ok.status_code == 200
        assert ok.json() == pipeline.health_payload
        pipeline.health_payload = {
            "ok": False,
            "qdrant": {
                "ok": False,
                "url": "http://qdrant.test:6333",
                "error": "ConnectionError: refused",
            },
            "neo4j": {"ok": True, "uri": "bolt://neo4j.test:7687", "ping": 1},
            "openai_key_configured": False,
        }
        failed = client.get("/health")

    assert failed.status_code == 503
    assert failed.json()["ok"] is False
    assert failed.json()["qdrant"]["error"] == "ConnectionError: refused"
    assert "search" not in pipeline.calls


def test_health_exception_is_503() -> None:
    pipeline = _Pipeline()
    pipeline.health_error = RuntimeError("neo4j down")
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"detail": "neo4j down"}


def test_no_ingest_route_and_method_mismatch() -> None:
    pipeline = _Pipeline()
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        assert client.post("/ingest").status_code == 404
        assert client.get("/search").status_code == 405
        assert client.post("/health").status_code == 405
        schema = client.get("/openapi.json").json()

    assert "/ingest" not in schema["paths"]
    assert {"get"} == set(schema["paths"]["/health"].keys())
    assert {"post"} == set(schema["paths"]["/search"].keys())
    assert pipeline.calls == []


def test_module_app_lifespan_constructs_and_closes(monkeypatch) -> None:
    created: list[str] = []
    closed: list[str] = []

    class FakePipeline:
        def __init__(self) -> None:
            created.append("init")

        def healthcheck(self) -> dict[str, object]:
            return {
                "ok": True,
                "qdrant": {"ok": True, "url": "http://qdrant.test:6333", "collections": []},
                "neo4j": {"ok": True, "uri": "bolt://neo4j.test:7687", "ping": 1},
                "openai_key_configured": False,
            }

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("masskb.api.RetrievalPipeline", FakePipeline)
    from masskb.api import app

    assert created == []
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert created == ["init"]
    assert closed == ["closed"]
