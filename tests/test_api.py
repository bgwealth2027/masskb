"""本机 HTTP：不连接 Qdrant、Neo4j 或 OpenAI，也不绑定端口。TestClient 不执行查询页脚本。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi.testclient import TestClient

from masskb.answer import AnswerStatus
from masskb.api import create_app
from masskb.ingest import IngestError, ingest_path
from masskb.pipeline import (
    AgentStep,
    AskResult,
    GraphStatus,
    IngestWriteResult,
    RerankStatus,
    RewriteStatus,
    SearchHit,
    SearchResult,
)


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
        self.ask_error: Exception | None = None
        self.ingest_error: Exception | None = None
        self.ingest_result: IngestWriteResult | None = None
        self.ask_result = _ask_result()

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

    def ask(self, query: str, **kwargs: object) -> AskResult:
        self.calls.append("ask")
        self.seen = {"query": query, **kwargs}
        if self.ask_error is not None:
            raise self.ask_error
        return self.ask_result

    def ingest(self, path: Path, **kwargs: object) -> IngestWriteResult:
        self.calls.append("ingest")
        self.seen = {"path": path, **kwargs}
        if self.ingest_error is not None:
            raise self.ingest_error
        assert self.ingest_result is not None
        return self.ingest_result

    def close(self) -> None:
        self.closed.append("closed")


def _ask_result(**overrides: object) -> AskResult:
    payload: dict[str, object] = {
        "query": "质量",
        "max_steps": 3,
        "model": "gpt-test",
        "stopped": "finish",
        "steps": (AgentStep(index=1, action="finish", ok=True),),
        "answer": AnswerStatus(
            ok=True,
            skipped=False,
            text="回答",
            citations=("c",),
            model="gpt-test",
        ),
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "rewrite": False,
        "rerank": False,
    }
    payload.update(overrides)
    return AskResult(**payload)  # type: ignore[arg-type]


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
        "graph_hops": False,
        "aliases": False,
    }
    assert "steps" not in response.json()
    assert "stopped" not in response.json()


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
        "graph_hops": True,
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
        "graph_hops": True,
        "aliases": False,
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
        {"query": "质量", "agent": True},
    ]
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        statuses = [client.post("/search", json=body).status_code for body in bodies]

    assert statuses == [422, 422, 422, 422, 422, 422, 422, 422]
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


def test_get_root_is_the_packaged_query_page() -> None:
    from importlib.resources import files

    pipeline = _Pipeline()
    packaged = files("masskb").joinpath("static", "index.html").read_text(encoding="utf-8")
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == packaged
    assert 'id="query"' in response.text
    assert 'id="answer" type="checkbox"' in response.text
    assert "type=\"checkbox\" checked" not in response.text
    assert "textContent" in response.text
    assert "没有命中" in response.text
    assert "正在检索" in response.text
    assert "answer.ok !== true" in response.text
    assert "回答失败：" in response.text
    assert 'answer.error === "no_hits"' in response.text
    assert "没有可引用的命中，未生成回答。" in response.text
    assert ': "未生成回答。"' in response.text
    assert "doc_id: null" in response.text
    assert "rerank: false" in response.text
    assert "rewrite: false" in response.text
    assert "graph_hops: false" in response.text
    assert "aliases: false" in response.text
    assert "openai_key_configured === true" in response.text
    assert "配置存在不表示回答会成功" in response.text
    assert 'fetch("/search"' in response.text
    assert 'fetch("/health"' in response.text
    assert "answer: wantAnswer" in response.text
    assert "hybrid: false" in response.text
    assert "top_k: 5" in response.text
    policy = response.headers["content-security-policy"]
    assert "connect-src 'self'" in policy
    assert "img-src 'none'" in policy
    for forbidden in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "/ingest",
        "/ask",
        "sk-",
        "OPENAI_API_KEY",
    ):
        assert forbidden not in response.text
    assert pipeline.calls == []


def test_ingest_route_methods_and_openapi() -> None:
    pipeline = _Pipeline()
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        assert client.get("/ingest").status_code == 405
        assert client.post("/ingest").status_code == 422
        assert client.get("/search").status_code == 405
        assert client.post("/health").status_code == 405
        schema = client.get("/openapi.json").json()

    assert {"post"} == set(schema["paths"]["/ingest"].keys())
    assert {"get"} == set(schema["paths"]["/health"].keys())
    assert {"post"} == set(schema["paths"]["/search"].keys())
    assert {"post"} == set(schema["paths"]["/ask"].keys())
    assert {"get"} == set(schema["paths"]["/"].keys())
    assert pipeline.calls == []


def _written(report: object) -> IngestWriteResult:
    return IngestWriteResult(
        report=report,  # type: ignore[arg-type]
        collection="masskb_chunks",
        vector_size=1536,
        points_upserted=1,
        graph_documents=1,
        graph_chunks=1,
        graph_skipped=False,
        entities_skipped=True,
    )


def test_post_ingest_write_matches_asdict(tmp_path: Path) -> None:
    target = tmp_path / "笔记.md"
    target.write_text("质量", encoding="utf-8")
    report = ingest_path(target, root=tmp_path)
    result = _written(report)
    pipeline = _Pipeline()
    pipeline.ingest_result = result
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ingest", json={"path": str(target), "root": str(tmp_path)})

    assert response.status_code == 200
    assert response.content == json.dumps(asdict(result), ensure_ascii=False).encode("utf-8")
    assert "质量".encode() in response.content
    assert b"\\u8d28" not in response.content
    assert "dry_run" not in response.json()
    assert pipeline.calls == ["ingest"]
    assert pipeline.seen == {
        "path": Path(str(target)).expanduser(),
        "root": Path(str(tmp_path)).expanduser(),
        "chunk_size": 800,
        "overlap": 100,
        "entities": False,
        "aliases": False,
        "drop_aliases": False,
    }


def test_post_ingest_forwards_flags(tmp_path: Path) -> None:
    target = tmp_path / "a.md"
    target.write_text("正文", encoding="utf-8")
    pipeline = _Pipeline()
    pipeline.ingest_result = _written(ingest_path(target, root=tmp_path, chunk_size=40, overlap=5))
    body = {
        "path": str(target),
        "root": str(tmp_path),
        "chunk_size": 40,
        "overlap": 5,
        "dry_run": False,
        "entities": True,
    }
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ingest", json=body)

    assert response.status_code == 200
    assert pipeline.calls == ["ingest"]
    assert pipeline.seen == {
        "path": Path(str(target)).expanduser(),
        "root": Path(str(tmp_path)).expanduser(),
        "chunk_size": 40,
        "overlap": 5,
        "entities": True,
        "aliases": False,
        "drop_aliases": False,
    }


def test_post_ingest_dry_run_does_not_call_pipeline(tmp_path: Path) -> None:
    target = tmp_path / "笔记.md"
    target.write_text("质量\n", encoding="utf-8")
    pipeline = _Pipeline()
    body = {
        "path": str(target),
        "root": str(tmp_path),
        "dry_run": True,
        "entities": True,
    }
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ingest", json=body)

    report = ingest_path(target, root=tmp_path)
    expected = {
        "dry_run": True,
        "entities": True,
        "aliases": False,
        "drop_aliases": False,
        "report": asdict(report),
    }
    assert response.status_code == 200
    assert response.content == json.dumps(expected, ensure_ascii=False).encode("utf-8")
    assert "质量".encode() in response.content
    assert b"\\u8d28" not in response.content
    assert "collection" not in response.json()
    assert pipeline.calls == []


def test_ingest_dry_run_error_is_422_without_pipeline(tmp_path: Path) -> None:
    pipeline = _Pipeline()
    missing = tmp_path / "missing.md"
    bad = tmp_path / "a.py"
    bad.write_text("x", encoding="utf-8")
    ok = tmp_path / "ok.md"
    ok.write_text("质量", encoding="utf-8")
    bodies = [
        {"path": str(missing), "dry_run": True},
        {"path": str(bad), "dry_run": True, "entities": True},
        {"path": str(ok), "dry_run": True, "chunk_size": 8, "overlap": 8},
    ]
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        statuses = [client.post("/ingest", json=body).status_code for body in bodies]

    assert statuses == [422, 422, 422]
    assert pipeline.calls == []


def test_ingest_validation_does_not_call_pipeline() -> None:
    pipeline = _Pipeline()
    bodies = [
        {},
        {"path": ""},
        {"path": "   "},
        {"path": "a.md", "chunk_size": 0},
        {"path": "a.md", "chunk_size": True},
        {"path": "a.md", "overlap": -1},
        {"path": "a.md", "overlap": True},
        {"path": "a.md", "root": ""},
        {"path": "a.md", "root": "   "},
        {"path": "a.md", "preview": 3},
        {"path": "a.md", "agent": True},
    ]
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        statuses = [client.post("/ingest", json=body).status_code for body in bodies]

    assert statuses == [422] * len(bodies)
    assert pipeline.calls == []


def test_ingest_value_error_is_422() -> None:
    pipeline = _Pipeline()
    pipeline.ingest_error = IngestError("路径不存在: a.md")
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ingest", json={"path": "a.md"})

    assert response.status_code == 422
    assert response.json() == {"detail": "路径不存在: a.md"}


def test_ingest_hard_failure_is_503_and_releases_lock() -> None:
    pipeline = _Pipeline()
    pipeline.ingest_error = RuntimeError("缺少真实 OPENAI_API_KEY")
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ingest", json={"path": "a.md"})
        assert response.status_code == 503
        assert response.json() == {"detail": "缺少真实 OPENAI_API_KEY"}
        health = client.get("/health")
        assert health.status_code == 200
    assert pipeline.closed == ["closed"]


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


def test_post_ask_matches_cli_json_and_defaults() -> None:
    result = _ask_result()
    pipeline = _Pipeline()
    pipeline.ask_result = result
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ask", json={"query": "质量"})

    assert response.status_code == 200
    assert response.content == json.dumps(asdict(result), ensure_ascii=False).encode("utf-8")
    assert "回答".encode() in response.content
    assert b"\\u56de" not in response.content
    assert pipeline.calls == ["ask"]
    assert pipeline.seen == {
        "query": "质量",
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "rewrite": False,
        "rerank": False,
        "graph_hops": False,
        "aliases": False,
        "max_steps": 3,
    }


def test_post_ask_forwards_flags() -> None:
    pipeline = _Pipeline()
    body = {
        "query": " 质量 ",
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": True,
        "rewrite": True,
        "rerank": True,
        "graph_hops": True,
        "aliases": False,
        "max_steps": 2,
    }
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ask", json=body)

    assert response.status_code == 200
    assert pipeline.seen == {
        "query": " 质量 ",
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": True,
        "rewrite": True,
        "rerank": True,
        "graph_hops": True,
        "aliases": False,
        "max_steps": 2,
    }
    assert "answer" not in pipeline.seen


def test_post_ask_soft_failure_stays_200() -> None:
    pipeline = _Pipeline()
    pipeline.ask_result = _ask_result(
        stopped="max_steps",
        answer=AnswerStatus(
            ok=False,
            skipped=False,
            text=None,
            citations=(),
            model="gpt-test",
            error="max_steps",
        ),
    )
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ask", json={"query": "质量"})

    assert response.status_code == 200
    assert response.json()["stopped"] == "max_steps"
    assert response.json()["answer"]["error"] == "max_steps"
    assert response.json()["answer"]["text"] is None


def test_ask_validation_does_not_call_pipeline() -> None:
    pipeline = _Pipeline()
    bodies = [
        {},
        {"query": "   "},
        {"query": "质量", "top_k": True},
        {"query": "质量", "doc_id": "   "},
        {"query": "质量", "max_steps": 4},
        {"query": "质量", "max_steps": 0},
        {"query": "质量", "max_steps": True},
        {"query": "质量", "answer": True},
        {"query": "质量", "agent": True},
    ]
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        statuses = [client.post("/ask", json=body).status_code for body in bodies]

    assert statuses == [422, 422, 422, 422, 422, 422, 422, 422, 422]
    assert pipeline.calls == []


def test_ask_value_error_is_422() -> None:
    pipeline = _Pipeline()
    pipeline.ask_error = ValueError("max_steps 必须是 1 到 3 的整数，收到 4")
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ask", json={"query": "质量"})

    assert response.status_code == 422
    assert response.json() == {"detail": "max_steps 必须是 1 到 3 的整数，收到 4"}


def test_ask_hard_failure_is_503_and_closes_once() -> None:
    pipeline = _Pipeline()
    pipeline.ask_error = RuntimeError("缺少真实 OPENAI_API_KEY")
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post("/ask", json={"query": "质量"})
        assert response.status_code == 503
        assert response.json() == {"detail": "缺少真实 OPENAI_API_KEY"}
    assert pipeline.closed == ["closed"]


def test_ingest_alias_flags_are_422_before_pipeline() -> None:
    pipeline = _Pipeline()
    bodies = [
        {"path": "a.md", "aliases": True},
        {"path": "a.md", "drop_aliases": True},
        {"path": "a.md", "entities": True, "aliases": True, "drop_aliases": True},
        {"path": "a.md", "dry_run": True, "aliases": True},
    ]
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        statuses = [client.post("/ingest", json=body).status_code for body in bodies]

    assert statuses == [422, 422, 422, 422]
    assert pipeline.calls == []


def test_ingest_alias_soft_failure_stays_200(tmp_path: Path) -> None:
    target = tmp_path / "a.md"
    target.write_text("孔子", encoding="utf-8")
    report = ingest_path(target, root=tmp_path)
    result = IngestWriteResult(
        report=report,
        collection="masskb_chunks",
        vector_size=1536,
        points_upserted=1,
        graph_documents=1,
        graph_chunks=1,
        graph_skipped=False,
        entities_skipped=False,
        graph_persons=2,
        graph_concepts=0,
        graph_mentions=2,
        aliases_skipped=False,
        aliases_ok=False,
        aliases_error="文档 d 的别名未写入：RuntimeError: down",
        graph_aliases=0,
        aliases_dropped=0,
    )
    pipeline = _Pipeline()
    pipeline.ingest_result = result
    with TestClient(create_app(pipeline)) as client:  # type: ignore[arg-type]
        response = client.post(
            "/ingest",
            json={"path": str(target), "root": str(tmp_path), "entities": True, "aliases": True},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["aliases_ok"] is False
    assert body["graph_mentions"] == 2
    assert "down" in body["aliases_error"]
    assert pipeline.calls == ["ingest"]
