"""检索编排冒烟：健康检查走真实 Pipeline，后端客户端全部 mock。"""

from __future__ import annotations

import pytest

from masskb.config import Settings
from masskb.pipeline import RetrievalPipeline


class _FakeQdrant:
    def __init__(self, **kwargs) -> None:
        self.closed = False

    def get_collections(self):
        collection = type("Collection", (), {"name": "docs"})()
        return type("Collections", (), {"collections": [collection]})()

    def close(self) -> None:
        self.closed = True


class _Session:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query: str):
        assert query == "RETURN 1 AS n"
        return type("Result", (), {"single": staticmethod(lambda: {"n": 1})})()


class _Driver:
    def __init__(self) -> None:
        self.closed = False

    def verify_connectivity(self) -> None:
        return None

    def session(self):
        return _Session()

    def close(self) -> None:
        self.closed = True


def test_pipeline_healthcheck_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    qdrant = _FakeQdrant()
    driver = _Driver()
    monkeypatch.setattr("masskb.vector_store.QdrantClient", lambda **kwargs: qdrant)
    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", lambda *args, **kwargs: driver)

    pipeline = RetrievalPipeline(Settings(openai_api_key="sk-your-api-key-here"))
    try:
        result = pipeline.healthcheck()
    finally:
        pipeline.close()

    assert result["ok"] is True
    assert result["qdrant"]["collections"] == ["docs"]
    assert result["qdrant"]["url"] == "http://localhost:6333"
    assert result["neo4j"]["ping"] == 1
    assert result["neo4j"]["uri"] == "bolt://localhost:7687"
    assert result["openai_key_configured"] is False
    assert qdrant.closed is True
    assert driver.closed is True


def test_pipeline_healthcheck_fails_if_either_backend_fails() -> None:
    pipeline = RetrievalPipeline(Settings(openai_api_key="sk-live-example"))
    pipeline.vector_store.healthcheck = lambda: {"ok": False, "url": "u", "error": "down"}
    pipeline.graph_store.healthcheck = lambda: {"ok": True, "uri": "b", "ping": 1}

    result = pipeline.healthcheck()

    assert result["ok"] is False
    assert result["openai_key_configured"] is True
    pipeline.close()


def test_search_is_not_implemented() -> None:
    pipeline = RetrievalPipeline(Settings())
    with pytest.raises(NotImplementedError, match="检索编排尚未实现"):
        pipeline.search("质量", top_k=3)
    pipeline.close()
