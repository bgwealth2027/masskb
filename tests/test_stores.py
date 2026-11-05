"""Qdrant / Neo4j 封装：替换底层客户端，不连真实服务。"""

from __future__ import annotations

import pytest

from masskb.config import Settings
from masskb.graph_store import GraphStore
from masskb.vector_store import VectorStore


class _FakeQdrant:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False

    def get_collections(self):
        collection = type("Collection", (), {"name": "docs"})()
        return type("Collections", (), {"collections": [collection]})()

    def close(self) -> None:
        self.closed = True


class _BrokenQdrant:
    def __init__(self, **kwargs) -> None:
        self.closed = False

    def get_collections(self):
        raise ConnectionError("refused")

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
        self.verified = False
        self.closed = False

    def verify_connectivity(self) -> None:
        self.verified = True

    def session(self):
        return _Session()

    def close(self) -> None:
        self.closed = True


class _BrokenDriver:
    def verify_connectivity(self) -> None:
        raise RuntimeError("bolt down")

    def close(self) -> None:
        return None


def test_vector_store_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[_FakeQdrant] = []

    def factory(**kwargs):
        client = _FakeQdrant(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr("masskb.vector_store.QdrantClient", factory)
    store = VectorStore(Settings(qdrant_url="http://qdrant.test:6333"))

    result = store.healthcheck()

    assert result == {
        "ok": True,
        "url": "http://qdrant.test:6333",
        "collections": ["docs"],
    }
    assert created[0].kwargs == {
        "url": "http://qdrant.test:6333",
        "prefer_grpc": False,
        "check_compatibility": False,
        "timeout": 10,
    }
    store.close()
    assert created[0].closed is True
    assert store._client is None


def test_vector_store_healthcheck_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[_BrokenQdrant] = []

    def factory(**kwargs):
        client = _BrokenQdrant(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr("masskb.vector_store.QdrantClient", factory)
    store = VectorStore(Settings(qdrant_url="http://qdrant.test:6333"))
    result = store.healthcheck()

    assert result["ok"] is False
    assert result["url"] == "http://qdrant.test:6333"
    assert "ConnectionError" in result["error"]
    assert "refused" in result["error"]
    store.close()
    assert created[0].closed is True


def test_vector_store_close_without_client() -> None:
    VectorStore(Settings()).close()


def test_graph_store_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    drivers: list[_Driver] = []

    def factory(uri, *, auth):
        assert uri == "bolt://neo4j.test:7687"
        assert auth == ("neo4j", "secret")
        driver = _Driver()
        drivers.append(driver)
        return driver

    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", factory)
    settings = Settings(
        neo4j_uri="bolt://neo4j.test:7687",
        neo4j_user="neo4j",
        neo4j_password="secret",
    )
    store = GraphStore(settings)

    assert store.healthcheck() == {
        "ok": True,
        "uri": "bolt://neo4j.test:7687",
        "ping": 1,
    }
    assert drivers[0].verified is True
    store.close()
    assert drivers[0].closed is True
    assert store._driver is None


def test_graph_store_healthcheck_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "masskb.graph_store.GraphDatabase.driver",
        lambda *args, **kwargs: _BrokenDriver(),
    )
    result = GraphStore(Settings(neo4j_uri="bolt://neo4j.test:7687")).healthcheck()
    assert result["ok"] is False
    assert result["uri"] == "bolt://neo4j.test:7687"
    assert "RuntimeError" in result["error"]
    assert "bolt down" in result["error"]


def test_graph_store_close_without_driver() -> None:
    GraphStore(Settings()).close()
