"""检索编排冒烟：健康检查、写入和检索走真实 Pipeline，后端客户端全部 mock。"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from masskb.config import Settings
from masskb.ingest import ingest_path
from masskb.pipeline import DimensionMismatchError, RetrievalPipeline
from masskb.vector_store import CollectionConfigError, point_id_for_chunk


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


def _search_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    embeddings: _Embeddings,
    *,
    present: bool = True,
    raw: list[dict] | None = None,
    seen: dict | None = None,
    search_error: Exception | None = None,
    require_error: Exception | None = None,
) -> tuple[RetrievalPipeline, list[str]]:
    _forbid_backends(monkeypatch)
    pipeline = RetrievalPipeline(
        Settings(openai_api_key="sk-live-example", embedding_model="embed-test")
    )
    pipeline.embeddings.embed_one = embeddings.embed_one
    calls: list[str] = []

    def has_collection(name: str = "masskb_chunks") -> bool:
        calls.append("has_collection")
        assert name == "masskb_chunks"
        return present

    def require_compatible_collection(name: str = "masskb_chunks", **kwargs) -> None:
        calls.append("require_compatible_collection")
        assert name == "masskb_chunks"
        if require_error is not None:
            raise require_error

    def search(vector, *, top_k: int = 5, doc_id: str | None = None, name: str = "masskb_chunks"):
        calls.append("search")
        assert name == "masskb_chunks"
        if seen is not None:
            seen["vector"] = list(vector)
            seen["top_k"] = top_k
            seen["doc_id"] = doc_id
        if search_error is not None:
            raise search_error
        return [] if raw is None else raw

    def upsert_points(*_args, **_kwargs) -> None:
        calls.append("upsert_points")
        raise AssertionError("搜索不应 upsert")

    def delete_by_doc_id(*_args, **_kwargs) -> None:
        calls.append("delete_by_doc_id")
        raise AssertionError("搜索不应删除")

    pipeline.vector_store.has_collection = has_collection
    pipeline.vector_store.require_compatible_collection = require_compatible_collection
    pipeline.vector_store.search = search
    pipeline.vector_store.upsert_points = upsert_points
    pipeline.vector_store.delete_by_doc_id = delete_by_doc_id
    return pipeline, calls


def test_search_embeds_query_and_maps_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = _Embeddings()
    seen: dict = {}
    raw = [
        {
            "score": 0.2,
            "payload": {
                "text": "第二段",
                "source": "b.md",
                "chunk_id": "c2",
                "doc_id": "d2",
                "index": 1,
                "start": 4,
                "end": 9,
            },
        },
        {
            "score": 0.9,
            "payload": {
                "text": "第一段",
                "source": "a.md",
                "chunk_id": "c1",
                "doc_id": "d1",
                "index": 0,
                "start": 0,
                "end": 4,
            },
        },
    ]
    pipeline, calls = _search_pipeline(monkeypatch, embeddings, raw=raw, seen=seen)

    result = pipeline.search("  质量  ", top_k=2)
    pipeline.close()

    assert embeddings.probes == ["  质量  "]
    assert "masskb" not in embeddings.probes
    assert seen == {"vector": [0.1] * 1536, "top_k": 2, "doc_id": None}
    assert calls == ["has_collection", "require_compatible_collection", "search"]
    assert pipeline.graph_store._driver is None
    assert result.query == "  质量  "
    assert result.top_k == 2
    assert result.collection == "masskb_chunks"
    assert result.embedding_model == "embed-test"
    assert result.doc_id is None
    assert [hit.score for hit in result.hits] == [0.2, 0.9]
    assert [hit.text for hit in result.hits] == ["第二段", "第一段"]
    assert result.hits[0].source == "b.md"
    assert result.hits[0].chunk_id == "c2"
    assert result.hits[0].doc_id == "d2"
    assert result.hits[0].index == 1
    assert result.hits[0].start == 4
    assert result.hits[0].end == 9


def test_search_missing_collection_returns_empty_after_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = _Embeddings()
    pipeline, calls = _search_pipeline(monkeypatch, embeddings, present=False)

    result = pipeline.search("质量")
    pipeline.close()

    assert embeddings.probes == ["质量"]
    assert calls == ["has_collection"]
    assert result.hits == ()
    assert result.collection == "masskb_chunks"
    assert result.embedding_model == "embed-test"
    assert result.doc_id is None
    assert result.top_k == 5


def test_search_empty_points_return_empty_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, calls = _search_pipeline(monkeypatch, _Embeddings(), raw=[])

    result = pipeline.search("质量", top_k=3)
    pipeline.close()

    assert result.hits == ()
    assert result.top_k == 3
    assert calls == ["has_collection", "require_compatible_collection", "search"]


def test_search_forwards_doc_id_and_empty_filter_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}
    pipeline, _calls = _search_pipeline(monkeypatch, _Embeddings(), raw=[], seen=seen)

    result = pipeline.search("质量", doc_id="abc")
    pipeline.close()

    assert seen["doc_id"] == "abc"
    assert result.doc_id == "abc"
    assert result.hits == ()


def test_search_aborts_when_query_dimension_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = _Embeddings(dim=2)
    pipeline, calls = _search_pipeline(monkeypatch, embeddings)

    with pytest.raises(DimensionMismatchError, match="未执行检索"):
        pipeline.search("质量")

    assert calls == []
    pipeline.close()


def test_search_missing_api_key_does_not_open_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _forbid_backends(monkeypatch)
    pipeline = RetrievalPipeline(Settings(openai_api_key="sk-your-api-key-here"))

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        pipeline.search("质量")

    assert created == []
    pipeline.close()


def test_search_aborts_when_collection_config_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, calls = _search_pipeline(
        monkeypatch,
        _Embeddings(),
        require_error=CollectionConfigError("mismatch"),
    )

    with pytest.raises(CollectionConfigError, match="mismatch"):
        pipeline.search("质量")

    assert calls == ["has_collection", "require_compatible_collection"]
    pipeline.close()


def test_search_rejects_blank_input_before_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings = _Embeddings()
    pipeline, calls = _search_pipeline(monkeypatch, embeddings)

    for query in ("", "   "):
        with pytest.raises(ValueError, match="查询不能为空"):
            pipeline.search(query)
    for top_k in (0, -1, True):
        with pytest.raises(ValueError, match="top_k 必须为正整数"):
            pipeline.search("质量", top_k=top_k)
    for doc_id in ("", "   "):
        with pytest.raises(ValueError, match="doc_id 不能为空"):
            pipeline.search("质量", doc_id=doc_id)

    assert embeddings.probes == []
    assert calls == []
    pipeline.close()


def test_search_keeps_hit_when_text_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = [{"score": 0.4, "payload": {"source": "a.md", "chunk_id": "c", "start": 1, "end": 2}}]
    pipeline, _calls = _search_pipeline(monkeypatch, _Embeddings(), raw=raw)

    hit = pipeline.search("质量").hits[0]
    pipeline.close()

    assert hit.score == 0.4
    assert hit.text is None
    assert hit.doc_id is None
    assert hit.index is None
    assert hit.source == "a.md"
    assert hit.chunk_id == "c"
    assert hit.start == 1
    assert hit.end == 2


def test_search_missing_collection_after_check_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, calls = _search_pipeline(
        monkeypatch,
        _Embeddings(),
        search_error=UnexpectedResponse(404, "Not Found", b"missing", None),
    )

    result = pipeline.search("质量")
    pipeline.close()

    assert result.hits == ()
    assert calls == ["has_collection", "require_compatible_collection", "search"]


def test_search_propagates_qdrant_errors_other_than_missing_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_store, _calls = _search_pipeline(monkeypatch, _Embeddings())
    missing_store.vector_store.has_collection = lambda name="masskb_chunks": (_ for _ in ()).throw(
        ConnectionError("refused")
    )
    with pytest.raises(ConnectionError, match="refused"):
        missing_store.search("质量")
    missing_store.close()

    server_error, _calls = _search_pipeline(
        monkeypatch,
        _Embeddings(),
        search_error=UnexpectedResponse(500, "Error", b"down", None),
    )
    with pytest.raises(UnexpectedResponse):
        server_error.search("质量")
    server_error.close()


_PAYLOAD_KEYS = {"doc_id", "chunk_id", "source", "index", "start", "end", "text"}


class _MemoryVectors:
    def __init__(self) -> None:
        self.points: dict[str, dict] = {}
        self.calls: list[str] = []
        self.present = True
        self.ensure_error: Exception | None = None

    def has_collection(self, name: str = "masskb_chunks") -> bool:
        self.calls.append("has_collection")
        assert name == "masskb_chunks"
        return self.present

    def ensure_collection(self, name: str = "masskb_chunks", *, vector_size: int = 1536) -> None:
        self.calls.append("ensure_collection")
        assert name == "masskb_chunks"
        assert vector_size == 1536
        if self.ensure_error is not None:
            raise self.ensure_error

    def upsert_points(self, ids, vectors, payloads, *, name: str = "masskb_chunks") -> None:
        self.calls.append("upsert_points")
        assert name == "masskb_chunks"
        assert len(ids) == len(vectors) == len(payloads)
        for point_id, vector, payload in zip(ids, vectors, payloads):
            self.points[point_id] = {"vector": list(vector), "payload": dict(payload)}

    def delete_by_doc_id(self, doc_id: str, *, keep_ids=(), name: str = "masskb_chunks") -> None:
        self.calls.append("delete_by_doc_id")
        assert name == "masskb_chunks"
        keep = set(keep_ids)
        for point_id in list(self.points):
            payload = self.points[point_id]["payload"]
            if payload.get("doc_id") == doc_id and point_id not in keep:
                del self.points[point_id]


class _Embeddings:
    def __init__(self, *, dim: int = 1536, chunk_dim: int | None = None) -> None:
        self.dim = dim
        self.chunk_dim = dim if chunk_dim is None else chunk_dim
        self.probes: list[str] = []
        self.batches: list[list[str]] = []

    def embed_one(self, text: str) -> list[float]:
        self.probes.append(text)
        return [0.1] * self.dim

    def embed(self, texts) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[0.2] * self.chunk_dim for _ in texts]


def _forbid_backends(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    created: list[str] = []

    def qdrant_factory(**kwargs):
        created.append("qdrant")
        raise AssertionError("不应创建 QdrantClient")

    def neo4j_factory(*args, **kwargs):
        created.append("neo4j")
        raise AssertionError("不应连接 Neo4j")

    def openai_factory(**kwargs):
        created.append("openai")
        raise AssertionError("不应创建 OpenAI")

    monkeypatch.setattr("masskb.vector_store.QdrantClient", qdrant_factory)
    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", neo4j_factory)
    monkeypatch.setattr("masskb.embeddings.OpenAI", openai_factory)
    return created


def _pipeline(
    monkeypatch: pytest.MonkeyPatch,
    embeddings: _Embeddings,
    store: _MemoryVectors,
) -> RetrievalPipeline:
    _forbid_backends(monkeypatch)
    pipeline = RetrievalPipeline(
        Settings(openai_api_key="sk-live-example", embedding_model="embed-test")
    )
    pipeline.embeddings.embed_one = embeddings.embed_one
    pipeline.embeddings.embed = embeddings.embed
    pipeline.vector_store.has_collection = store.has_collection
    pipeline.vector_store.ensure_collection = store.ensure_collection
    pipeline.vector_store.upsert_points = store.upsert_points
    pipeline.vector_store.delete_by_doc_id = store.delete_by_doc_id
    return pipeline


def test_ingest_probes_before_write_and_skips_probe_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    embeddings = _Embeddings()
    store = _MemoryVectors()
    order: list[str] = []

    def embed_one(text: str) -> list[float]:
        order.append("probe")
        return embeddings.embed_one(text)

    def ensure_collection(*args, **kwargs) -> None:
        order.append("ensure")
        store.ensure_collection(*args, **kwargs)

    def upsert_points(*args, **kwargs) -> None:
        order.append("upsert")
        store.upsert_points(*args, **kwargs)

    def delete_by_doc_id(*args, **kwargs) -> None:
        order.append("delete")
        store.delete_by_doc_id(*args, **kwargs)

    pipeline = _pipeline(monkeypatch, embeddings, store)
    pipeline.embeddings.embed_one = embed_one
    pipeline.vector_store.ensure_collection = ensure_collection
    pipeline.vector_store.upsert_points = upsert_points
    pipeline.vector_store.delete_by_doc_id = delete_by_doc_id

    result = pipeline.ingest(path, root=tmp_path)
    pipeline.close()

    chunk = result.report.files[0].chunks[0]
    stored = next(iter(store.points.values()))
    assert order == ["probe", "ensure", "upsert", "delete"]
    assert embeddings.probes == ["masskb"]
    assert embeddings.batches == [["hello"]]
    assert result.points_upserted == 1
    assert result.collection == "masskb_chunks"
    assert result.vector_size == 1536
    assert len(store.points) == 1
    assert set(store.points) == {point_id_for_chunk(chunk.chunk_id)}
    assert set(stored["payload"]) == _PAYLOAD_KEYS
    assert stored["payload"]["text"] == "hello"
    assert stored["payload"]["text"] != "masskb"
    assert "openai_api_key" not in stored["payload"]


def test_ingest_aborts_when_probe_dimension_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    created = _forbid_backends(monkeypatch)
    pipeline = RetrievalPipeline(Settings(openai_api_key="sk-live-example", embedding_model="embed-test"))
    pipeline.embeddings.embed_one = lambda text: [0.1, 0.2]

    with pytest.raises(DimensionMismatchError, match="未创建集合，也未写入"):
        pipeline.ingest(path, root=tmp_path)

    assert created == []
    pipeline.close()


def test_ingest_missing_api_key_does_not_open_qdrant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    created = _forbid_backends(monkeypatch)
    pipeline = RetrievalPipeline(Settings(openai_api_key="sk-your-api-key-here"))

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        pipeline.ingest(path, root=tmp_path)

    assert created == []
    pipeline.close()


def test_ingest_aborts_when_collection_config_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    store = _MemoryVectors()
    store.ensure_error = CollectionConfigError("mismatch")
    pipeline = _pipeline(monkeypatch, _Embeddings(), store)

    with pytest.raises(CollectionConfigError):
        pipeline.ingest(path, root=tmp_path)

    assert "upsert_points" not in store.calls
    assert "delete_by_doc_id" not in store.calls
    pipeline.close()


def test_ingest_embeds_in_batches_of_64(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "long.txt"
    path.write_text("测" * 650, encoding="utf-8")
    embeddings = _Embeddings()
    store = _MemoryVectors()
    pipeline = _pipeline(monkeypatch, embeddings, store)

    result = pipeline.ingest(path, root=tmp_path, chunk_size=10, overlap=0)
    pipeline.close()

    expected = [chunk.text for chunk in ingest_path(path, root=tmp_path, chunk_size=10, overlap=0).files[0].chunks]
    assert [len(batch) for batch in embeddings.batches] == [64, 1]
    assert [text for batch in embeddings.batches for text in batch] == expected
    assert result.points_upserted == 65
    assert len(store.points) == 65


def test_ingest_rerun_overwrites_same_ids_and_drops_stale_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    store = _MemoryVectors()
    pipeline = _pipeline(monkeypatch, _Embeddings(), store)

    first = pipeline.ingest(path, root=tmp_path)
    first_ids = set(store.points)
    path.write_text("world\n", encoding="utf-8")
    second = pipeline.ingest(path, root=tmp_path)

    assert set(store.points) == first_ids
    assert len(store.points) == first.points_upserted == second.points_upserted == 1
    assert next(iter(store.points.values()))["payload"]["text"] == "world"

    path.write_text("abcdefghij" * 4, encoding="utf-8")
    wide = pipeline.ingest(path, root=tmp_path, chunk_size=20, overlap=0)
    wide_ids = set(store.points)
    assert wide_ids.isdisjoint(first_ids)
    assert len(wide_ids) == wide.points_upserted == 2
    narrow = pipeline.ingest(path, root=tmp_path, chunk_size=8, overlap=0)
    assert set(store.points).isdisjoint(wide_ids)
    assert len(store.points) == narrow.points_upserted == 5
    pipeline.close()


def test_ingest_empty_file_skips_embedding_and_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    embeddings = _Embeddings()
    store = _MemoryVectors()
    store.present = False
    pipeline = _pipeline(monkeypatch, embeddings, store)

    result = pipeline.ingest(path, root=tmp_path)
    pipeline.close()

    assert result.points_upserted == 0
    assert store.calls == ["has_collection"]
    assert embeddings.probes == []
    assert embeddings.batches == []


def test_ingest_empty_file_deletes_existing_doc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    doc_id = ingest_path(path, root=tmp_path).files[0].doc_id
    store = _MemoryVectors()
    store.points["stale"] = {"vector": [0.0], "payload": {"doc_id": doc_id, "text": "old"}}
    embeddings = _Embeddings()
    pipeline = _pipeline(monkeypatch, embeddings, store)

    result = pipeline.ingest(path, root=tmp_path)
    pipeline.close()

    assert result.points_upserted == 0
    assert store.points == {}
    assert "ensure_collection" not in store.calls
    assert store.calls.count("delete_by_doc_id") == 1
    assert embeddings.probes == []


def test_ingest_mixed_empty_and_text_clears_empty_doc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "empty.txt").write_text("", encoding="utf-8")
    (root / "full.txt").write_text("hi\n", encoding="utf-8")
    empty_id = ingest_path(root / "empty.txt", root=root).files[0].doc_id
    store = _MemoryVectors()
    store.points["stale"] = {"vector": [0.0], "payload": {"doc_id": empty_id, "text": "old"}}
    pipeline = _pipeline(monkeypatch, _Embeddings(), store)

    result = pipeline.ingest(root, root=root)
    pipeline.close()

    assert result.points_upserted == 1
    assert "stale" not in store.points
    assert len(store.points) == 1
    assert store.calls[0] == "ensure_collection"
    assert store.calls.count("delete_by_doc_id") == 2


def test_ingest_bad_chunk_dimension_does_not_upsert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    store = _MemoryVectors()
    pipeline = _pipeline(monkeypatch, _Embeddings(chunk_dim=3), store)

    with pytest.raises(DimensionMismatchError, match="未写入该文档"):
        pipeline.ingest(path, root=tmp_path)

    assert "ensure_collection" in store.calls
    assert "upsert_points" not in store.calls
    assert "delete_by_doc_id" not in store.calls
    pipeline.close()
