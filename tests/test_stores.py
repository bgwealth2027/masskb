"""Qdrant / Neo4j 封装：替换底层客户端，不连真实服务。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, Modifier, PayloadSchemaType, SparseVector

from masskb.aliases import AliasLink
from masskb.config import Settings
from masskb.entities import EntityMention, entity_id
from masskb.graph_store import (
    GRAPH_HOP_CANDIDATES,
    READ_CONNECTION_ACQUISITION_TIMEOUT,
    READ_MAX_TRANSACTION_RETRY_TIME,
    GraphStore,
    GraphStoreError,
    rank_related_rows,
)
from masskb.ingest import Chunk, Document
from masskb.vector_store import (
    SEARCH_PAYLOAD_FIELDS,
    SPARSE_VECTOR_NAME,
    CollectionConfigError,
    VectorStore,
    point_id_for_chunk,
)


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


class _RecordingQdrant:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.collections: list[str] = []
        self.info: object | None = None
        self.index_error: Exception | None = None

    def get_collections(self):
        self.calls.append(("get_collections", {}))
        cols = [type("C", (), {"name": name})() for name in self.collections]
        return type("Collections", (), {"collections": cols})()

    def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs))
        self.collections.append(kwargs["collection_name"])

    def get_collection(self, collection_name: str, **kwargs):
        self.calls.append(("get_collection", {"collection_name": collection_name, **kwargs}))
        assert self.info is not None
        return self.info

    def create_payload_index(self, **kwargs):
        self.calls.append(("create_payload_index", kwargs))
        if self.index_error is not None:
            raise self.index_error

    def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def delete_collection(self, **kwargs):
        self.calls.append(("delete_collection", kwargs))

    def query_points(self, **kwargs):
        self.calls.append(("query_points", kwargs))
        return getattr(self, "query_response", SimpleNamespace(points=[]))

    def retrieve(self, **kwargs):
        self.calls.append(("retrieve", kwargs))
        return getattr(self, "retrieve_points", [])


def _bind_qdrant(monkeypatch: pytest.MonkeyPatch, client: _RecordingQdrant) -> VectorStore:
    monkeypatch.setattr("masskb.vector_store.QdrantClient", lambda **kwargs: client)
    return VectorStore(Settings(qdrant_url="http://qdrant.test:6333"))


def _names(client: _RecordingQdrant) -> list[str]:
    return [name for name, _ in client.calls]


def _collection_info(vectors: object, sparse: object | None = "default") -> SimpleNamespace:
    if sparse == "default":
        sparse_vectors: object = {SPARSE_VECTOR_NAME: SimpleNamespace(modifier=Modifier.IDF)}
    else:
        sparse_vectors = sparse
    params = SimpleNamespace(vectors=vectors, sparse_vectors=sparse_vectors)
    return SimpleNamespace(config=SimpleNamespace(params=params))


def test_point_id_for_chunk_is_stable_uuid5() -> None:
    assert point_id_for_chunk("0" * 64) == "a163952d-6b3f-5386-add1-9c46269a2f8c"
    assert point_id_for_chunk("0" * 64) == point_id_for_chunk("0" * 64)
    assert point_id_for_chunk("1" * 64) != point_id_for_chunk("0" * 64)


def test_ensure_collection_creates_cosine_1536(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    store = _bind_qdrant(monkeypatch, client)

    store.ensure_collection()

    created = [kwargs for name, kwargs in client.calls if name == "create_collection"]
    assert len(created) == 1
    assert created[0]["collection_name"] == "masskb_chunks"
    assert created[0]["vectors_config"].size == 1536
    assert created[0]["vectors_config"].distance == Distance.COSINE
    sparse = created[0]["sparse_vectors_config"]
    assert set(sparse) == {SPARSE_VECTOR_NAME}
    assert sparse[SPARSE_VECTOR_NAME].modifier == Modifier.IDF
    indexed = [kwargs for name, kwargs in client.calls if name == "create_payload_index"]
    assert indexed[0]["field_name"] == "doc_id"
    assert indexed[0]["field_schema"] == PayloadSchemaType.KEYWORD
    assert "get_collection" not in _names(client)
    assert "delete_collection" not in _names(client)


def test_ensure_collection_skips_create_when_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    client.collections = ["masskb_chunks"]
    client.info = _collection_info(SimpleNamespace(size=1536, distance=Distance.COSINE))
    store = _bind_qdrant(monkeypatch, client)

    store.ensure_collection()

    assert "create_collection" not in _names(client)
    assert "get_collection" in _names(client)
    assert "create_payload_index" in _names(client)
    assert "delete_collection" not in _names(client)


@pytest.mark.parametrize(
    "vectors",
    [
        SimpleNamespace(size=3, distance=Distance.COSINE),
        SimpleNamespace(size=1536, distance=Distance.EUCLID),
        {"body": SimpleNamespace(size=1536, distance=Distance.COSINE)},
    ],
)
def test_ensure_collection_rejects_incompatible(
    monkeypatch: pytest.MonkeyPatch,
    vectors: object,
) -> None:
    client = _RecordingQdrant()
    client.collections = ["masskb_chunks"]
    client.info = _collection_info(vectors)
    store = _bind_qdrant(monkeypatch, client)

    with pytest.raises(CollectionConfigError, match="未写入"):
        store.ensure_collection()

    forbidden = {"create_collection", "upsert", "delete", "delete_collection", "create_payload_index"}
    assert forbidden.isdisjoint(_names(client))


def test_ensure_collection_rejects_missing_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    client.collections = ["masskb_chunks"]
    client.info = _collection_info(
        SimpleNamespace(size=1536, distance=Distance.COSINE),
        sparse=None,
    )
    store = _bind_qdrant(monkeypatch, client)

    with pytest.raises(CollectionConfigError, match="bm25"):
        store.ensure_collection()

    forbidden = {"create_collection", "upsert", "delete", "delete_collection", "create_payload_index"}
    assert forbidden.isdisjoint(_names(client))


def test_ensure_collection_rejects_wrong_sparse_modifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingQdrant()
    client.collections = ["masskb_chunks"]
    client.info = _collection_info(
        SimpleNamespace(size=1536, distance=Distance.COSINE),
        sparse={SPARSE_VECTOR_NAME: SimpleNamespace(modifier=None)},
    )
    store = _bind_qdrant(monkeypatch, client)

    with pytest.raises(CollectionConfigError, match="未写入"):
        store.ensure_collection()

    assert "delete_collection" not in _names(client)
    assert "upsert" not in _names(client)
    assert "delete" not in _names(client)


def test_ensure_collection_ignores_existing_payload_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingQdrant()
    client.index_error = RuntimeError("Payload index already exists")
    store = _bind_qdrant(monkeypatch, client)

    store.ensure_collection()

    assert "create_collection" in _names(client)
    assert "create_payload_index" in _names(client)


def test_ensure_collection_reports_other_index_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    client.index_error = RuntimeError("connection refused")
    store = _bind_qdrant(monkeypatch, client)

    with pytest.raises(RuntimeError, match="connection refused"):
        store.ensure_collection()


def test_upsert_points_batches_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    store = _bind_qdrant(monkeypatch, client)
    ids = [f"id-{index}" for index in range(65)]
    vectors = [[float(index)] for index in range(65)]
    payloads = [{"doc_id": "d", "n": index} for index in range(65)]

    store.upsert_points(ids, vectors, payloads)

    upserts = [kwargs for name, kwargs in client.calls if name == "upsert"]
    assert [len(item["points"]) for item in upserts] == [64, 1]
    assert all(item["wait"] is True for item in upserts)
    assert upserts[0]["collection_name"] == "masskb_chunks"
    assert upserts[0]["points"][0].id == "id-0"
    assert upserts[0]["points"][0].vector == [0.0]
    assert upserts[0]["points"][0].payload == {"doc_id": "d", "n": 0}
    assert upserts[1]["points"][0].id == "id-64"
    assert "delete_collection" not in _names(client)


def test_upsert_points_writes_dense_and_sparse_on_the_same_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingQdrant()
    store = _bind_qdrant(monkeypatch, client)

    store.upsert_points(
        ["id-0", "id-1"],
        [[0.1, 0.2], [0.3]],
        [{"doc_id": "d"}, {"doc_id": "d"}],
        [([1, 3], [0.5, 0.2]), ([], [])],
    )

    point = client.calls[0][1]["points"]
    assert point[0].vector[""] == [0.1, 0.2]
    assert point[0].vector[SPARSE_VECTOR_NAME] == SparseVector(indices=[1, 3], values=[0.5, 0.2])
    assert point[1].vector == [0.3]
    assert "delete_collection" not in _names(client)


def test_upsert_points_rejects_uneven_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    store = _bind_qdrant(monkeypatch, client)

    with pytest.raises(ValueError, match="长度必须一致"):
        store.upsert_points(["a"], [[0.0]], [{"doc_id": "d"}], [([1], [0.2]), ([2], [0.3])])

    assert "upsert" not in _names(client)


def test_upsert_points_rejects_uneven_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    store = _bind_qdrant(monkeypatch, client)

    with pytest.raises(ValueError, match="长度必须一致"):
        store.upsert_points(["a"], [[0.0]], [])

    assert "upsert" not in _names(client)


def test_delete_by_doc_id_keeps_current_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    store = _bind_qdrant(monkeypatch, client)

    store.delete_by_doc_id("doc-1", keep_ids=["p1", "p2"])

    deleted = [kwargs for name, kwargs in client.calls if name == "delete"]
    assert len(deleted) == 1
    assert deleted[0]["wait"] is True
    assert deleted[0]["collection_name"] == "masskb_chunks"
    filt = deleted[0]["points_selector"].filter
    assert filt.must[0].key == "doc_id"
    assert filt.must[0].match.value == "doc-1"
    assert list(filt.must_not[0].has_id) == ["p1", "p2"]
    assert "delete_collection" not in _names(client)


def test_delete_by_doc_id_without_keep_ids_matches_doc_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _RecordingQdrant()
    store = _bind_qdrant(monkeypatch, client)

    store.delete_by_doc_id("doc-1")

    filt = client.calls[0][1]["points_selector"].filter
    assert filt.must[0].match.value == "doc-1"
    assert filt.must_not is None
    assert "delete_collection" not in _names(client)


class _Score(float):
    """用来确认分数会被收成内置 float。"""


def test_search_queries_nearest_neighbors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    client.query_response = SimpleNamespace(
        points=[
            SimpleNamespace(score=_Score(0.2), payload={"text": "b", "chunk_id": "c2"}),
            SimpleNamespace(score=0.9, payload=None),
        ]
    )
    store = _bind_qdrant(monkeypatch, client)

    result = store.search((0.1, 0.2), top_k=3)

    queries = [kwargs for name, kwargs in client.calls if name == "query_points"]
    assert len(queries) == 1
    assert queries[0]["collection_name"] == "masskb_chunks"
    assert queries[0]["query"] == [0.1, 0.2]
    assert queries[0]["limit"] == 3
    assert queries[0]["query_filter"] is None
    assert queries[0]["with_payload"] == list(SEARCH_PAYLOAD_FIELDS)
    assert queries[0]["with_vectors"] is False
    assert "using" not in queries[0]
    assert result == [
        {"score": 0.2, "payload": {"text": "b", "chunk_id": "c2"}},
        {"score": 0.9, "payload": {}},
    ]
    assert type(result[0]["score"]) is float
    forbidden = {"create_collection", "upsert", "delete", "delete_collection", "create_payload_index"}
    assert forbidden.isdisjoint(_names(client))


def test_search_filters_by_doc_id(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    store = _bind_qdrant(monkeypatch, client)

    assert store.search([0.0], doc_id="doc-1") == []

    filt = client.calls[0][1]["query_filter"]
    assert filt.must[0].key == "doc_id"
    assert filt.must[0].match.value == "doc-1"
    assert filt.must_not is None
    forbidden = {"create_collection", "upsert", "delete", "delete_collection", "create_payload_index"}
    assert forbidden.isdisjoint(_names(client))


def test_search_sparse_uses_bm25_and_doc_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    client.query_response = SimpleNamespace(
        points=[SimpleNamespace(score=_Score(1.5), payload={"chunk_id": "c", "text": "质量"})]
    )
    store = _bind_qdrant(monkeypatch, client)

    result = store.search_sparse([3, 8], [0.2, 0.4], top_k=4, doc_id="doc-1")

    query = client.calls[0][1]
    assert query["using"] == SPARSE_VECTOR_NAME
    assert query["limit"] == 4
    assert query["query"] == SparseVector(indices=[3, 8], values=[0.2, 0.4])
    assert query["with_payload"] == list(SEARCH_PAYLOAD_FIELDS)
    assert query["with_vectors"] is False
    assert query["query_filter"].must[0].key == "doc_id"
    assert query["query_filter"].must[0].match.value == "doc-1"
    assert result == [{"score": 1.5, "payload": {"chunk_id": "c", "text": "质量"}}]
    forbidden = {"create_collection", "upsert", "delete", "delete_collection", "create_payload_index"}
    assert forbidden.isdisjoint(_names(client))


def test_sparse_ready_reads_bm25_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    client.info = _collection_info(SimpleNamespace(size=1536, distance=Distance.COSINE))
    store = _bind_qdrant(monkeypatch, client)

    assert store.sparse_ready() is True

    client.info = _collection_info(
        SimpleNamespace(size=1536, distance=Distance.COSINE),
        sparse=None,
    )
    assert store.sparse_ready() is False
    assert "delete_collection" not in _names(client)


def test_require_compatible_collection_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    client.info = _collection_info(SimpleNamespace(size=1536, distance=Distance.COSINE))
    store = _bind_qdrant(monkeypatch, client)

    store.require_compatible_collection()

    assert _names(client) == ["get_collection"]


class _GraphResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def consume(self):
        return None

    def single(self):
        return self._rows[0] if self._rows else None


class _GraphSession:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.queries: list[tuple[str, dict]] = []
        self.write_calls = 0
        self.read_calls = 0
        self.rows = [] if rows is None else rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute_write(self, work):
        self.write_calls += 1
        return work(self)

    def execute_read(self, work):
        self.read_calls += 1
        return work(self)

    def run(self, query, **params):
        self.queries.append((query, params))
        return _GraphResult(self.rows)


class _GraphDriver:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.session_obj = _GraphSession(rows)
        self.sessions = 0
        self.session_kwargs: list[dict] = []
        self.closed = False

    def session(self, **kwargs):
        self.sessions += 1
        self.session_kwargs.append(dict(kwargs))
        return self.session_obj

    def close(self) -> None:
        self.closed = True


def _bind_graph(monkeypatch: pytest.MonkeyPatch, driver: _GraphDriver) -> GraphStore:
    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", lambda *args, **kwargs: driver)
    return GraphStore(Settings(neo4j_uri="bolt://neo4j.test:7687"))


def _skeleton_doc(*chunks: Chunk) -> Document:
    return Document(source=chunks[0].source, doc_id=chunks[0].doc_id, chunks=chunks)


def _skeleton_chunk(
    chunk_id: str,
    *,
    index: int = 0,
    start: int = 0,
    end: int = 1,
    text: str = "正文不应出现在语句里",
) -> Chunk:
    return Chunk(
        doc_id='doc"1',
        chunk_id=chunk_id,
        source='a".txt',
        index=index,
        start=start,
        end=end,
        text=text,
    )


def test_ensure_schema_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)

    store.ensure_schema()
    store.ensure_schema()

    assert driver.sessions == 1
    assert driver.session_obj.write_calls == 1
    assert driver.session_obj.read_calls == 0
    queries = [query for query, _params in driver.session_obj.queries]
    assert len(queries) == 3
    combined = "\n".join(queries)
    assert "masskb_document_doc_id" in combined
    assert "masskb_chunk_chunk_id" in combined
    assert "masskb_chunk_doc_id" in combined
    assert combined.count("IF NOT EXISTS") == 3
    assert "FOR (d:Document)" in combined
    assert "FOR (c:Chunk)" in combined
    assert "Person" not in combined
    assert "Concept" not in combined
    assert "apoc" not in combined.lower()


def test_upsert_document_merges_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)
    first = _skeleton_chunk('chunk"1', index=0, start=0, end=2)
    second = _skeleton_chunk("chunk-2", index=1, start=2, end=4)
    doc = _skeleton_doc(first, second)

    assert store.upsert_document(doc) == 2
    assert store.upsert_document(doc) == 2

    assert driver.session_obj.write_calls == 2
    assert len(driver.session_obj.queries) == 2
    query, params = driver.session_obj.queries[0]
    assert driver.session_obj.queries[1][1] == params
    assert "MERGE (d:Document" in query
    assert "MERGE (c:Chunk" in query
    assert "HAS_CHUNK" in query
    assert "NEXT_CHUNK" in query
    assert "DETACH DELETE" in query
    assert "OPTIONAL MATCH" in query
    assert "MENTIONS" in query
    assert ":Person" in query
    assert ":Concept" in query
    assert _keeps_alias_endpoints(query)
    assert "target_id" in query
    assert "apoc" not in query.lower()
    assert "CREATE (" not in query
    assert first.text not in query
    assert first.chunk_id not in query
    assert params["doc_id"] == first.doc_id
    assert params["source"] == first.source
    assert params["keep_ids"] == [first.chunk_id, second.chunk_id]
    assert params["chunks"] == [
        {"chunk_id": first.chunk_id, "index": 0, "start": 0, "end": 2},
        {"chunk_id": second.chunk_id, "index": 1, "start": 2, "end": 4},
    ]
    assert "text" not in params["chunks"][0]


def test_upsert_document_accepts_a_single_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)
    chunk = _skeleton_chunk("only")

    assert store.upsert_document(_skeleton_doc(chunk)) == 1

    _query, params = driver.session_obj.queries[0]
    assert params["keep_ids"] == ["only"]
    assert driver.session_obj.write_calls == 1


def test_delete_call_is_unit_subquery_and_upsert_returns_before_linking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """删除语句不能再以带 RETURN 的 CALL 收尾；更新语句仍要靠这个 RETURN 保住后继行。"""
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)
    chunk = _skeleton_chunk("only")

    store.upsert_document(_skeleton_doc(chunk))
    store.delete_document(chunk.doc_id)

    upsert_query = driver.session_obj.queries[0][0]
    delete_query = driver.session_obj.queries[1][0]
    returned = "RETURN count(*) AS removed_orphans"
    assert returned in upsert_query
    after_return = upsert_query.split(returned, 1)[1]
    assert "MERGE (prev)-[:NEXT_CHUNK]->(nxt)" in after_return
    assert _keeps_alias_endpoints(upsert_query)
    call_at = delete_query.rfind("CALL {")
    assert call_at != -1
    assert "RETURN" not in delete_query[call_at:]
    assert _keeps_alias_endpoints(delete_query)
    assert "DETACH DELETE person" not in delete_query
    assert "DETACH DELETE concept" not in delete_query


def test_delete_document_detaches_chunks_and_document(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)

    store.delete_document('doc"1')

    assert driver.session_obj.write_calls == 1
    assert driver.session_obj.read_calls == 0
    query, params = driver.session_obj.queries[0]
    assert "DETACH DELETE" in query
    assert "Chunk" in query
    assert "Document" in query
    assert "MENTIONS" in query
    assert ":Person" in query
    assert ":Concept" in query
    assert _keeps_alias_endpoints(query)
    assert params == {"doc_id": 'doc"1'}
    assert 'doc"1' not in query


def test_neighbor_chunk_ids_maps_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver(
        rows=[
            {"chunk_id": "c1", "prev_chunk_id": None, "next_chunk_id": "c2"},
            {"chunk_id": "c2", "prev_chunk_id": "c1", "next_chunk_id": None},
        ]
    )
    store = _bind_graph(monkeypatch, driver)

    found = store.neighbor_chunk_ids(["c1", "c3"])

    assert found == {"c1": (None, "c2"), "c2": ("c1", None)}
    assert driver.session_obj.read_calls == 1
    assert driver.session_obj.write_calls == 0
    query, params = driver.session_obj.queries[0]
    assert "NEXT_CHUNK" in query
    assert "UNWIND $chunk_ids" in query
    assert params == {"chunk_ids": ["c1", "c3"]}


def test_neighbor_chunk_ids_rejects_duplicate_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver(
        rows=[
            {"chunk_id": "c", "prev_chunk_id": "a", "next_chunk_id": "b"},
            {"chunk_id": "c", "prev_chunk_id": "a2", "next_chunk_id": "b2"},
        ]
    )
    store = _bind_graph(monkeypatch, driver)

    with pytest.raises(GraphStoreError, match="不唯一"):
        store.neighbor_chunk_ids(["c"])


def test_read_sessions_bound_retry_and_writes_keep_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)
    chunk = _skeleton_chunk("c1")
    mention = EntityMention(
        chunk_id="c1",
        kind="person",
        name="甲",
        entity_id=entity_id("person", "甲"),
    )

    store.ensure_schema()
    store.ensure_entity_schema()
    store.upsert_document(_skeleton_doc(chunk))
    store.replace_mentions(chunk.doc_id, ["c1"], [mention])
    store.replace_aliases(chunk.doc_id, ())
    store.delete_document(chunk.doc_id)
    assert driver.session_kwargs
    assert driver.session_kwargs == [{}] * len(driver.session_kwargs)

    driver.session_obj.rows = [
        {
            "chunk_id": "c1",
            "prev_chunk_id": None,
            "next_chunk_id": "c2",
            "via": "next_chunk",
            "shared": 0,
        }
    ]
    store.neighbor_chunk_ids(["c1"])
    store.related_chunks(["c1"], doc_id=chunk.doc_id)
    store.related_chunks(["c1"], doc_id=chunk.doc_id, aliases=True)
    assert READ_CONNECTION_ACQUISITION_TIMEOUT == 2.0
    assert READ_MAX_TRANSACTION_RETRY_TIME == 0.0
    assert driver.session_kwargs[-3:] == [
        {
            "connection_acquisition_timeout": 2.0,
            "max_transaction_retry_time": 0.0,
        },
        {
            "connection_acquisition_timeout": 2.0,
            "max_transaction_retry_time": 0.0,
        },
        {
            "connection_acquisition_timeout": 2.0,
            "max_transaction_retry_time": 0.0,
        },
    ]


def test_neighbor_chunk_ids_empty_does_not_open_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("不应连接 Neo4j")

    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", boom)
    store = GraphStore(Settings())

    assert store.neighbor_chunk_ids([]) == {}
    assert store._driver is None


def test_ensure_entity_schema_runs_once_and_stays_out_of_skeleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)

    store.ensure_schema()
    store.ensure_entity_schema()
    store.ensure_entity_schema()

    assert driver.sessions == 2
    entity_queries = [query for query, _params in driver.session_obj.queries[3:]]
    assert len(entity_queries) == 2
    combined = "\n".join(entity_queries)
    assert "masskb_person_person_id" in combined
    assert "masskb_concept_concept_id" in combined
    assert "FOR (p:Person)" in combined
    assert "FOR (c:Concept)" in combined
    assert combined.count("IF NOT EXISTS") == 2
    assert "apoc" not in combined.lower()
    skeleton = "\n".join(query for query, _params in driver.session_obj.queries[:3])
    assert "Person" not in skeleton


def test_replace_mentions_merges_and_retires_old_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver(rows=[{"old_targets": ["old-person", "pid"]}])
    store = _bind_graph(monkeypatch, driver)
    mention = EntityMention(chunk_id="c1", kind="person", name="孔子", entity_id="pid")

    assert store.replace_mentions("doc-1", ["c1", "c1"], [mention]) == (1, 0, 1)

    assert driver.session_obj.write_calls == 1
    clear_query, clear_params = driver.session_obj.queries[0]
    merge_query, merge_params = driver.session_obj.queries[1]
    orphan_query, orphan_params = driver.session_obj.queries[2]
    assert "DELETE" in clear_query
    assert "MENTIONS" in clear_query
    assert clear_params == {"chunk_ids": ["c1"]}
    assert "MERGE (person:Person" in merge_query
    assert "MERGE (concept:Concept" in merge_query
    assert "CREATE (" not in merge_query
    assert "apoc" not in merge_query.lower()
    assert "孔子" not in merge_query
    assert "ALIAS_OF" not in merge_query
    assert "ALIAS_OF" not in clear_query
    assert _keeps_alias_endpoints(orphan_query)
    assert "DETACH DELETE" not in orphan_query
    assert merge_params["mentions"] == [
        {"chunk_id": "c1", "kind": "person", "entity_id": "pid", "name": "孔子"}
    ]
    assert ":Person" in orphan_query
    assert ":Concept" in orphan_query
    assert "MENTIONS" in orphan_query
    assert orphan_params == {"target_ids": ["old-person"]}


def test_replace_mentions_empty_list_clears_without_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver(rows=[{"old_targets": ["gone"]}])
    store = _bind_graph(monkeypatch, driver)

    assert store.replace_mentions("doc-1", ["c1"], []) == (0, 0, 0)

    assert len(driver.session_obj.queries) == 2
    assert "MERGE (person:Person" not in driver.session_obj.queries[0][0]
    assert driver.session_obj.queries[1][1] == {"target_ids": ["gone"]}


def test_replace_mentions_keeps_target_that_is_still_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _GraphDriver(rows=[{"old_targets": ["pid"]}])
    store = _bind_graph(monkeypatch, driver)
    mention = EntityMention(chunk_id="c1", kind="concept", name="仁", entity_id="pid")

    assert store.replace_mentions("doc-1", ["c1"], [mention]) == (0, 1, 1)

    assert len(driver.session_obj.queries) == 2


def test_replace_mentions_empty_chunks_do_not_open_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("不应连接 Neo4j")

    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", boom)
    store = GraphStore(Settings())

    assert store.replace_mentions("doc-1", [], []) == (0, 0, 0)
    assert store._driver is None


def test_replace_mentions_rejects_unknown_chunk_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)
    mention = EntityMention(chunk_id="other", kind="person", name="孔子", entity_id="pid")

    with pytest.raises(ValueError, match="不在本次"):
        store.replace_mentions("doc-1", ["c1"], [mention])

    assert store._driver is None
    assert driver.sessions == 0


def test_replace_mentions_rejects_unknown_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)
    mention = EntityMention(chunk_id="c1", kind="place", name="孔子", entity_id="pid")

    with pytest.raises(ValueError, match="未知实体种类"):
        store.replace_mentions("doc-1", ["c1"], [mention])

    assert store._driver is None


def test_related_chunks_reads_existing_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver(
        rows=[
            {"chunk_id": "b", "via": "person", "shared": 1},
            {"chunk_id": "b", "via": "next_chunk", "shared": 0},
            {"chunk_id": "a", "via": "concept", "shared": 1},
            {"chunk_id": "a", "via": "concept", "shared": 1},
            {"chunk_id": 'seed"1', "via": "person", "shared": 1},
        ]
    )
    store = _bind_graph(monkeypatch, driver)

    found = store.related_chunks(['seed"1', 'seed"1'], doc_id='doc"1')

    assert driver.session_obj.read_calls == 1
    assert driver.session_obj.write_calls == 0
    query, params = driver.session_obj.queries[0]
    assert "NEXT_CHUNK" in query
    assert "MENTIONS" in query
    assert ":Person" in query
    assert ":Concept" in query
    assert "ORDER BY" in query
    assert "LIMIT 2" in query
    assert "LIMIT 8" in query
    assert "apoc" not in query.lower()
    assert "ALIAS_OF" not in query
    assert params == {"chunk_ids": ['seed"1'], "doc_id": 'doc"1'}
    assert 'doc"1' not in query
    assert 'seed"1' not in query
    assert [(item.chunk_id, item.mention_count, item.via_next) for item in found] == [
        ("a", 2, False),
        ("b", 1, True),
    ]


def test_rank_related_rows_caps_candidates() -> None:
    rows = [
        {"chunk_id": f"c{index:02d}", "via": "person", "shared": 1} for index in range(10)
    ]
    ranked = rank_related_rows(rows, seeds=["c00"])

    assert len(ranked) == GRAPH_HOP_CANDIDATES
    assert ranked[0].chunk_id == "c01"
    assert all(item.chunk_id != "c00" for item in ranked)


def test_related_chunks_empty_does_not_open_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("不应连接 Neo4j")

    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", boom)
    store = GraphStore(Settings())

    assert store.related_chunks([], doc_id=None) == ()
    assert store._driver is None


def test_get_by_chunk_ids_retrieves_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _RecordingQdrant()
    client.retrieve_points = [
        SimpleNamespace(payload={"chunk_id": "c1", "text": "甲", "doc_id": "d"}),
        SimpleNamespace(payload=None),
        SimpleNamespace(payload={"text": "没有 id"}),
        SimpleNamespace(payload={"chunk_id": "c1", "text": "重复"}),
    ]
    store = _bind_qdrant(monkeypatch, client)

    found = store.get_by_chunk_ids(["c1", "c1", "c2"])

    retrieve = [kwargs for name, kwargs in client.calls if name == "retrieve"]
    assert len(retrieve) == 1
    assert retrieve[0]["collection_name"] == "masskb_chunks"
    assert retrieve[0]["ids"] == [point_id_for_chunk("c1"), point_id_for_chunk("c2")]
    assert retrieve[0]["with_payload"] == list(SEARCH_PAYLOAD_FIELDS)
    assert retrieve[0]["with_vectors"] is False
    assert found == {"c1": {"chunk_id": "c1", "text": "甲", "doc_id": "d"}}
    forbidden = {
        "create_collection",
        "upsert",
        "delete",
        "delete_collection",
        "create_payload_index",
        "query_points",
    }
    assert forbidden.isdisjoint(_names(client))


def test_get_by_chunk_ids_empty_does_not_open_client() -> None:
    store = VectorStore(Settings())

    assert store.get_by_chunk_ids([]) == {}
    assert store._client is None


def _keeps_alias_endpoints(query: str) -> bool:
    """孤儿清理可以删掉空节点，但不能拆掉仍连着别名的表面节点。"""
    person = "NOT (()-[:MENTIONS]->(person)) AND NOT (person)-[:ALIAS_OF]-()"
    concept = "NOT (()-[:MENTIONS]->(concept)) AND NOT (concept)-[:ALIAS_OF]-()"
    return person in query and concept in query and "DETACH DELETE person" not in query


def _alias_link(kind: str = "person") -> AliasLink:
    return AliasLink(
        kind=kind,
        alias_name="孔子",
        canonical_name="孔丘",
        alias_id=entity_id(kind, "孔子"),
        canonical_id=entity_id(kind, "孔丘"),
    )


def test_replace_aliases_touches_only_this_document(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)
    link = _alias_link()
    concept = AliasLink(
        kind="concept",
        alias_name="仁义",
        canonical_name="仁",
        alias_id=entity_id("concept", "仁义"),
        canonical_id=entity_id("concept", "仁"),
    )

    assert store.replace_aliases("doc-a", [link, concept]) == 2

    delete_query, delete_params = driver.session_obj.queries[0]
    person_query, person_params = driver.session_obj.queries[1]
    concept_query, concept_params = driver.session_obj.queries[2]
    assert "ALIAS_OF {doc_id: $doc_id}" in delete_query
    assert "MENTIONS" not in delete_query
    assert "DETACH DELETE" not in delete_query
    assert delete_params == {"doc_id": "doc-a"}
    assert "MERGE (alias)-[:ALIAS_OF {doc_id: $doc_id}]->(canonical)" in person_query
    assert ":Person" in person_query
    assert person_params["doc_id"] == "doc-a"
    assert person_params["links"] == [
        {"alias_id": link.alias_id, "canonical_id": link.canonical_id}
    ]
    assert ":Concept" in concept_query
    assert concept_params["doc_id"] == "doc-a"
    assert "apoc" not in person_query.lower()
    assert _keeps_alias_endpoints(driver.session_obj.queries[3][0])
    assert driver.session_obj.write_calls == 1

    assert store.replace_aliases("doc-b", []) == 0

    doc_b_queries = driver.session_obj.queries[-2:]
    clear_query, clear_params = doc_b_queries[0]
    sweep_query, sweep_params = doc_b_queries[1]
    assert clear_params == {"doc_id": "doc-b"}
    assert "doc-a" not in clear_query
    assert "MENTIONS" not in clear_query
    assert _keeps_alias_endpoints(sweep_query)
    assert sweep_params == {}
    assert len(driver.session_obj.queries) == 6
    assert driver.session_obj.write_calls == 2


def test_replace_aliases_rejects_bad_links_before_write(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _GraphDriver()
    store = _bind_graph(monkeypatch, driver)
    bad = AliasLink(
        kind="place",
        alias_name="孔子",
        canonical_name="孔丘",
        alias_id="a",
        canonical_id="b",
    )

    with pytest.raises(ValueError, match="未知别名种类"):
        store.replace_aliases("doc-a", [bad])

    assert store._driver is None
    assert driver.sessions == 0


def test_related_chunks_alias_hop_is_one_edge_and_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """别名边不按写入它的 doc_id 过滤。分块过滤仍在，所以带 doc_id 时跨文档分块进不来。"""
    driver = _GraphDriver(rows=[{"chunk_id": "chunk-a", "via": "person", "shared": 1}])
    store = _bind_graph(monkeypatch, driver)

    found = store.related_chunks(["chunk-b"], doc_id=None, aliases=True)

    query, params = driver.session_obj.queries[0]
    assert query.count("[:ALIAS_OF]-") == 2
    assert "ALIAS_OF*" not in query
    assert "*1" not in query
    assert "ALIAS_OF {doc_id" not in query
    assert "rel.doc_id" not in query
    assert "other.doc_id = $doc_id" in query
    assert "LIMIT 4" in query
    assert "LIMIT 8" in query
    assert "LIMIT 2" in query
    assert "apoc" not in query.lower()
    assert params == {"chunk_ids": ["chunk-b"], "doc_id": None}
    assert found[0].chunk_id == "chunk-a"
    assert driver.session_obj.write_calls == 0


def test_related_chunks_alias_flag_off_keeps_the_old_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _GraphDriver(rows=[])
    store = _bind_graph(monkeypatch, driver)

    assert store.related_chunks(["seed"], doc_id="doc-b", aliases=False) == ()

    query, params = driver.session_obj.queries[0]
    assert "ALIAS_OF" not in query
    assert params == {"chunk_ids": ["seed"], "doc_id": "doc-b"}


def test_related_chunks_aliases_empty_does_not_open_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("不应连接 Neo4j")

    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", boom)
    store = GraphStore(Settings())

    assert store.related_chunks([], doc_id=None, aliases=True) == ()
    assert store._driver is None
