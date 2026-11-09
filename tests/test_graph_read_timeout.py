"""图读路径的超时边界。

默认 pytest 只跑离线部分：会话配置、对 ``127.0.0.1:1`` 的拒绝连接，以及本机临时套接字。
这些测试不探测、不写入已有数据库。

真实库集成必须显式打开，例如::

    python -m pytest tests/test_graph_read_timeout.py \
        --graph-read-bolt bolt://127.0.0.1:17687 \
        --graph-read-user neo4j \
        --graph-read-password '<临时库密码>'

请把地址指到空的临时库。该用例会确保 masskb 约束存在，写入本次唯一的文档、分块和人物，
读完后只删除这些节点。未传入 ``--graph-read-bolt`` 时跳过，并且不打开套接字。
"""

from __future__ import annotations

import socket
import threading
import time
import uuid

import pytest
from neo4j.exceptions import ConfigurationError, ServiceUnavailable

from masskb.config import Settings
from masskb.entities import EntityMention, entity_id
from masskb.graph_store import GraphStore
from masskb.ingest import Chunk, Document

_DELETE_THESE_CHUNKS = """
MATCH (c:Chunk)
WHERE c.chunk_id IN $chunk_ids
DETACH DELETE c
"""

_DELETE_THIS_DOCUMENT = """
MATCH (d:Document {doc_id: $doc_id})
DETACH DELETE d
"""

_DELETE_THIS_PERSON_IF_UNATTACHED = """
MATCH (person:Person {person_id: $entity_id})
WHERE NOT (()-[:MENTIONS]->(person)) AND NOT (person)-[:ALIAS_OF]-()
DELETE person
"""


def _refused_settings() -> Settings:
    return Settings(
        neo4j_uri="bolt://127.0.0.1:1",
        neo4j_user="neo4j",
        neo4j_password="unused",
    )


def test_read_session_overrides_leave_write_budget_on_the_same_driver() -> None:
    store = GraphStore(_refused_settings())
    try:
        with store.driver.session() as session:
            assert session._config.max_transaction_retry_time == 30.0
            assert session._config.connection_acquisition_timeout == 60.0
        with store._read_session() as session:
            assert session._config.max_transaction_retry_time == 0.0
            assert session._config.connection_acquisition_timeout == 2.0
        with store.driver.session() as session:
            assert session._config.max_transaction_retry_time == 30.0
            assert session._config.connection_acquisition_timeout == 60.0
        with pytest.raises(ConfigurationError):
            store.driver.session(connection_timeout=1.0)
    finally:
        store.close()


def test_refused_bolt_reads_fail_fast_and_health_stays_down() -> None:
    store = GraphStore(_refused_settings())
    try:
        started = time.perf_counter()
        with pytest.raises(ServiceUnavailable):
            store.neighbor_chunk_ids(["chunk-a"])
        neighbor_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        with pytest.raises(ServiceUnavailable):
            store.related_chunks(["chunk-a"], doc_id="doc")
        related_elapsed = time.perf_counter() - started

        started = time.perf_counter()
        health = store.healthcheck()
        health_elapsed = time.perf_counter() - started
    finally:
        store.close()

    assert neighbor_elapsed < 1.0
    assert related_elapsed < 1.0
    assert health_elapsed < 1.0
    assert health["ok"] is False
    assert health["uri"] == "bolt://127.0.0.1:1"
    assert "ServiceUnavailable" in health["error"]


def test_stalled_handshake_is_bounded_by_acquisition_timeout() -> None:
    """对端接受 TCP 后不完成 Bolt 握手。等待应落在获取期限附近，而不是 30 秒连接超时。"""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    accepted = threading.Event()
    release = threading.Event()

    def hold() -> None:
        try:
            listener.settimeout(8)
            conn, _addr = listener.accept()
        except OSError:
            return
        accepted.set()
        try:
            release.wait(8)
        finally:
            conn.close()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    store = GraphStore(
        Settings(
            neo4j_uri=f"bolt://127.0.0.1:{port}",
            neo4j_user="neo4j",
            neo4j_password="unused",
        )
    )
    try:
        started = time.perf_counter()
        with pytest.raises(ServiceUnavailable):
            store.neighbor_chunk_ids(["chunk-a"])
        elapsed = time.perf_counter() - started
    finally:
        release.set()
        listener.close()
        store.close()
        thread.join(timeout=2)

    assert accepted.is_set()
    assert 1.0 <= elapsed < 4.0


def test_cleanup_statements_name_only_this_run() -> None:
    assert "c.chunk_id IN $chunk_ids" in _DELETE_THESE_CHUNKS
    assert "d:Document {doc_id: $doc_id}" in _DELETE_THIS_DOCUMENT
    assert "person:Person {person_id: $entity_id}" in _DELETE_THIS_PERSON_IF_UNATTACHED
    assert "MATCH (person:Person)\n" not in _DELETE_THIS_PERSON_IF_UNATTACHED
    assert "DETACH DELETE person" not in _DELETE_THIS_PERSON_IF_UNATTACHED


def test_opt_in_neo4j_neighbor_and_related_queries(request: pytest.FixtureRequest) -> None:
    bolt = request.config.getoption("--graph-read-bolt").strip()
    if bolt == "":
        pytest.skip("未传入 --graph-read-bolt。默认不探测、不写入数据库。")
    password = request.config.getoption("--graph-read-password")
    if password == "":
        pytest.fail("已指定 --graph-read-bolt，还需要 --graph-read-password。不会用空密码连接。")
    user = request.config.getoption("--graph-read-user").strip() or "neo4j"
    run_id = uuid.uuid4().hex
    doc_id = f"masskb-read-timeout-{run_id}"
    chunk_a = f"masskb-read-timeout-{run_id}-a"
    chunk_b = f"masskb-read-timeout-{run_id}-b"
    person_name = f"masskb-person-{run_id}"
    person = entity_id("person", person_name)
    control_person = entity_id("person", f"masskb-control-{run_id}")
    settings = Settings(neo4j_uri=bolt, neo4j_user=user, neo4j_password=password)
    store = GraphStore(settings)
    source = f"read-timeout-{run_id}.txt"
    first = Chunk(
        doc_id=doc_id,
        chunk_id=chunk_a,
        source=source,
        index=0,
        start=0,
        end=1,
        text=person_name,
    )
    second = Chunk(
        doc_id=doc_id,
        chunk_id=chunk_b,
        source=source,
        index=1,
        start=1,
        end=2,
        text=person_name,
    )
    mentions = (
        EntityMention(
            chunk_id=chunk_a,
            kind="person",
            name=person_name,
            entity_id=person,
        ),
        EntityMention(
            chunk_id=chunk_b,
            kind="person",
            name=person_name,
            entity_id=person,
        ),
    )
    try:
        _create_control_person(store, control_person, f"masskb-control-{run_id}")
        store.ensure_schema()
        store.upsert_document(Document(source=source, doc_id=doc_id, chunks=(first, second)))
        store.ensure_entity_schema()
        store.replace_mentions(doc_id, [chunk_a, chunk_b], mentions)
        with store.driver.session() as session:
            assert session._config.max_transaction_retry_time == 30.0
            assert session._config.connection_acquisition_timeout == 60.0

        started = time.perf_counter()
        neighbors = store.neighbor_chunk_ids([chunk_a, chunk_b])
        neighbor_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        related = store.related_chunks([chunk_a], doc_id=doc_id)
        related_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        related_aliases = store.related_chunks([chunk_a], doc_id=doc_id, aliases=True)
        alias_elapsed = time.perf_counter() - started

        assert neighbors[chunk_a] == (None, chunk_b)
        assert neighbors[chunk_b] == (chunk_a, None)
        matched = [item for item in related if item.chunk_id == chunk_b]
        assert len(matched) == 1
        assert matched[0].via_next is True
        assert matched[0].mention_count >= 1
        assert any(item.chunk_id == chunk_b for item in related_aliases)
        assert neighbor_elapsed < 5
        assert related_elapsed < 5
        assert alias_elapsed < 5

        _remove_this_run(store, doc_id=doc_id, chunk_ids=(chunk_a, chunk_b), person_id=person)
        left = _count_this_run(
            store,
            doc_id=doc_id,
            chunk_ids=(chunk_a, chunk_b),
            person_id=person,
        )
        assert left == 0
        assert _count_person(store, control_person) == 1
    finally:
        _remove_this_run(store, doc_id=doc_id, chunk_ids=(chunk_a, chunk_b), person_id=person)
        _remove_this_run(store, doc_id=None, chunk_ids=(), person_id=control_person)
        store.close()


def _create_control_person(store: GraphStore, person_id: str, name: str) -> None:
    def work(tx):
        tx.run(
            "MERGE (person:Person {person_id: $entity_id}) SET person.name = $name",
            entity_id=person_id,
            name=name,
        ).consume()

    with store.driver.session() as session:
        session.execute_write(work)


def _remove_this_run(
    store: GraphStore,
    *,
    doc_id: str | None,
    chunk_ids: tuple[str, ...],
    person_id: str,
) -> None:
    """只删除本次 id。人物仅在已经没有任何提及和别名边时删除。"""

    def work(tx):
        if chunk_ids:
            tx.run(_DELETE_THESE_CHUNKS, chunk_ids=list(chunk_ids)).consume()
        if doc_id is not None:
            tx.run(_DELETE_THIS_DOCUMENT, doc_id=doc_id).consume()
        tx.run(_DELETE_THIS_PERSON_IF_UNATTACHED, entity_id=person_id).consume()

    with store.driver.session() as session:
        session.execute_write(work)


def _count_this_run(
    store: GraphStore,
    *,
    doc_id: str,
    chunk_ids: tuple[str, ...],
    person_id: str,
) -> int:
    def work(tx):
        record = tx.run(
            """
            MATCH (n)
            WHERE n.doc_id = $doc_id
               OR n.chunk_id IN $chunk_ids
               OR n.person_id = $entity_id
            RETURN count(n) AS n
            """,
            doc_id=doc_id,
            chunk_ids=list(chunk_ids),
            entity_id=person_id,
        ).single()
        return 0 if record is None else int(record["n"])

    with store.driver.session() as session:
        return session.execute_read(work)


def _count_person(store: GraphStore, person_id: str) -> int:
    def work(tx):
        record = tx.run(
            "MATCH (person:Person {person_id: $entity_id}) RETURN count(person) AS n",
            entity_id=person_id,
        ).single()
        return 0 if record is None else int(record["n"])

    with store.driver.session() as session:
        return session.execute_read(work)
