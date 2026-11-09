"""图读路径的超时边界。

离线测试检查会话配置。拒绝连接和卡住的握手使用本机套接字。
在线测试只在 ``127.0.0.1:17687`` 已有临时 Neo4j 时运行，端口没开就跳过。
跳过不是在线成功。启动示例（数据目录用完即删，不要指向已有库）::

    bin/neo4j-admin dbms set-initial-password masskb-read-timeout-test
    # server.bolt.listen_address=127.0.0.1:17687
    bin/neo4j console
"""

from __future__ import annotations

import socket
import threading
import time

import pytest
from neo4j.exceptions import ConfigurationError, ServiceUnavailable

from masskb.config import Settings
from masskb.entities import EntityMention, entity_id
from masskb.graph_store import GraphStore
from masskb.ingest import Chunk, Document

_LOOPBACK_HOST = "127.0.0.1"
_LOOPBACK_BOLT_PORT = 17687
_LOOPBACK_BOLT = f"bolt://{_LOOPBACK_HOST}:{_LOOPBACK_BOLT_PORT}"
_LOOPBACK_PASSWORD = "masskb-read-timeout-test"
_FIXTURE_DOC_ID = "masskb-read-timeout-fixture"


def _refused_settings() -> Settings:
    return Settings(
        neo4j_uri="bolt://127.0.0.1:1",
        neo4j_user="neo4j",
        neo4j_password="unused",
    )


def _loopback_neo4j_open() -> bool:
    probe = socket.socket()
    probe.settimeout(0.2)
    try:
        return probe.connect_ex((_LOOPBACK_HOST, _LOOPBACK_BOLT_PORT)) == 0
    finally:
        probe.close()


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
    listener.bind((_LOOPBACK_HOST, 0))
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
            neo4j_uri=f"bolt://{_LOOPBACK_HOST}:{port}",
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


@pytest.mark.skipif(
    not _loopback_neo4j_open(),
    reason="127.0.0.1:17687 没有临时 Neo4j。跳过不是在线查询成功。",
)
def test_loopback_neo4j_neighbor_and_related_queries() -> None:
    settings = Settings(
        neo4j_uri=_LOOPBACK_BOLT,
        neo4j_user="neo4j",
        neo4j_password=_LOOPBACK_PASSWORD,
    )
    store = GraphStore(settings)
    source = "read-timeout-fixture.txt"
    first = Chunk(
        doc_id=_FIXTURE_DOC_ID,
        chunk_id="read-timeout-a",
        source=source,
        index=0,
        start=0,
        end=1,
        text="甲",
    )
    second = Chunk(
        doc_id=_FIXTURE_DOC_ID,
        chunk_id="read-timeout-b",
        source=source,
        index=1,
        start=1,
        end=2,
        text="甲",
    )
    person = entity_id("person", "甲")
    mentions = (
        EntityMention(
            chunk_id=first.chunk_id,
            kind="person",
            name="甲",
            entity_id=person,
        ),
        EntityMention(
            chunk_id=second.chunk_id,
            kind="person",
            name="甲",
            entity_id=person,
        ),
    )
    try:
        _remove_fixture(store)
        store.ensure_schema()
        store.upsert_document(
            Document(source=source, doc_id=_FIXTURE_DOC_ID, chunks=(first, second))
        )
        store.ensure_entity_schema()
        store.replace_mentions(
            _FIXTURE_DOC_ID,
            [first.chunk_id, second.chunk_id],
            mentions,
        )
        with store.driver.session() as session:
            assert session._config.max_transaction_retry_time == 30.0
            assert session._config.connection_acquisition_timeout == 60.0

        started = time.perf_counter()
        neighbors = store.neighbor_chunk_ids([first.chunk_id, second.chunk_id])
        neighbor_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        related = store.related_chunks([first.chunk_id], doc_id=_FIXTURE_DOC_ID)
        related_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        related_aliases = store.related_chunks(
            [first.chunk_id],
            doc_id=_FIXTURE_DOC_ID,
            aliases=True,
        )
        alias_elapsed = time.perf_counter() - started
        assert neighbors[first.chunk_id] == (None, second.chunk_id)
        assert neighbors[second.chunk_id] == (first.chunk_id, None)
        matched = [item for item in related if item.chunk_id == second.chunk_id]
        assert len(matched) == 1
        assert matched[0].via_next is True
        assert matched[0].mention_count >= 1
        assert any(item.chunk_id == second.chunk_id for item in related_aliases)
        assert neighbor_elapsed < 5
        assert related_elapsed < 5
        assert alias_elapsed < 5
    finally:
        _remove_fixture(store)
        store.close()


def _remove_fixture(store: GraphStore) -> None:
    """删掉本测试写入的节点。

    ``delete_document`` 的语句在 Neo4j 5.26 上以带 RETURN 的 CALL 结尾，
    服务端会拒绝。这里只用本次夹具的删除，不改产品里的写入语句。
    """

    def work(tx):
        tx.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) DETACH DELETE c",
            doc_id=_FIXTURE_DOC_ID,
        ).consume()
        tx.run(
            "MATCH (d:Document {doc_id: $doc_id}) DETACH DELETE d",
            doc_id=_FIXTURE_DOC_ID,
        ).consume()
        tx.run(
            """
            MATCH (person:Person)
            WHERE NOT (()-[:MENTIONS]->(person)) AND NOT (person)-[:ALIAS_OF]-()
            DELETE person
            """
        ).consume()

    with store.driver.session() as session:
        session.execute_write(work)
