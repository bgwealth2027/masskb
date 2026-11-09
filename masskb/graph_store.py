"""Neo4j 图数据库客户端封装。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from neo4j import Driver, GraphDatabase

from masskb.aliases import AliasLink
from masskb.config import Settings, get_settings
from masskb.entities import EntityMention
from masskb.ingest import Document

# 单独的语句，避免和数据写入放在同一个显式事务里。
_SCHEMA_STATEMENTS = (
    """
    CREATE CONSTRAINT masskb_document_doc_id IF NOT EXISTS
    FOR (d:Document) REQUIRE d.doc_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT masskb_chunk_chunk_id IF NOT EXISTS
    FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
    """,
    """
    CREATE INDEX masskb_chunk_doc_id IF NOT EXISTS
    FOR (c:Chunk) ON (c.doc_id)
    """,
)

_ENTITY_SCHEMA_STATEMENTS = (
    """
    CREATE CONSTRAINT masskb_person_person_id IF NOT EXISTS
    FOR (p:Person) REQUIRE p.person_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT masskb_concept_concept_id IF NOT EXISTS
    FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE
    """,
)

# 过期块和旧 NEXT_CHUNK 用 OPTIONAL MATCH。普通 MATCH 没命中时会让整条查询停掉，
# 首次写入就建不出前后关系。
_UPSERT_DOCUMENT = """
MERGE (d:Document {doc_id: $doc_id})
SET d.source = $source
WITH d
UNWIND $chunks AS row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.doc_id = $doc_id,
    c.source = $source,
    c.index = row.index,
    c.start = row.start,
    c.end = row.end
MERGE (d)-[:HAS_CHUNK]->(c)
WITH DISTINCT d
OPTIONAL MATCH (stale:Chunk {doc_id: $doc_id})
WHERE NOT stale.chunk_id IN $keep_ids
OPTIONAL MATCH (stale)-[mention:MENTIONS]->()
WITH d,
     collect(DISTINCT stale) AS stale_nodes,
     collect(DISTINCT mention.target_id) AS stale_targets
FOREACH (n IN [node IN stale_nodes WHERE node IS NOT NULL] | DETACH DELETE n)
WITH d, [target IN stale_targets WHERE target IS NOT NULL] AS stale_targets
CALL {
  WITH stale_targets
  UNWIND (CASE WHEN size(stale_targets) = 0 THEN [null] ELSE stale_targets END) AS eid
  WITH eid WHERE eid IS NOT NULL
  OPTIONAL MATCH (person:Person {person_id: eid})
  WHERE NOT (()-[:MENTIONS]->(person)) AND NOT (person)-[:ALIAS_OF]-()
  DELETE person
  WITH eid
  OPTIONAL MATCH (concept:Concept {concept_id: eid})
  WHERE NOT (()-[:MENTIONS]->(concept)) AND NOT (concept)-[:ALIAS_OF]-()
  DELETE concept
  RETURN count(*) AS removed_orphans
}
WITH d
OPTIONAL MATCH (c:Chunk {doc_id: $doc_id})-[rel:NEXT_CHUNK]-()
WITH d, collect(DISTINCT rel) AS old_rels
FOREACH (r IN [item IN old_rels WHERE item IS NOT NULL] | DELETE r)
WITH d
MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
WITH c ORDER BY c.index ASC
WITH collect(c) AS chunks
UNWIND range(0, size(chunks) - 2) AS i
WITH chunks[i] AS prev, chunks[i + 1] AS nxt
MERGE (prev)-[:NEXT_CHUNK]->(nxt)
"""

_DELETE_DOCUMENT = """
OPTIONAL MATCH (c:Chunk {doc_id: $doc_id})
OPTIONAL MATCH (c)-[mention:MENTIONS]->()
WITH collect(DISTINCT c) AS chunks, collect(DISTINCT mention.target_id) AS targets
FOREACH (n IN [node IN chunks WHERE node IS NOT NULL] | DETACH DELETE n)
WITH [target IN targets WHERE target IS NOT NULL] AS targets
OPTIONAL MATCH (d:Document {doc_id: $doc_id})
DETACH DELETE d
WITH targets
CALL {
  WITH targets
  UNWIND (CASE WHEN size(targets) = 0 THEN [null] ELSE targets END) AS eid
  WITH eid WHERE eid IS NOT NULL
  OPTIONAL MATCH (person:Person {person_id: eid})
  WHERE NOT (()-[:MENTIONS]->(person)) AND NOT (person)-[:ALIAS_OF]-()
  DELETE person
  WITH eid
  OPTIONAL MATCH (concept:Concept {concept_id: eid})
  WHERE NOT (()-[:MENTIONS]->(concept)) AND NOT (concept)-[:ALIAS_OF]-()
  DELETE concept
  RETURN count(*) AS removed_orphans
}
"""

# 先收旧 target_id，再删这些 chunk 上的边。空 mentions 也要跑这条，用来清空。
_CLEAR_MENTIONS = """
UNWIND $chunk_ids AS cid
OPTIONAL MATCH (:Chunk {chunk_id: cid})-[old:MENTIONS]->()
WITH collect(DISTINCT old.target_id) AS raw_targets
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
OPTIONAL MATCH (c)-[old:MENTIONS]->()
DELETE old
WITH DISTINCT raw_targets
RETURN [target IN raw_targets WHERE target IS NOT NULL] AS old_targets
"""

_MERGE_MENTIONS = """
UNWIND $mentions AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
FOREACH (_ IN CASE WHEN row.kind = 'person' THEN [1] ELSE [] END |
  MERGE (person:Person {person_id: row.entity_id})
  SET person.name = row.name
  MERGE (c)-[person_edge:MENTIONS]->(person)
  SET person_edge.kind = 'person', person_edge.target_id = row.entity_id
)
FOREACH (_ IN CASE WHEN row.kind = 'concept' THEN [1] ELSE [] END |
  MERGE (concept:Concept {concept_id: row.entity_id})
  SET concept.name = row.name
  MERGE (c)-[concept_edge:MENTIONS]->(concept)
  SET concept_edge.kind = 'concept', concept_edge.target_id = row.entity_id
)
"""

# 没有提及、但还连着别名边的表面节点必须留下。DETACH DELETE 会把 ALIAS_OF 一起拆掉，
# 而这一步发生在别名事务之前，写入失败也补不回旧边。
_DELETE_ORPHANS = """
UNWIND $target_ids AS eid
OPTIONAL MATCH (person:Person {person_id: eid})
WHERE NOT (()-[:MENTIONS]->(person)) AND NOT (person)-[:ALIAS_OF]-()
DELETE person
WITH eid
OPTIONAL MATCH (concept:Concept {concept_id: eid})
WHERE NOT (()-[:MENTIONS]->(concept)) AND NOT (concept)-[:ALIAS_OF]-()
DELETE concept
"""

# 只在别名事务里清掉两端都已落空的表面节点。和边的删除一起回滚。
_DELETE_ENTITIES_WITHOUT_MENTIONS_OR_ALIASES = """
MATCH (person:Person)
WHERE NOT (()-[:MENTIONS]->(person)) AND NOT (person)-[:ALIAS_OF]-()
DELETE person
WITH count(*) AS removed_persons
MATCH (concept:Concept)
WHERE NOT (()-[:MENTIONS]->(concept)) AND NOT (concept)-[:ALIAS_OF]-()
DELETE concept
"""

_NEIGHBORS = """
UNWIND $chunk_ids AS cid
MATCH (c:Chunk {chunk_id: cid})
OPTIONAL MATCH (prev:Chunk)-[:NEXT_CHUNK]->(c)
OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(nxt:Chunk)
RETURN c.chunk_id AS chunk_id,
       prev.chunk_id AS prev_chunk_id,
       nxt.chunk_id AS next_chunk_id
"""

# 每个种子的支路各自 LIMIT，避免热门概念把读查询撑开。数字是常量，不是用户输入。
GRAPH_HOP_CANDIDATES = 8
_RELATED_CHUNKS = """
UNWIND $chunk_ids AS cid
MATCH (seed:Chunk {chunk_id: cid})
CALL {
  WITH seed
  MATCH (seed)-[:NEXT_CHUNK]-(near:Chunk)
  WHERE ($doc_id IS NULL OR near.doc_id = $doc_id)
    AND NOT near.chunk_id IN $chunk_ids
  WITH near
  ORDER BY near.chunk_id ASC
  LIMIT 2
  RETURN near.chunk_id AS chunk_id, 'next_chunk' AS via, 0 AS shared
  UNION ALL
  WITH seed
  MATCH (seed)-[:MENTIONS]->(entity:Person)<-[:MENTIONS]-(other:Chunk)
  WHERE other.chunk_id <> seed.chunk_id
    AND NOT other.chunk_id IN $chunk_ids
    AND ($doc_id IS NULL OR other.doc_id = $doc_id)
  WITH other
  ORDER BY other.chunk_id ASC
  LIMIT 8
  RETURN other.chunk_id AS chunk_id, 'person' AS via, 1 AS shared
  UNION ALL
  WITH seed
  MATCH (seed)-[:MENTIONS]->(entity:Concept)<-[:MENTIONS]-(other:Chunk)
  WHERE other.chunk_id <> seed.chunk_id
    AND NOT other.chunk_id IN $chunk_ids
    AND ($doc_id IS NULL OR other.doc_id = $doc_id)
  WITH other
  ORDER BY other.chunk_id ASC
  LIMIT 8
  RETURN other.chunk_id AS chunk_id, 'concept' AS via, 1 AS shared
}
RETURN chunk_id, via, shared
"""

# 别名支路各自 LIMIT 4，仍是常量。只走一条 ALIAS_OF，不沿第二跳串到同组的其他表面名。
# 边的 doc_id 不参与过滤：表面节点是全局的，文档 A 写下的别名在未按 doc_id 过滤时
# 也能从文档 B 的同名分块走出去。分块本身仍服从 $doc_id。
_RELATED_CHUNKS_ALIASES = """
UNWIND $chunk_ids AS cid
MATCH (seed:Chunk {chunk_id: cid})
CALL {
  WITH seed
  MATCH (seed)-[:NEXT_CHUNK]-(near:Chunk)
  WHERE ($doc_id IS NULL OR near.doc_id = $doc_id)
    AND NOT near.chunk_id IN $chunk_ids
  WITH near
  ORDER BY near.chunk_id ASC
  LIMIT 2
  RETURN near.chunk_id AS chunk_id, 'next_chunk' AS via, 0 AS shared
  UNION ALL
  WITH seed
  MATCH (seed)-[:MENTIONS]->(entity:Person)<-[:MENTIONS]-(other:Chunk)
  WHERE other.chunk_id <> seed.chunk_id
    AND NOT other.chunk_id IN $chunk_ids
    AND ($doc_id IS NULL OR other.doc_id = $doc_id)
  WITH other
  ORDER BY other.chunk_id ASC
  LIMIT 8
  RETURN other.chunk_id AS chunk_id, 'person' AS via, 1 AS shared
  UNION ALL
  WITH seed
  MATCH (seed)-[:MENTIONS]->(entity:Concept)<-[:MENTIONS]-(other:Chunk)
  WHERE other.chunk_id <> seed.chunk_id
    AND NOT other.chunk_id IN $chunk_ids
    AND ($doc_id IS NULL OR other.doc_id = $doc_id)
  WITH other
  ORDER BY other.chunk_id ASC
  LIMIT 8
  RETURN other.chunk_id AS chunk_id, 'concept' AS via, 1 AS shared
  UNION ALL
  WITH seed
  MATCH (seed)-[:MENTIONS]->(:Person)-[:ALIAS_OF]-(:Person)<-[:MENTIONS]-(other:Chunk)
  WHERE other.chunk_id <> seed.chunk_id
    AND NOT other.chunk_id IN $chunk_ids
    AND ($doc_id IS NULL OR other.doc_id = $doc_id)
  WITH other
  ORDER BY other.chunk_id ASC
  LIMIT 4
  RETURN other.chunk_id AS chunk_id, 'person' AS via, 1 AS shared
  UNION ALL
  WITH seed
  MATCH (seed)-[:MENTIONS]->(:Concept)-[:ALIAS_OF]-(:Concept)<-[:MENTIONS]-(other:Chunk)
  WHERE other.chunk_id <> seed.chunk_id
    AND NOT other.chunk_id IN $chunk_ids
    AND ($doc_id IS NULL OR other.doc_id = $doc_id)
  WITH other
  ORDER BY other.chunk_id ASC
  LIMIT 4
  RETURN other.chunk_id AS chunk_id, 'concept' AS via, 1 AS shared
}
RETURN chunk_id, via, shared
"""

_DELETE_DOC_ALIASES = """
MATCH ()-[rel:ALIAS_OF {doc_id: $doc_id}]->()
DELETE rel
"""

_MERGE_PERSON_ALIASES = """
UNWIND $links AS row
MATCH (alias:Person {person_id: row.alias_id})
MATCH (canonical:Person {person_id: row.canonical_id})
MERGE (alias)-[:ALIAS_OF {doc_id: $doc_id}]->(canonical)
"""

_MERGE_CONCEPT_ALIASES = """
UNWIND $links AS row
MATCH (alias:Concept {concept_id: row.alias_id})
MATCH (canonical:Concept {concept_id: row.canonical_id})
MERGE (alias)-[:ALIAS_OF {doc_id: $doc_id}]->(canonical)
"""


class GraphStoreError(RuntimeError):
    """骨架读回的前后邻不唯一。"""


@dataclass(frozen=True)
class RelatedChunk:
    """一条图扩展候选。`mention_count` 是人物边与概念边条数之和。"""

    chunk_id: str
    mention_count: int
    via_next: bool


def rank_related_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[str],
    limit: int = GRAPH_HOP_CANDIDATES,
) -> tuple[RelatedChunk, ...]:
    """合并同一分块的多行，丢掉种子，再按提及数、前后边和 chunk_id 排序。"""
    seed_set = set(seeds)
    merged: dict[str, list[int | bool]] = {}
    order: list[str] = []
    for row in rows:
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id == "" or chunk_id in seed_set:
            continue
        via = row.get("via")
        if chunk_id not in merged:
            merged[chunk_id] = [0, False]
            order.append(chunk_id)
        if via == "next_chunk":
            merged[chunk_id][1] = True
        elif via in {"person", "concept"}:
            shared = row.get("shared")
            if isinstance(shared, int) and not isinstance(shared, bool) and shared > 0:
                add = shared
            else:
                add = 1
            merged[chunk_id][0] = int(merged[chunk_id][0]) + add
    ranked = [
        RelatedChunk(
            chunk_id=chunk_id,
            mention_count=int(merged[chunk_id][0]),
            via_next=bool(merged[chunk_id][1]),
        )
        for chunk_id in order
    ]
    ranked.sort(key=lambda item: (-item.mention_count, 0 if item.via_next else 1, item.chunk_id))
    return tuple(ranked[:limit])


class GraphStore:
    """Neo4j 薄封装：连通性检查、文档骨架、可选提及和别名，以及只读的相关分块。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._driver: Driver | None = None
        self._schema_ready = False
        self._entity_schema_ready = False

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
        return self._driver

    def healthcheck(self) -> dict[str, Any]:
        """检查 Neo4j Bolt 是否可用。成功返回 {"ok": True, ...}。"""
        try:
            self.driver.verify_connectivity()
            with self.driver.session() as session:
                record = session.run("RETURN 1 AS n").single()
                value = record["n"] if record else None
            return {
                "ok": True,
                "uri": self.settings.neo4j_uri,
                "ping": value,
            }
        except Exception as exc:  # noqa: BLE001 — healthcheck 需要吞并并报告
            return {
                "ok": False,
                "uri": self.settings.neo4j_uri,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def ensure_schema(self) -> None:
        """创建文档骨架的唯一约束和 doc_id 索引。同一实例只发一次。"""
        if self._schema_ready:
            return

        def work(tx: Any) -> None:
            for statement in _SCHEMA_STATEMENTS:
                tx.run(statement).consume()

        with self.driver.session() as session:
            session.execute_write(work)
        self._schema_ready = True

    def ensure_entity_schema(self) -> None:
        """创建 Person / Concept 唯一约束。同一实例只发一次，不影响骨架语句。"""
        if self._entity_schema_ready:
            return

        def work(tx: Any) -> None:
            for statement in _ENTITY_SCHEMA_STATEMENTS:
                tx.run(statement).consume()

        with self.driver.session() as session:
            session.execute_write(work)
        self._entity_schema_ready = True

    def upsert_document(self, doc: Document) -> int:
        """MERGE 一篇文档的骨架，并按 index 重建 NEXT_CHUNK。返回分块数。"""
        if not doc.chunks:
            raise ValueError("空文档应使用 delete_document")
        chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "index": chunk.index,
                "start": chunk.start,
                "end": chunk.end,
            }
            for chunk in doc.chunks
        ]
        keep_ids = [chunk.chunk_id for chunk in doc.chunks]

        def work(tx: Any) -> None:
            tx.run(
                _UPSERT_DOCUMENT,
                doc_id=doc.doc_id,
                source=doc.source,
                chunks=chunks,
                keep_ids=keep_ids,
            ).consume()

        with self.driver.session() as session:
            session.execute_write(work)
        return len(doc.chunks)

    def delete_document(self, doc_id: str) -> None:
        """删除该 doc_id 的 Chunk 和 Document。库里没有时也成功。"""

        def work(tx: Any) -> None:
            tx.run(_DELETE_DOCUMENT, doc_id=doc_id).consume()

        with self.driver.session() as session:
            session.execute_write(work)

    def neighbor_chunk_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> dict[str, tuple[str | None, str | None]]:
        """读取一跳前后邻。图里没有的 id 不出现在结果里。空输入不打开驱动。"""
        unique: list[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk_id)
        if not unique:
            return {}

        def work(tx: Any) -> list[dict[str, Any]]:
            result = tx.run(_NEIGHBORS, chunk_ids=unique)
            return [
                {
                    "chunk_id": record["chunk_id"],
                    "prev_chunk_id": record["prev_chunk_id"],
                    "next_chunk_id": record["next_chunk_id"],
                }
                for record in result
            ]

        with self.driver.session() as session:
            rows = session.execute_read(work)

        found: dict[str, tuple[str | None, str | None]] = {}
        for row in rows:
            chunk_id = row["chunk_id"]
            if chunk_id in found:
                raise GraphStoreError(f"分块 {chunk_id} 的前后邻不唯一")
            found[chunk_id] = (row["prev_chunk_id"], row["next_chunk_id"])
        return found

    def replace_mentions(
        self,
        doc_id: str,
        chunk_ids: Sequence[str],
        mentions: Sequence[EntityMention],
    ) -> tuple[int, int, int]:
        """替换这些 chunk 上的 MENTIONS。返回 (persons, concepts, mentions)。

        计数来自校验后的入参，不回读 Neo4j。`chunk_ids` 为空且没有提及时不打开驱动。
        提及指向本次列表之外的分块时抛出 ValueError，不写。
        同一事务内先删旧边，再 MERGE 节点和边，最后只删除本次旧目标里已经没有提及、
        也没有 ALIAS_OF 的 Person / Concept。仍连着别名边的节点留下，语句不删除别名边。
        `doc_id` 只用于调用方标识这篇文档，语句不按它扫描其他文档。
        """
        unique: list[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk_id)
        allowed = set(unique)
        rows: list[dict[str, str]] = []
        for mention in mentions:
            if mention.kind not in {"person", "concept"}:
                raise ValueError(f"文档 {doc_id} 含有未知实体种类 {mention.kind}")
            if mention.chunk_id not in allowed:
                raise ValueError(
                    f"文档 {doc_id} 的提及分块 {mention.chunk_id} 不在本次文档中"
                )
            rows.append(
                {
                    "chunk_id": mention.chunk_id,
                    "kind": mention.kind,
                    "entity_id": mention.entity_id,
                    "name": mention.name,
                }
            )
        if not unique:
            return (0, 0, 0)

        persons = {row["entity_id"] for row in rows if row["kind"] == "person"}
        concepts = {row["entity_id"] for row in rows if row["kind"] == "concept"}
        new_ids = persons | concepts

        def work(tx: Any) -> None:
            record = tx.run(_CLEAR_MENTIONS, chunk_ids=unique).single()
            old_targets = [] if record is None else list(record["old_targets"] or [])
            if rows:
                tx.run(_MERGE_MENTIONS, mentions=rows).consume()
            retired: list[str] = []
            retired_seen: set[str] = set()
            for target_id in old_targets:
                if target_id and target_id not in new_ids and target_id not in retired_seen:
                    retired_seen.add(target_id)
                    retired.append(target_id)
            if retired:
                tx.run(_DELETE_ORPHANS, target_ids=retired).consume()

        with self.driver.session() as session:
            session.execute_write(work)
        return len(persons), len(concepts), len(rows)

    def replace_aliases(self, doc_id: str, links: Sequence[AliasLink]) -> int:
        """替换该文档写下的 ALIAS_OF。返回准备写入的边数。

        空列表也会打开驱动，只删除 `doc_id` 等于本文档的别名边。
        不改 MENTIONS，也不删除其他文档的别名边。
        同一事务末尾只删除已经既没有提及也没有别名边的表面节点。事务失败时边和节点一起留下。
        端点必须已经是 Person / Concept。本方法不创建表面节点，也不回读确认。
        """
        persons: list[dict[str, str]] = []
        concepts: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for link in links:
            if link.kind not in {"person", "concept"}:
                raise ValueError(f"文档 {doc_id} 含有未知别名种类 {link.kind}")
            if link.alias_id == "" or link.canonical_id == "" or link.alias_id == link.canonical_id:
                raise ValueError(f"文档 {doc_id} 含有空的或自指向的别名")
            key = (link.kind, link.alias_id, link.canonical_id)
            if key in seen:
                raise ValueError(f"文档 {doc_id} 含有重复别名")
            seen.add(key)
            row = {"alias_id": link.alias_id, "canonical_id": link.canonical_id}
            if link.kind == "person":
                persons.append(row)
            else:
                concepts.append(row)

        def work(tx: Any) -> None:
            tx.run(_DELETE_DOC_ALIASES, doc_id=doc_id).consume()
            if persons:
                tx.run(_MERGE_PERSON_ALIASES, links=persons, doc_id=doc_id).consume()
            if concepts:
                tx.run(_MERGE_CONCEPT_ALIASES, links=concepts, doc_id=doc_id).consume()
            tx.run(_DELETE_ENTITIES_WITHOUT_MENTIONS_OR_ALIASES).consume()

        with self.driver.session() as session:
            session.execute_write(work)
        return len(persons) + len(concepts)

    def related_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        doc_id: str | None,
        aliases: bool = False,
    ) -> tuple[RelatedChunk, ...]:
        """只读前后块和同人物、同概念的其他分块。空输入不打开驱动。

        `aliases` 为假时使用不含 ALIAS_OF 的语句。为真时额外走一跳别名，
        每支路最多 4 条。不建约束，也不写节点。排序和 8 条上限在进程内完成。
        """
        unique: list[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk_id)
        if not unique:
            return ()
        statement = _RELATED_CHUNKS_ALIASES if aliases else _RELATED_CHUNKS

        def work(tx: Any) -> list[dict[str, Any]]:
            result = tx.run(statement, chunk_ids=unique, doc_id=doc_id)
            return [
                {
                    "chunk_id": record["chunk_id"],
                    "via": record["via"],
                    "shared": record["shared"],
                }
                for record in result
            ]

        with self.driver.session() as session:
            rows = session.execute_read(work)
        return rank_related_rows(rows, seeds=unique)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
