"""检索编排：向量入库与文档骨架同步，以及同一模型上的近邻检索。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client.http.exceptions import UnexpectedResponse

from masskb.config import Settings, get_settings
from masskb.embeddings import EmbeddingClient
from masskb.graph_store import GraphStore
from masskb.ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
    Document,
    IngestReport,
    ingest_path,
)
from masskb.vector_store import (
    DEFAULT_COLLECTION,
    DEFAULT_VECTOR_SIZE,
    VectorStore,
    point_id_for_chunk,
)

EMBED_BATCH_SIZE = 64
_PROBE_TEXT = "masskb"


class DimensionMismatchError(RuntimeError):
    """探测、分块或查询向量的维度与约定的 1536 不一致。"""


class GraphSyncError(RuntimeError):
    """向量已按文档提交，图谱事务未完成。整次 ingest 可重试。"""


@dataclass(frozen=True)
class SearchHit:
    """一条近邻命中。`score` 来自 Qdrant，正文来自 payload，前后 id 来自图。"""

    score: float
    text: str | None
    source: str | None
    chunk_id: str | None
    doc_id: str | None
    index: int | None
    start: int | None
    end: int | None
    prev_chunk_id: str | None
    next_chunk_id: str | None


@dataclass(frozen=True)
class GraphStatus:
    """这次检索有没有补上前后邻。`skipped` 表示没有去查 Neo4j。"""

    ok: bool
    skipped: bool
    error: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """一次向量检索。`doc_id` 是过滤条件，未过滤时为 None。"""

    query: str
    top_k: int
    collection: str
    embedding_model: str
    doc_id: str | None
    hits: tuple[SearchHit, ...]
    graph: GraphStatus


@dataclass(frozen=True)
class IngestWriteResult:
    """一次写入的结果。`points_upserted` 是本次 upsert 的点数。"""

    report: IngestReport
    collection: str
    vector_size: int
    points_upserted: int
    graph_documents: int
    graph_chunks: int
    graph_skipped: bool


def _validate_search(query: str, top_k: int, doc_id: str | None) -> None:
    if not isinstance(query, str) or query.strip() == "":
        raise ValueError("查询不能为空")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k 必须为正整数，收到 {top_k!r}")
    if doc_id is not None and (not isinstance(doc_id, str) or doc_id.strip() == ""):
        raise ValueError("doc_id 不能为空")


def _search_result(
    embedding_model: str,
    query: str,
    top_k: int,
    doc_id: str | None,
    hits: tuple[SearchHit, ...],
    graph: GraphStatus,
) -> SearchResult:
    return SearchResult(
        query=query,
        top_k=top_k,
        collection=DEFAULT_COLLECTION,
        embedding_model=embedding_model,
        doc_id=doc_id,
        hits=hits,
        graph=graph,
    )


def _skipped_graph() -> GraphStatus:
    return GraphStatus(ok=True, skipped=True, error=None)


def _search_hit(raw: dict[str, Any]) -> SearchHit:
    payload = raw.get("payload") or {}
    return SearchHit(
        score=float(raw["score"]),
        text=payload.get("text"),
        source=payload.get("source"),
        chunk_id=payload.get("chunk_id"),
        doc_id=payload.get("doc_id"),
        index=payload.get("index"),
        start=payload.get("start"),
        end=payload.get("end"),
        prev_chunk_id=None,
        next_chunk_id=None,
    )


def _hit_with_neighbors(
    hit: SearchHit,
    prev_chunk_id: str | None,
    next_chunk_id: str | None,
) -> SearchHit:
    return SearchHit(
        score=hit.score,
        text=hit.text,
        source=hit.source,
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        index=hit.index,
        start=hit.start,
        end=hit.end,
        prev_chunk_id=prev_chunk_id,
        next_chunk_id=next_chunk_id,
    )


def _graph_failure(doc_id: str, exc: Exception) -> GraphSyncError:
    return GraphSyncError(
        f"文档 {doc_id} 的 Qdrant 已写入，Neo4j 骨架未完成，可重试。"
        f"{type(exc).__name__}: {exc}"
    )


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "index": chunk.index,
        "start": chunk.start,
        "end": chunk.end,
        "text": chunk.text,
    }


class RetrievalPipeline:
    """组装向量、图谱与 Embedding。`ingest` 先写 Qdrant 再同步骨架。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = VectorStore(self.settings)
        self.graph_store = GraphStore(self.settings)
        self.embeddings = EmbeddingClient(self.settings)

    def healthcheck(self) -> dict[str, Any]:
        """检查向量库与图库连通性（不强制测 embedding）。"""
        qdrant = self.vector_store.healthcheck()
        neo4j = self.graph_store.healthcheck()
        return {
            "ok": bool(qdrant.get("ok") and neo4j.get("ok")),
            "qdrant": qdrant,
            "neo4j": neo4j,
            "openai_key_configured": self.settings.has_real_openai_key,
        }

    def ingest(
        self,
        path: Path,
        *,
        root: Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> IngestWriteResult:
        """切分并写入 Qdrant，再同步这篇文档的 Neo4j 骨架。不做检索。

        有分块时先探测一条向量的维度，再确保集合存在。维度或集合配置不符时中止，
        不 upsert，也不删除，也不连接 Neo4j。探测用的那条向量不会写入。
        每个文档的向量 upsert 和过期点删除都成功后，才写该文档的图。
        """
        report = ingest_path(path, root=root, chunk_size=chunk_size, overlap=overlap)
        if not any(doc.chunks for doc in report.files):
            return self._clear_docs_if_collection_exists(report)

        probe = self.embeddings.embed_one(_PROBE_TEXT)
        if len(probe) != DEFAULT_VECTOR_SIZE:
            raise self._dimension_error(len(probe), before_collection=True)

        self.vector_store.ensure_collection()
        upserted = 0
        graph_documents = 0
        graph_chunks = 0
        for doc in report.files:
            if not doc.chunks:
                self.vector_store.delete_by_doc_id(doc.doc_id)
                self._delete_graph_document(doc.doc_id)
                graph_documents += 1
                continue
            vectors = self._embed_document(doc.chunks)
            ids = [point_id_for_chunk(chunk.chunk_id) for chunk in doc.chunks]
            payloads = [_payload(chunk) for chunk in doc.chunks]
            self.vector_store.upsert_points(ids, vectors, payloads)
            self.vector_store.delete_by_doc_id(doc.doc_id, keep_ids=ids)
            graph_chunks += self._sync_graph_document(doc)
            graph_documents += 1
            upserted += len(ids)
        return self._result(
            report,
            upserted,
            graph_documents=graph_documents,
            graph_chunks=graph_chunks,
            graph_skipped=False,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_id: str | None = None,
    ) -> SearchResult:
        """用与入库相同的模型做近邻检索，再按 chunk_id 补前后邻。

        先生成查询向量，再看集合是否存在。集合不存在、没有点或过滤后没有命中时，
        返回空的 `hits`，并且不连接 Neo4j。缺 Key、维度不符、集合配置不符或 Qdrant
        连不上时抛错。Neo4j 补邻接失败时仍返回向量命中。
        """
        _validate_search(query, top_k, doc_id)
        vector = self.embeddings.embed_one(query)
        if len(vector) != DEFAULT_VECTOR_SIZE:
            raise DimensionMismatchError(
                f"向量维度不符：模型 {self.settings.embedding_model} 返回 {len(vector)} 维，"
                f"期望 {DEFAULT_VECTOR_SIZE}。未执行检索。"
            )
        if not self.vector_store.has_collection():
            return _search_result(
                self.settings.embedding_model,
                query,
                top_k,
                doc_id,
                (),
                _skipped_graph(),
            )
        self.vector_store.require_compatible_collection()
        try:
            raw_hits = self.vector_store.search(vector, top_k=top_k, doc_id=doc_id)
        except UnexpectedResponse as exc:
            if exc.status_code != 404:
                raise
            raw_hits = []
        hits = tuple(_search_hit(item) for item in raw_hits)
        return self._attach_neighbors(
            _search_result(
                self.settings.embedding_model,
                query,
                top_k,
                doc_id,
                hits,
                _skipped_graph(),
            )
        )

    def close(self) -> None:
        self.vector_store.close()
        self.graph_store.close()

    def _clear_docs_if_collection_exists(self, report: IngestReport) -> IngestWriteResult:
        """没有可写入的分块时不探测、不建集合。集合已在则先删点，再删骨架。"""
        if not self.vector_store.has_collection() or not report.files:
            return self._result(
                report,
                0,
                graph_documents=0,
                graph_chunks=0,
                graph_skipped=True,
            )
        for doc in report.files:
            self.vector_store.delete_by_doc_id(doc.doc_id)
            self._delete_graph_document(doc.doc_id)
        return self._result(
            report,
            0,
            graph_documents=len(report.files),
            graph_chunks=0,
            graph_skipped=False,
        )

    def _sync_graph_document(self, doc: Document) -> int:
        try:
            self.graph_store.ensure_schema()
            return self.graph_store.upsert_document(doc)
        except GraphSyncError:
            raise
        except Exception as exc:  # 图失败不回滚已写入的向量
            raise _graph_failure(doc.doc_id, exc) from exc

    def _delete_graph_document(self, doc_id: str) -> None:
        try:
            self.graph_store.delete_document(doc_id)
        except GraphSyncError:
            raise
        except Exception as exc:  # 点已删，骨架失败时可重试
            raise _graph_failure(doc_id, exc) from exc

    def _attach_neighbors(self, result: SearchResult) -> SearchResult:
        chunk_ids: list[str] = []
        seen: set[str] = set()
        for hit in result.hits:
            if hit.chunk_id and hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                chunk_ids.append(hit.chunk_id)
        if not chunk_ids:
            return result
        try:
            neighbors = self.graph_store.neighbor_chunk_ids(chunk_ids)
        except Exception as exc:  # noqa: BLE001 — 邻接失败仍返回向量命中
            cleared = tuple(_hit_with_neighbors(hit, None, None) for hit in result.hits)
            status = GraphStatus(
                ok=False,
                skipped=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return _search_result(
                result.embedding_model,
                result.query,
                result.top_k,
                result.doc_id,
                cleared,
                status,
            )
        enriched = []
        for hit in result.hits:
            pair = neighbors.get(hit.chunk_id) if hit.chunk_id else None
            if pair is None:
                enriched.append(hit)
            else:
                enriched.append(_hit_with_neighbors(hit, pair[0], pair[1]))
        return _search_result(
            result.embedding_model,
            result.query,
            result.top_k,
            result.doc_id,
            tuple(enriched),
            GraphStatus(ok=True, skipped=False, error=None),
        )

    def _embed_document(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        vectors: list[list[float]] = []
        texts = [chunk.text for chunk in chunks]
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            embedded = self.embeddings.embed(batch)
            if len(embedded) != len(batch):
                raise DimensionMismatchError(
                    f"embedding 返回 {len(embedded)} 条，期望 {len(batch)} 条。未写入该文档。"
                )
            for vector in embedded:
                if len(vector) != DEFAULT_VECTOR_SIZE:
                    raise self._dimension_error(len(vector), before_collection=False)
            vectors.extend(embedded)
        return vectors

    def _dimension_error(self, actual: int, *, before_collection: bool) -> DimensionMismatchError:
        suffix = "未创建集合，也未写入。" if before_collection else "未写入该文档。"
        model = self.settings.embedding_model
        return DimensionMismatchError(
            f"向量维度不符：模型 {model} 返回 {actual} 维，期望 {DEFAULT_VECTOR_SIZE}。{suffix}"
        )

    @staticmethod
    def _result(
        report: IngestReport,
        points_upserted: int,
        *,
        graph_documents: int,
        graph_chunks: int,
        graph_skipped: bool,
    ) -> IngestWriteResult:
        return IngestWriteResult(
            report=report,
            collection=DEFAULT_COLLECTION,
            vector_size=DEFAULT_VECTOR_SIZE,
            points_upserted=points_upserted,
            graph_documents=graph_documents,
            graph_chunks=graph_chunks,
            graph_skipped=graph_skipped,
        )
