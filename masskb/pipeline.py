"""检索编排：向量入库，以及尚未实现的检索入口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from masskb.config import Settings, get_settings
from masskb.embeddings import EmbeddingClient
from masskb.graph_store import GraphStore
from masskb.ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    Chunk,
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
    """探测或分块向量的维度与约定的 1536 不一致。"""


@dataclass(frozen=True)
class IngestWriteResult:
    """一次向量写入的结果。`points_upserted` 是本次 upsert 的点数。"""

    report: IngestReport
    collection: str
    vector_size: int
    points_upserted: int


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
    """组装向量、图谱与 Embedding。`ingest` 只写 Qdrant；`search` 仍未实现。"""

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
        """切分并写入 Qdrant。不写 Neo4j，不做检索。

        有分块时先探测一条向量的维度，再确保集合存在。维度或集合配置不符时中止，
        不 upsert，也不删除。探测用的那条向量不会写入。
        """
        report = ingest_path(path, root=root, chunk_size=chunk_size, overlap=overlap)
        if not any(doc.chunks for doc in report.files):
            return self._clear_docs_if_collection_exists(report)

        probe = self.embeddings.embed_one(_PROBE_TEXT)
        if len(probe) != DEFAULT_VECTOR_SIZE:
            raise self._dimension_error(len(probe), before_collection=True)

        self.vector_store.ensure_collection()
        upserted = 0
        for doc in report.files:
            if not doc.chunks:
                self.vector_store.delete_by_doc_id(doc.doc_id)
                continue
            vectors = self._embed_document(doc.chunks)
            ids = [point_id_for_chunk(chunk.chunk_id) for chunk in doc.chunks]
            payloads = [_payload(chunk) for chunk in doc.chunks]
            self.vector_store.upsert_points(ids, vectors, payloads)
            self.vector_store.delete_by_doc_id(doc.doc_id, keep_ids=ids)
            upserted += len(ids)
        return self._result(report, upserted)

    def search(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        """检索占位：尚未实现真实混合检索。"""
        raise NotImplementedError(
            "检索编排尚未实现。请先配置真实 OPENAI_API_KEY，再接入 embedding + Qdrant/Neo4j 查询。"
        )

    def close(self) -> None:
        self.vector_store.close()
        self.graph_store.close()

    def _clear_docs_if_collection_exists(self, report: IngestReport) -> IngestWriteResult:
        """没有可写入的分块时不探测、不建集合。集合已在则按 doc_id 清掉旧点。"""
        if self.vector_store.has_collection():
            for doc in report.files:
                self.vector_store.delete_by_doc_id(doc.doc_id)
        return self._result(report, 0)

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
    def _result(report: IngestReport, points_upserted: int) -> IngestWriteResult:
        return IngestWriteResult(
            report=report,
            collection=DEFAULT_COLLECTION,
            vector_size=DEFAULT_VECTOR_SIZE,
            points_upserted=points_upserted,
        )
