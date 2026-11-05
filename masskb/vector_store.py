"""Qdrant 向量库客户端封装。"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from masskb.config import Settings, get_settings

DEFAULT_COLLECTION = "masskb_chunks"
DEFAULT_VECTOR_SIZE = 1536
UPSERT_BATCH_SIZE = 64
POINT_NAMESPACE = uuid.UUID("c0ffee00-1111-5222-8333-445566778899")


def point_id_for_chunk(chunk_id: str) -> str:
    """Qdrant point id。`chunk_id` 是 64 位十六进制，这里收成稳定的 UUID5。"""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


class CollectionConfigError(RuntimeError):
    """集合已存在，但维度、距离或向量形态与约定不一致。"""


SEARCH_PAYLOAD_FIELDS = (
    "doc_id",
    "chunk_id",
    "source",
    "index",
    "start",
    "end",
    "text",
)


class VectorStore:
    """Qdrant 薄封装：连通性检查、建集合、批量 upsert、按文档删除、近邻查询。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: QdrantClient | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                prefer_grpc=False,
                check_compatibility=False,
                timeout=10,
            )
        return self._client

    def healthcheck(self) -> dict[str, Any]:
        """检查 Qdrant 是否就绪。成功返回 {"ok": True, ...}。"""
        try:
            # get_collections 会真正打到服务端；比仅建客户端更可靠
            collections = self.client.get_collections()
            names = [c.name for c in collections.collections]
            return {
                "ok": True,
                "url": self.settings.qdrant_url,
                "collections": names,
            }
        except Exception as exc:  # noqa: BLE001 — healthcheck 需要吞并并报告
            return {
                "ok": False,
                "url": self.settings.qdrant_url,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def has_collection(self, name: str = DEFAULT_COLLECTION) -> bool:
        """集合是否已存在。只读 `get_collections`，不创建。"""
        listed = self.client.get_collections()
        return any(item.name == name for item in listed.collections)

    def ensure_collection(
        self,
        name: str = DEFAULT_COLLECTION,
        *,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        """集合不存在则按余弦距离创建；已存在则核对配置。不一致时不改集合。"""
        if self.has_collection(name):
            self.require_compatible_collection(name, vector_size=vector_size)
        else:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        self._ensure_doc_id_index(name)

    def upsert_points(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
        *,
        name: str = DEFAULT_COLLECTION,
    ) -> None:
        """按批 upsert。相同 point id 由 Qdrant 覆盖。"""
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids、vectors、payloads 长度必须一致")
        points = [
            PointStruct(id=point_id, vector=list(vector), payload=dict(payload))
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[start : start + UPSERT_BATCH_SIZE]
            self.client.upsert(collection_name=name, points=batch, wait=True)

    def delete_by_doc_id(
        self,
        doc_id: str,
        *,
        keep_ids: Sequence[str] = (),
        name: str = DEFAULT_COLLECTION,
    ) -> None:
        """删除该 `doc_id` 的点。`keep_ids` 非空时保留这些 point id。"""
        must = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        if keep_ids:
            selector = Filter(must=must, must_not=[HasIdCondition(has_id=list(keep_ids))])
        else:
            selector = Filter(must=must)
        self.client.delete(
            collection_name=name,
            points_selector=FilterSelector(filter=selector),
            wait=True,
        )

    def require_compatible_collection(
        self,
        name: str = DEFAULT_COLLECTION,
        *,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        """核对已有集合的维度与余弦距离。不创建集合，也不建索引。"""
        self._require_compatible_collection(name, vector_size)

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        name: str = DEFAULT_COLLECTION,
    ) -> list[dict[str, Any]]:
        """余弦近邻查询。不创建集合，也不写入。集合不存在时由 Qdrant 报错。"""
        query_filter = None
        if doc_id is not None:
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            )
        response = self.client.query_points(
            collection_name=name,
            query=list(vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=list(SEARCH_PAYLOAD_FIELDS),
            with_vectors=False,
        )
        hits: list[dict[str, Any]] = []
        for point in response.points:
            payload = getattr(point, "payload", None)
            hits.append({"score": float(point.score), "payload": dict(payload or {})})
        return hits

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _require_compatible_collection(self, name: str, vector_size: int) -> None:
        info = self.client.get_collection(name)
        vectors = info.config.params.vectors
        actual = _read_vector_params(vectors)
        if actual is None or actual != (vector_size, Distance.COSINE):
            raise CollectionConfigError(
                f"集合 {name} 的向量配置与约定不符："
                f"期望 size={vector_size}、distance={Distance.COSINE.value}，"
                f"实际为 {_describe_vectors(vectors)}。未写入。"
            )

    def _ensure_doc_id_index(self, name: str) -> None:
        try:
            self.client.create_payload_index(
                collection_name=name,
                field_name="doc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            # 索引已存在时保持集合不变，其它错误继续抛出。
            if "already exists" not in str(exc).lower():
                raise


def _read_vector_params(vectors: Any) -> tuple[Any, Any] | None:
    """只接受单向量配置。命名向量或缺少 size/distance 时返回 None。"""
    if isinstance(vectors, dict):
        return None
    size = getattr(vectors, "size", None)
    distance = getattr(vectors, "distance", None)
    if size is None or distance is None:
        return None
    return size, distance


def _describe_vectors(vectors: Any) -> str:
    if isinstance(vectors, dict):
        return "命名向量"
    size = getattr(vectors, "size", None)
    distance = getattr(vectors, "distance", None)
    if size is None and distance is None:
        return type(vectors).__name__
    return f"size={size} distance={distance}"
