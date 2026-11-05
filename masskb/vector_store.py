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
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from masskb.config import Settings, get_settings

DEFAULT_COLLECTION = "masskb_chunks"
DEFAULT_VECTOR_SIZE = 1536
SPARSE_VECTOR_NAME = "bm25"
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
    """Qdrant 薄封装：连通性检查、建集合、批量 upsert、按文档删除、近邻、稀疏查询和按 id 读 payload。"""

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
        """集合不存在则创建未命名稠密向量和 bm25 稀疏槽；已存在则核对。不一致时不改集合。"""
        if self.has_collection(name):
            self.require_compatible_collection(name, vector_size=vector_size)
            self._require_sparse_config(name)
        else:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
                },
            )
        self._ensure_doc_id_index(name)

    def upsert_points(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
        sparse_vectors: Sequence[tuple[Sequence[int], Sequence[float]]] | None = None,
        *,
        name: str = DEFAULT_COLLECTION,
    ) -> None:
        """按批 upsert。相同 point id 由 Qdrant 覆盖稠密向量和稀疏向量。"""
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids、vectors、payloads 长度必须一致")
        if sparse_vectors is not None and len(sparse_vectors) != len(ids):
            raise ValueError("ids、vectors、payloads、sparse_vectors 长度必须一致")
        points = [
            PointStruct(
                id=point_id,
                vector=_point_vector(vector, None if sparse_vectors is None else sparse_vectors[index]),
                payload=dict(payload),
            )
            for index, (point_id, vector, payload) in enumerate(zip(ids, vectors, payloads))
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
        response = self.client.query_points(
            collection_name=name,
            query=list(vector),
            query_filter=_doc_filter(doc_id),
            limit=top_k,
            with_payload=list(SEARCH_PAYLOAD_FIELDS),
            with_vectors=False,
        )
        return _hits_from_response(response)

    def search_sparse(
        self,
        indices: Sequence[int],
        values: Sequence[float],
        *,
        top_k: int = 5,
        doc_id: str | None = None,
        name: str = DEFAULT_COLLECTION,
    ) -> list[dict[str, Any]]:
        """查询名为 bm25 的稀疏向量。不创建集合，也不写入。"""
        response = self.client.query_points(
            collection_name=name,
            query=SparseVector(indices=list(indices), values=[float(item) for item in values]),
            using=SPARSE_VECTOR_NAME,
            query_filter=_doc_filter(doc_id),
            limit=top_k,
            with_payload=list(SEARCH_PAYLOAD_FIELDS),
            with_vectors=False,
        )
        return _hits_from_response(response)

    def get_by_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        name: str = DEFAULT_COLLECTION,
    ) -> dict[str, dict[str, Any]]:
        """按 chunk_id 读取 payload。不创建集合，也不写入。空输入不打开客户端。"""
        unique: list[str] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk_id)
        if not unique:
            return {}
        points = self.client.retrieve(
            collection_name=name,
            ids=[point_id_for_chunk(chunk_id) for chunk_id in unique],
            with_payload=list(SEARCH_PAYLOAD_FIELDS),
            with_vectors=False,
        )
        found: dict[str, dict[str, Any]] = {}
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            chunk_id = payload.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id and chunk_id not in found:
                found[chunk_id] = payload
        return found

    def sparse_ready(self, name: str = DEFAULT_COLLECTION) -> bool:
        """集合里是否已有 bm25 槽。只读，不创建，也不改配置。"""
        return SPARSE_VECTOR_NAME in _sparse_map(self.client.get_collection(name))

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

    def _require_sparse_config(self, name: str) -> None:
        """稠密配置已经核对过。缺 bm25 或 modifier 不是 IDF 时不改集合、不删除。"""
        info = self.client.get_collection(name)
        params = _sparse_map(info).get(SPARSE_VECTOR_NAME)
        if params is None:
            raise CollectionConfigError(
                f"集合 {name} 没有稀疏向量 {SPARSE_VECTOR_NAME}。"
                "请删除该集合后重新执行 ingest。未写入。"
            )
        modifier = getattr(params, "modifier", None)
        if modifier != Modifier.IDF:
            actual = getattr(modifier, "value", modifier)
            raise CollectionConfigError(
                f"集合 {name} 的稀疏向量 {SPARSE_VECTOR_NAME} 与约定不符："
                f"期望 modifier={Modifier.IDF.value}，实际为 {actual}。未写入。"
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


def _doc_filter(doc_id: str | None) -> Filter | None:
    if doc_id is None:
        return None
    return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])


def _hits_from_response(response: Any) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for point in response.points:
        payload = getattr(point, "payload", None)
        hits.append({"score": float(point.score), "payload": dict(payload or {})})
    return hits


def _sparse_map(info: Any) -> dict[str, Any]:
    sparse = getattr(info.config.params, "sparse_vectors", None)
    if not sparse:
        return {}
    if isinstance(sparse, dict):
        return sparse
    return dict(sparse)


def _point_vector(
    dense: Sequence[float],
    sparse: tuple[Sequence[int], Sequence[float]] | None,
) -> list[float] | dict[str, Any]:
    dense_list = list(dense)
    if sparse is None:
        return dense_list
    indices, values = sparse
    if len(indices) != len(values):
        raise ValueError("稀疏下标与权重长度必须一致")
    if not indices:
        return dense_list
    return {
        "": dense_list,
        SPARSE_VECTOR_NAME: SparseVector(
            indices=list(indices),
            values=[float(item) for item in values],
        ),
    }


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
