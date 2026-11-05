"""Qdrant 向量库客户端封装。"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from masskb.config import Settings, get_settings


class VectorStore:
    """Qdrant 薄封装：连通性检查 + 后续检索扩展点。"""

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

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
