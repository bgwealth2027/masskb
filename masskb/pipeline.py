"""检索编排占位：向量检索 + 图谱 + LLM 的串联入口。"""

from __future__ import annotations

from typing import Any

from masskb.config import Settings, get_settings
from masskb.embeddings import EmbeddingClient
from masskb.graph_store import GraphStore
from masskb.vector_store import VectorStore


class RetrievalPipeline:
    """最小编排骨架：后续在此接入混合检索与 Agent。"""

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

    def search(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        """检索占位：尚未实现真实混合检索。"""
        raise NotImplementedError(
            "检索编排尚未实现。请先配置真实 OPENAI_API_KEY，再接入 embedding + Qdrant/Neo4j 查询。"
        )

    def close(self) -> None:
        self.vector_store.close()
        self.graph_store.close()
