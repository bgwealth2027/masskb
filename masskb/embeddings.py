"""OpenAI 兼容 Embeddings 客户端。"""

from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from masskb.config import Settings, get_settings


class EmbeddingClient:
    """通过 OpenAI-compatible API 生成向量（不依赖本地 sentence-transformers）。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.has_real_openai_key:
                raise RuntimeError(
                    "缺少真实 OPENAI_API_KEY：请在 .env 中填入可用的 OpenAI 兼容 API Key。"
                )
            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.llm_base_url,
            )
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """对一批文本做 embedding，返回向量列表。"""
        if not texts:
            return []
        resp = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=list(texts),
        )
        # OpenAI 按 index 排序返回
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [item.embedding for item in ordered]

    def embed_one(self, text: str) -> list[float]:
        vectors = self.embed([text])
        return vectors[0]
