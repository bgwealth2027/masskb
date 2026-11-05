"""OpenAI 兼容 Embeddings 客户端。"""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from masskb.config import Settings, get_settings


def _dedicated_base_url(settings: Settings) -> str:
    return (settings.embedding_base_url or "").strip()


class EmbeddingClient:
    """通过 OpenAI-compatible API 生成向量，不在进程内加载模型。

    `embedding_base_url` 为空时使用 `LLM_BASE_URL`，并要求真实 API Key。
    设置后只把嵌入请求发到该地址，不再要求真实 Key。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            dedicated = _dedicated_base_url(self.settings)
            if dedicated:
                api_key = (self.settings.openai_api_key or "").strip() or "local"
                base_url = dedicated
            else:
                if not self.settings.has_real_openai_key:
                    raise RuntimeError(
                        "缺少真实 OPENAI_API_KEY：请在 .env 中填入可用的 OpenAI 兼容 API Key。"
                    )
                api_key = self.settings.openai_api_key
                base_url = self.settings.llm_base_url
            self._client = OpenAI(api_key=api_key, base_url=base_url)
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
