"""从环境变量 / .env 加载配置。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 默认从仓库根目录加载 .env（masskb/ 的上一级）
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _ensure_local_no_proxy() -> None:
    """保证本机 Docker 服务不走系统 HTTP 代理（如 macOS 系统代理）。

    不修改 Clash/VPN 本身，只把 localhost / 127.0.0.1 加入 NO_PROXY，
    避免 httpx / qdrant-client 把本地请求误送进代理导致超时。
    """
    hosts = ("localhost", "127.0.0.1", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        current = [p.strip() for p in os.environ.get(key, "").split(",") if p.strip()]
        merged = list(dict.fromkeys([*current, *hosts]))
        os.environ[key] = ",".join(merged)


_ensure_local_no_proxy()


class Settings(BaseModel):
    """运行时配置。"""

    qdrant_url: str = Field(default="http://localhost:6333")
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")
    openai_api_key: str = Field(default="")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    embedding_base_url: str = Field(default="")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536, ge=1)
    llm_model: str = Field(default="gpt-4o-mini")

    @property
    def has_real_openai_key(self) -> bool:
        key = (self.openai_api_key or "").strip()
        return bool(key) and key != "sk-your-api-key-here" and not key.startswith("sk-your-")


def _embedding_base_url() -> str:
    """空白视为未设置，嵌入继续走 LLM_BASE_URL。"""
    return os.getenv("EMBEDDING_BASE_URL", "").strip()


def _embedding_dimensions() -> int:
    """未设置时保持 1536。空字符串、非整数和小于 1 的值直接失败。"""
    if "EMBEDDING_DIMENSIONS" not in os.environ:
        return 1536
    raw = os.environ["EMBEDDING_DIMENSIONS"].strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"EMBEDDING_DIMENSIONS 必须是大于等于 1 的整数，实际为 {raw!r}。"
        ) from exc
    if value < 1:
        raise ValueError(f"EMBEDDING_DIMENSIONS 必须是大于等于 1 的整数，实际为 {value}。")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        embedding_base_url=_embedding_base_url(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dimensions=_embedding_dimensions(),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )
