"""把用户查询改写成一条检索文本。

解析是纯函数。聊天客户端由调用方在真正补全时创建。
本模块不创建 Qdrant、Neo4j 或 OpenAI 客户端。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

MAX_REWRITE_CHARS = 500

SYSTEM_PROMPT = (
    "你是检索查询改写器。把用户问题改写成一条更适合向量检索和关键词检索的查询。\n"
    "只根据用户问题本身，不要补充问题之外的事实，不要回答问题。\n"
    "不要输出多条查询。若原文已经适合检索，原样返回。\n"
    '只输出一个 JSON 对象：{"query": "..."}。\n'
    "不要 Markdown。"
)


@dataclass(frozen=True)
class RewriteParse:
    """解析结果。失败时 `text` 为空，调用方改用原文检索。"""

    ok: bool
    text: str | None
    error: str | None = None


def build_rewrite_messages(query: str) -> tuple[str, str]:
    """系统提示词和用户消息。用户问题原样放入，不做预处理。"""
    return SYSTEM_PROMPT, f"问题：{query}"


def parse_rewrite(raw: str) -> RewriteParse:
    """解析模型输出。`query` 会去掉首尾空白。

    空白、超长、非法 JSON 或形状不对时失败。与用户原文相同也算成功。
    """
    try:
        data = json.loads(_unwrap_json_fence(raw))
    except json.JSONDecodeError:
        return _failed("invalid_json")
    if not isinstance(data, dict) or not isinstance(data.get("query"), str):
        return _failed("invalid_shape")
    text = data["query"].strip()
    if text == "":
        return _failed("empty")
    if len(text) > MAX_REWRITE_CHARS:
        return _failed("too_long")
    return RewriteParse(ok=True, text=text, error=None)


def _unwrap_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        return text[len("```json") : -len("```")].strip()
    return text


def _failed(error: str) -> RewriteParse:
    return RewriteParse(ok=False, text=None, error=error)
