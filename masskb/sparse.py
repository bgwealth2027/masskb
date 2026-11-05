"""进程内稀疏编码。

中文按重叠二字组，英文和数字按词。下标是 crc32，值是饱和词频。
IDF 由 Qdrant 的 modifier 在查询时乘上，这里不算文档频率。
本模块不创建 Qdrant、Neo4j 或 OpenAI 客户端。
"""

from __future__ import annotations

import unicodedata
import zlib

BM25_K1 = 1.2

# 计划里的汉字、平假名、片假名和谚文。不引入分词库。
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0x3040, 0x309F),
    (0x30A0, 0x30FF),
    (0xAC00, 0xD7AF),
)


def encode_text(text: str) -> tuple[list[int], list[float]]:
    """把正文编成升序下标和饱和词频。空结果表示这个点不带稀疏向量。"""
    tokens = _tokens(unicodedata.normalize("NFKC", text).lower())
    if not tokens:
        return [], []
    counts: dict[int, int] = {}
    for token in tokens:
        index = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
        counts[index] = counts.get(index, 0) + 1
    indices = sorted(counts)
    values = [_saturated_tf(counts[index]) for index in indices]
    return indices, values


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    buffer: list[str] = []
    mode: str | None = None

    def flush() -> None:
        nonlocal buffer, mode
        if buffer:
            if mode == "cjk":
                tokens.extend(_cjk_tokens("".join(buffer)))
            else:
                word = "".join(buffer)
                if len(word) >= 2:
                    tokens.append(word)
        buffer = []
        mode = None

    for char in text:
        if _is_cjk(char):
            if mode != "cjk":
                flush()
            mode = "cjk"
            buffer.append(char)
        elif char.isalnum():
            if mode != "word":
                flush()
            mode = "word"
            buffer.append(char)
        else:
            flush()
    flush()
    return tokens


def _cjk_tokens(run: str) -> list[str]:
    if len(run) == 1:
        return [run]
    return [run[index : index + 2] for index in range(len(run) - 1)]


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def _saturated_tf(tf: int) -> float:
    return tf * (BM25_K1 + 1) / (tf + BM25_K1)
