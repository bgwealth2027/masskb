"""稀疏编码：只做本地分词，不连接外部服务。"""

from __future__ import annotations

import inspect

import pytest

from masskb import sparse
from masskb.sparse import BM25_K1, encode_text


def test_chinese_bigrams_are_deterministic_and_sorted() -> None:
    first = encode_text("质量检测")
    assert first == encode_text("质量检测")
    indices, values = first
    assert len(indices) == 3
    assert indices == sorted(indices)
    assert values == [1.0, 1.0, 1.0]


def test_single_cjk_character_is_kept() -> None:
    indices, values = encode_text("质")
    assert len(indices) == 1
    assert values == [1.0]


def test_latin_tokens_are_lowercased_and_short_tokens_drop() -> None:
    hello, _values = encode_text("Hello")
    again, _again_values = encode_text("hello")
    assert hello == again
    assert encode_text("a") == ([], [])
    assert encode_text("") == ([], [])
    assert encode_text("   ") == ([], [])


def test_crc32_collision_sums_term_frequency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sparse.zlib, "crc32", lambda _data: 7)

    indices, values = encode_text("ab cd")

    assert indices == [7]
    assert values[0] == pytest.approx(2 * (BM25_K1 + 1) / (2 + BM25_K1))


def test_module_does_not_construct_external_clients() -> None:
    source = inspect.getsource(sparse)
    assert "qdrant_client" not in source
    assert "import openai" not in source
    assert "GraphDatabase" not in source
