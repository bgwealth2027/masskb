"""查询改写提示词与解析。不创建外部客户端。"""

from __future__ import annotations

from pathlib import Path

from masskb.rewrite import MAX_REWRITE_CHARS, SYSTEM_PROMPT, build_rewrite_messages, parse_rewrite


def test_prompt_contains_query_only() -> None:
    system, user = build_rewrite_messages("  质量  ")

    assert system == SYSTEM_PROMPT
    assert '{"query": "..."}' in system
    assert "不要输出多条查询" in system
    assert user == "问题：  质量  "
    assert "chunk_id" not in user
    assert "chunk_id" not in system


def test_parser_strips_and_accepts_same_text() -> None:
    parsed = parse_rewrite('{"query": "  质量  ", "note": "忽略"}')

    assert parsed.ok is True
    assert parsed.error is None
    assert parsed.text == "质量"

    same = parse_rewrite('{"query": "质量"}')
    assert same.ok is True
    assert same.text == "质量"


def test_parser_accepts_json_fence_and_rejects_other_fences() -> None:
    fenced = '```json\n{"query": "product quality"}\n```'
    parsed = parse_rewrite(fenced)
    assert parsed.ok is True
    assert parsed.text == "product quality"

    other = '```\n{"query": "product quality"}\n```'
    rejected = parse_rewrite(other)
    assert rejected.ok is False
    assert rejected.error == "invalid_json"
    assert rejected.text is None


def test_parser_rejects_invalid_json_and_shape() -> None:
    invalid = parse_rewrite("不是 JSON")
    assert invalid.ok is False
    assert invalid.error == "invalid_json"
    assert invalid.text is None

    missing = parse_rewrite('{"text": "质量"}')
    assert missing.error == "invalid_shape"

    not_string = parse_rewrite('{"query": ["质量"]}')
    assert not_string.error == "invalid_shape"

    top_level = parse_rewrite('["质量"]')
    assert top_level.error == "invalid_shape"

    bare = parse_rewrite('"质量"')
    assert bare.error == "invalid_shape"


def test_parser_rejects_empty_and_too_long() -> None:
    empty = parse_rewrite('{"query": "   "}')
    assert empty.ok is False
    assert empty.error == "empty"
    assert empty.text is None

    limit = parse_rewrite('{"query": "%s"}' % ("质" * MAX_REWRITE_CHARS))
    assert limit.ok is True
    assert limit.text is not None
    assert len(limit.text) == MAX_REWRITE_CHARS

    over = parse_rewrite('{"query": "%s"}' % ("质" * (MAX_REWRITE_CHARS + 1)))
    assert over.ok is False
    assert over.error == "too_long"
    assert over.text is None


def test_module_does_not_reference_external_clients() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("masskb", "rewrite.py").read_text(
        encoding="utf-8"
    )
    assert "openai" not in source
    assert "QdrantClient" not in source
    assert "GraphDatabase" not in source
