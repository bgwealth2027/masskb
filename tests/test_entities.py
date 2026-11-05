"""人物与概念名称的规范化、过滤和提示词。不创建外部客户端。"""

from __future__ import annotations

import hashlib

import pytest

from masskb.entities import (
    SYSTEM_PROMPT,
    EntityParseError,
    build_entity_messages,
    entity_id,
    normalize_entity_name,
    parse_entity_response,
)


def test_entity_id_is_stable_sha256() -> None:
    kind = "person"
    name = "孔子"
    digest = hashlib.sha256(f"{kind}:{name}".encode()).hexdigest()

    assert entity_id(kind, name) == digest
    assert entity_id(kind, name) == entity_id(kind, name)
    assert entity_id("concept", name) != digest


def test_normalize_nfkc_casefold_and_whitespace() -> None:
    assert normalize_entity_name("  ＡBC  ") == "abc"
    assert normalize_entity_name("孔  子") == "孔 子"
    assert normalize_entity_name("孔子") == "孔子"


def test_normalize_rejects_blank_stopwords_punctuation_and_length() -> None:
    assert normalize_entity_name("   ") is None
    assert normalize_entity_name("之") is None
    assert normalize_entity_name("所以") is None
    assert normalize_entity_name("，") is None
    assert normalize_entity_name("孔\n子") is None
    assert normalize_entity_name("仁" * 33) is None
    assert normalize_entity_name("仁") == "仁"
    assert normalize_entity_name("道") == "道"


def test_parser_keeps_substring_names_and_drops_the_rest() -> None:
    raw = '{"persons": [" 孔子 ", "之", "编造", "孔子"], "concepts": ["仁义", "所以"], "extra": 1}'

    mentions, dropped = parse_entity_response(raw, chunk_id="c1", text="孔子讲仁义。")

    assert dropped == 4
    assert [item.name for item in mentions] == ["孔子", "仁义"]
    assert [item.kind for item in mentions] == ["person", "concept"]
    assert mentions[0].chunk_id == "c1"
    assert mentions[0].entity_id == entity_id("person", "孔子")
    assert mentions[1].entity_id == entity_id("concept", "仁义")


def test_parser_caps_each_kind_at_eight_and_keeps_order() -> None:
    names = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬"]
    raw = '{"persons": [' + ", ".join(f'"{name}"' for name in names) + '], "concepts": ["甲"]}'

    mentions, dropped = parse_entity_response(
        raw,
        chunk_id="c",
        text="甲乙丙丁戊己庚辛壬",
    )

    assert [item.name for item in mentions if item.kind == "person"] == names[:8]
    assert [item.name for item in mentions if item.kind == "concept"] == ["甲"]
    assert dropped == 1
    assert entity_id("person", "甲") != entity_id("concept", "甲")


def test_parser_accepts_json_fence_and_empty_lists() -> None:
    fenced = '```json\n{"persons": [], "concepts": []}\n```'

    mentions, dropped = parse_entity_response(fenced, chunk_id="c", text="正文")

    assert mentions == ()
    assert dropped == 0


def test_parser_rejects_invalid_payloads() -> None:
    with pytest.raises(EntityParseError, match="invalid_json"):
        parse_entity_response("不是 JSON", chunk_id="c", text="正文")
    with pytest.raises(EntityParseError, match="invalid_shape"):
        parse_entity_response('{"persons": []}', chunk_id="c", text="正文")
    with pytest.raises(EntityParseError, match="invalid_shape"):
        parse_entity_response('{"persons": "孔子", "concepts": []}', chunk_id="c", text="孔子")
    with pytest.raises(EntityParseError, match="invalid_shape"):
        parse_entity_response('{"persons": [1], "concepts": []}', chunk_id="c", text="1")


def test_prompt_contains_only_this_chunk() -> None:
    system, user = build_entity_messages("c1", "孔子讲仁义。")

    assert system == SYSTEM_PROMPT
    assert "不可信" in system
    assert "c1" in user
    assert "孔子讲仁义。" in user
    assert "另一块" not in user
    assert "另一块" not in system


def test_casefold_matches_ascii_inside_the_chunk() -> None:
    mentions, dropped = parse_entity_response(
        '{"persons": ["LI BAI"], "concepts": []}',
        chunk_id="c",
        text="Li Bai wrote",
    )

    assert dropped == 0
    assert mentions[0].name == "li bai"


def test_module_does_not_construct_external_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("不应创建外部客户端")

    monkeypatch.setattr("masskb.embeddings.OpenAI", boom)
    monkeypatch.setattr("masskb.answer.OpenAI", boom)
    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", boom)
    monkeypatch.setattr("masskb.vector_store.QdrantClient", boom)

    parse_entity_response(
        '{"persons": ["孔子"], "concepts": []}',
        chunk_id="c",
        text="孔子",
    )