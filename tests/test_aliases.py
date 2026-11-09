"""别名配对是纯函数。不创建 Qdrant、Neo4j 或 OpenAI 客户端。"""

from __future__ import annotations

import json

from masskb.aliases import (
    MAX_ALIAS_PAIRS,
    AliasParseError,
    alias_flag_error,
    build_alias_messages,
    parse_alias_response,
)
from masskb.entities import entity_id


def test_flag_error_requires_entities_and_rejects_both_switches() -> None:
    assert alias_flag_error(entities=False, aliases=False, drop_aliases=False) is None
    assert alias_flag_error(entities=True, aliases=False, drop_aliases=False) is None
    assert alias_flag_error(entities=True, aliases=True, drop_aliases=False) is None
    assert alias_flag_error(entities=True, aliases=False, drop_aliases=True) is None
    assert alias_flag_error(entities=True, aliases=True, drop_aliases=True) is not None
    assert "entities" in (alias_flag_error(entities=False, aliases=True, drop_aliases=False) or "")
    assert "entities" in (alias_flag_error(entities=False, aliases=False, drop_aliases=True) or "")


def test_parse_pairs_only_names_from_this_document() -> None:
    raw = '{"persons": [["孔子", "孔丘"], ["孔子", "仲尼"]], "concepts": [["仁", "仁义"]]}'
    links, dropped = parse_alias_response(
        raw,
        persons=("孔子", "孔丘"),
        concepts=("仁",),
    )

    assert dropped == 2
    assert len(links) == 1
    link = links[0]
    assert link.kind == "person"
    assert link.canonical_name == "孔丘"
    assert link.alias_name == "孔子"
    assert link.canonical_id == entity_id("person", "孔丘")
    assert link.alias_id == entity_id("person", "孔子")
    assert link.alias_id != link.canonical_id
    assert "doc" not in link.alias_id


def test_parse_rejects_name_that_belongs_to_another_document() -> None:
    """别的文档里出现过的表面名，不在本次允许名单里就不能写边。"""
    links, dropped = parse_alias_response(
        '{"persons": [["孔子", "孔丘"]], "concepts": []}',
        persons=("孔子",),
        concepts=(),
    )

    assert links == ()
    assert dropped == 1


def test_parse_does_not_chain_shared_names() -> None:
    links, dropped = parse_alias_response(
        '{"persons": [["孔子", "孔丘"], ["孔子", "仲尼"]], "concepts": []}',
        persons=("孔子", "孔丘", "仲尼"),
        concepts=(),
    )

    pairs = {(item.alias_name, item.canonical_name) for item in links}
    assert dropped == 0
    assert pairs == {("孔子", "孔丘"), ("孔子", "仲尼")}
    assert ("孔丘", "仲尼") not in pairs
    assert ("仲尼", "孔丘") not in pairs


def test_parse_keeps_person_and_concept_apart() -> None:
    links, dropped = parse_alias_response(
        '{"persons": [["仁", "仁义"]], "concepts": [["仁", "仁义"]]}',
        persons=("仁",),
        concepts=("仁", "仁义"),
    )

    assert dropped == 1
    assert len(links) == 1
    assert links[0].kind == "concept"
    assert links[0].alias_id == entity_id("concept", links[0].alias_name)
    assert links[0].alias_id != entity_id("person", links[0].alias_name)


def test_parse_drops_self_pairs_duplicates_and_extras() -> None:
    groups: list[object] = [["甲", "乙"], ["甲", "乙"], ["甲", "甲"], ["丙", "丁", "戊"], "甲"]
    groups.extend(["甲", f"名{index}"] for index in range(MAX_ALIAS_PAIRS))
    raw = json.dumps({"persons": groups, "concepts": []}, ensure_ascii=False)
    persons = ["甲", "乙", *[f"名{index}" for index in range(MAX_ALIAS_PAIRS)]]
    links, dropped = parse_alias_response(raw, persons=persons, concepts=())

    assert len(links) == MAX_ALIAS_PAIRS
    assert dropped == 5
    assert len({(item.alias_id, item.canonical_id) for item in links}) == MAX_ALIAS_PAIRS


def test_parse_accepts_json_fence_and_rejects_bad_shape() -> None:
    fenced = '```json\n{"persons": [["孔子", "孔丘"]], "concepts": []}\n```'
    links, dropped = parse_alias_response(fenced, persons=("孔子", "孔丘"), concepts=())
    assert dropped == 0
    assert links[0].canonical_name == "孔丘"

    for raw in (
        "不是 JSON",
        '{"persons": []}',
        '{"persons": "孔子", "concepts": []}',
    ):
        try:
            parse_alias_response(raw, persons=("孔子", "孔丘"), concepts=())
        except AliasParseError:
            continue
        raise AssertionError(raw)


def test_messages_treat_names_as_data() -> None:
    system, user = build_alias_messages(("孔子", "孔丘"), ("仁",))

    assert "不可信数据" in system
    assert "不要翻译" in system
    assert "孔子" not in system
    assert '"persons": ["孔子", "孔丘"]' in user
    assert '"concepts": ["仁"]' in user
