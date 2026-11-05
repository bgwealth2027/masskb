"""重排提示词与顺序解析。不创建外部客户端。"""

from __future__ import annotations

from pathlib import Path

from masskb.rerank import SYSTEM_PROMPT, build_rerank_messages, parse_rerank_order


class _Hit:
    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.next_chunk_id = "neighbor-id"
        self.prev_chunk_id = "prev-id"


def test_prompt_contains_query_chunk_id_and_text_only() -> None:
    system, user = build_rerank_messages("  质量  ", [_Hit("c1", "正文")])

    assert system == SYSTEM_PROMPT
    assert "不可信数据" in system
    assert '{"order": ["chunk_id", ...]}' in system
    assert "问题：  质量  " in user
    assert "chunk_id: c1" in user
    assert "text: 正文" in user
    assert "neighbor-id" not in user
    assert "prev-id" not in user
    assert "neighbor-id" not in system


def test_parser_permutes_and_appends_missing_ids() -> None:
    raw = '{"order": ["c3", "nope", "c3", "c1", 7], "note": "忽略"}'

    parsed = parse_rerank_order(raw, ["c1", "c2", "c3"])

    assert parsed.ok is True
    assert parsed.error is None
    assert parsed.ids == ("c3", "c1", "c2")


def test_parser_empty_order_keeps_original_ids() -> None:
    parsed = parse_rerank_order('{"order": []}', ["c1", "c2"])

    assert parsed.ok is True
    assert parsed.ids == ("c1", "c2")


def test_parser_accepts_json_fence_and_rejects_other_fences() -> None:
    fenced = '```json\n{"order": ["c2", "c1"]}\n```'
    parsed = parse_rerank_order(fenced, ["c1", "c2"])
    assert parsed.ok is True
    assert parsed.ids == ("c2", "c1")

    other = '```\n{"order": ["c2"]}\n```'
    rejected = parse_rerank_order(other, ["c1", "c2"])
    assert rejected.ok is False
    assert rejected.error == "invalid_json"
    assert rejected.ids == ()


def test_parser_rejects_invalid_json_and_shape() -> None:
    invalid = parse_rerank_order("不是 JSON", ["c1"])
    assert invalid.ok is False
    assert invalid.error == "invalid_json"
    assert invalid.ids == ()

    missing = parse_rerank_order('{"ids": ["c1"]}', ["c1"])
    assert missing.error == "invalid_shape"

    not_list = parse_rerank_order('{"order": "c1"}', ["c1"])
    assert not_list.error == "invalid_shape"

    top_level = parse_rerank_order('["c1"]', ["c1"])
    assert top_level.error == "invalid_shape"


def test_module_does_not_reference_external_clients() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("masskb", "rerank.py").read_text(
        encoding="utf-8"
    )
    assert "openai" not in source
    assert "QdrantClient" not in source
    assert "GraphDatabase" not in source
