"""工具循环的提示词、动作解析和观察文本。不创建外部客户端。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from masskb.agent import (
    MAX_TOOL_QUERY_CHARS,
    SYSTEM_PROMPT,
    build_agent_messages,
    format_health_observation,
    format_search_observation,
    parse_agent_action,
)
from masskb.answer import parse_answer


def test_prompt_contains_question_and_observation_without_neighbors() -> None:
    system, user = build_agent_messages(
        "质量问题",
        ["- chunk_id: c\n  text: 命中正文"],
        top_k=5,
        doc_id=None,
        hybrid=False,
        rewrite=True,
        rerank=False,
        graph_hops=True,
    )

    assert system == SYSTEM_PROMPT
    assert "不可信数据" in system
    assert '{"action":"search","query":"检索文本"}' in system
    assert "healthcheck" in system
    assert "finish" in system
    assert "top_k" in system
    assert "search 不会再生成回答" in system
    assert "问题：质量问题" in user
    assert "命中正文" in user
    assert "chunk_id: c" in user
    assert "prev_chunk_id" not in user
    assert "hybrid：false" in user
    assert "rewrite：true" in user
    assert "rerank：false" in user
    assert "graph_hops：true" in user
    assert "graph_hops" in system
    assert "doc_id：null" in user


def test_search_observation_hides_neighbor_ids_and_textless_hits() -> None:
    hit = SimpleNamespace(
        chunk_id="keep",
        text="命中正文",
        score=0.5,
        prev_chunk_id="prev-id",
        next_chunk_id="next-id",
    )
    empty = SimpleNamespace(
        chunk_id="empty",
        text=None,
        score=0.1,
        prev_chunk_id="prev-id",
        next_chunk_id="next-id",
    )
    skipped = SimpleNamespace(ok=True, skipped=True, error=None, text=None)
    result = SimpleNamespace(
        hits=(hit, empty),
        score_kind="cosine",
        rewrite=SimpleNamespace(ok=True, skipped=False, error=None, text="改写后"),
        hybrid=skipped,
        rerank=skipped,
        graph=SimpleNamespace(ok=False, skipped=False, error="bolt"),
        graph_hops=SimpleNamespace(ok=True, skipped=True, error=None),
    )

    text = format_search_observation("质量", result)

    assert "命中正文" in text
    assert "chunk_id: keep" in text
    assert "改写后" in text
    assert "bolt" in text
    assert "score_kind: cosine" in text
    assert "graph_hops: ok=true skipped=true error=null" in text
    assert "prev_chunk_id" not in text
    assert "prev-id" not in text
    assert "next-id" not in text
    assert "empty" not in text


def test_health_observation_keeps_payload() -> None:
    text = format_health_observation({"ok": False, "qdrant": {"error": "拒绝"}})

    assert "healthcheck" in text
    assert "拒绝" in text
    assert "\\u" not in text


def test_parser_accepts_three_actions_and_strips_search_query() -> None:
    search = parse_agent_action('{"action":"search","query":"  质量  "}')
    assert search.ok is True
    assert search.kind == "search"
    assert search.query == "质量"
    assert search.error is None

    health = parse_agent_action('{"action":"healthcheck"}')
    assert health.ok is True
    assert health.kind == "healthcheck"
    assert health.query is None

    finish = parse_agent_action(
        '{"action":"finish","abstain":true,"answer":null,"citations":[]}'
    )
    assert finish.ok is True
    assert finish.kind == "finish"
    assert finish.raw is not None


def test_parser_accepts_json_fence_and_rejects_other_fences() -> None:
    fenced = '```json\n{"action":"search","query":"product quality"}\n```'
    parsed = parse_agent_action(fenced)
    assert parsed.ok is True
    assert parsed.query == "product quality"

    other = '```\n{"action":"search","query":"product quality"}\n```'
    rejected = parse_agent_action(other)
    assert rejected.ok is False
    assert rejected.error == "invalid_json"


def test_parser_rejects_invalid_json_shape_and_unknown_action() -> None:
    invalid = parse_agent_action("不是 JSON")
    assert invalid.ok is False
    assert invalid.error == "invalid_json"
    assert invalid.kind is None

    missing = parse_agent_action('{"query":"质量"}')
    assert missing.error == "invalid_shape"

    extra = parse_agent_action('{"action":"search","query":"质量","top_k":1}')
    assert extra.error == "invalid_shape"

    health_extra = parse_agent_action('{"action":"healthcheck","query":"质量"}')
    assert health_extra.error == "invalid_shape"

    not_string = parse_agent_action('{"action":"search","query":["质量"]}')
    assert not_string.error == "invalid_shape"

    unknown = parse_agent_action('{"action":"ingest"}')
    assert unknown.error == "unknown_action"

    top_level = parse_agent_action('["search"]')
    assert top_level.error == "invalid_shape"


def test_parser_rejects_empty_and_too_long_search_query() -> None:
    empty = parse_agent_action('{"action":"search","query":"   "}')
    assert empty.ok is False
    assert empty.error == "empty"
    assert empty.query is None

    limit = parse_agent_action(
        '{"action":"search","query":"%s"}' % ("质" * MAX_TOOL_QUERY_CHARS)
    )
    assert limit.ok is True
    assert limit.query is not None
    assert len(limit.query) == MAX_TOOL_QUERY_CHARS

    over = parse_agent_action(
        '{"action":"search","query":"%s"}' % ("质" * (MAX_TOOL_QUERY_CHARS + 1))
    )
    assert over.ok is False
    assert over.error == "too_long"


def test_finish_raw_is_checked_by_parse_answer() -> None:
    raw = (
        '```json\n{"action":"finish","abstain":false,"answer":"答",'
        '"citations":["keep","missing"]}\n```'
    )
    action = parse_agent_action(raw)
    assert action.ok is True
    assert action.raw == raw

    kept = parse_answer(action.raw or "", {"keep"})
    assert kept.ok is True
    assert kept.text == "答"
    assert kept.citations == ("keep",)

    uncited = parse_answer(action.raw or "", set())
    assert uncited.ok is False
    assert uncited.error == "uncited"
    assert uncited.text is None

    abstain_raw = '{"action":"finish","abstain":true,"answer":null,"citations":[]}'
    abstain = parse_agent_action(abstain_raw)
    parsed = parse_answer(abstain.raw or "", {"keep"})
    assert parsed.ok is True
    assert parsed.text is None
    assert parsed.citations == ()


def test_module_does_not_reference_external_clients() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("masskb", "agent.py").read_text(
        encoding="utf-8"
    )
    assert "openai" not in source
    assert "QdrantClient" not in source
    assert "GraphDatabase" not in source
