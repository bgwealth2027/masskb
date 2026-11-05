"""回答提示词、引用校验和聊天客户端。不访问网络。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from masskb.answer import ChatClient, build_answer_messages, parse_answer
from masskb.config import Settings


class _Hit:
    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text
        self.next_chunk_id = "neighbor-id"


def test_prompt_contains_only_hit_text_and_chunk_id() -> None:
    system, user = build_answer_messages("质量", [_Hit("c1", "正文")])

    assert "chunk_id" in system
    assert "c1" in user
    assert "正文" in user
    assert "质量" in user
    assert "neighbor-id" not in user
    assert "neighbor-id" not in system


def test_parser_drops_unknown_ids_and_keeps_order() -> None:
    raw = '{"abstain": false, "answer": "有依据", "citations": ["c2", "nope", "c1", "c2"]}'

    parsed = parse_answer(raw, {"c1", "c2"})

    assert parsed.ok is True
    assert parsed.text == "有依据"
    assert parsed.citations == ("c2", "c1")


def test_parser_abstain_drops_citations() -> None:
    raw = '```json\n{"abstain": true, "answer": "不要", "citations": ["c1"]}\n```'

    parsed = parse_answer(raw, {"c1"})

    assert parsed.ok is True
    assert parsed.text is None
    assert parsed.citations == ()
    assert parsed.error is None


def test_parser_rejects_uncited_and_invalid_payloads() -> None:
    uncited = parse_answer(
        '{"abstain": false, "answer": "秘密", "citations": ["other"]}',
        {"c1"},
    )
    invalid = parse_answer("不是 JSON", {"c1"})
    shape = parse_answer('{"answer": "x"}', {"c1"})

    assert uncited.ok is False
    assert uncited.error == "uncited"
    assert uncited.text is None
    assert invalid.error == "invalid_json"
    assert invalid.text is None
    assert shape.error == "invalid_shape"


def test_chat_client_uses_llm_model_without_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[SimpleNamespace] = []

    def factory(**kwargs):
        completions = SimpleNamespace(kwargs=None)

        def create(**sent):
            completions.kwargs = sent
            message = SimpleNamespace(content="ok")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        completions.create = create
        client = SimpleNamespace(kwargs=kwargs, chat=SimpleNamespace(completions=completions))
        created.append(client)
        return client

    monkeypatch.setattr("masskb.answer.OpenAI", factory)
    settings = Settings(
        openai_api_key="sk-live-example",
        llm_model="chat-test",
        llm_base_url="http://llm.test/v1",
    )

    assert ChatClient(settings).complete("系统", "用户") == "ok"

    sent = created[0].chat.completions.kwargs
    assert created[0].kwargs["base_url"] == "http://llm.test/v1"
    assert sent["model"] == "chat-test"
    assert sent["temperature"] == 0
    assert "response_format" not in sent
    assert sent["messages"][0] == {"role": "system", "content": "系统"}
    assert sent["messages"][1] == {"role": "user", "content": "用户"}


def test_chat_client_missing_key_does_not_construct_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**_kwargs):
        raise AssertionError("不应创建 OpenAI")

    monkeypatch.setattr("masskb.answer.OpenAI", boom)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ChatClient(Settings(openai_api_key="sk-your-api-key-here")).complete("系统", "用户")
