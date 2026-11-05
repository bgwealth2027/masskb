"""Embedding 客户端：不发起真实 HTTP。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from masskb.answer import ChatClient
from masskb.config import Settings
from masskb.embeddings import EmbeddingClient


def test_embed_empty_skips_api() -> None:
    client = EmbeddingClient(Settings(openai_api_key=""))
    assert client.embed([]) == []


def test_embed_requires_real_api_key() -> None:
    client = EmbeddingClient(Settings(openai_api_key="sk-your-api-key-here"))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client.embed(["hello"])


def test_client_is_built_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("masskb.embeddings.OpenAI", FakeOpenAI)
    settings = Settings(
        openai_api_key="sk-live-abc",
        llm_base_url="https://gateway.example/v1",
    )
    client = EmbeddingClient(settings)

    assert client.client is client.client
    assert captured == {
        "api_key": "sk-live-abc",
        "base_url": "https://gateway.example/v1",
    }


def test_embed_sorts_by_index_and_embed_one() -> None:
    settings = Settings(openai_api_key="sk-live-abc", embedding_model="embed-test")
    client = EmbeddingClient(settings)
    calls: list[dict] = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[0.2, 0.3]),
                    SimpleNamespace(index=0, embedding=[0.0, 0.1]),
                ]
            )

    client._client = SimpleNamespace(embeddings=FakeEmbeddings())

    assert client.embed(["a", "b"]) == [[0.0, 0.1], [0.2, 0.3]]
    assert calls[0] == {"model": "embed-test", "input": ["a", "b"]}
    assert client.embed_one("a") == [0.0, 0.1]


def test_dedicated_embedding_url_skips_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("masskb.embeddings.OpenAI", FakeOpenAI)
    client = EmbeddingClient(
        Settings(
            openai_api_key="",
            llm_base_url="https://llm.example/v1",
            embedding_base_url="http://embed:8080/v1",
        )
    )

    assert client.client is client.client
    assert captured == {"api_key": "local", "base_url": "http://embed:8080/v1"}


def test_dedicated_embedding_url_keeps_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("masskb.embeddings.OpenAI", FakeOpenAI)
    client = EmbeddingClient(
        Settings(
            openai_api_key="sk-live-abc",
            llm_base_url="https://llm.example/v1",
            embedding_base_url="  http://embed:8080/v1  ",
        )
    )

    assert captured == {}
    assert client.client
    assert captured == {"api_key": "sk-live-abc", "base_url": "http://embed:8080/v1"}


def test_chat_client_ignores_embedding_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("masskb.answer.OpenAI", FakeOpenAI)
    client = ChatClient(
        Settings(
            openai_api_key="sk-live-abc",
            llm_base_url="https://llm.example/v1",
            embedding_base_url="http://embed:8080/v1",
        )
    )

    assert client.client
    assert captured == {"api_key": "sk-live-abc", "base_url": "https://llm.example/v1"}


def test_chat_client_still_requires_real_key() -> None:
    client = ChatClient(
        Settings(
            openai_api_key="",
            embedding_base_url="http://embed:8080/v1",
        )
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _ = client.client


def test_import_masskb_does_not_load_local_embedding_runtime() -> None:
    import sys

    import masskb

    assert masskb.__version__
    assert "fastembed" not in sys.modules
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules
