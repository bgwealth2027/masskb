"""配置加载：只读环境变量，不访问 Qdrant / Neo4j / OpenAI。"""

from __future__ import annotations

import os

import pytest

from masskb import __version__
from masskb.config import Settings, _ensure_local_no_proxy, get_settings


def test_package_version() -> None:
    assert __version__ == "0.1.0"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("", False),
        ("   ", False),
        ("sk-your-api-key-here", False),
        ("  sk-your-api-key-here  ", False),
        ("sk-your-placeholder", False),
        ("sk-proj-example", True),
        ("test-key", True),
    ],
)
def test_has_real_openai_key(key: str, expected: bool) -> None:
    assert Settings(openai_api_key=key).has_real_openai_key is expected


def test_get_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("NEO4J_URI", "bolt://neo4j.test:7687")
    monkeypatch.setenv("NEO4J_USER", "reader")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-example")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embed.test/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-test")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    monkeypatch.setenv("LLM_MODEL", "chat-test")

    settings = get_settings()

    assert settings.qdrant_url == "http://qdrant.test:6333"
    assert settings.neo4j_uri == "bolt://neo4j.test:7687"
    assert settings.neo4j_user == "reader"
    assert settings.neo4j_password == "secret"
    assert settings.openai_api_key == "sk-live-example"
    assert settings.llm_base_url == "https://llm.example/v1"
    assert settings.embedding_base_url == "http://embed.test/v1"
    assert settings.embedding_model == "embed-test"
    assert settings.embedding_dimensions == 512
    assert settings.llm_model == "chat-test"
    assert settings.has_real_openai_key is True


def test_get_settings_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "QDRANT_URL",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = get_settings()

    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == ""
    assert settings.openai_api_key == ""
    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.embedding_base_url == ""
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.has_real_openai_key is False


def test_blank_embedding_base_url_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BASE_URL", "   ")
    assert get_settings().embedding_base_url == ""


@pytest.mark.parametrize("raw", ["0", "-1", "no", "1.5", ""])
def test_get_settings_rejects_bad_embedding_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", raw)
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        get_settings()


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://one:6333")
    first = get_settings()
    monkeypatch.setenv("QDRANT_URL", "http://two:6333")
    assert get_settings() is first
    assert get_settings().qdrant_url == "http://one:6333"

    get_settings.cache_clear()
    assert get_settings().qdrant_url == "http://two:6333"


def test_local_no_proxy_includes_loopback() -> None:
    _ensure_local_no_proxy()
    for key in ("NO_PROXY", "no_proxy"):
        hosts = [part.strip() for part in os.environ[key].split(",")]
        for host in ("localhost", "127.0.0.1", "::1"):
            assert host in hosts
        before = os.environ[key]
        _ensure_local_no_proxy()
        assert os.environ[key] == before
