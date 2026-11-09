"""嵌入边车协议：不下载模型，不导入 fastembed。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from openai.types import CreateEmbeddingResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL = "BAAI/bge-small-zh-v1.5"


def _server() -> ModuleType:
    path = ROOT / "embed" / "server.py"
    spec = importlib.util.spec_from_file_location("masskb_embed_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_import_does_not_load_fastembed() -> None:
    before = set(sys.modules)
    module = _server()
    assert "fastembed" not in sys.modules
    assert "torch" not in sys.modules
    loaded = set(sys.modules) - before
    assert "fastembed" not in loaded
    source = (ROOT / "embed" / "server.py").read_text(encoding="utf-8")
    head, _, _tail = source.partition("def load_embedder")
    assert "import fastembed" not in head
    assert "from fastembed" not in head
    assert module.load_embedder.__module__ == "masskb_embed_server"


def test_embeddings_response_keeps_order_and_openai_shape() -> None:
    module = _server()

    def fake(texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]

    status, body = module.embeddings_response(
        {"model": MODEL, "input": ["bb", "a"]},
        model=MODEL,
        embed=fake,
    )
    assert status == 200
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert body["data"][0]["embedding"] == [2.0, 0.0]
    assert body["data"][1]["embedding"] == [1.0, 0.0]
    parsed = CreateEmbeddingResponse.model_validate(body)
    assert parsed.model == MODEL
    assert parsed.data[0].embedding == [2.0, 0.0]


def test_embeddings_response_accepts_a_single_string() -> None:
    module = _server()
    status, body = module.embeddings_response(
        {"model": MODEL, "input": "ab"},
        model=MODEL,
        embed=lambda texts: [[0.5, 0.25]],
    )
    assert status == 200
    assert body["data"][0]["index"] == 0
    assert body["data"][0]["embedding"] == [0.5, 0.25]
    CreateEmbeddingResponse.model_validate(body)


def test_embeddings_response_rejects_other_models_without_embedding() -> None:
    module = _server()

    def boom(_texts: list[str]) -> list[list[float]]:
        raise AssertionError("不应嵌入")

    status, body = module.embeddings_response(
        {"model": "text-embedding-3-small", "input": ["a"]},
        model=MODEL,
        embed=boom,
    )
    assert status == 400
    assert "text-embedding-3-small" in body["error"]


def test_embeddings_response_empty_input() -> None:
    module = _server()

    def boom(_texts: list[str]) -> list[list[float]]:
        raise AssertionError("不应嵌入")

    status, body = module.embeddings_response(
        {"model": MODEL, "input": []},
        model=MODEL,
        embed=boom,
    )
    assert status == 200
    assert body["data"] == []
    parsed = CreateEmbeddingResponse.model_validate(body)
    assert parsed.data == []


def test_embeddings_response_count_mismatch_is_500() -> None:
    module = _server()
    status, body = module.embeddings_response(
        {"model": MODEL, "input": ["a", "b"]},
        model=MODEL,
        embed=lambda _texts: [[0.1, 0.2]],
    )
    assert status == 500
    assert "2" in body["error"]


def test_health_is_not_ready_until_dimensions_exist() -> None:
    module = _server()
    status, body = module.health_response(ready=False, model=MODEL, dimensions=None)
    assert status == 503
    assert body["ok"] is False
    status, body = module.health_response(ready=True, model=MODEL, dimensions=512)
    assert status == 200
    assert body == {"ok": True, "model": MODEL, "dimensions": 512}


def test_listen_address_defaults_match_container_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server()
    for name in ("HOST", "PORT", "EMBED_HOST", "EMBED_PORT"):
        monkeypatch.setenv(name, "127.0.0.1" if name.endswith("HOST") else "8081")
    assert module.listen_address([]) == ("0.0.0.0", 8080)


def test_listen_address_accepts_loopback_8081() -> None:
    module = _server()
    assert module.listen_address(["--host", "127.0.0.1", "--port", "8081"]) == (
        "127.0.0.1",
        8081,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--port", "0"],
        ["--port", "65536"],
        ["--port", "no"],
        ["--host", ""],
        ["--host", "   "],
    ],
)
def test_listen_address_rejects_bad_values(argv: list[str]) -> None:
    module = _server()
    with pytest.raises(SystemExit) as caught:
        module.listen_address(argv)
    assert caught.value.code == 2


def test_container_entrypoint_keeps_default_listen() -> None:
    dockerfile = (ROOT / "embed" / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["python", "server.py"]' in dockerfile
    assert "--host" not in dockerfile
    assert "--port" not in dockerfile
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:8081:8080" in compose


def _patch_server(monkeypatch: pytest.MonkeyPatch, module: ModuleType, seen: dict[str, object]) -> None:
    class FakeServer:
        def __init__(self, address: tuple[str, int], state: object) -> None:
            seen["address"] = address
            seen["state"] = state

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            seen["shutdown"] = True

    monkeypatch.setattr(module, "EmbedHTTPServer", FakeServer)


def test_main_uses_loopback_without_loading_fastembed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server()
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    seen: dict[str, object] = {}
    _patch_server(monkeypatch, module, seen)

    def fake_load(model_name: str) -> tuple[object, int]:
        assert model_name == MODEL

        def embed(texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

        return embed, 512

    monkeypatch.setattr(module, "load_embedder", fake_load)
    module.main(["--host", "127.0.0.1", "--port", "8081"])
    assert seen["address"] == ("127.0.0.1", 8081)
    state = seen["state"]
    assert isinstance(state, module.ServerState)
    assert state.ready is True
    assert state.dimensions == 512
    assert state.model == MODEL
    assert seen.get("shutdown") is not True
    assert "fastembed" not in sys.modules


def test_main_default_address_stays_all_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server()
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    seen: dict[str, object] = {}
    _patch_server(monkeypatch, module, seen)
    monkeypatch.setattr(module, "load_embedder", lambda _model: (lambda texts: [[0.0]], 1))
    module.main([])
    assert seen["address"] == ("0.0.0.0", 8080)
    state = seen["state"]
    assert isinstance(state, module.ServerState)
    assert state.ready is True
    assert "fastembed" not in sys.modules


def test_main_shuts_down_when_loader_fails_before_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _server()
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    seen: dict[str, object] = {}
    _patch_server(monkeypatch, module, seen)

    def boom(_model_name: str) -> tuple[object, int]:
        raise RuntimeError("探测失败")

    monkeypatch.setattr(module, "load_embedder", boom)
    with pytest.raises(SystemExit) as caught:
        module.main(["--host", "127.0.0.1", "--port", "8081"])
    assert caught.value.code == "探测失败"
    assert seen["address"] == ("127.0.0.1", 8081)
    state = seen["state"]
    assert isinstance(state, module.ServerState)
    assert state.ready is False
    assert seen.get("shutdown") is True
    assert "fastembed" not in sys.modules
