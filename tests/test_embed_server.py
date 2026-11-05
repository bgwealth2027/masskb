"""嵌入边车协议：不下载模型，不导入 fastembed。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
