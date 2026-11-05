"""OpenAI 兼容的嵌入 HTTP 边车。

`fastembed` 只在进程启动并加载模型时导入。协议函数可以离线测试。
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

EmbedFn = Callable[[list[str]], list[list[float]]]
_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


class ServerState:
    """进程内的加载状态。模型未就绪时健康检查返回 503。"""

    def __init__(self, model: str) -> None:
        self.model = model
        self.ready = False
        self.dimensions: int | None = None
        self.embed: EmbedFn | None = None


def health_response(
    *,
    ready: bool,
    model: str,
    dimensions: int | None,
) -> tuple[int, dict[str, Any]]:
    if not ready or dimensions is None:
        return 503, {"ok": False, "model": model, "dimensions": None}
    return 200, {"ok": True, "model": model, "dimensions": dimensions}


def embeddings_response(
    payload: dict[str, Any],
    *,
    model: str,
    embed: EmbedFn,
) -> tuple[int, dict[str, Any]]:
    """按已加载的模型生成向量。请求里的 model 必须相同，避免临时下载别的模型。"""
    requested = payload.get("model")
    if requested != model:
        return 400, {"error": f"模型不符：请求 {requested!r}，服务已加载 {model}。"}
    raw_input = payload.get("input")
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list) and all(isinstance(item, str) for item in raw_input):
        texts = list(raw_input)
    else:
        return 400, {"error": "input 必须是字符串或字符串数组"}
    if not texts:
        return 200, _embedding_body(model, [])
    try:
        vectors = embed(texts)
    except Exception as exc:  # noqa: BLE001 — 返回给调用方，不让 HTTP 线程退出
        return 500, {"error": f"{type(exc).__name__}: {exc}"}
    if len(vectors) != len(texts):
        return 500, {"error": f"embedding 返回 {len(vectors)} 条，期望 {len(texts)} 条。"}
    data: list[dict[str, Any]] = []
    for index, vector in enumerate(vectors):
        if not isinstance(vector, (list, tuple)) or not vector:
            return 500, {"error": "embedding 返回了空向量"}
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": [float(value) for value in vector],
            }
        )
    return 200, _embedding_body(model, data)


def load_embedder(model_name: str) -> tuple[EmbedFn, int]:
    """加载一个 ONNX 模型。名字不在该版本的支持表里时失败，不改用别的模型。"""
    from fastembed import TextEmbedding

    known = {item["model"] for item in TextEmbedding.list_supported_models()}
    if model_name not in known:
        raise RuntimeError(f"不支持的嵌入模型：{model_name}")
    embedding = TextEmbedding(model_name=model_name)
    probed = list(embedding.embed(["masskb"]))
    if len(probed) != 1:
        raise RuntimeError("嵌入探测返回条数不符")
    probe = [float(value) for value in probed[0]]
    if not probe:
        raise RuntimeError("嵌入探测返回空向量")

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = list(embedding.embed(texts))
        if len(vectors) != len(texts):
            raise RuntimeError(f"embedding 返回 {len(vectors)} 条，期望 {len(texts)} 条。")
        return [[float(value) for value in item] for item in vectors]

    return embed, len(probe)


def _embedding_body(model: str, data: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


class EmbedHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: ServerState) -> None:
        self.state = state
        super().__init__(server_address, EmbedHandler)


class EmbedHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if _path(self.path) != "/health":
            self._send(404, {"error": "not found"})
            return
        state = self._state()
        status, body = health_response(
            ready=state.ready,
            model=state.model,
            dimensions=state.dimensions,
        )
        self._send(status, body)

    def do_POST(self) -> None:
        if _path(self.path) != "/v1/embeddings":
            self._send(404, {"error": "not found"})
            return
        payload, error = _read_json(self)
        if error is not None:
            self._send(400, {"error": error})
            return
        assert payload is not None
        state = self._state()
        if not state.ready or state.embed is None:
            self._send(503, {"error": "模型尚未加载"})
            return
        status, body = embeddings_response(payload, model=state.model, embed=state.embed)
        self._send(status, body)

    def _state(self) -> ServerState:
        server = self.server
        assert isinstance(server, EmbedHTTPServer)
        return server.state

    def _send(self, status: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> None:
    model_name = os.environ.get("EMBEDDING_MODEL", _DEFAULT_MODEL).strip()
    if not model_name:
        raise SystemExit("EMBEDDING_MODEL 为空")
    state = ServerState(model_name)
    server = EmbedHTTPServer(("0.0.0.0", 8080), state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        embed, dimensions = load_embedder(model_name)
    except Exception as exc:
        server.shutdown()
        raise SystemExit(str(exc)) from exc
    state.embed = embed
    state.dimensions = dimensions
    state.ready = True
    print(f"embed ready model={model_name} dimensions={dimensions}", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()


def _path(raw: str) -> str:
    return raw.split("?", 1)[0]


def _read_json(handler: BaseHTTPRequestHandler) -> tuple[dict[str, Any] | None, str | None]:
    length_raw = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_raw)
    except ValueError:
        return None, "Content-Length 不是整数"
    if length < 0:
        return None, "Content-Length 不是整数"
    raw = handler.rfile.read(length) if length else b""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "请求体不是 JSON"
    if not isinstance(payload, dict):
        return None, "请求体必须是 JSON 对象"
    return payload, None


if __name__ == "__main__":
    main()
