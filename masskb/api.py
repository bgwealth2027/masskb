"""只读 HTTP：健康检查、检索，以及默认不运行的工具循环。入库仍走 CLI。

导入本模块只创建 FastAPI 应用，不连接 Qdrant、Neo4j 或 OpenAI。
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from masskb.agent import MAX_AGENT_STEPS
from masskb.pipeline import RetrievalPipeline


class UTF8JSONResponse(JSONResponse):
    """与 CLI 一样保留中文，不转成 \\uXXXX。"""

    def render(self, content: object) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")


class SearchRequest(BaseModel):
    """对应 `masskb search` 的参数。未知字段拒绝，避免误当成入库接口。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, strict=True)
    doc_id: str | None = None
    hybrid: bool = False
    answer: bool = False
    rerank: bool = False
    rewrite: bool = False

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("查询不能为空")
        return value

    @field_validator("doc_id")
    @classmethod
    def doc_id_not_blank(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError("doc_id 不能为空")
        return value


class AskRequest(BaseModel):
    """对应 `masskb ask` 的参数。未知字段拒绝。没有 `answer` 字段。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, strict=True)
    doc_id: str | None = None
    hybrid: bool = False
    rewrite: bool = False
    rerank: bool = False
    max_steps: int = Field(default=MAX_AGENT_STEPS, ge=1, le=MAX_AGENT_STEPS, strict=True)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if value.strip() == "":
            raise ValueError("查询不能为空")
        return value

    @field_validator("doc_id")
    @classmethod
    def doc_id_not_blank(cls, value: str | None) -> str | None:
        if value is not None and value.strip() == "":
            raise ValueError("doc_id 不能为空")
        return value


def create_app(pipeline: RetrievalPipeline | None = None) -> FastAPI:
    """`pipeline` 为空时，在 lifespan 里构造唯一的 `RetrievalPipeline`。

    传入实例时直接复用，并在退出时 `close()` 一次。健康检查、检索和 ask 共用一把锁。
    `ask` 内部再调用 `search` 与 `healthcheck` 时不再加锁。
    """
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = pipeline if pipeline is not None else RetrievalPipeline()
        app.state.pipeline = owned
        try:
            yield
        finally:
            owned.close()

    app = FastAPI(lifespan=lifespan, default_response_class=UTF8JSONResponse)

    @app.get("/health")
    def health(request: Request) -> UTF8JSONResponse:
        pipe = request.app.state.pipeline
        try:
            with lock:
                payload = pipe.healthcheck()
        except Exception as exc:  # noqa: BLE001 — 健康检查失败仍返回 JSON
            return UTF8JSONResponse({"detail": str(exc)}, status_code=503)
        status = 200 if payload.get("ok") else 503
        return UTF8JSONResponse(payload, status_code=status)

    @app.post("/search")
    def search(body: SearchRequest, request: Request) -> UTF8JSONResponse:
        pipe = request.app.state.pipeline
        try:
            with lock:
                result = pipe.search(
                    body.query,
                    top_k=body.top_k,
                    doc_id=body.doc_id,
                    hybrid=body.hybrid,
                    answer=body.answer,
                    rerank=body.rerank,
                    rewrite=body.rewrite,
                )
        except ValueError as exc:
            return UTF8JSONResponse({"detail": str(exc)}, status_code=422)
        except Exception as exc:  # noqa: BLE001 — 与 CLI 的 [FAIL] 同一段文字
            return UTF8JSONResponse({"detail": str(exc)}, status_code=503)
        return UTF8JSONResponse(asdict(result), status_code=200)

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> UTF8JSONResponse:
        pipe = request.app.state.pipeline
        try:
            with lock:
                result = pipe.ask(
                    body.query,
                    top_k=body.top_k,
                    doc_id=body.doc_id,
                    hybrid=body.hybrid,
                    rewrite=body.rewrite,
                    rerank=body.rerank,
                    max_steps=body.max_steps,
                )
        except ValueError as exc:
            return UTF8JSONResponse({"detail": str(exc)}, status_code=422)
        except Exception as exc:  # noqa: BLE001 — 与 CLI 的 [FAIL] 同一段文字
            return UTF8JSONResponse({"detail": str(exc)}, status_code=503)
        return UTF8JSONResponse(asdict(result), status_code=200)

    return app


app = create_app()
