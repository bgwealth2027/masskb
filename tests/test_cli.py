"""CLI 与 scripts/healthcheck.py：不连接真实服务。"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from masskb.__main__ import main
from masskb.answer import AnswerStatus
from masskb.pipeline import (
    AgentStep,
    AskResult,
    DimensionMismatchError,
    GraphStatus,
    IngestWriteResult,
    RerankStatus,
    RewriteStatus,
    SearchHit,
    SearchResult,
)

_ROOT = Path(__file__).resolve().parents[1]


def _result(*, ok: bool, openai: bool) -> dict:
    qdrant = (
        {"ok": True, "url": "http://qdrant.test:6333", "collections": ["docs"]}
        if ok
        else {
            "ok": False,
            "url": "http://qdrant.test:6333",
            "error": "ConnectionError: refused",
        }
    )
    return {
        "ok": ok,
        "qdrant": qdrant,
        "neo4j": {"ok": True, "uri": "bolt://neo4j.test:7687", "ping": 1},
        "openai_key_configured": openai,
    }


def _patch(monkeypatch: pytest.MonkeyPatch, result: dict, closed: list[str]) -> None:
    class Pipeline:
        def healthcheck(self):
            return result

        def close(self):
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", Pipeline)


def test_healthcheck_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed: list[str] = []
    _patch(monkeypatch, _result(ok=True, openai=False), closed)

    assert main(["healthcheck"]) == 0

    out = capsys.readouterr().out
    assert "[OK]  Qdrant" in out
    assert "[OK]  Neo4j" in out
    assert "OPENAI_API_KEY 未配置真实值" in out
    payload = json.loads(out.split("---", 1)[1])
    assert payload["ok"] is True
    assert payload["qdrant"]["collections"] == ["docs"]
    assert closed == ["closed"]


def test_healthcheck_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed: list[str] = []
    _patch(monkeypatch, _result(ok=False, openai=True), closed)

    assert main(["healthcheck"]) == 1

    out = capsys.readouterr().out
    assert "[FAIL] Qdrant" in out
    assert "[OK]  Neo4j" in out
    assert "OPENAI_API_KEY 已配置" in out
    assert closed == ["closed"]


def test_default_command_is_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []
    _patch(monkeypatch, _result(ok=True, openai=False), closed)
    assert main([]) == 0
    assert closed == ["closed"]


def test_closes_when_healthcheck_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class Boom:
        def healthcheck(self):
            raise RuntimeError("boom")

        def close(self):
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", Boom)
    with pytest.raises(RuntimeError, match="boom"):
        main(["healthcheck"])
    assert closed == ["closed"]


def test_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "healthcheck" in out
    assert "search" in out
    assert "serve" in out


def _forbid_external_clients(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    created: list[str] = []

    def forbid(name: str):
        def _boom(*_args, **_kwargs):
            created.append(name)
            raise AssertionError(f"不应创建 {name}")

        return _boom

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", forbid("RetrievalPipeline"))
    monkeypatch.setattr("masskb.vector_store.VectorStore", forbid("VectorStore"))
    monkeypatch.setattr("masskb.vector_store.QdrantClient", forbid("QdrantClient"))
    monkeypatch.setattr("masskb.graph_store.GraphStore", forbid("GraphStore"))
    monkeypatch.setattr("masskb.graph_store.GraphDatabase.driver", forbid("Neo4jDriver"))
    monkeypatch.setattr("masskb.embeddings.EmbeddingClient", forbid("EmbeddingClient"))
    monkeypatch.setattr("masskb.embeddings.OpenAI", forbid("OpenAI"))
    monkeypatch.setattr("masskb.config.get_settings", forbid("get_settings"))
    return created


def test_ingest_dry_run_prints_counts_without_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "note.md").write_text("# 标题\n\n" + ("甲" * 120), encoding="utf-8")
    (root / "skip.py").write_text("x = 1\n", encoding="utf-8")
    created = _forbid_external_clients(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    code = main(
        [
            "ingest",
            "corpus",
            "--dry-run",
            "--root",
            str(root),
            "--chunk-size",
            "50",
            "--overlap",
            "10",
            "--preview",
            "1",
        ]
    )

    assert code == 0
    assert created == []
    captured = capsys.readouterr()
    assert captured.err == ""
    out = captured.out
    assert out.startswith("=== MassKB Ingest (dry-run) ===\n")
    assert "files: 1\nchunks: 4\n" in out
    assert "source: note.md" in out
    assert "# 标题" in out
    assert "... 还有 3 个分块未展示" in out
    assert "[INFO] dry-run：未连接 Qdrant、Neo4j 或 OpenAI。" in out
    assert "未写入向量库" not in out


def test_ingest_without_dry_run_writes_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    closed: list[str] = []

    class FakePipeline:
        def ingest(
            self,
            source: Path,
            *,
            root: Path | None = None,
            chunk_size: int = 800,
            overlap: int = 100,
            entities: bool = False,
        ):
            assert entities is False
            from masskb.ingest import ingest_path

            report = ingest_path(source, root=root, chunk_size=chunk_size, overlap=overlap)
            return IngestWriteResult(
                report=report,
                collection="masskb_chunks",
                vector_size=1536,
                points_upserted=report.chunk_count,
                graph_documents=report.file_count,
                graph_chunks=report.chunk_count,
                graph_skipped=False,
            )

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["ingest", str(path), "--root", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert out.startswith("=== MassKB Ingest ===\n")
    assert "files: 1\nchunks: 1\n" in out
    assert "hello" in out
    assert "(dry-run)" not in out
    assert "collection: masskb_chunks" in out
    assert "vector_size: 1536" in out
    assert "upserted: 1" in out
    assert "sparse: bm25" in out
    assert "graph_documents: 1" in out
    assert "graph_chunks: 1" in out
    assert "[INFO] 已写入 Qdrant 集合 masskb_chunks，并同步 Neo4j 文档骨架。未执行检索。" in out
    assert "entities:" not in out
    assert "graph_persons:" not in out
    assert closed == ["closed"]


def test_ingest_write_failure_exits_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    closed: list[str] = []

    class FakePipeline:
        def ingest(self, *_args, **_kwargs):
            raise DimensionMismatchError("向量维度不符")

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["ingest", str(path), "--root", str(tmp_path)]) == 1
    assert "[FAIL] 向量维度不符" in capsys.readouterr().err
    assert closed == ["closed"]


def test_ingest_bad_input_does_not_construct_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    created: list[str] = []

    class FakePipeline:
        def __init__(self) -> None:
            created.append("init")

        def close(self) -> None:
            created.append("close")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)
    py_file = tmp_path / "a.py"
    py_file.write_text("print(1)\n", encoding="utf-8")
    assert main(["ingest", str(py_file)]) == 1

    missing = tmp_path / "missing.md"
    assert main(["ingest", str(missing)]) == 1

    bad = tmp_path / "bad.txt"
    bad.write_bytes("中文".encode("gbk"))
    assert main(["ingest", str(bad)]) == 1

    assert created == []
    assert "[FAIL]" in capsys.readouterr().err


def test_ingest_dry_run_doc_id_stable_across_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "root"
    nested = root / "sub"
    nested.mkdir(parents=True)
    target = nested / "a.md"
    target.write_text("hello\n", encoding="utf-8")
    _forbid_external_clients(monkeypatch)

    monkeypatch.chdir(tmp_path)
    assert main(["ingest", str(target), "--dry-run", "--root", str(root)]) == 0
    first = capsys.readouterr().out

    monkeypatch.chdir(nested)
    assert main(["ingest", "a.md", "--dry-run", "--root", str(root)]) == 0
    second = capsys.readouterr().out

    assert "source: sub/a.md" in first
    assert "source: sub/a.md" in second
    doc_line = next(line for line in first.splitlines() if line.startswith("doc_id:"))
    chunk_line = next(line for line in first.splitlines() if "chunk_id=" in line)
    assert doc_line in second
    assert chunk_line in second


def test_ingest_rejects_bad_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    py_file = tmp_path / "a.py"
    py_file.write_text("print(1)\n", encoding="utf-8")
    assert main(["ingest", str(py_file), "--dry-run"]) == 1
    assert ".txt" in capsys.readouterr().err

    missing = tmp_path / "missing.md"
    assert main(["ingest", str(missing), "--dry-run"]) == 1
    assert "不存在" in capsys.readouterr().err

    bad = tmp_path / "bad.txt"
    bad.write_bytes("中文".encode("gbk"))
    assert main(["ingest", str(bad), "--dry-run"]) == 1
    assert "UTF-8" in capsys.readouterr().err

    ok = tmp_path / "ok.txt"
    ok.write_text("hi\n", encoding="utf-8")
    assert main(["ingest", str(ok), "--dry-run", "--chunk-size", "8", "--overlap", "8"]) == 1
    assert "overlap" in capsys.readouterr().err
    assert main(["ingest", str(ok), "--dry-run", "--preview", "-1"]) == 1


def test_ingest_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ingest", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--dry-run" in out
    assert "--entities" in out
    assert "chunk-size" in out
    assert "文档骨架" in out


def test_search_prints_json_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed: list[str] = []
    seen: dict = {}

    class FakePipeline:
        def search(
            self,
            query: str,
            *,
            top_k: int = 5,
            doc_id: str | None = None,
            hybrid: bool = False,
            answer: bool = False,
            rerank: bool = False,
            rewrite: bool = False,
        ) -> SearchResult:
            seen["query"] = query
            seen["top_k"] = top_k
            seen["doc_id"] = doc_id
            seen["hybrid"] = hybrid
            seen["answer"] = answer
            seen["rerank"] = rerank
            seen["rewrite"] = rewrite
            return SearchResult(
                query=query,
                top_k=top_k,
                collection="masskb_chunks",
                embedding_model="embed-test",
                doc_id=doc_id,
                hits=(
                    SearchHit(
                        score=0.5,
                        text="原文",
                        source="a.md",
                        chunk_id="c",
                        doc_id="d",
                        index=0,
                        start=0,
                        end=2,
                        prev_chunk_id=None,
                        next_chunk_id="c-next",
                    ),
                ),
                graph=GraphStatus(ok=True, skipped=False),
            )

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["search", "质量", "--top-k", "3", "--doc-id", "abc"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload == {
        "query": "质量",
        "top_k": 3,
        "collection": "masskb_chunks",
        "embedding_model": "embed-test",
        "doc_id": "abc",
        "hits": [
            {
                "score": 0.5,
                "text": "原文",
                "source": "a.md",
                "chunk_id": "c",
                "doc_id": "d",
                "index": 0,
                "start": 0,
                "end": 2,
                "prev_chunk_id": None,
                "next_chunk_id": "c-next",
                "dense_score": None,
                "sparse_score": None,
                "dense_rank": None,
                "sparse_rank": None,
            }
        ],
        "graph": {"ok": True, "skipped": False, "error": None},
        "mode": "dense",
        "score_kind": "cosine",
        "hybrid": {
            "ok": True,
            "skipped": True,
            "error": None,
            "rrf_k": None,
            "candidate_k": None,
        },
        "answer": {
            "ok": True,
            "skipped": True,
            "text": None,
            "citations": [],
            "model": None,
            "error": None,
        },
        "rerank": {
            "ok": True,
            "skipped": True,
            "error": None,
            "model": None,
        },
        "rewrite": {
            "ok": True,
            "skipped": True,
            "text": None,
            "model": None,
            "error": None,
        },
    }
    assert seen == {
        "query": "质量",
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": False,
        "answer": False,
        "rerank": False,
        "rewrite": False,
    }
    assert closed == ["closed"]


def test_search_empty_hits_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def search(
            self,
            query: str,
            *,
            top_k: int = 5,
            doc_id: str | None = None,
            hybrid: bool = False,
            answer: bool = False,
            rerank: bool = False,
            rewrite: bool = False,
        ) -> SearchResult:
            return SearchResult(
                query=query,
                top_k=top_k,
                collection="masskb_chunks",
                embedding_model="text-embedding-3-small",
                doc_id=doc_id,
                hits=(),
                graph=GraphStatus(ok=True, skipped=True),
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["search", "质量"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"] == []
    assert payload["doc_id"] is None
    assert payload["top_k"] == 5
    assert payload["query"] == "质量"
    assert payload["graph"] == {"ok": True, "skipped": True, "error": None}


def test_search_failure_exits_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed: list[str] = []

    class FakePipeline:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("缺少真实 OPENAI_API_KEY")

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["search", "质量"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[FAIL] 缺少真实 OPENAI_API_KEY" in captured.err
    assert closed == ["closed"]


def test_search_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["search", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--top-k" in out
    assert "--doc-id" in out
    assert "--hybrid" in out
    assert "--answer" in out
    assert "--rerank" in out
    assert "--rewrite" in out
    assert "--max-steps" not in out
    assert "chunk_id" in out


def test_search_answer_failure_prints_json_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def search(self, *_args, **_kwargs) -> SearchResult:
            return SearchResult(
                query="质量",
                top_k=5,
                collection="masskb_chunks",
                embedding_model="embed-test",
                doc_id=None,
                hits=(
                    SearchHit(
                        score=0.5,
                        text="原文",
                        source="a.md",
                        chunk_id="c",
                        doc_id="d",
                        index=0,
                        start=0,
                        end=2,
                        prev_chunk_id=None,
                        next_chunk_id=None,
                    ),
                ),
                graph=GraphStatus(ok=True, skipped=False),
                answer=AnswerStatus(
                    ok=False,
                    skipped=False,
                    text=None,
                    citations=(),
                    model="gpt-4o-mini",
                    error="uncited",
                ),
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["search", "质量", "--answer"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"][0]["chunk_id"] == "c"
    assert payload["answer"]["ok"] is False
    assert payload["answer"]["error"] == "uncited"
    assert payload["answer"]["text"] is None


def test_search_rerank_flag_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict = {}

    class FakePipeline:
        def search(self, query: str, **kwargs) -> SearchResult:
            seen["query"] = query
            seen.update(kwargs)
            return SearchResult(
                query=query,
                top_k=kwargs["top_k"],
                collection="masskb_chunks",
                embedding_model="embed-test",
                doc_id=None,
                hits=(),
                graph=GraphStatus(ok=True, skipped=True),
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["search", "质量", "--rerank"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert seen["rerank"] is True
    assert seen["hybrid"] is False
    assert seen["answer"] is False
    assert payload["rerank"]["skipped"] is True
    assert payload["hits"] == []


def test_search_rerank_failure_prints_json_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def search(self, *_args, **_kwargs) -> SearchResult:
            return SearchResult(
                query="质量",
                top_k=5,
                collection="masskb_chunks",
                embedding_model="embed-test",
                doc_id=None,
                hits=(
                    SearchHit(
                        score=0.4,
                        text="原文",
                        source="a.md",
                        chunk_id="c",
                        doc_id="d",
                        index=0,
                        start=0,
                        end=2,
                        prev_chunk_id=None,
                        next_chunk_id=None,
                    ),
                ),
                graph=GraphStatus(ok=True, skipped=False),
                score_kind="cosine",
                rerank=RerankStatus(
                    ok=False,
                    skipped=False,
                    error="invalid_json",
                    model="gpt-4o-mini",
                ),
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["search", "质量", "--rerank"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["hits"][0]["chunk_id"] == "c"
    assert payload["hits"][0]["score"] == 0.4
    assert payload["score_kind"] == "cosine"
    assert payload["rerank"]["ok"] is False
    assert payload["rerank"]["error"] == "invalid_json"


def test_search_rewrite_flag_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict = {}

    class FakePipeline:
        def search(self, query: str, **kwargs) -> SearchResult:
            seen["query"] = query
            seen.update(kwargs)
            return SearchResult(
                query=query,
                top_k=kwargs["top_k"],
                collection="masskb_chunks",
                embedding_model="embed-test",
                doc_id=None,
                hits=(),
                graph=GraphStatus(ok=True, skipped=True),
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["search", "质量", "--rewrite"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert seen["rewrite"] is True
    assert seen["hybrid"] is False
    assert seen["rerank"] is False
    assert seen["answer"] is False
    assert payload["rewrite"]["skipped"] is True
    assert payload["rewrite"]["text"] is None
    assert payload["hits"] == []


def test_search_rewrite_failure_prints_json_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def search(self, *_args, **_kwargs) -> SearchResult:
            return SearchResult(
                query="质量",
                top_k=5,
                collection="masskb_chunks",
                embedding_model="embed-test",
                doc_id=None,
                hits=(
                    SearchHit(
                        score=0.4,
                        text="原文",
                        source="a.md",
                        chunk_id="c",
                        doc_id="d",
                        index=0,
                        start=0,
                        end=2,
                        prev_chunk_id=None,
                        next_chunk_id=None,
                    ),
                ),
                graph=GraphStatus(ok=True, skipped=False),
                rewrite=RewriteStatus(
                    ok=False,
                    skipped=False,
                    text=None,
                    model="gpt-4o-mini",
                    error="invalid_json",
                ),
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["search", "质量", "--rewrite"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "质量"
    assert payload["hits"][0]["chunk_id"] == "c"
    assert payload["rewrite"]["ok"] is False
    assert payload["rewrite"]["error"] == "invalid_json"
    assert payload["rewrite"]["text"] is None


def test_ingest_entities_prints_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("孔子\n", encoding="utf-8")
    seen: dict = {}

    class FakePipeline:
        def ingest(
            self,
            source: Path,
            *,
            root: Path | None = None,
            chunk_size: int = 800,
            overlap: int = 100,
            entities: bool = False,
        ):
            from masskb.ingest import ingest_path

            seen["entities"] = entities
            report = ingest_path(source, root=root, chunk_size=chunk_size, overlap=overlap)
            return IngestWriteResult(
                report=report,
                collection="masskb_chunks",
                vector_size=1536,
                points_upserted=report.chunk_count,
                graph_documents=1,
                graph_chunks=report.chunk_count,
                graph_skipped=False,
                entities_skipped=False,
                graph_persons=1,
                graph_concepts=1,
                graph_mentions=2,
                entities_dropped=3,
            )

        def close(self) -> None:
            seen["closed"] = True

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["ingest", str(path), "--root", str(tmp_path), "--entities"]) == 0

    out = capsys.readouterr().out
    assert seen["entities"] is True
    assert seen["closed"] is True
    assert "entities: on" in out
    assert "graph_persons: 1" in out
    assert "graph_concepts: 1" in out
    assert "graph_mentions: 2" in out
    assert "entities_dropped: 3" in out
    assert "并替换本批文档的人物/概念提及" in out
    assert "并同步 Neo4j 文档骨架。未执行检索。" not in out


def test_ingest_dry_run_with_entities_creates_no_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("孔子\n", encoding="utf-8")
    created = _forbid_external_clients(monkeypatch)

    assert main(["ingest", str(path), "--dry-run", "--entities", "--root", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert created == []
    assert "[INFO] dry-run：未连接 Qdrant、Neo4j 或 OpenAI。" in out
    assert "[INFO] --entities 在 dry-run 中不抽取。" in out
    assert "entities: on" not in out


def test_ingest_entities_failure_exits_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "a.txt"
    path.write_text("孔子\n", encoding="utf-8")
    closed: list[str] = []

    class FakePipeline:
        def ingest(self, *_args, **_kwargs):
            from masskb.pipeline import EntitySyncError

            raise EntitySyncError("文档 d 的 Qdrant 与文档骨架已保留，实体未写入，可重试。")

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["ingest", str(path), "--entities"]) == 1
    assert "骨架已保留" in capsys.readouterr().err
    assert closed == ["closed"]


def test_search_missing_query_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["search"])
    assert exc.value.code == 2


def test_serve_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--host" in out
    assert "--port" in out
    assert "[api]" in out


def test_serve_rejects_port_out_of_range() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--port", "0"])
    assert exc.value.code == 2


def test_serve_runs_uvicorn_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(app: object, host: str, port: int) -> None:
        seen["app"] = app
        seen["host"] = host
        seen["port"] = port

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    assert main(["serve"]) == 0

    from masskb.api import app

    assert seen["app"] is app
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8080

    assert main(["serve", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 9000


def test_serve_missing_extra_exits_with_hint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_import = __import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked)
    assert main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "[FAIL]" in err
    assert '.[api]' in err


def test_importing_cli_does_not_import_fastapi() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys\n"
        "import masskb.__main__\n"
        "assert 'fastapi' not in sys.modules\n"
        "assert 'uvicorn' not in sys.modules\n"
        "assert 'masskb.api' not in sys.modules\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_script_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []
    _patch(monkeypatch, _result(ok=True, openai=False), closed)
    path = _ROOT / "scripts" / "healthcheck.py"
    spec = importlib.util.spec_from_file_location("masskb_healthcheck_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main() == 0
    assert closed == ["closed"]


def _ask_hit() -> SearchHit:
    return SearchHit(
        score=0.5,
        text="原文",
        source="a.md",
        chunk_id="c",
        doc_id="d",
        index=0,
        start=0,
        end=2,
        prev_chunk_id=None,
        next_chunk_id="c-next",
    )


def _ask_result() -> AskResult:
    hit = _ask_hit()
    search = SearchResult(
        query="质量",
        top_k=5,
        collection="masskb_chunks",
        embedding_model="embed-test",
        doc_id=None,
        hits=(hit,),
        graph=GraphStatus(ok=True, skipped=False),
    )
    return AskResult(
        query="质量",
        max_steps=3,
        model="gpt-test",
        stopped="finish",
        steps=(
            AgentStep(index=1, action="search", ok=True, query="质量", search=search),
            AgentStep(index=2, action="finish", ok=True),
        ),
        answer=AnswerStatus(
            ok=True,
            skipped=False,
            text="回答",
            citations=("c",),
            model="gpt-test",
        ),
        top_k=5,
        doc_id=None,
        hybrid=False,
        rewrite=False,
        rerank=False,
    )


def test_ask_prints_json_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed: list[str] = []
    seen: dict[str, object] = {}
    expected = _ask_result()

    class FakePipeline:
        def ask(
            self,
            query: str,
            *,
            top_k: int = 5,
            doc_id: str | None = None,
            hybrid: bool = False,
            rewrite: bool = False,
            rerank: bool = False,
            max_steps: int = 3,
        ) -> AskResult:
            seen["query"] = query
            seen["top_k"] = top_k
            seen["doc_id"] = doc_id
            seen["hybrid"] = hybrid
            seen["rewrite"] = rewrite
            seen["rerank"] = rerank
            seen["max_steps"] = max_steps
            return expected

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["ask", "质量"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "回答" in captured.out
    assert "\\u56de" not in captured.out
    payload = json.loads(captured.out)
    assert payload == json.loads(json.dumps(asdict(expected), ensure_ascii=False))
    assert payload["steps"][0]["search"]["answer"]["skipped"] is True
    assert payload["query"] == "质量"
    assert seen == {
        "query": "质量",
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "rewrite": False,
        "rerank": False,
        "max_steps": 3,
    }
    assert closed == ["closed"]


def test_ask_forwards_flags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    class FakePipeline:
        def ask(self, query: str, **kwargs: object) -> AskResult:
            seen["query"] = query
            seen["kwargs"] = kwargs
            return _ask_result()

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert (
        main(
            [
                "ask",
                "质量",
                "--top-k",
                "3",
                "--doc-id",
                "abc",
                "--hybrid",
                "--rewrite",
                "--rerank",
                "--max-steps",
                "2",
            ]
        )
        == 0
    )
    assert seen["query"] == "质量"
    assert seen["kwargs"] == {
        "top_k": 3,
        "doc_id": "abc",
        "hybrid": True,
        "rewrite": True,
        "rerank": True,
        "max_steps": 2,
    }
    assert capsys.readouterr().err == ""


def test_ask_max_steps_soft_failure_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def ask(self, *_args: object, **_kwargs: object) -> AskResult:
            result = _ask_result()
            return AskResult(
                query=result.query,
                max_steps=result.max_steps,
                model=result.model,
                stopped="max_steps",
                steps=result.steps,
                answer=AnswerStatus(
                    ok=False,
                    skipped=False,
                    text=None,
                    citations=(),
                    model="gpt-test",
                    error="max_steps",
                ),
                top_k=result.top_k,
                doc_id=result.doc_id,
                hybrid=False,
                rewrite=False,
                rerank=False,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["ask", "质量"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stopped"] == "max_steps"
    assert payload["answer"]["error"] == "max_steps"
    assert payload["answer"]["text"] is None


def test_ask_failure_exits_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed: list[str] = []

    class FakePipeline:
        def ask(self, *_args: object, **_kwargs: object) -> AskResult:
            raise RuntimeError("缺少真实 OPENAI_API_KEY")

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["ask", "质量"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[FAIL] 缺少真实 OPENAI_API_KEY" in captured.err
    assert closed == ["closed"]


def test_ask_rejects_out_of_range_steps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def ask(self, *_args: object, **kwargs: object) -> AskResult:
            raise ValueError(f"max_steps 必须是 1 到 3 的整数，收到 {kwargs['max_steps']!r}")

        def close(self) -> None:
            return None

    monkeypatch.setattr("masskb.__main__.RetrievalPipeline", FakePipeline)

    assert main(["ask", "质量", "--max-steps", "4"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "max_steps" in captured.err


def test_ask_help_and_rejected_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["ask", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--top-k" in out
    assert "--doc-id" in out
    assert "--hybrid" in out
    assert "--rewrite" in out
    assert "--rerank" in out
    assert "--max-steps" in out
    assert "--answer" not in out

    with pytest.raises(SystemExit) as missing:
        main(["ask"])
    assert missing.value.code == 2

    with pytest.raises(SystemExit) as answer:
        main(["ask", "质量", "--answer"])
    assert answer.value.code == 2
