"""CLI 与 scripts/healthcheck.py：不连接真实服务。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from masskb.__main__ import main
from masskb.pipeline import DimensionMismatchError, IngestWriteResult, SearchHit, SearchResult

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
        def ingest(self, source: Path, *, root: Path | None = None, chunk_size: int = 800, overlap: int = 100):
            from masskb.ingest import ingest_path

            report = ingest_path(source, root=root, chunk_size=chunk_size, overlap=overlap)
            return IngestWriteResult(
                report=report,
                collection="masskb_chunks",
                vector_size=1536,
                points_upserted=report.chunk_count,
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
    assert "[INFO] 已写入 Qdrant 集合 masskb_chunks。未连接 Neo4j，未执行检索。" in out
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
    assert "chunk-size" in out


def test_search_prints_json_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed: list[str] = []
    seen: dict = {}

    class FakePipeline:
        def search(self, query: str, *, top_k: int = 5, doc_id: str | None = None) -> SearchResult:
            seen["query"] = query
            seen["top_k"] = top_k
            seen["doc_id"] = doc_id
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
                    ),
                ),
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
            }
        ],
    }
    assert seen == {"query": "质量", "top_k": 3, "doc_id": "abc"}
    assert closed == ["closed"]


def test_search_empty_hits_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePipeline:
        def search(self, query: str, *, top_k: int = 5, doc_id: str | None = None) -> SearchResult:
            return SearchResult(
                query=query,
                top_k=top_k,
                collection="masskb_chunks",
                embedding_model="text-embedding-3-small",
                doc_id=doc_id,
                hits=(),
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


def test_search_missing_query_exits_2() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["search"])
    assert exc.value.code == 2


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
