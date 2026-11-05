"""CLI 与 scripts/healthcheck.py：不连接真实服务。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from masskb.__main__ import main

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
    assert "healthcheck" in capsys.readouterr().out


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
