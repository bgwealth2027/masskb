"""网关对着真实 loopback 检索链。不传 --live-* 时不探测、不写入。"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from test_live_stack import (
    _assert_listener_is_loopback,
    _child_env,
    _free_high_port,
    _live_target,
    _settings,
)

from masskb.graph_store import GraphStore
from masskb.ingest import ingest_path, repo_root
from masskb.vector_store import VectorStore

_PRIVATE = "私密标记不应出网关。"


def _request(method: str, url: str, payload: dict[str, object] | None = None) -> tuple[int, object]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("accept", "application/json")
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        body: object = json.loads(raw) if raw else {}
        return exc.code, body


def _wait_ok(url: str) -> tuple[int, dict]:
    deadline = time.monotonic() + 30
    last: tuple[int, object] = (0, {})
    while time.monotonic() < deadline:
        try:
            last = _request("GET", url)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)
            continue
        if last[0] == 200 and isinstance(last[1], dict):
            return last[0], last[1]
        time.sleep(0.2)
    assert isinstance(last[1], dict)
    return last[0], last[1]


def _search_body(query: str) -> dict[str, object]:
    return {
        "query": query,
        "top_k": 5,
        "doc_id": None,
        "hybrid": False,
        "answer": False,
        "rerank": False,
        "rewrite": False,
        "graph_hops": False,
        "aliases": False,
    }


def _stop(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_gate_live_loopback_roundtrip(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    target = _live_target(request)
    if target is None:
        pytest.skip("未传入 --live-qdrant、--live-bolt 和 --live-embed。默认不探测、不写入数据库。")
    settings = _settings(target)
    env = _child_env(settings)
    env["PYTHONPATH"] = str(repo_root())
    graph = GraphStore(settings)
    vector = VectorStore(settings)
    serve: subprocess.Popen[str] | None = None
    gate: subprocess.Popen[str] | None = None
    doc_ids: list[str] = []
    try:
        sample_root = repo_root() / "masskb" / "samples"
        documents = ingest_path(sample_root, root=sample_root).files
        assert {item.source for item in documents} == {"lanheliu.md", "qingpingguo.md"}
        sample_chunks = [chunk for doc in documents for chunk in doc.chunks]
        assert len(sample_chunks) == 4
        secret_path = tmp_path / "secret-live.md"
        secret_path.write_text(_PRIVATE + "\n", encoding="utf-8")
        secret = ingest_path(secret_path).files[0]
        assert secret.source.startswith("/")
        doc_ids.extend([item.doc_id for item in documents])
        doc_ids.append(secret.doc_id)

        ingested = subprocess.run(
            [sys.executable, "-m", "masskb", "ingest", str(sample_root), "--root", str(sample_root)],
            cwd=repo_root(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert ingested.returncode == 0, ingested.stderr
        hidden = subprocess.run(
            [sys.executable, "-m", "masskb", "ingest", str(secret_path)],
            cwd=repo_root(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert hidden.returncode == 0, hidden.stderr

        serve_port = _free_high_port()
        held = socket.socket()
        held.bind(("127.0.0.1", serve_port))
        try:
            gate_port = _free_high_port()
        finally:
            held.close()
        assert gate_port != serve_port
        serve_log = (tmp_path / "serve.err").open("w", encoding="utf-8")
        serve = subprocess.Popen(
            [sys.executable, "-m", "masskb", "serve", "--host", "127.0.0.1", "--port", str(serve_port)],
            cwd=repo_root(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=serve_log,
            text=True,
        )
        serve_base = f"http://127.0.0.1:{serve_port}"
        serve_status, serve_health = _wait_ok(f"{serve_base}/health")
        assert serve_status == 200, serve_health
        assert serve_health["qdrant"]["url"] == settings.qdrant_url
        assert serve_health["neo4j"]["uri"] == settings.neo4j_uri
        _assert_listener_is_loopback(serve_port)

        gate_log = (tmp_path / "gate.err").open("w", encoding="utf-8")
        gate = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "masskb",
                "gate",
                "--host",
                "127.0.0.1",
                "--port",
                str(gate_port),
                "--upstream",
                serve_base,
            ],
            cwd=repo_root(),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=gate_log,
            text=True,
        )
        gate_base = f"http://127.0.0.1:{gate_port}"
        gate_status, gate_health = _wait_ok(f"{gate_base}/health")
        assert gate_status == 200, gate_health
        assert set(gate_health) == {"ok", "openai_key_configured"}
        assert gate_health["ok"] is True
        assert gate_health["openai_key_configured"] is False
        rendered_health = json.dumps(gate_health, ensure_ascii=False)
        assert settings.qdrant_url not in rendered_health
        assert settings.neo4j_uri not in rendered_health
        _assert_listener_is_loopback(gate_port)

        page_req = urllib.request.Request(f"{gate_base}/")
        with urllib.request.urlopen(page_req, timeout=10) as page:
            html = page.read().decode()
            assert page.status == 200
        assert 'fetch("/search"' in html
        assert 'fetch("/health"' in html

        status, found = _request("POST", f"{gate_base}/search", _search_body("青苹果"))
        assert status == 200, found
        assert isinstance(found, dict)
        assert set(found) == {"query", "hits"}
        hits = found["hits"]
        assert isinstance(hits, list) and hits
        assert all(set(hit) == {"source", "chunk_id", "text"} for hit in hits)
        assert hits[0]["source"] == "qingpingguo.md"
        river_status, river = _request("POST", f"{gate_base}/search", _search_body("蓝河流"))
        assert river_status == 200, river
        assert isinstance(river, dict)
        assert river["hits"][0]["source"] == "lanheliu.md"
        assert any(hit["source"] == "qingpingguo.md" and "虚构果园" in hit["text"] for hit in hits)
        repeated: list[str] = []
        for chunk in sample_chunks:
            first_status, first = _request("POST", f"{gate_base}/search", _search_body(chunk.text))
            second_status, second = _request("POST", f"{gate_base}/search", _search_body(chunk.text))
            assert first_status == 200 and second_status == 200, (first, second)
            assert isinstance(first, dict) and isinstance(second, dict)
            assert [item["chunk_id"] for item in first["hits"]] == [
                item["chunk_id"] for item in second["hits"]
            ]
            matched = [item for item in first["hits"] if item["chunk_id"] == chunk.chunk_id]
            assert matched and matched[0]["text"] == chunk.text
            assert matched[0]["source"] == chunk.source
            repeated.append(chunk.chunk_id)
        assert len(set(repeated)) == 4
        encoded = json.dumps(found, ensure_ascii=False)
        assert _PRIVATE not in encoded
        assert secret.source not in encoded
        assert "graph" not in found
        assert "error" not in encoded

        hidden_status, hidden_found = _request("POST", f"{gate_base}/search", _search_body(_PRIVATE))
        assert hidden_status == 200, hidden_found
        assert isinstance(hidden_found, dict)
        assert hidden_found["query"] == _PRIVATE
        hidden_hits = hidden_found["hits"]
        assert isinstance(hidden_hits, list)
        assert all(hit["source"] in {"qingpingguo.md", "lanheliu.md"} for hit in hidden_hits)
        assert all(_PRIVATE not in hit["text"] for hit in hidden_hits)
        assert secret.source not in json.dumps(hidden_found, ensure_ascii=False)

        direct_status, direct = _request("POST", f"{serve_base}/search", _search_body(_PRIVATE))
        assert direct_status == 200, direct
        assert isinstance(direct, dict)
        direct_text = json.dumps(direct, ensure_ascii=False)
        assert _PRIVATE in direct_text
        assert secret.source in direct_text
        assert "graph" in direct

        for method, path, payload in (
            ("POST", "/ingest", {"path": str(secret_path)}),
            ("POST", "/ask", _search_body("青苹果")),
            ("GET", "/openapi.json", None),
            ("POST", "/v1/embeddings", {"input": "青苹果"}),
        ):
            code, body = _request(method, f"{gate_base}{path}", payload)
            assert code in {403, 404, 405}, (path, code, body)
            assert secret.source not in json.dumps(body, ensure_ascii=False)

        refused, refused_body = _request(
            "POST",
            f"{gate_base}/search",
            _search_body("青苹果") | {"answer": True},
        )
        assert refused == 403, refused_body
        print(
            f"LIVE gate_health={gate_status} gate_hits={len(hits)} "
            f"serve_still_has_graph={('graph' in direct)} loopback=1",
            flush=True,
        )
    finally:
        _stop(gate)
        _stop(serve)
        for doc_id in doc_ids:
            try:
                vector.delete_by_doc_id(doc_id)
            except Exception as exc:  # noqa: BLE001 — 清理失败留下原因，不掩盖断言
                print(f"LIVE gate cleanup_vector={type(exc).__name__}: {exc}", flush=True)
        if doc_ids:
            try:
                with graph.driver.session() as session:
                    session.run(
                        "MATCH (d:Document) WHERE d.doc_id IN $ids DETACH DELETE d",
                        ids=doc_ids,
                    ).consume()
            except Exception as exc:  # noqa: BLE001
                print(f"LIVE gate cleanup_graph={type(exc).__name__}: {exc}", flush=True)
        graph.close()
        vector.close()
