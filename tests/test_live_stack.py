"""显式的 Qdrant + Neo4j + 本地嵌入全链路。

默认 pytest 不进入此用例的连接代码。要跑真实验收，需要同时传入::

    python -m pytest tests/test_live_stack.py \
        --live-qdrant http://127.0.0.1:16333 \
        --live-bolt bolt://127.0.0.1:17687 \
        --live-neo4j-user neo4j \
        --live-neo4j-password '<临时库密码>' \
        --live-embed http://127.0.0.1:18081/v1 \
        --live-embedding-model BAAI/bge-small-zh-v1.5 \
        --live-embedding-dimensions 512

这些值会组装成一份独立的 ``Settings``，再传给本进程里的客户端。
CLI 和 HTTP 进程只拿到由这份 ``Settings`` 写出的环境，从而在子进程里再建成同一组配置。
地址必须是 loopback 上的高位端口。库不是空的时会直接失败，避免扫到别的数据。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from masskb.aliases import AliasLink
from masskb.config import Settings
from masskb.embeddings import EmbeddingClient
from masskb.entities import EntityMention, entity_id
from masskb.graph_store import GraphStore
from masskb.ingest import ingest_path, repo_root
from masskb.vector_store import VectorStore

_FORBIDDEN_PORTS = frozenset({6333, 6334, 7474, 7687, 8080, 8081})
_DELETE_THESE_CHUNKS = """
MATCH (c:Chunk)
WHERE c.chunk_id IN $chunk_ids
DETACH DELETE c
"""
_DELETE_THESE_DOCUMENTS = """
MATCH (d:Document)
WHERE d.doc_id IN $doc_ids
DETACH DELETE d
"""
_DELETE_THESE_ALIAS_EDGES = """
MATCH ()-[rel:ALIAS_OF]->()
WHERE rel.doc_id IN $doc_ids
DELETE rel
"""
_DELETE_THIS_PERSON_IF_UNATTACHED = """
MATCH (person:Person {person_id: $entity_id})
WHERE NOT (()-[:MENTIONS]->(person)) AND NOT (person)-[:ALIAS_OF]-()
DELETE person
"""
_DELETE_THIS_CONCEPT_IF_UNATTACHED = """
MATCH (concept:Concept {concept_id: $entity_id})
WHERE NOT (()-[:MENTIONS]->(concept)) AND NOT (concept)-[:ALIAS_OF]-()
DELETE concept
"""


@dataclass(frozen=True)
class _LiveTarget:
    qdrant_url: str
    bolt: str
    user: str
    password: str
    embed_base_url: str
    embedding_model: str
    embedding_dimensions: int


def _live_target(request: pytest.FixtureRequest) -> _LiveTarget | None:
    qdrant = request.config.getoption("--live-qdrant").strip()
    bolt = request.config.getoption("--live-bolt").strip()
    embed = request.config.getoption("--live-embed").strip()
    password = request.config.getoption("--live-neo4j-password")
    dimensions = request.config.getoption("--live-embedding-dimensions").strip()
    if qdrant == "" and bolt == "" and embed == "" and password == "" and dimensions == "":
        return None
    missing = [
        name
        for name, value in (
            ("--live-qdrant", qdrant),
            ("--live-bolt", bolt),
            ("--live-neo4j-password", password),
            ("--live-embed", embed),
            ("--live-embedding-dimensions", dimensions),
        )
        if value == ""
    ]
    if missing:
        pytest.fail("显式集成缺少 " + "、".join(missing) + "。不会改连其它库。")
    _require_loopback(qdrant, schemes=("http",))
    _require_loopback(bolt, schemes=("bolt",))
    _require_loopback(embed, schemes=("http",))
    try:
        parsed_dimensions = int(dimensions)
    except ValueError:
        pytest.fail("--live-embedding-dimensions 必须是整数。")
    if parsed_dimensions < 1:
        pytest.fail("--live-embedding-dimensions 必须大于等于 1。")
    model = request.config.getoption("--live-embedding-model").strip()
    if model == "":
        pytest.fail("--live-embedding-model 不能为空。")
    user = request.config.getoption("--live-neo4j-user").strip() or "neo4j"
    return _LiveTarget(
        qdrant_url=qdrant,
        bolt=bolt,
        user=user,
        password=password,
        embed_base_url=embed,
        embedding_model=model,
        embedding_dimensions=parsed_dimensions,
    )


def _require_loopback(url: str, *, schemes: tuple[str, ...]) -> None:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme not in schemes
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is None
        or parsed.port < 10000
        or parsed.port in _FORBIDDEN_PORTS
    ):
        pytest.fail(f"地址必须是 loopback 上的高位端口：{url}")


def _settings(target: _LiveTarget) -> Settings:
    return Settings(
        qdrant_url=target.qdrant_url,
        neo4j_uri=target.bolt,
        neo4j_user=target.user,
        neo4j_password=target.password,
        openai_api_key="",
        llm_base_url="http://127.0.0.1:9/v1",
        embedding_base_url=target.embed_base_url,
        embedding_model=target.embedding_model,
        embedding_dimensions=target.embedding_dimensions,
        llm_model="unused",
    )


def _child_env(settings: Settings) -> dict[str, str]:
    """只带这份 Settings 里的连接信息，不继承当前进程里的库地址或 Key。"""
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": "localhost,127.0.0.1,::1",
        "no_proxy": "localhost,127.0.0.1,::1",
        "QDRANT_URL": settings.qdrant_url,
        "NEO4J_URI": settings.neo4j_uri,
        "NEO4J_USER": settings.neo4j_user,
        "NEO4J_PASSWORD": settings.neo4j_password,
        "OPENAI_API_KEY": "",
        "LLM_BASE_URL": settings.llm_base_url,
        "LLM_MODEL": settings.llm_model,
        "EMBEDDING_BASE_URL": settings.embedding_base_url,
        "EMBEDDING_MODEL": settings.embedding_model,
        "EMBEDDING_DIMENSIONS": str(settings.embedding_dimensions),
    }


def _run_masskb(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "masskb", *args],
        cwd=repo_root(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _request(method: str, url: str, payload: dict[str, object] | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("accept", "application/json")
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode()
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        body = json.loads(raw) if raw else {}
        return exc.code, body


def _free_high_port() -> int:
    for port in range(19080, 19180):
        if port in _FORBIDDEN_PORTS:
            continue
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        else:
            sock.close()
            return port
    pytest.fail("没有可用的 loopback 高位端口来启动 HTTP。")


def _wait_http(url: str) -> tuple[int, dict]:
    deadline = time.monotonic() + 30
    last: tuple[int, dict] = (0, {})
    while time.monotonic() < deadline:
        try:
            last = _request("GET", url)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)
            continue
        if last[0] == 200:
            return last
        time.sleep(0.2)
    return last


def _scalar(store: GraphStore, query: str, **params: object) -> int:
    def work(tx):
        record = tx.run(query, **params).single()
        return 0 if record is None else int(record["n"])

    with store.driver.session() as session:
        return session.execute_read(work)


def _listen_addresses(port: int) -> list[str]:
    """从 /proc 读出正在监听该端口的地址。状态 0A 是 LISTEN。"""
    found: list[str] = []
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        text = Path(path).read_text(encoding="utf-8")
        for line in text.splitlines()[1:]:
            columns = line.split()
            local, state = columns[1], columns[3]
            ip_hex, port_hex = local.split(":")
            if int(port_hex, 16) != port or state != "0A":
                continue
            raw = bytes.fromhex(ip_hex)
            if len(raw) == 4:
                found.append(socket.inet_ntoa(raw[::-1]))
            else:
                found.append(socket.inet_ntop(socket.AF_INET6, raw))
    return found


def _assert_listener_is_loopback(port: int) -> None:
    addresses = _listen_addresses(port)
    assert addresses, f"端口 {port} 没有在监听"
    assert set(addresses) <= {"127.0.0.1", "::1"}


def _search(base: str, query: str, doc_id: str) -> tuple[int, dict]:
    return _request(
        "POST",
        f"{base}/search",
        {
            "query": query,
            "top_k": 5,
            "doc_id": doc_id,
            "hybrid": False,
            "answer": False,
            "rerank": False,
            "rewrite": False,
            "graph_hops": False,
            "aliases": False,
        },
    )


def _hits_by_id(payload: dict) -> dict[str, dict]:
    hits = payload.get("hits")
    assert isinstance(hits, list)
    found = {}
    for hit in hits:
        assert isinstance(hit, dict)
        chunk_id = hit.get("chunk_id")
        assert isinstance(chunk_id, str) and chunk_id != ""
        found[chunk_id] = hit
    return found


def _assert_page(url: str, query: str, expected_text: str) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        pytest.fail(f"显式集成需要 playwright 才能验收网页：{exc}")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        try:
            page = browser.new_page()
            opened = page.goto(url, wait_until="domcontentloaded")
            assert opened is not None
            assert opened.status == 200
            page.wait_for_function(
                "() => (document.querySelector('#service-status').textContent || '')"
                ".includes('未配置回答')",
                timeout=20000,
            )
            assert page.locator("#answer-control").is_hidden()
            page.locator("#query").fill(query)
            page.locator("#submit").click()
            page.wait_for_function(
                "() => document.querySelector('#status').textContent === '检索完成'",
                timeout=30000,
            )
            assert expected_text in page.locator("#hits").inner_text()
            assert page.locator("#answer-section").is_hidden()
            target = Path("/opt/cursor/artifacts")
            if target.is_dir():
                page.screenshot(path=str(target / "live-search.png"), full_page=True)
        finally:
            browser.close()


def _cleanup(
    graph: GraphStore,
    vector: VectorStore,
    *,
    doc_ids: list[str],
    chunk_ids: list[str],
    people: list[str],
    concepts: list[str],
) -> None:
    def work(tx) -> None:
        if chunk_ids:
            tx.run(_DELETE_THESE_CHUNKS, chunk_ids=chunk_ids).consume()
        if doc_ids:
            tx.run(_DELETE_THESE_DOCUMENTS, doc_ids=doc_ids).consume()
            tx.run(_DELETE_THESE_ALIAS_EDGES, doc_ids=doc_ids).consume()
        for person_id in people:
            tx.run(_DELETE_THIS_PERSON_IF_UNATTACHED, entity_id=person_id).consume()
        for concept_id in concepts:
            tx.run(_DELETE_THIS_CONCEPT_IF_UNATTACHED, entity_id=concept_id).consume()

    try:
        with graph.driver.session() as session:
            session.execute_write(work)
    except Exception as exc:  # noqa: BLE001 — 清理失败要留下原因，同时继续删点
        print(f"LIVE cleanup_graph_error={type(exc).__name__}: {exc}", flush=True)
    for doc_id in doc_ids:
        try:
            vector.delete_by_doc_id(doc_id)
        except Exception as exc:  # noqa: BLE001 — 集合可能还没建出来
            print(f"LIVE cleanup_vector_error={type(exc).__name__}: {exc}", flush=True)


def test_live_qdrant_neo4j_embed_roundtrip(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    target = _live_target(request)
    if target is None:
        pytest.skip("未传入 --live-qdrant、--live-bolt 和 --live-embed。默认不探测、不写入数据库。")
    settings = _settings(target)
    env_file = repo_root() / ".env"
    env_existed = env_file.exists()
    env_mtime = env_file.stat().st_mtime_ns if env_existed else None
    graph = GraphStore(settings)
    vector = VectorStore(settings)
    server: subprocess.Popen[str] | None = None
    stdout_handle = None
    stderr_handle = None
    doc_ids: list[str] = []
    chunk_ids: list[str] = []
    people: list[str] = []
    concepts: list[str] = []
    try:
        assert graph.settings is settings
        assert vector.settings is settings
        assert settings.qdrant_url == target.qdrant_url
        assert settings.neo4j_uri == target.bolt
        assert settings.neo4j_user == target.user
        assert settings.neo4j_password == target.password
        assert settings.embedding_base_url == target.embed_base_url
        assert settings.embedding_model == target.embedding_model
        assert settings.embedding_dimensions == target.embedding_dimensions
        assert settings.openai_api_key == ""
        assert settings.has_real_openai_key is False
        with graph.driver.session() as session:
            assert session._config.max_transaction_retry_time == 30.0
            assert session._config.connection_acquisition_timeout == 60.0
        assert _scalar(graph, "MATCH (n) RETURN count(n) AS n") == 0
        probe = EmbeddingClient(settings).embed_one("masskb")
        assert len(probe) == settings.embedding_dimensions
        env = _child_env(settings)
        run = uuid.uuid4().hex
        apple = f"甲段标记{run}青苹果句子用于命中。"
        river = f"乙段标记{run}蓝河流句子用于邻接。"
        apple_updated = f"{apple}更新后加长一段。"
        river_updated = f"{river}更新后加长一段。"
        control_text = f"对照文档{run}只保留不删除。"
        sample_path = tmp_path / "sample-live.txt"
        control_path = tmp_path / "control-live.txt"
        sample_path.write_text(f"{apple}\n\n{river}\n", encoding="utf-8")
        control_path.write_text(f"{control_text}\n", encoding="utf-8")
        sample = ingest_path(sample_path, root=tmp_path).files[0]
        control = ingest_path(control_path, root=tmp_path).files[0]
        assert len(sample.chunks) >= 2
        assert len(control.chunks) == 1
        first, second = sample.chunks[0], sample.chunks[1]
        doc_ids.extend([sample.doc_id, control.doc_id])
        chunk_ids.extend(chunk.chunk_id for chunk in (*sample.chunks, *control.chunks))

        ingested = _run_masskb(["ingest", str(sample_path), "--root", str(tmp_path)], env)
        assert ingested.returncode == 0, ingested.stderr
        print(f"LIVE ingest_exit={ingested.returncode}", flush=True)
        controlled = _run_masskb(["ingest", str(control_path), "--root", str(tmp_path)], env)
        assert controlled.returncode == 0, controlled.stderr
        assert _scalar(
            graph,
            """
            MATCH (:Chunk {chunk_id: $left})-[:NEXT_CHUNK]->(:Chunk {chunk_id: $right})
            RETURN count(*) AS n
            """,
            left=first.chunk_id,
            right=second.chunk_id,
        ) == 1

        port = _free_high_port()
        stdout_handle = (tmp_path / "serve.out").open("w", encoding="utf-8")
        stderr_handle = (tmp_path / "serve.err").open("w", encoding="utf-8")
        server = subprocess.Popen(
            [sys.executable, "-m", "masskb", "serve", "--host", "127.0.0.1", "--port", str(port)],
            cwd=repo_root(),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        base = f"http://127.0.0.1:{port}"
        health_status, health = _wait_http(f"{base}/health")
        assert health_status == 200, health
        assert health["ok"] is True
        assert health["qdrant"]["ok"] is True
        assert health["neo4j"]["ok"] is True
        assert health["qdrant"]["url"] == settings.qdrant_url
        assert health["neo4j"]["uri"] == settings.neo4j_uri
        assert health["neo4j"]["ping"] == 1
        assert health["openai_key_configured"] is False
        _assert_listener_is_loopback(port)
        print(
            "LIVE health_http="
            f"{health_status} qdrant_ok={health['qdrant']['ok']} "
            f"neo4j_ok={health['neo4j']['ok']}",
            flush=True,
        )

        search_status, found = _search(base, apple, sample.doc_id)
        assert search_status == 200, found
        hits = _hits_by_id(found)
        assert hits[first.chunk_id]["text"] == apple
        assert hits[first.chunk_id]["next_chunk_id"] == second.chunk_id
        assert hits[first.chunk_id]["prev_chunk_id"] is None
        assert hits[second.chunk_id]["text"] == river
        assert hits[second.chunk_id]["prev_chunk_id"] == first.chunk_id
        assert hits[second.chunk_id]["next_chunk_id"] is None
        assert found["graph"]["ok"] is True
        assert found["graph"]["skipped"] is False
        assert found["graph"]["error"] is None
        assert found["answer"]["skipped"] is True
        print(
            f"LIVE search_http={search_status} hits={len(hits)} "
            f"next={hits[first.chunk_id]['next_chunk_id'] == second.chunk_id}",
            flush=True,
        )
        _assert_page(f"{base}/", apple, apple)
        print("LIVE page_http=200 page_hit=1", flush=True)

        sample_path.write_text(f"{apple_updated}\n\n{river_updated}\n", encoding="utf-8")
        updated = ingest_path(sample_path, root=tmp_path).files[0]
        assert updated.doc_id == sample.doc_id
        assert len(updated.chunks) >= 2
        new_first, new_second = updated.chunks[0], updated.chunks[1]
        assert new_first.chunk_id != first.chunk_id
        assert new_second.chunk_id != second.chunk_id
        chunk_ids.extend(chunk.chunk_id for chunk in updated.chunks)
        updated_run = _run_masskb(["ingest", str(sample_path), "--root", str(tmp_path)], env)
        assert updated_run.returncode == 0, updated_run.stderr
        print(f"LIVE update_exit={updated_run.returncode}", flush=True)
        assert vector.get_by_chunk_ids([first.chunk_id, second.chunk_id]) == {}
        assert _scalar(
            graph,
            "MATCH (c:Chunk) WHERE c.chunk_id IN $ids RETURN count(c) AS n",
            ids=[first.chunk_id, second.chunk_id],
        ) == 0
        assert _scalar(
            graph,
            """
            MATCH (:Chunk {chunk_id: $left})-[:NEXT_CHUNK]->(:Chunk {chunk_id: $right})
            RETURN count(*) AS n
            """,
            left=new_first.chunk_id,
            right=new_second.chunk_id,
        ) == 1
        updated_status, updated_found = _search(base, apple_updated, sample.doc_id)
        assert updated_status == 200, updated_found
        updated_hits = _hits_by_id(updated_found)
        assert updated_hits[new_first.chunk_id]["text"] == apple_updated
        assert updated_hits[new_first.chunk_id]["next_chunk_id"] == new_second.chunk_id
        assert first.chunk_id not in updated_hits
        assert second.chunk_id not in updated_hits

        exclusive_person = entity_id("person", f"独享甲{run[:8]}")
        shared_person = entity_id("person", f"共享乙{run[:8]}")
        alias_person = entity_id("person", f"别名丙{run[:8]}")
        canonical_person = entity_id("person", f"规范丁{run[:8]}")
        exclusive_concept = entity_id("concept", f"独享概念{run[:8]}")
        people.extend([exclusive_person, shared_person, alias_person, canonical_person])
        concepts.append(exclusive_concept)
        graph.ensure_entity_schema()
        graph.replace_mentions(
            sample.doc_id,
            [chunk.chunk_id for chunk in updated.chunks],
            [
                EntityMention(
                    chunk_id=new_first.chunk_id,
                    kind="person",
                    name=f"独享甲{run[:8]}",
                    entity_id=exclusive_person,
                ),
                EntityMention(
                    chunk_id=new_first.chunk_id,
                    kind="person",
                    name=f"共享乙{run[:8]}",
                    entity_id=shared_person,
                ),
                EntityMention(
                    chunk_id=new_second.chunk_id,
                    kind="person",
                    name=f"别名丙{run[:8]}",
                    entity_id=alias_person,
                ),
                EntityMention(
                    chunk_id=new_second.chunk_id,
                    kind="concept",
                    name=f"独享概念{run[:8]}",
                    entity_id=exclusive_concept,
                ),
            ],
        )
        graph.replace_mentions(
            control.doc_id,
            [control.chunks[0].chunk_id],
            [
                EntityMention(
                    chunk_id=control.chunks[0].chunk_id,
                    kind="person",
                    name=f"共享乙{run[:8]}",
                    entity_id=shared_person,
                ),
                EntityMention(
                    chunk_id=control.chunks[0].chunk_id,
                    kind="person",
                    name=f"规范丁{run[:8]}",
                    entity_id=canonical_person,
                ),
            ],
        )
        graph.replace_aliases(
            sample.doc_id,
            [
                AliasLink(
                    kind="person",
                    alias_name=f"别名丙{run[:8]}",
                    canonical_name=f"规范丁{run[:8]}",
                    alias_id=alias_person,
                    canonical_id=canonical_person,
                )
            ],
        )

        sample_path.write_text("", encoding="utf-8")
        deleted = _run_masskb(["ingest", str(sample_path), "--root", str(tmp_path)], env)
        assert deleted.returncode == 0, deleted.stderr
        assert "SyntaxError" not in deleted.stderr
        print(f"LIVE delete_exit={deleted.returncode}", flush=True)
        assert vector.get_by_chunk_ids(
            [first.chunk_id, second.chunk_id, new_first.chunk_id, new_second.chunk_id]
        ) == {}
        assert vector.get_by_chunk_ids([control.chunks[0].chunk_id]) != {}
        assert _scalar(
            graph,
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) AS n",
            doc_id=sample.doc_id,
        ) == 0
        assert _scalar(
            graph,
            "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) AS n",
            doc_id=sample.doc_id,
        ) == 0
        assert _scalar(
            graph,
            "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) AS n",
            doc_id=control.doc_id,
        ) == 1
        assert _scalar(
            graph,
            "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN count(c) AS n",
            chunk_id=control.chunks[0].chunk_id,
        ) == 1
        assert _scalar(
            graph,
            "MATCH (person:Person {person_id: $entity_id}) RETURN count(person) AS n",
            entity_id=exclusive_person,
        ) == 0
        assert _scalar(
            graph,
            "MATCH (concept:Concept {concept_id: $entity_id}) RETURN count(concept) AS n",
            entity_id=exclusive_concept,
        ) == 0
        assert _scalar(
            graph,
            """
            MATCH (:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(:Person {person_id: $entity_id})
            RETURN count(*) AS n
            """,
            chunk_id=control.chunks[0].chunk_id,
            entity_id=shared_person,
        ) == 1
        assert _scalar(
            graph,
            """
            MATCH (:Person {person_id: $alias})-[rel:ALIAS_OF {doc_id: $doc_id}]->
                  (:Person {person_id: $canonical})
            RETURN count(rel) AS n
            """,
            alias=alias_person,
            canonical=canonical_person,
            doc_id=sample.doc_id,
        ) == 1
        gone_status, gone = _search(base, apple_updated, sample.doc_id)
        assert gone_status == 200, gone
        assert gone["hits"] == []
        kept_status, kept = _search(base, control_text, control.doc_id)
        assert kept_status == 200, kept
        kept_hits = _hits_by_id(kept)
        assert kept_hits[control.chunks[0].chunk_id]["text"] == control_text
        print(
            "LIVE kept_control=1 kept_shared=1 kept_alias=1 "
            "removed_exclusive_person=1 removed_exclusive_concept=1",
            flush=True,
        )
        assert env_file.exists() is env_existed
        if env_existed:
            assert env_file.stat().st_mtime_ns == env_mtime
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
        _cleanup(
            graph,
            vector,
            doc_ids=doc_ids,
            chunk_ids=chunk_ids,
            people=people,
            concepts=concepts,
        )
        graph.close()
        vector.close()
