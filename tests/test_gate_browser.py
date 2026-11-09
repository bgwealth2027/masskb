"""用真实浏览器核对查询页和网关投影。未安装 Playwright 时跳过，不连数据库。"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from masskb.gate import GateConfig, bind_gate, load_sample_catalog

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


class _Switch:
    def __init__(self) -> None:
        self.mode = "catalog"
        self.hold = threading.Event()
        self.hold.set()
        self.calls = 0
        self.catalog = load_sample_catalog()

    def payload(self, body: bytes) -> dict[str, object]:
        if self.mode == "empty":
            return {"hits": []}
        requested = json.loads(body.decode("utf-8"))
        doc_id = requested.get("doc_id")
        return {
            "hits": [
                {
                    "source": item.source,
                    "chunk_id": item.chunk_id,
                    "doc_id": item.doc_id,
                    "text": item.text,
                }
                for item in self.catalog
                if item.doc_id == doc_id
            ]
        }


def _server(switch: _Switch) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if not switch.hold.wait(15):
                self._send(503, {"ok": False, "detail": "held"})
                return
            self._send(200, {"ok": True, "qdrant": {"url": "secret"}, "neo4j": {"uri": "bolt://x"}})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            switch.calls += 1
            if not switch.hold.wait(15):
                self._send(503, {"detail": "held"})
                return
            self._send(200, switch.payload(body))

        def _send(self, status: int, payload: dict[str, object]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _gate(upstream: ThreadingHTTPServer, *, wait: float = 5, limit: int = 2) -> tuple[object, int]:
    host, up_port = upstream.server_address[:2]
    gate = bind_gate(
        GateConfig(
            upstream=f"http://{host}:{up_port}",
            client_wait_seconds=wait,
            upstream_timeout_seconds=3,
            admission_limit=limit,
        ),
        "127.0.0.1",
        0,
    )
    return gate, gate.server_address[1]


def test_browser_page_matches_gate_statuses() -> None:
    switch = _Switch()
    upstream = _server(switch)
    gate, port = _gate(upstream)
    catalog = switch.catalog
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path="/usr/local/bin/google-chrome",
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
            page.wait_for_function(
                "() => (document.querySelector('#service-status').textContent || '')"
                ".includes('未配置回答')",
                timeout=10000,
            )
            assert page.locator("#answer-control").is_hidden()
            assert page.locator("#answer-section").is_hidden()

            def submit(query: str) -> None:
                page.locator("#query").fill(query)
                page.locator("#submit").click()

            submit("青苹果")
            page.wait_for_function(
                "() => document.querySelector('#status').textContent === '检索完成'",
                timeout=10000,
            )
            shown = page.locator("#hits").inner_text()
            for item in catalog:
                assert item.text in shown
            assert page.locator("#answer-section").is_hidden()
            first = page.locator("#hits li").all_inner_texts()
            assert len(first) == 4
            submit("青苹果")
            page.wait_for_function(
                "() => document.querySelector('#status').textContent === '检索完成'",
                timeout=10000,
            )
            assert page.locator("#hits li").all_inner_texts() == first

            switch.mode = "empty"
            submit("没有这段")
            page.wait_for_function(
                "() => document.querySelector('#empty').hidden === false",
                timeout=10000,
            )
            assert page.locator("#status").inner_text() == "检索完成"

            invalid = page.evaluate(
                """async (url) => {
                    const response = await fetch(url, {
                        method: 'POST',
                        headers: {'content-type': 'application/json', accept: 'application/json'},
                        body: '{"answer": false, "answer": true}'
                    });
                    return {status: response.status, detail: (await response.json()).detail};
                }""",
                f"http://127.0.0.1:{port}/search",
            )
            assert invalid == {"status": 422, "detail": "请求无效"}

            slow_switch = _Switch()
            slow_switch.hold.clear()
            slow_up = _server(slow_switch)
            slow_gate, slow_port = _gate(slow_up, wait=0.3, limit=2)
            try:
                page.goto(f"http://127.0.0.1:{slow_port}/", wait_until="domcontentloaded")
                page.locator("#query").fill("青苹果")
                page.locator("#submit").click()
                page.wait_for_function(
                    "() => (document.querySelector('#status').textContent || '').includes('检索超时')",
                    timeout=10000,
                )
            finally:
                slow_switch.hold.set()
                slow_gate.finish()
                slow_up.shutdown()
                slow_up.server_close()

            busy_switch = _Switch()
            busy_switch.hold.clear()
            busy_up = _server(busy_switch)
            busy_gate, busy_port = _gate(busy_up, wait=2, limit=1)
            try:
                body = json.dumps(
                    {
                        "query": "占住",
                        "top_k": 5,
                        "doc_id": None,
                        "hybrid": False,
                        "answer": False,
                        "rerank": False,
                        "rewrite": False,
                        "graph_hops": False,
                        "aliases": False,
                    }
                ).encode()
                probe = socket.create_connection(("127.0.0.1", busy_port), timeout=3)
                probe.sendall(
                    (
                        f"POST /search HTTP/1.1\r\nHost: 127.0.0.1:{busy_port}\r\n"
                        "Content-Type: application/json\r\n"
                        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
                    ).encode()
                    + body
                )
                for _ in range(50):
                    if busy_switch.calls:
                        break
                    threading.Event().wait(0.05)
                assert busy_switch.calls >= 1
                page.goto(f"http://127.0.0.1:{busy_port}/", wait_until="domcontentloaded")
                page.locator("#query").fill("青苹果")
                page.locator("#submit").click()
                page.wait_for_function(
                    "() => (document.querySelector('#status').textContent || '').includes('服务正忙')",
                    timeout=10000,
                )
                probe.close()
            finally:
                busy_switch.hold.set()
                busy_gate.finish()
                busy_up.shutdown()
                busy_up.server_close()
            browser.close()
    finally:
        switch.hold.set()
        gate.finish()
        upstream.shutdown()
        upstream.server_close()
