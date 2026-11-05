"""Neo4j 图数据库客户端封装。"""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase, Driver

from masskb.config import Settings, get_settings


class GraphStore:
    """Neo4j 薄封装：Bolt 连通性检查 + 后续图查询扩展点。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
        return self._driver

    def healthcheck(self) -> dict[str, Any]:
        """检查 Neo4j Bolt 是否可用。成功返回 {"ok": True, ...}。"""
        try:
            self.driver.verify_connectivity()
            with self.driver.session() as session:
                record = session.run("RETURN 1 AS n").single()
                value = record["n"] if record else None
            return {
                "ok": True,
                "uri": self.settings.neo4j_uri,
                "ping": value,
            }
        except Exception as exc:  # noqa: BLE001 — healthcheck 需要吞并并报告
            return {
                "ok": False,
                "uri": self.settings.neo4j_uri,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
