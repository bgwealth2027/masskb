"""共享夹具：避免配置缓存在用例之间串扰。"""

from __future__ import annotations

import pytest

from masskb.config import get_settings


def pytest_addoption(parser: pytest.Parser) -> None:
    """图读集成的开关。不设置产品环境变量，默认 pytest 不使用它们。"""
    parser.addoption(
        "--graph-read-bolt",
        default="",
        help="显式连接这个 Bolt 地址做图读集成。默认空，不探测、不写入。请指向空的临时库。",
    )
    parser.addoption(
        "--graph-read-user",
        default="neo4j",
        help="与 --graph-read-bolt 一起使用的用户名。不是产品配置。",
    )
    parser.addoption(
        "--graph-read-password",
        default="",
        help="与 --graph-read-bolt 一起使用的密码。不是产品配置。",
    )
    parser.addoption(
        "--live-qdrant",
        default="",
        help="显式全链路使用的 Qdrant HTTP 地址。默认空，不探测、不写入。",
    )
    parser.addoption(
        "--live-bolt",
        default="",
        help="显式全链路使用的 Neo4j Bolt 地址。默认空，不探测、不写入。",
    )
    parser.addoption(
        "--live-neo4j-user",
        default="neo4j",
        help="与 --live-bolt 一起写入隔离 Settings 的用户名。",
    )
    parser.addoption(
        "--live-neo4j-password",
        default="",
        help="与 --live-bolt 一起写入隔离 Settings 的密码。",
    )
    parser.addoption(
        "--live-embed",
        default="",
        help="显式全链路使用的嵌入 API 根，含 /v1。默认空，不探测。",
    )
    parser.addoption(
        "--live-embedding-model",
        default="BAAI/bge-small-zh-v1.5",
        help="与 --live-embed 一起写入隔离 Settings 的模型名。",
    )
    parser.addoption(
        "--live-embedding-dimensions",
        default="",
        help="与 --live-embed 一起写入隔离 Settings 的向量维度。",
    )


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
