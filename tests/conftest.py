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


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
