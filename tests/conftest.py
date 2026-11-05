"""共享夹具：避免配置缓存在用例之间串扰。"""

from __future__ import annotations

import pytest

from masskb.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
