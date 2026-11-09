#!/usr/bin/env python3
"""独立健康检查脚本：验证 Qdrant + Neo4j 连通。

用法（在仓库根目录）:
  .venv/bin/python scripts/healthcheck.py
  # 或先 pip install -e . 后:
  python -m masskb healthcheck
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证未安装包时也能从仓库根导入 masskb
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from masskb.__main__ import cmd_healthcheck


def main() -> int:
    return cmd_healthcheck(argparse.Namespace())


if __name__ == "__main__":
    sys.exit(main())
