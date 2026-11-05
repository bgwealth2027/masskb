"""CLI 入口：python -m masskb  或  masskb"""

from __future__ import annotations

import argparse
import json
import sys

from masskb.pipeline import RetrievalPipeline


def cmd_healthcheck(_: argparse.Namespace) -> int:
    pipeline = RetrievalPipeline()
    try:
        result = pipeline.healthcheck()
    finally:
        pipeline.close()

    qdrant_ok = result["qdrant"].get("ok")
    neo4j_ok = result["neo4j"].get("ok")

    print("=== MassKB Healthcheck ===")
    if qdrant_ok:
        cols = result["qdrant"].get("collections", [])
        print(f"[OK]  Qdrant  {result['qdrant'].get('url')}  collections={cols}")
    else:
        print(f"[FAIL] Qdrant  {result['qdrant'].get('error')}")

    if neo4j_ok:
        print(f"[OK]  Neo4j   {result['neo4j'].get('uri')}  ping={result['neo4j'].get('ping')}")
    else:
        print(f"[FAIL] Neo4j   {result['neo4j'].get('error')}")

    if result.get("openai_key_configured"):
        print("[INFO] OPENAI_API_KEY 已配置（本检查未调用 embedding）")
    else:
        print("[INFO] OPENAI_API_KEY 未配置真实值 — embedding/问答暂不可用（不阻塞本检查）")

    print("---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="masskb", description="MassKB 检索中间件 CLI")
    sub = parser.add_subparsers(dest="command")

    health = sub.add_parser("healthcheck", help="检查 Qdrant + Neo4j 连通性")
    health.set_defaults(func=cmd_healthcheck)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # 默认跑 healthcheck，便于 `python -m masskb`
        args = parser.parse_args(["healthcheck"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
