"""CLI 入口：python -m masskb  或  masskb"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from masskb.ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    IngestError,
    IngestReport,
    ingest_path,
)
from masskb.pipeline import IngestWriteResult, RetrievalPipeline

_PREVIEW_CHARS = 80


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


def _preview_text(text: str, limit: int = _PREVIEW_CHARS) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _format_report(
    report: IngestReport,
    *,
    dry_run: bool,
    preview_limit: int,
    written: IngestWriteResult | None = None,
) -> str:
    header = "=== MassKB Ingest (dry-run) ===" if dry_run else "=== MassKB Ingest ==="
    lines = [
        header,
        f"files: {report.file_count}",
        f"chunks: {report.chunk_count}",
        "---",
    ]
    for doc in report.files:
        lines.append(f"source: {doc.source}")
        lines.append(f"doc_id: {doc.doc_id}")
        lines.append(f"file_chunks: {len(doc.chunks)}")
        shown = doc.chunks[:preview_limit]
        for chunk in shown:
            lines.append(
                f"[{chunk.index}] chunk_id={chunk.chunk_id} "
                f"start={chunk.start} end={chunk.end}"
            )
            lines.append(f"    {_preview_text(chunk.text)}")
        hidden = len(doc.chunks) - len(shown)
        if hidden:
            lines.append(f"... 还有 {hidden} 个分块未展示")
    if dry_run:
        lines.append("[INFO] dry-run：未连接 Qdrant、Neo4j 或 OpenAI。")
    elif written is not None:
        if report.files:
            lines.append("---")
        lines.append(f"collection: {written.collection}")
        lines.append(f"vector_size: {written.vector_size}")
        lines.append(f"upserted: {written.points_upserted}")
        lines.append(
            f"[INFO] 已写入 Qdrant 集合 {written.collection}。未连接 Neo4j，未执行检索。"
        )
    return "\n".join(lines)


def cmd_ingest(args: argparse.Namespace) -> int:
    """切分本地文本。--dry-run 只预览；否则写入 Qdrant。"""
    if args.preview < 0:
        print("[FAIL] preview 不能为负", file=sys.stderr)
        return 1
    path = Path(args.path).expanduser()
    root = Path(args.root).expanduser() if args.root is not None else None
    try:
        report = ingest_path(
            path,
            root=root,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except IngestError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(_format_report(report, dry_run=True, preview_limit=args.preview))
        return 0

    pipeline = RetrievalPipeline()
    try:
        try:
            result = pipeline.ingest(
                path,
                root=root,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        except Exception as exc:  # noqa: BLE001 — CLI 把写入失败转成退出码
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        print(
            _format_report(
                result.report,
                dry_run=False,
                preview_limit=args.preview,
                written=result,
            )
        )
        return 0
    finally:
        pipeline.close()


def cmd_search(args: argparse.Namespace) -> int:
    """查询向量近邻，并把 JSON 打到 stdout。失败信息只写 stderr。"""
    pipeline = RetrievalPipeline()
    try:
        try:
            result = pipeline.search(args.query, top_k=args.top_k, doc_id=args.doc_id)
        except Exception as exc:  # noqa: BLE001 — CLI 把检索失败转成退出码
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0
    finally:
        pipeline.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="masskb", description="MassKB 检索中间件 CLI")
    sub = parser.add_subparsers(dest="command")

    health = sub.add_parser("healthcheck", help="检查 Qdrant + Neo4j 连通性")
    health.set_defaults(func=cmd_healthcheck)

    ingest = sub.add_parser(
        "ingest",
        help="切分 UTF-8 的 .txt/.md；默认写入 Qdrant。--dry-run 只预览",
    )
    ingest.add_argument("path", type=Path, help="要切分的 .txt/.md 文件或目录")
    ingest.add_argument(
        "--dry-run",
        action="store_true",
        help="只做本地预览，不连接 Qdrant / Neo4j / OpenAI",
    )
    ingest.add_argument(
        "--root",
        type=Path,
        default=None,
        help="计算 doc_id 的路径根；默认仓库根（仓库外文件用绝对路径）",
    )
    ingest.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"超长段落的字符窗口（默认 {DEFAULT_CHUNK_SIZE}）",
    )
    ingest.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"相邻窗口重叠的字符数（默认 {DEFAULT_CHUNK_OVERLAP}）",
    )
    ingest.add_argument(
        "--preview",
        type=int,
        default=3,
        help="每个文件最多展示的分块条数（默认 3）",
    )
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser(
        "search",
        help="用与入库相同的模型做 Qdrant 近邻检索，并把 JSON 打到 stdout",
    )
    search.add_argument("query", help="查询文本")
    search.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回的命中条数（默认 5）",
    )
    search.add_argument(
        "--doc-id",
        default=None,
        help="只检索该 doc_id；不传则搜索整个集合",
    )
    search.set_defaults(func=cmd_search)

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
