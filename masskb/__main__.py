"""CLI 入口：python -m masskb  或  masskb"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from masskb.agent import MAX_AGENT_STEPS
from masskb.aliases import alias_flag_error
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

    settings = getattr(pipeline, "settings", None)
    embedding_url = ""
    if settings is not None:
        embedding_url = (getattr(settings, "embedding_base_url", "") or "").strip()
    if result.get("openai_key_configured"):
        print("[INFO] OPENAI_API_KEY 已配置（本检查未调用 embedding）")
    elif embedding_url:
        print(
            "[INFO] OPENAI_API_KEY 未配置真实值 — 问答暂不可用；"
            "嵌入地址已单独配置（本检查未调用）"
        )
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
    entities: bool = False,
    aliases: bool = False,
    drop_aliases: bool = False,
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
        if entities:
            lines.append("[INFO] --entities 在 dry-run 中不抽取。")
        if aliases:
            lines.append("[INFO] --aliases 在 dry-run 中不合并。")
        if drop_aliases:
            lines.append("[INFO] --drop-aliases 在 dry-run 中不删除。")
    elif written is not None:
        if report.files:
            lines.append("---")
        lines.append(f"collection: {written.collection}")
        lines.append(f"vector_size: {written.vector_size}")
        lines.append("sparse: bm25")
        lines.append(f"upserted: {written.points_upserted}")
        lines.append(f"graph_documents: {written.graph_documents}")
        lines.append(f"graph_chunks: {written.graph_chunks}")
        if not written.entities_skipped:
            lines.append("entities: on")
            lines.append(f"graph_persons: {written.graph_persons}")
            lines.append(f"graph_concepts: {written.graph_concepts}")
            lines.append(f"graph_mentions: {written.graph_mentions}")
            lines.append(f"entities_dropped: {written.entities_dropped}")
        if not written.aliases_skipped:
            lines.append("aliases: drop" if drop_aliases else "aliases: on")
            lines.append(f"graph_aliases: {written.graph_aliases}")
            lines.append(f"aliases_dropped: {written.aliases_dropped}")
            if not written.aliases_ok:
                lines.append(f"[INFO] 别名未写入：{written.aliases_error}")
        if written.graph_skipped:
            lines.append("[INFO] 没有可写入的分块，且集合不存在。未连接 Neo4j，未执行检索。")
        elif written.entities_skipped:
            lines.append(
                f"[INFO] 已写入 Qdrant 集合 {written.collection}，"
                "并同步 Neo4j 文档骨架。未执行检索。"
            )
        else:
            lines.append(
                f"[INFO] 已写入 Qdrant 集合 {written.collection}，"
                "已同步 Neo4j 文档骨架，并替换本批文档的人物/概念提及。未执行检索。"
            )
    return "\n".join(lines)


def cmd_ingest(args: argparse.Namespace) -> int:
    """切分本地文本。--dry-run 只预览；否则写入 Qdrant 并同步文档骨架。"""
    rejected = alias_flag_error(
        entities=args.entities,
        aliases=args.aliases,
        drop_aliases=args.drop_aliases,
    )
    if rejected is not None:
        print(f"[FAIL] {rejected}", file=sys.stderr)
        return 1
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
        print(
            _format_report(
                report,
                dry_run=True,
                preview_limit=args.preview,
                entities=args.entities,
                aliases=args.aliases,
                drop_aliases=args.drop_aliases,
            )
        )
        return 0

    pipeline = RetrievalPipeline()
    try:
        try:
            result = pipeline.ingest(
                path,
                root=root,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                entities=args.entities,
                aliases=args.aliases,
                drop_aliases=args.drop_aliases,
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
                entities=args.entities,
                aliases=args.aliases,
                drop_aliases=args.drop_aliases,
            )
        )
        return 0
    finally:
        pipeline.close()


def _listen_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port 必须是 1 到 65535 的整数") from exc
    if port < 1 or port > 65535:
        raise argparse.ArgumentTypeError("port 必须是 1 到 65535 的整数")
    return port


def cmd_serve(args: argparse.Namespace) -> int:
    """启动 HTTP。FastAPI 与 uvicorn 在这里才导入。"""
    try:
        import uvicorn

        from masskb.api import app
    except ImportError:
        print(
            '[FAIL] 未安装 API 额外依赖。请执行 pip install -e ".[api]"',
            file=sys.stderr,
        )
        return 1
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as exc:  # noqa: BLE001 — CLI 把启动失败转成退出码
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """查询向量近邻，并把 JSON 打到 stdout。向量失败只写 stderr。"""
    pipeline = RetrievalPipeline()
    try:
        try:
            result = pipeline.search(
                args.query,
                top_k=args.top_k,
                doc_id=args.doc_id,
                hybrid=args.hybrid,
                answer=args.answer,
                rerank=args.rerank,
                rewrite=args.rewrite,
                graph_hops=args.graph_hops,
                aliases=args.aliases,
            )
        except Exception as exc:  # noqa: BLE001 — CLI 把检索失败转成退出码
            print(f"[FAIL] {exc}", file=sys.stderr)
            return 1
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0
    finally:
        pipeline.close()


def cmd_ask(args: argparse.Namespace) -> int:
    """按有界工具循环回答问题，并把 JSON 打到 stdout。硬失败只写 stderr。"""
    pipeline = RetrievalPipeline()
    try:
        try:
            result = pipeline.ask(
                args.query,
                top_k=args.top_k,
                doc_id=args.doc_id,
                hybrid=args.hybrid,
                rewrite=args.rewrite,
                rerank=args.rerank,
                graph_hops=args.graph_hops,
                aliases=args.aliases,
                max_steps=args.max_steps,
            )
        except Exception as exc:  # noqa: BLE001 — CLI 把失败转成退出码
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
        help="切分 .txt/.md，写入 Qdrant 并同步 Neo4j 文档骨架。--dry-run 只预览",
        description="切分 .txt/.md，写入 Qdrant 并同步 Neo4j 文档骨架。--dry-run 只预览",
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
    ingest.add_argument(
        "--entities",
        action="store_true",
        help="用 LLM_MODEL 抽取人物与概念并写入 Neo4j；默认关闭",
    )
    ingest.add_argument(
        "--aliases",
        action="store_true",
        help="把本篇已抽取的人物/概念表面名合并为别名；默认关闭，必须和 --entities 一起使用",
    )
    ingest.add_argument(
        "--drop-aliases",
        action="store_true",
        help="删除本篇文档写下的别名边；默认关闭，必须和 --entities 一起使用",
    )
    ingest.set_defaults(func=cmd_ingest)

    search = sub.add_parser(
        "search",
        help="用与入库相同的模型做 Qdrant 近邻检索，JSON 中附带相邻 chunk_id",
        description="用与入库相同的模型做 Qdrant 近邻检索，JSON 中附带相邻 chunk_id",
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
    search.add_argument(
        "--hybrid",
        action="store_true",
        help="融合稠密近邻与 bm25 稀疏名次；默认只做余弦近邻",
    )
    search.add_argument(
        "--answer",
        action="store_true",
        help="用 LLM_MODEL 根据命中分块生成带 chunk_id 引用的回答；默认关闭",
    )
    search.add_argument(
        "--rerank",
        action="store_true",
        help="用 LLM_MODEL 对已截断的 top_k 做 listwise 重排；默认关闭",
    )
    search.add_argument(
        "--rewrite",
        action="store_true",
        help="用 LLM_MODEL 在检索前把查询改写成一条检索文本；默认关闭",
    )
    search.add_argument(
        "--graph-hops",
        action="store_true",
        help="沿已有前后边和人物/概念提及最多替换 1 条命中；默认关闭",
    )
    search.add_argument(
        "--aliases",
        action="store_true",
        help="图扩展时再走一跳人物/概念别名；默认关闭，且只在 --graph-hops 打开时生效",
    )
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser(
        "ask",
        help="用最多 3 步调用检索和健康检查，再给出带引用的回答",
        description=(
            "用最多 3 步调用现有检索和健康检查，再按本轮命中的 chunk_id 收束回答。"
            "默认不进入 search。"
        ),
    )
    ask.add_argument("query", help="问题文本")
    ask.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="每次 search 返回的命中条数（默认 5）",
    )
    ask.add_argument(
        "--doc-id",
        default=None,
        help="每次 search 只检索该 doc_id；不传则搜索整个集合",
    )
    ask.add_argument(
        "--hybrid",
        action="store_true",
        help="每次 search 融合稠密近邻与 bm25 稀疏名次；默认关闭",
    )
    ask.add_argument(
        "--rewrite",
        action="store_true",
        help="每次 search 在检索前把查询改写成一条检索文本；默认关闭",
    )
    ask.add_argument(
        "--rerank",
        action="store_true",
        help="每次 search 对已截断的 top_k 做 listwise 重排；默认关闭",
    )
    ask.add_argument(
        "--graph-hops",
        action="store_true",
        help="每次 search 沿已有前后边和人物/概念提及最多替换 1 条命中；默认关闭",
    )
    ask.add_argument(
        "--aliases",
        action="store_true",
        help="每次 search 在图扩展时再走一跳别名；默认关闭，且只在 --graph-hops 打开时生效",
    )
    ask.add_argument(
        "--max-steps",
        type=int,
        default=MAX_AGENT_STEPS,
        help=f"代理步数，1 到 {MAX_AGENT_STEPS}（默认 {MAX_AGENT_STEPS}）",
    )
    ask.set_defaults(func=cmd_ask)

    serve = sub.add_parser(
        "serve",
        help='启动 HTTP（需要 pip install -e ".[api]"）',
        description=(
            "启动 HTTP：GET /health、POST /search、POST /ask 与 POST /ingest。"
            "CLI 入库仍可使用 ingest 子命令。"
            '需要 pip install -e ".[api]"。'
        ),
    )
    serve.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    serve.add_argument(
        "--port",
        type=_listen_port,
        default=8080,
        help="监听端口（默认 8080）",
    )
    serve.set_defaults(func=cmd_serve)

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
