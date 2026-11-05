"""本地文档切分：UTF-8 的 .txt / .md → 带稳定 id 与字符偏移的分块。

本模块只读本地文件，不创建 Qdrant、Neo4j 或 OpenAI 客户端。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

_ALLOWED_SUFFIXES = {".txt", ".md"}


class IngestError(ValueError):
    """输入不符合本地切分契约（扩展名、编码、路径或窗口参数）。"""


@dataclass(frozen=True)
class Chunk:
    """一个分块。`start`/`end` 是换行规范化后文本上的字符偏移，左闭右开。"""

    doc_id: str
    chunk_id: str
    source: str
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Document:
    """一个源文件及其全部分块。空文件的 `chunks` 为空，但 `doc_id` 仍然稳定。"""

    source: str
    doc_id: str
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class IngestReport:
    """一次切分的全部文件，按 `source` 字典序排列。"""

    files: tuple[Document, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def chunk_count(self) -> int:
        return sum(len(doc.chunks) for doc in self.files)


def repo_root() -> Path:
    """仓库根目录（`masskb/` 的上一级）。

    与 `config` 加载 `.env` 时使用的根目录相同。文件落在仓库内时，
    `doc_id` 相对此目录计算，因此与当前工作目录无关。
    """
    return Path(__file__).resolve().parent.parent


def make_doc_id(source: str) -> str:
    """文档 id：路径键的 SHA-256 十六进制。与文件内容无关。"""
    return hashlib.sha256(source.encode()).hexdigest()


def make_chunk_id(doc_id: str, start: int, end: int) -> str:
    """分块 id：`sha256(f"{doc_id}:{start}:{end}")`。

    只由文档 id 和字符偏移决定。同一路径、同一内容重复切分，id 不变。
    """
    payload = f"{doc_id}:{start}:{end}".encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_newlines(text: str) -> str:
    """把 CRLF / CR 收成 LF。偏移量都相对规范化后的文本。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Document:
    """把一段已读入的文本切成稳定分块。

    `source` 是路径键（仓库内为相对仓库根的 POSIX 路径，否则为绝对 POSIX 路径，
    或调用方通过 `root` 指定的相对路径）。先按空行分段；某一段长于
    `chunk_size` 时，再按字符窗口切分，相邻窗口重叠 `overlap` 个字符。
    短段落不会为了填满窗口而合并。
    """
    _validate_window(chunk_size, overlap)
    key = _validate_source(source)
    normalized = normalize_newlines(text)
    doc_id = make_doc_id(key)
    spans: list[tuple[int, int]] = []
    for para_start, para_end in _paragraph_spans(normalized):
        spans.extend(_window_spans(para_start, para_end, chunk_size, overlap))

    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=make_chunk_id(doc_id, start, end),
                source=key,
                index=index,
                start=start,
                end=end,
                text=normalized[start:end],
            )
        )
    return Document(source=key, doc_id=doc_id, chunks=tuple(chunks))


def ingest_path(
    path: Path,
    *,
    root: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> IngestReport:
    """切分一个 UTF-8 `.txt`/`.md` 文件，或目录下所有此类文件。

    目录中的其他扩展名会被跳过。显式传入不支持的文件则报错。
    `root` 给定时，路径键相对该目录；否则仓库内文件相对仓库根，
    仓库外文件使用解析后的绝对 POSIX 路径。两种方式都与 cwd 无关。
    """
    _validate_window(chunk_size, overlap)
    files = _discover(path)
    documents = [
        _load_file(item, root=root, chunk_size=chunk_size, overlap=overlap) for item in files
    ]
    documents.sort(key=lambda doc: doc.source)
    seen: set[str] = set()
    for doc in documents:
        if doc.source in seen:
            raise IngestError(f"重复的文档路径键: {doc.source}")
        seen.add(doc.source)
    return IngestReport(files=tuple(documents))


def _validate_window(chunk_size: int, overlap: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise IngestError(f"chunk_size 必须为正整数，收到 {chunk_size!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise IngestError(f"overlap 不能为负，收到 {overlap!r}")
    if overlap >= chunk_size:
        raise IngestError(
            f"overlap 必须小于 chunk_size（overlap={overlap}, chunk_size={chunk_size}）"
        )


def _validate_source(source: str) -> str:
    if not isinstance(source, str) or source.strip() == "":
        raise IngestError("source 不能为空")
    return source


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """按空行切段。空行指去掉空白后为空的行。段内单个换行保留。

    返回的区间不含段尾换行，且是原文（已规范化）的精确切片。
    """
    spans: list[tuple[int, int]] = []
    para_start: int | None = None
    para_end: int | None = None
    for content_start, content_end in _iter_lines(text):
        if text[content_start:content_end].strip() == "":
            if para_start is not None and para_end is not None:
                spans.append((para_start, para_end))
            para_start = None
            para_end = None
            continue
        if para_start is None:
            para_start = content_start
        para_end = content_end
    if para_start is not None and para_end is not None:
        spans.append((para_start, para_end))
    return spans


def _iter_lines(text: str) -> list[tuple[int, int]]:
    """每行的内容区间，不含行尾的 `\n`。末尾换行不会多出一行。"""
    lines: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        nl = text.find("\n", start)
        if nl == -1:
            lines.append((start, n))
            break
        lines.append((start, nl))
        start = nl + 1
    return lines


def _window_spans(
    start: int,
    end: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if end <= start:
        return []
    if end - start <= chunk_size:
        return [(start, end)]
    step = chunk_size - overlap
    spans: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        win_end = min(pos + chunk_size, end)
        spans.append((pos, win_end))
        if win_end >= end:
            break
        pos += step
    return spans


def _discover(path: Path) -> list[Path]:
    if not path.exists():
        raise IngestError(f"路径不存在: {path}")
    if path.is_file():
        if not _is_supported(path):
            raise IngestError(f"仅支持 UTF-8 的 .txt 与 .md 文件，收到: {path.name}")
        return [path]
    if path.is_dir():
        return [item for item in path.rglob("*") if item.is_file() and _is_supported(item)]
    raise IngestError(f"不是文件或目录: {path}")


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in _ALLOWED_SUFFIXES


def _load_file(
    path: Path,
    *,
    root: Path | None,
    chunk_size: int,
    overlap: int,
) -> Document:
    return chunk_text(
        _read_utf8(path),
        source=source_key(path, root),
        chunk_size=chunk_size,
        overlap=overlap,
    )


def source_key(path: Path, root: Path | None) -> str:
    """计算与 cwd 无关的路径键。"""
    resolved = path.resolve()
    if root is not None:
        base = _as_directory(root)
        try:
            relative = resolved.relative_to(base)
        except ValueError as exc:
            raise IngestError(f"文件不在指定根目录内: {resolved}（根目录: {base}）") from exc
        return relative.as_posix()
    repo = repo_root()
    try:
        return resolved.relative_to(repo).as_posix()
    except ValueError:
        return resolved.as_posix()


def _as_directory(root: Path) -> Path:
    base = root.expanduser().resolve()
    if not base.is_dir():
        raise IngestError(f"根目录不存在或不是目录: {root}")
    return base


def _read_utf8(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IngestError(f"无法读取文件: {path}") from exc
    # utf-8-sig 去掉可选 BOM，正文偏移从正文第一个字符算起。
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestError(f"仅支持 UTF-8 文本: {path}") from exc
