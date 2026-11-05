"""本地切分：空文件、短文件、超长段、重叠、Markdown 标题与稳定 id。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from masskb.ingest import (
    IngestError,
    chunk_text,
    ingest_path,
    make_chunk_id,
    make_doc_id,
    repo_root,
)


def _ids(doc) -> list[str]:
    return [chunk.chunk_id for chunk in doc.chunks]


def test_empty_file_yields_doc_id_and_no_chunks(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    blank = tmp_path / "blank.txt"
    blank.write_text("\n\n  \n\t\n", encoding="utf-8")

    report = ingest_path(tmp_path, root=tmp_path)

    assert report.file_count == 2
    assert report.chunk_count == 0
    assert [doc.source for doc in report.files] == ["blank.txt", "empty.md"]
    assert all(doc.chunks == () for doc in report.files)
    assert report.files[1].doc_id == make_doc_id("empty.md")
    assert ingest_path(empty, root=tmp_path).files[0] == report.files[1]


def test_short_file_single_chunk(tmp_path: Path) -> None:
    path = tmp_path / "short.txt"
    path.write_text("你好，世界\n", encoding="utf-8")

    doc = ingest_path(path, root=tmp_path).files[0]
    chunk = doc.chunks[0]

    assert len(doc.chunks) == 1
    assert chunk.text == "你好，世界"
    assert (chunk.start, chunk.end, chunk.index) == (0, 5, 0)
    assert chunk.source == "short.txt"
    assert chunk.doc_id == doc.doc_id == make_doc_id("short.txt")
    assert chunk.chunk_id == make_chunk_id(doc.doc_id, 0, 5)
    assert chunk.chunk_id == hashlib.sha256(f"{doc.doc_id}:0:5".encode()).hexdigest()


def test_blank_lines_separate_segments_without_merging() -> None:
    doc = chunk_text("aa\n\n\nbb\ncc", source="s.txt", chunk_size=100, overlap=10)

    assert [chunk.text for chunk in doc.chunks] == ["aa", "bb\ncc"]
    assert (doc.chunks[0].start, doc.chunks[0].end) == (0, 2)
    assert (doc.chunks[1].start, doc.chunks[1].end) == (5, 10)
    assert doc.chunks[1].text == "bb\ncc"

    spaced = chunk_text("aa\n \nbb", source="s.txt", chunk_size=100, overlap=0)
    assert [chunk.text for chunk in spaced.chunks] == ["aa", "bb"]


def test_overlong_segment_splits_on_character_window() -> None:
    text = "测" * 25
    doc = chunk_text(text, source="long.md", chunk_size=10, overlap=0)

    assert [(chunk.start, chunk.end) for chunk in doc.chunks] == [(0, 10), (10, 20), (20, 25)]
    assert "".join(chunk.text for chunk in doc.chunks) == text
    assert [chunk.index for chunk in doc.chunks] == [0, 1, 2]
    assert all(len(chunk.text) <= 10 for chunk in doc.chunks)
    assert all(text[chunk.start : chunk.end] == chunk.text for chunk in doc.chunks)

    mixed = ("a" * 10) + "\n\n" + ("b" * 10)
    parts = chunk_text(mixed, source="mix.txt", chunk_size=6, overlap=0)
    assert [chunk.text for chunk in parts.chunks] == ["aaaaaa", "aaaa", "bbbbbb", "bbbb"]
    assert [(chunk.start, chunk.end) for chunk in parts.chunks] == [
        (0, 6),
        (6, 10),
        (12, 18),
        (18, 22),
    ]
    assert all(set(chunk.text) <= {"a"} or set(chunk.text) <= {"b"} for chunk in parts.chunks)


def test_overlap_windows_share_characters() -> None:
    text = "0123456789" * 3
    doc = chunk_text(text, source="ov.txt", chunk_size=10, overlap=4)

    assert [(chunk.start, chunk.end) for chunk in doc.chunks] == [
        (0, 10),
        (6, 16),
        (12, 22),
        (18, 28),
        (24, 30),
    ]
    for prev, nxt in zip(doc.chunks, doc.chunks[1:]):
        assert prev.text[-4:] == nxt.text[:4]
        assert text[nxt.start : nxt.end] == nxt.text

    covered = [0] * len(text)
    for chunk in doc.chunks:
        for index in range(chunk.start, chunk.end):
            covered[index] += 1
    assert all(count >= 1 for count in covered)
    assert covered[6:10] == [2, 2, 2, 2]

    separated = ("a" * 10) + "\n\n" + ("b" * 10)
    parts = chunk_text(separated, source="p.txt", chunk_size=8, overlap=3)
    assert [chunk.text for chunk in parts.chunks] == ["a" * 8, "a" * 5, "b" * 8, "b" * 5]
    assert parts.chunks[0].text[-3:] == parts.chunks[1].text[:3]
    assert set(parts.chunks[2].text) == {"b"}
    assert parts.chunks[1].end == 10
    assert parts.chunks[2].start == 12


def test_markdown_headings_are_preserved() -> None:
    text = "# 总览\n\n引言还在。\n\n## 细节\n\n- 条目一\n- 条目二\n"
    doc = chunk_text(text, source="说明.md")

    assert [chunk.text for chunk in doc.chunks] == [
        "# 总览",
        "引言还在。",
        "## 细节",
        "- 条目一\n- 条目二",
    ]
    for chunk in doc.chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert chunk.text.strip() != ""

    glued = chunk_text("# 无空行\n紧挨着的正文", source="说明.md")
    assert len(glued.chunks) == 1
    assert glued.chunks[0].text == "# 无空行\n紧挨着的正文"
    assert glued.chunks[0].start == 0


def test_repeated_chunking_same_path_keeps_identical_chunk_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "corpus"
    folder = root / "notes"
    folder.mkdir(parents=True)
    target = folder / "page.md"
    target.write_text("# 标题\n\n正文还在。\n", encoding="utf-8")

    monkeypatch.chdir(root)
    first = ingest_path(Path("notes/page.md"), root=root)
    second = ingest_path(Path("notes/page.md"), root=root)
    assert first == second
    assert _ids(first.files[0]) == _ids(second.files[0])
    assert first.files[0].source == "notes/page.md"

    monkeypatch.chdir(folder)
    from_nested = ingest_path(Path("page.md"), root=root)
    monkeypatch.chdir(tmp_path)
    from_absolute = ingest_path(target, root=root)

    assert _ids(from_nested.files[0]) == _ids(first.files[0])
    assert _ids(from_absolute.files[0]) == _ids(first.files[0])
    assert from_nested.files[0].doc_id == from_absolute.files[0].doc_id == first.files[0].doc_id

    via_dir = ingest_path(root, root=root)
    assert via_dir.files[0] == first.files[0]


def test_repo_file_ids_independent_of_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    readme = repo_root() / "README.md"
    monkeypatch.chdir(tmp_path)
    from_abs = ingest_path(readme)
    monkeypatch.chdir(repo_root())
    from_rel = ingest_path(Path("README.md"))
    again = ingest_path(Path("README.md"))

    assert from_abs == from_rel == again
    assert from_abs.files[0].source == "README.md"
    assert _ids(from_abs.files[0]) == _ids(again.files[0])
    assert Path(__file__).resolve().parents[1] == repo_root()


def test_outside_repo_ids_use_absolute_path(tmp_path: Path) -> None:
    left = tmp_path / "a" / "n.txt"
    right = tmp_path / "b" / "n.txt"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("same\n", encoding="utf-8")
    right.write_text("same\n", encoding="utf-8")

    left_doc = ingest_path(left).files[0]
    right_doc = ingest_path(right).files[0]

    assert left_doc.source == left.resolve().as_posix()
    assert right_doc.source == right.resolve().as_posix()
    assert left_doc.doc_id != right_doc.doc_id
    assert left_doc.chunks[0].text == right_doc.chunks[0].text
    assert repo_root() not in left.resolve().parents


def test_newline_style_and_bom_do_not_change_ids(tmp_path: Path) -> None:
    lf = chunk_text("甲\n\n乙", source="n.md", chunk_size=20, overlap=0)
    crlf = chunk_text("甲\r\n\r\n乙", source="n.md", chunk_size=20, overlap=0)
    assert lf == crlf

    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhello\n")
    doc = ingest_path(path, root=tmp_path).files[0]
    assert doc.chunks[0].text == "hello"
    assert (doc.chunks[0].start, doc.chunks[0].end) == (0, 5)


def test_directory_keeps_only_txt_and_md(tmp_path: Path) -> None:
    root = tmp_path / "in"
    (root / "nested").mkdir(parents=True)
    (root / "a.md").write_text("甲\n", encoding="utf-8")
    (root / "nested" / "b.TXT").write_text("乙\n", encoding="utf-8")
    (root / "c.py").write_text("pass\n", encoding="utf-8")
    (root / "note.markdown").write_text("nope\n", encoding="utf-8")
    (root / "empty.txt").write_bytes(b"")

    report = ingest_path(root, root=root)

    assert [doc.source for doc in report.files] == ["a.md", "empty.txt", "nested/b.TXT"]
    assert report.file_count == 3
    assert report.chunk_count == 2
    assert report.files[0].chunks[0].text == "甲"
    assert report.files[0].chunks[0].index == 0
    assert report.files[2].chunks[0].text == "乙"
    assert report.files[2].chunks[0].index == 0


def test_rejects_unsupported_extension_and_non_utf8(tmp_path: Path) -> None:
    py_file = tmp_path / "a.py"
    py_file.write_text("print(1)\n", encoding="utf-8")
    with pytest.raises(IngestError, match=r"\.txt"):
        ingest_path(py_file)

    markdown = tmp_path / "a.markdown"
    markdown.write_text("hello\n", encoding="utf-8")
    with pytest.raises(IngestError, match=r"\.md"):
        ingest_path(markdown)

    bad = tmp_path / "bad.txt"
    bad.write_bytes("中文".encode("gbk"))
    with pytest.raises(IngestError, match="UTF-8"):
        ingest_path(bad)

    with pytest.raises(IngestError, match="不存在"):
        ingest_path(tmp_path / "missing.md")

    outside = tmp_path / "keep.txt"
    outside.write_text("x\n", encoding="utf-8")
    other = tmp_path / "root"
    other.mkdir()
    with pytest.raises(IngestError, match="根目录"):
        ingest_path(outside, root=other)


def test_window_parameters_must_be_sane() -> None:
    with pytest.raises(IngestError, match="chunk_size"):
        chunk_text("a", source="a.txt", chunk_size=0, overlap=0)
    with pytest.raises(IngestError, match="overlap"):
        chunk_text("a", source="a.txt", chunk_size=8, overlap=8)
    with pytest.raises(IngestError, match="overlap"):
        chunk_text("a", source="a.txt", chunk_size=8, overlap=-1)
    with pytest.raises(IngestError, match="source"):
        chunk_text("a", source="  ")
