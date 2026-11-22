"""Hostile SQLite stores, scratch cleanup, parse caps and inventory scope wording."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from nachos_core.reconciliation import acquire
from nachos_core.reconciliation.acquire import SourceSpec, parse_flatfile, read_current_sources
from nachos_core.reconciliation.boundary import AcquisitionError, SourceBoundary
from nachos_core.reconciliation.inventory import build_inventory

DB = "nachos/memory.db"
ENTRIES = "CREATE TABLE entries (key TEXT, title TEXT, summary TEXT, category TEXT, body TEXT);"
ROW = "INSERT INTO entries VALUES ('k','t','s','c','b');"


def _db(root: Path, script: str, raw: bytes | None = None) -> None:
    path = root / DB
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_bytes(raw)
        return
    conn = sqlite3.connect(path)
    conn.executescript(script)
    conn.commit()
    conn.close()


def _read(root: Path):
    boundary = SourceBoundary(root, (DB,))
    return read_current_sources(boundary, SourceSpec("sqlite"))


@pytest.mark.parametrize(
    "script",
    [
        "CREATE TABLE real (key TEXT, title TEXT, summary TEXT, category TEXT, body TEXT);"
        " INSERT INTO real VALUES ('k','t','s','c','b');"
        " CREATE VIEW entries AS SELECT * FROM real;",
        "CREATE VIEW entries AS SELECT 'k' AS key, 't' AS title, 's' AS summary,"
        " 'c' AS category, 'b' AS body;",
        "CREATE VIEW entries AS SELECT 'k' AS key, 't' AS title, 's' AS summary,"
        " 'c' AS category, hex(randomblob(4)) AS body;",
        "CREATE VIRTUAL TABLE entries USING fts5(key, title, summary, category, body);",
        "CREATE TABLE entries (key INTEGER, title TEXT, summary TEXT, category TEXT, body TEXT);"
        " INSERT INTO entries VALUES (1,'t','s','c','b');",
        "CREATE TABLE entries (key TEXT, title TEXT);",
        "CREATE TABLE ENTRIES_OTHER (x);",
    ],
)
def test_hostile_schemas_fail_closed_with_stable_reason(tmp_path, script):
    _db(tmp_path, script)
    with pytest.raises(AcquisitionError) as caught:
        _read(tmp_path)
    assert str(caught.value) == "invalid_sqlite"


@pytest.mark.parametrize(
    "raw", [b"SQLite format 3\0" + b"\xff" * 300, b"SQLite format 3\0", b"SQLite format 3\0" * 50]
)
def test_corrupt_database_fails_closed(tmp_path, raw):
    _db(tmp_path, "", raw)
    with pytest.raises(AcquisitionError) as caught:
        _read(tmp_path)
    assert str(caught.value) in {"invalid_sqlite", "invalid_sqlite_header", "helper_failed"}


def test_triggers_extra_tables_and_pragma_like_content_are_inert(tmp_path):
    _db(
        tmp_path,
        ENTRIES + " CREATE TABLE audit (x);"  # noqa: S608 -- synthetic SQL for a test database
        " CREATE TRIGGER t1 AFTER INSERT ON entries BEGIN INSERT INTO audit VALUES (1); END;"
        " CREATE TRIGGER t2 BEFORE UPDATE ON entries BEGIN SELECT RAISE(ABORT, 'x'); END;"
        " CREATE VIEW unrelated AS SELECT * FROM audit;"
        " INSERT INTO entries VALUES ('PRAGMA writable_schema=1;', 'ATTACH DATABASE 1',"
        " 'DROP TABLE entries', 'c', '; --');",
    )
    found = _read(tmp_path)
    assert list(found) == ["memory:PRAGMA writable_schema=1;"]
    raw = (tmp_path / DB).read_bytes()
    conn = sqlite3.connect(tmp_path / DB)
    assert conn.execute("SELECT count(*) FROM audit").fetchone() == (1,)
    conn.close()
    assert (tmp_path / DB).read_bytes() == raw


def test_non_utf8_text_is_rejected(tmp_path):
    _db(tmp_path, ENTRIES + ROW)
    conn = sqlite3.connect(tmp_path / DB)
    conn.text_factory = bytes
    conn.execute("UPDATE entries SET body = ?", (b"\xff\xfe\x80",))
    conn.commit()
    conn.close()
    with pytest.raises(AcquisitionError, match="invalid_sqlite"):
        _read(tmp_path)


def _tree(root: Path):
    def ident(p: Path):
        st = p.lstat()
        return (st.st_ino, st.st_size, st.st_mtime_ns, p.read_bytes() if p.is_file() else None)

    return {str(p.relative_to(root)): ident(p) for p in sorted(root.rglob("*"))}


def test_live_wal_database_is_read_without_touching_the_source(tmp_path):
    _db(tmp_path, ENTRIES)
    path = tmp_path / DB
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(ROW.replace("'k','t','s','c','b'", "'live','t','s','c','only in the wal'"))
    writer.commit()
    try:
        assert Path(str(path) + "-wal").stat().st_size > 0
        before = _tree(tmp_path)
        found = _read(tmp_path)
        assert _tree(tmp_path) == before
    finally:
        writer.close()
    assert [item.identity for item in found.values()] == ["memory:live"]


def test_snapshot_directory_is_removed_after_reading_a_wal_database(tmp_path, monkeypatch):
    _db(tmp_path, ENTRIES + ROW)
    writer = sqlite3.connect(tmp_path / DB)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("UPDATE entries SET body = 'newer'")
    writer.commit()
    made = []
    real = tempfile.mkdtemp

    def spy(*args, **kwargs):
        made.append(Path(real(*args, **kwargs)))
        return str(made[-1])

    monkeypatch.setattr(tempfile, "mkdtemp", spy)
    try:
        _read(tmp_path)
    finally:
        writer.close()
    assert made
    assert not any(path.exists() for path in made)


def test_flatfile_record_cap_applies_during_parsing(monkeypatch):
    text = "".join(f"## n{i}\nbody\n" for i in range(5))
    assert len(parse_flatfile(text)) == 5
    with pytest.raises(AcquisitionError, match="record_limit"):
        parse_flatfile(text, max_records=4)
    built = []
    real = acquire._flat_record

    def count(*args):
        built.append(1)
        return real(*args)

    monkeypatch.setattr(acquire, "_flat_record", count)
    monkeypatch.setattr(acquire, "MAX_RECORDS", 2)
    with pytest.raises(AcquisitionError, match="record_limit"):
        parse_flatfile(text)
    assert len(built) == 3  # stopped at the first record over the cap, not after all five


def test_inventory_states_it_is_only_the_supplied_explicit_labels(tmp_path):
    (tmp_path / "nachos").mkdir()
    (tmp_path / "nachos/memory.md").write_text("## A\nx\n", encoding="utf-8")
    (tmp_path / "skills/other").mkdir(parents=True)
    (tmp_path / "skills/other/SKILL.md").write_text("never listed", encoding="utf-8")
    boundary = SourceBoundary(tmp_path, ("nachos/memory.md",))
    inv = build_inventory(boundary, {"id": "p", "root": str(tmp_path)}, SourceSpec("flatfile"))
    text = " ".join(inv["coverage"]["limitations"])
    assert "explicit bounded labels" in text
    assert "not a discovery of the whole profile" in text
    assert "outside those labels were not examined" in text
    assert [s["locator"] for s in inv["sources"]] == ["nachos/memory.md"]
    assert "never listed" not in str(inv)
    assert inv["coverage"]["discovered"] is True
    assert [s["coverage"]["limitations"] for s in inv["sources"]] == [
        inv["coverage"]["limitations"]
    ]
