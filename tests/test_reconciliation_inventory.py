"""Synthetic-only inventory acquisition tests (disposable tmp_path profiles)."""

import os
import sqlite3
from pathlib import Path

import pytest

from nachos_core.reconciliation import acquire
from nachos_core.reconciliation.acquire import SourceSpec, read_current_sources
from nachos_core.reconciliation.boundary import AcquisitionError, SourceBoundary
from nachos_core.reconciliation.inventory import build_inventory
from nachos_core.reconciliation.models import validate_document
from nachos_core.store.md_store import MDStore

DB = "nachos/memory.db"
MD = "nachos/memory.md"
SKILL = "skills/demo/SKILL.md"
REF = "skills/demo/ref.md"
ROWS = [
    ("k1", "Alpha", "alpha summary", "general", "alpha summary\nmore"),
    ("k2", "Beta", "beta summary", "ops", "beta summary"),
]


def _make_db(root: Path, rows=ROWS, wal=False) -> Path:
    path = root / DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE entries (key TEXT PRIMARY KEY, title TEXT, summary TEXT,"
        " category TEXT, body TEXT)"
    )
    conn.executemany("INSERT INTO entries VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


def _make_md(root: Path, text: str = "# ops\n\n## Beta Note\nbeta line\nmore\n") -> Path:
    path = root / MD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _files(root: Path) -> None:
    (root / "skills/demo").mkdir(parents=True)
    (root / SKILL).write_text("# Demo\nbody\n", encoding="utf-8")
    (root / REF).write_text("ref text\n", encoding="utf-8")


def _boundary(root: Path, *labels: str) -> SourceBoundary:
    return SourceBoundary(root, allowed_labels=labels)


def _state(root: Path):
    out = {}
    for dirpath, dirs, files in os.walk(root):
        for name in sorted(dirs + files):
            full = Path(dirpath) / name
            st = full.lstat()
            data = full.read_bytes() if full.is_file() else None
            out[str(full)] = (
                st.st_mode,
                st.st_ino,
                st.st_size,
                st.st_mtime_ns,
                st.st_ctime_ns,
                data,
            )
    return out


def _profile(root: Path) -> dict:
    return {"id": "p", "root": str(root)}


def test_sqlite_round_trip_with_files(tmp_path):
    _make_db(tmp_path)
    _files(tmp_path)
    boundary = _boundary(tmp_path, DB, SKILL, REF)
    spec = SourceSpec("sqlite", (SKILL,), (REF,))
    inv = build_inventory(boundary, _profile(tmp_path), spec)
    validate_document(inv, expected_profile=_profile(tmp_path))
    by_identity = {s["identity"]: s for s in inv["sources"]}
    assert set(by_identity) == {"memory:k1", "memory:k2", f"skill:{SKILL}", f"reference:{REF}"}
    assert by_identity["memory:k1"]["before"]["body"] == "alpha summary\nmore"
    assert by_identity["memory:k1"]["locator"] == DB
    assert by_identity[f"skill:{SKILL}"]["before"] == "# Demo\nbody\n"
    assert by_identity[f"reference:{REF}"]["role"] == "reference"


def test_flatfile_round_trip(tmp_path):
    _make_md(tmp_path)
    boundary = _boundary(tmp_path, MD)
    found = read_current_sources(boundary, SourceSpec("flatfile"))
    record = found["memory:beta-note"]
    assert record.before == {
        "key": "beta-note",
        "title": "Beta Note",
        "summary": "beta line",
        "category": "ops",
        "body": "beta line\nmore",
    }
    assert record.locator == MD


def test_flatfile_parse_matches_md_store(tmp_path):
    text = (
        "pre\n# Cat A\n\n## One\nsum\nbody\n\n## Two Words!\n\nlater\n# Cat B\n## Three\nx\r\ny\n"
    )
    path = _make_md(tmp_path, text)
    store = MDStore(path)
    expected = {k: (r.title, r.summary, r.category, r.body) for k, r in store._read().items()}
    got = {
        r["key"]: (r["title"], r["summary"], r["category"], r["body"])
        for r in acquire.parse_flatfile(path.read_text(encoding="utf-8"))
    }
    assert got == expected


def test_duplicate_flatfile_keys_rejected(tmp_path):
    _make_md(tmp_path, "## a b\nx\n## a-b\ny\n")
    with pytest.raises(AcquisitionError, match="duplicate_key"):
        read_current_sources(_boundary(tmp_path, MD), SourceSpec("flatfile"))


@pytest.mark.parametrize("store", ["sqlite", "flatfile"])
def test_missing_store_errors_and_creates_nothing(tmp_path, store):
    label = DB if store == "sqlite" else MD
    before = _state(tmp_path)
    with pytest.raises(AcquisitionError):
        build_inventory(
            _boundary(tmp_path, label),
            _profile(tmp_path),
            SourceSpec(store),
        )
    assert _state(tmp_path) == before
    assert list(tmp_path.iterdir()) == []


def test_missing_skill_errors(tmp_path):
    _make_md(tmp_path)
    with pytest.raises(AcquisitionError, match="source_unavailable"):
        read_current_sources(_boundary(tmp_path, MD, SKILL), SourceSpec("flatfile", (SKILL,)))


def test_wal_database_is_readable_and_not_modified(tmp_path):
    path = _make_db(tmp_path, wal=True)
    before = _state(tmp_path)
    found = read_current_sources(_boundary(tmp_path, DB), SourceSpec("sqlite"))
    assert found
    assert _state(tmp_path) == before
    assert path.exists()


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        ("../x.md", "invalid_label"),
        ("/etc/passwd", "invalid_label"),
        ("skills/.env", "credential_path"),
        ("skills/token.md", "credential_path"),
    ],
)
def test_bad_labels_rejected(tmp_path, label, reason):
    _make_md(tmp_path)
    with pytest.raises(AcquisitionError, match=reason):
        SourceBoundary(tmp_path, allowed_labels=(MD, label))


def test_label_must_be_allowlisted(tmp_path):
    _make_md(tmp_path)
    (tmp_path / "other.md").write_text("x")
    with pytest.raises(AcquisitionError, match="not_allowlisted"):
        read_current_sources(_boundary(tmp_path, MD), SourceSpec("flatfile", ("other.md",)))


def test_symlinked_skill_rejected(tmp_path):
    _make_md(tmp_path)
    (tmp_path / "real.md").write_text("secret-ish")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "link.md").symlink_to(tmp_path / "real.md")
    with pytest.raises(AcquisitionError, match="source_unavailable"):
        read_current_sources(
            _boundary(tmp_path, MD, "skills/link.md"), SourceSpec("flatfile", ("skills/link.md",))
        )


def test_symlinked_directory_rejected(tmp_path):
    _make_md(tmp_path)
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "SKILL.md").write_text("x")
    (tmp_path / "skills").symlink_to(real)
    with pytest.raises(AcquisitionError, match="source_unavailable"):
        read_current_sources(
            _boundary(tmp_path, MD, "skills/SKILL.md"), SourceSpec("flatfile", ("skills/SKILL.md",))
        )


def test_spec_validation(tmp_path):
    _make_md(tmp_path)
    boundary = _boundary(tmp_path, MD, SKILL)
    with pytest.raises(AcquisitionError, match="invalid_store"):
        read_current_sources(boundary, SourceSpec("other"))
    with pytest.raises(AcquisitionError, match="invalid_spec"):
        read_current_sources(boundary, SourceSpec("flatfile", (SKILL,), (SKILL,)))
    with pytest.raises(AcquisitionError, match="invalid_spec"):
        read_current_sources(boundary, SourceSpec("flatfile", (MD,)))
    too_many = tuple(f"s/{i}.md" for i in range(acquire.MAX_LABELS + 1))
    with pytest.raises(AcquisitionError, match="invalid_spec"):
        read_current_sources(boundary, SourceSpec("flatfile", too_many))


def test_non_utf8_and_nul_files_rejected(tmp_path):
    _make_md(tmp_path)
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills/bad.md").write_bytes(b"\xff\xfe bad")
    (tmp_path / "skills/nul.md").write_bytes(b"a\0b")
    for name in ("bad", "nul"):
        label = f"skills/{name}.md"
        with pytest.raises(AcquisitionError, match="invalid_encoding"):
            read_current_sources(_boundary(tmp_path, MD, label), SourceSpec("flatfile", (label,)))


def test_oversized_file_rejected(tmp_path):
    _make_md(tmp_path)
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills/big.md").write_bytes(b"a" * (acquire.MAX_FILE_BYTES + 1))
    with pytest.raises(AcquisitionError, match="size_limit"):
        read_current_sources(
            _boundary(tmp_path, MD, "skills/big.md"), SourceSpec("flatfile", ("skills/big.md",))
        )


def test_total_budget_enforced(tmp_path):
    _make_md(tmp_path)
    (tmp_path / "skills").mkdir()
    labels = []
    for i in range(5):
        (tmp_path / f"skills/f{i}.md").write_bytes(b"a" * (acquire.MAX_FILE_BYTES - 10))
        labels.append(f"skills/f{i}.md")
    with pytest.raises(AcquisitionError, match="size_limit"):
        read_current_sources(
            _boundary(tmp_path, MD, *labels), SourceSpec("flatfile", tuple(labels))
        )


def test_sqlite_hostile_rows(tmp_path):
    hostile = [
        ("../../etc/passwd", "<script>alert(1)</script>", "$(rm -rf /)", "c;DROP", "# x\n\x01"),
        ("big", "t", "s", "c", "b" * 100_000),
    ]
    _make_db(tmp_path, hostile)
    found = read_current_sources(_boundary(tmp_path, DB), SourceSpec("sqlite"))
    assert found["memory:../../etc/passwd"].locator == DB
    assert found["memory:big"].char_size > 100_000


def test_sqlite_nul_and_non_text_rejected(tmp_path):
    _make_db(tmp_path, [("k", "t", "s", "c", "a\0b")])
    with pytest.raises(AcquisitionError, match="invalid_encoding"):
        read_current_sources(_boundary(tmp_path, DB), SourceSpec("sqlite"))


def test_sqlite_non_str_column_rejected(tmp_path):
    _make_db(tmp_path, [("k", "t", "s", "c", b"blob")])
    with pytest.raises(AcquisitionError, match="invalid_sqlite"):
        read_current_sources(_boundary(tmp_path, DB), SourceSpec("sqlite"))


def test_sqlite_wrong_schema_rejected(tmp_path):
    path = tmp_path / DB
    path.parent.mkdir()
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE other (x TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(AcquisitionError, match="invalid_sqlite"):
        read_current_sources(_boundary(tmp_path, DB), SourceSpec("sqlite"))


def test_sqlite_non_utf8_text_rejected(tmp_path):
    path = _make_db(tmp_path)
    conn = sqlite3.connect(path)
    conn.text_factory = bytes
    conn.execute("UPDATE entries SET title = ? WHERE key = 'k1'", (b"\xff\xfe",))
    conn.commit()
    conn.close()
    with pytest.raises(AcquisitionError, match="invalid_sqlite"):
        read_current_sources(_boundary(tmp_path, DB), SourceSpec("sqlite"))


def test_record_limit(tmp_path, monkeypatch):
    _make_db(tmp_path)
    monkeypatch.setattr(acquire, "MAX_RECORDS", 1)
    with pytest.raises(AcquisitionError, match="record_limit"):
        read_current_sources(_boundary(tmp_path, DB), SourceSpec("sqlite"))


def test_authorizer_denies_everything_else():
    assert acquire._authorizer(sqlite3.SQLITE_SELECT, None, None) == sqlite3.SQLITE_OK
    assert acquire._authorizer(sqlite3.SQLITE_READ, "entries", "body") == sqlite3.SQLITE_OK
    assert acquire._authorizer(sqlite3.SQLITE_READ, "entries", "rowid") == sqlite3.SQLITE_DENY
    assert acquire._authorizer(sqlite3.SQLITE_READ, "other", "body") == sqlite3.SQLITE_DENY
    assert acquire._authorizer(sqlite3.SQLITE_INSERT, "entries", None) == sqlite3.SQLITE_DENY
    assert acquire._authorizer(sqlite3.SQLITE_ATTACH, "x", None) == sqlite3.SQLITE_DENY


def test_unknown_metadata_stays_unknown(tmp_path):
    _make_db(tmp_path)
    _files(tmp_path)
    inv = build_inventory(
        _boundary(tmp_path, DB, SKILL),
        _profile(tmp_path),
        SourceSpec("sqlite", (SKILL,)),
    )
    for source in inv["sources"]:
        assert source["metadata"]["enabled"] is None
        assert source["metadata"]["provenance"] == []
        assert source["metadata"]["usage"] == {"status": "unknown", "references": []}
        cov = source["coverage"]
        assert (cov["discovered"], cov["hashed"]) == (True, True)
        assert cov["content_searched"] is False
        assert cov["section_reviewed"] is False
        assert cov["body_reviewed"] is False
        assert cov["limitations"]
    assert inv["coverage"]["body_reviewed"] is False
    assert inv["coverage"]["limitations"]


def test_profile_must_match_boundary_root(tmp_path):
    _make_md(tmp_path)
    boundary = _boundary(tmp_path, MD)
    for bad in ({"id": "p", "root": "/elsewhere"}, {"id": "p"}, {"id": 1, "root": str(tmp_path)}):
        with pytest.raises(AcquisitionError, match="profile_mismatch"):
            build_inventory(boundary, bad, SourceSpec("flatfile"))


def test_deterministic_and_sources_unchanged(tmp_path):
    _make_db(tmp_path)
    _files(tmp_path)
    boundary = _boundary(tmp_path, DB, SKILL, REF)
    spec = SourceSpec("sqlite", (SKILL,), (REF,))
    before = _state(tmp_path)
    first = build_inventory(boundary, _profile(tmp_path), spec)
    second = build_inventory(boundary, _profile(tmp_path), spec)
    assert first == second
    assert first["digest"] == second["digest"]
    assert _state(tmp_path) == before
    ids = [s["id"] for s in first["sources"]]
    assert len(set(ids)) == len(ids)
    assert all(i.startswith("src-") for i in ids)


def test_sqlite_source_is_only_opened_read_only(tmp_path, monkeypatch):
    _make_db(tmp_path)
    targets = []
    real = sqlite3.connect

    def spy(target, *args, **kwargs):
        targets.append(str(target))
        return real(target, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy)
    read_current_sources(_boundary(tmp_path, DB), SourceSpec("sqlite"))
    assert targets
    assert all(t.endswith("?mode=ro") for t in targets if str(tmp_path) in t)
