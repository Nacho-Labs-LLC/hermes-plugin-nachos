"""Offline maintenance must preserve source files, including WAL-mode SQLite databases."""

import importlib.metadata
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.store import SqliteStore
from nachos_hermes import maintenance


def snapshot(home):
    return {
        str(p.relative_to(home)): (p.stat().st_mtime_ns, p.read_bytes()) if p.is_file() else None
        for p in home.rglob("*")
    }


def maintain(home, check="all", *options):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "nachos_hermes.maintenance",
            "--profile-home",
            str(home),
            "--json",
            *options,
            check,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_offline_report_preserves_bytes_and_unknown_telemetry(tmp_path):
    path = tmp_path / "nachos" / "memory.db"
    store = SqliteStore(path)
    for key in ("one", "two", "unknown"):
        store.put(
            key,
            title="Deployment notes",
            summary="blue green deployment procedure",
            category="ops",
            body="blue green deployment instructions",
        )
    store.close()
    before = snapshot(tmp_path)
    result = maintain(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["checks"]["duplicates"]["count"] == 3
    assert report["checks"]["unused"]["untracked_keys"] == ["one", "two", "unknown"]
    assert report["checks"]["unused"]["telemetry_available"] is False
    assert report["checks"]["inbox"]["pending_candidates"] is None
    assert snapshot(tmp_path) == before


def test_usage_inbox_and_orphan_rows_are_reported_without_sync(tmp_path):
    path = tmp_path / "nachos" / "memory.db"
    store = SqliteStore(path)
    for key in ("one", "two", "unknown"):
        store.put(
            key,
            title="Deployment notes",
            summary="blue green deployment procedure",
            category="ops",
            body="blue green deployment instructions",
        )
    store.close()
    index = MemoryIndex(index_path_for(path))
    index.record_seen(["one", "two", "deleted-entry"])
    index.record_recall("two")
    index.upsert("deleted-entry", "old text", "old-model", [1.0])
    index.add_inbox(
        session_id="s",
        title="Candidate",
        summary="deployment fact",
        category="ops",
        body="private candidate body",
    )
    index.close()
    conn = sqlite3.connect(index_path_for(path))
    conn.execute("UPDATE usage SET first_seen = 1")
    conn.commit()
    conn.close()
    pending = path.parent / "inbox" / "pending"
    pending.mkdir(parents=True)
    (pending / "job.json.gz").write_bytes(b"not read by the report")
    before = snapshot(tmp_path)
    result = maintain(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert [r["key"] for r in report["checks"]["unused"]["items"]] == ["one"]
    assert report["checks"]["unused"]["untracked_keys"] == ["unknown"]
    assert report["checks"]["inbox"]["pending_candidates"] == 1
    assert report["checks"]["inbox"]["pending_jobs"] == 1
    assert "private candidate body" not in result.stdout
    assert snapshot(tmp_path) == before


def test_wal_uncheckpointed_audit_reads_wal_and_preserves_bytes_and_mtimes(tmp_path):
    store_path = tmp_path / "nachos" / "memory.db"
    SqliteStore(store_path).close()
    index_path = index_path_for(store_path)
    MemoryIndex(index_path).close()

    store_writer = sqlite3.connect(store_path)
    index_writer = sqlite3.connect(index_path)
    try:
        assert store_writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        store_writer.execute("PRAGMA wal_autocheckpoint=0")
        store_writer.execute(
            "INSERT INTO entries (key, title, summary, category, body) VALUES (?, ?, ?, ?, ?)",
            ("wal-bad", "Bad WAL entry", "nothing", "ops", "Real deployment runbook"),
        )
        store_writer.execute(
            "INSERT INTO entries (key, title, summary, category, body) VALUES (?, ?, ?, ?, ?)",
            (
                "wal-dup-a",
                "Deployment notes",
                "blue green deployment procedure",
                "ops",
                "blue green deployment instructions",
            ),
        )
        store_writer.execute(
            "INSERT INTO entries (key, title, summary, category, body) VALUES (?, ?, ?, ?, ?)",
            (
                "wal-dup-b",
                "Deployment notes",
                "blue green deployment procedure",
                "ops",
                "blue green deployment instructions",
            ),
        )
        store_writer.commit()
        # Leave an uncommitted row in the WAL to verify it is NOT surfaced.
        store_writer.execute(
            "INSERT INTO entries (key, title, summary, category, body) VALUES (?, ?, ?, ?, ?)",
            ("uncommitted", "Uncommitted", "nothing", "ops", "Should not be visible"),
        )

        assert index_writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        index_writer.execute("PRAGMA wal_autocheckpoint=0")
        index_writer.execute(
            "INSERT INTO usage (key, first_seen, recall_count, hit_count) VALUES (?, 1, 0, 0)",
            ("wal-bad",),
        )
        index_writer.execute(
            "INSERT INTO inbox (created_ts, session_id, title, summary, category, body) "
            "VALUES (1, 's1', 'WAL candidate', 'wal fact', 'ops', 'wal body')"
        )
        index_writer.commit()

        # Verify WAL and SHM sidecars exist with un-checkpointed writes.
        for db_path in (store_path, index_path):
            wal = Path(f"{db_path}-wal")
            shm = Path(f"{db_path}-shm")
            assert wal.is_file()
            assert wal.stat().st_size > 0
            assert shm.is_file()
            assert shm.stat().st_size > 0

        time.sleep(0.02)
        before = snapshot(tmp_path)

        result = maintain(tmp_path)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert [item["key"] for item in report["checks"]["summaries"]["items"]] == ["wal-bad"]
        assert report["checks"]["duplicates"]["count"] == 1
        assert [item["key"] for item in report["checks"]["unused"]["items"]] == ["wal-bad"]
        assert report["checks"]["unused"]["untracked_keys"] == ["wal-dup-a", "wal-dup-b"]
        assert report["checks"]["inbox"]["pending_candidates"] == 1
        assert "uncommitted" not in result.stdout

        # Original db, -wal, and -shm bytes AND mtimes must be bit-for-bit identical.
        assert snapshot(tmp_path) == before
    finally:
        store_writer.rollback()
        store_writer.close()
        index_writer.close()


def test_wal_checkpointed_and_wal_without_shm_preserve_source_directory(tmp_path):
    # Case 1: clean checkpointed WAL DB (WAL header bytes, no -wal or -shm on disk).
    clean_home = tmp_path / "clean"
    clean_db = clean_home / "nachos" / "memory.db"
    store = SqliteStore(clean_db)
    store.put("k1", title="T", summary="nothing", category="ops", body="real body text")
    store.close()
    conn = sqlite3.connect(clean_db)
    assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    conn.close()
    assert not Path(f"{clean_db}-wal").exists()
    assert not Path(f"{clean_db}-shm").exists()

    before_clean = snapshot(clean_home)
    res_clean = maintain(clean_home, "summaries")
    assert res_clean.returncode == 0, res_clean.stderr
    assert json.loads(res_clean.stdout)["checks"]["summaries"]["count"] == 1
    assert snapshot(clean_home) == before_clean

    # Case 2: un-checkpointed -wal file without a -shm file (e.g. copied or post-crash).
    noshm_home = tmp_path / "noshm"
    noshm_db = noshm_home / "nachos" / "memory.db"
    SqliteStore(noshm_db).close()
    writer = sqlite3.connect(noshm_db)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute(
        "INSERT INTO entries (key, title, summary, category, body) VALUES (?, ?, ?, ?, ?)",
        ("k2", "T2", "nothing", "ops", "wal body without shm"),
    )
    writer.commit()
    db_bytes = noshm_db.read_bytes()
    wal_bytes = Path(f"{noshm_db}-wal").read_bytes()
    writer.close()
    noshm_db.write_bytes(db_bytes)
    Path(f"{noshm_db}-wal").write_bytes(wal_bytes)
    assert not Path(f"{noshm_db}-shm").exists()

    before_noshm = snapshot(noshm_home)
    res_noshm = maintain(noshm_home, "summaries")
    assert res_noshm.returncode == 0, res_noshm.stderr
    assert [i["key"] for i in json.loads(res_noshm.stdout)["checks"]["summaries"]["items"]] == [
        "k2"
    ]
    assert snapshot(noshm_home) == before_noshm


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory write permissions")
def test_wal_in_readonly_directory_audits_via_temp_snapshot(tmp_path):
    db_path = tmp_path / "nachos" / "memory.db"
    store = SqliteStore(db_path)
    store.put("k1", title="T", summary="nothing", category="ops", body="real body text")
    store.close()
    conn = sqlite3.connect(db_path)
    assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
    conn.close()

    nachos_dir = db_path.parent
    os.chmod(nachos_dir, 0o500)
    try:
        result = maintain(tmp_path, "summaries")
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["checks"]["summaries"]["count"] == 1
    finally:
        os.chmod(nachos_dir, 0o700)


def test_readonly_wal_resolves_symlink_and_cleans_up_tempdir(tmp_path):
    actual = tmp_path / "actual.db"
    SqliteStore(actual).close()
    writer = sqlite3.connect(actual)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO entries (key, title, summary, category, body) VALUES (?, ?, ?, ?, ?)",
            ("sym-k", "Sym", "sym summary", "ops", "sym body"),
        )
        writer.commit()
        alias = tmp_path / "alias.db"
        alias.symlink_to(actual)
        store = SqliteStore(alias, read_only=True)
        temp_path = Path(store._temp_dir.name) if store._temp_dir else None
        assert temp_path is not None
        assert temp_path.is_dir()
        assert [row[0] for row in store.list()] == ["sym-k"]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.put("bad", title="t", summary="s", category="c", body="b")
        store.close()
        assert not temp_path.exists()
    finally:
        writer.close()


def test_readonly_connections_reject_writes(tmp_path):
    path = tmp_path / "memory.db"
    SqliteStore(path).close()
    MemoryIndex(index_path_for(path)).close()
    store = SqliteStore(path, read_only=True)
    index = MemoryIndex(index_path_for(path), read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.put("bad", title="t", summary="s", category="c", body="b")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            index.record_seen(["bad"])
    finally:
        store.close()
        index.close()


@pytest.mark.parametrize(
    "target", ["missing", "relative", "symlink", "corrupt", "dangling-wal", "escaped-wal"]
)
def test_unsafe_targets_return_json_error(tmp_path, target):
    home = tmp_path / "profile"
    if target == "relative":
        home = Path("relative-profile")
    elif target == "symlink":
        other = tmp_path / "other"
        other.mkdir()
        home.mkdir()
        (home / "nachos").symlink_to(other, target_is_directory=True)
    elif target == "corrupt":
        (home / "nachos").mkdir(parents=True)
        (home / "nachos" / "memory.db").write_bytes(b"bad database")
    elif target == "dangling-wal":
        db = home / "nachos" / "memory.db"
        SqliteStore(db).close()
        Path(f"{db}-wal").symlink_to(home / "nachos" / "nonexistent-wal")
    elif target == "escaped-wal":
        db = home / "nachos" / "memory.db"
        SqliteStore(db).close()
        outside = tmp_path / "outside-wal"
        outside.write_bytes(b"outside")
        Path(f"{db}-wal").symlink_to(outside)
    result = maintain(home)
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["error"]
    assert report["report_only"] is True
    assert "checks" not in report


@pytest.mark.parametrize("days", ["-1", "nan", "inf"])
def test_invalid_days(tmp_path, days):
    result = maintain(tmp_path, "unused", "--days", days)
    assert result.returncode == 2
    assert "--days must be finite and nonnegative" in json.loads(result.stdout)["error"]


def test_flatfile_report(tmp_path):
    path = tmp_path / "nachos" / "memory.md"
    path.parent.mkdir()
    path.write_text("# ops\n\n## Broken summary\nnothing\nActual deployment instructions\n")
    before = snapshot(tmp_path)
    result = maintain(tmp_path, "all", "--store", "flatfile")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["checks"]["summaries"]["count"] == 1
    assert snapshot(tmp_path) == before


def test_human_summary_report_never_constructs_provider_or_initializes_source(
    tmp_path, capsys, monkeypatch
):
    path = tmp_path / "nachos" / "memory.db"
    store = SqliteStore(path)
    store.put(
        "bad",
        title="Bad summary",
        summary="nothing",
        category="ops",
        body="Real deployment instructions",
    )
    store.close()
    before = snapshot(tmp_path)

    def forbidden(*_args, **_kwargs):
        pytest.fail("report attempted source schema initialization or runtime import")

    class RejectRuntimeImports:
        def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002 -- import hook signature
            if fullname.startswith(("hermes_constants", "agent.", "sentence_transformers")):
                forbidden()

    guard = RejectRuntimeImports()
    monkeypatch.setattr(SqliteStore, "_init_schema", forbidden)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "wrong-profile"))
    sys.meta_path.insert(0, guard)
    try:
        assert maintenance.main(["--profile-home", str(tmp_path), "summaries"]) == 0
    finally:
        sys.meta_path.remove(guard)
    output = capsys.readouterr().out
    assert "summaries: 1" in output
    assert "bad" in output
    assert "index unavailable" in output
    assert snapshot(tmp_path) == before


def test_console_script_entry_point_runs_outside_repo(tmp_path):
    entry_point = next(
        ep
        for ep in importlib.metadata.entry_points(group="console_scripts")
        if ep.name == "nachos-maintain"
    )
    assert entry_point.load() is maintenance.main

    script = Path(sys.executable).parent / "nachos-maintain"
    assert script.is_file()
    home = tmp_path / "profile"
    SqliteStore(home / "nachos" / "memory.db").close()
    output = subprocess.run(
        [str(script), "--profile-home", str(home), "--json", "all"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.stdout)["checks"]["summaries"]["count"] == 0
