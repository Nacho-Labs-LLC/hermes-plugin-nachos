"""SqliteStore — default local store driver (stdlib sqlite3 only)."""

from __future__ import annotations

import builtins
import os
import sqlite3
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

from ..sqlite_util import close_sqlite, open_read_only_sqlite
from .base import Entry, EntryMeta, MemoryStore


class SqliteStore(MemoryStore):
    """Single-table sqlite store. Zero external deps.

    Schema: entries(key PK, title, summary, category, body, created_at,
    updated_at, superseded_by, approx_timestamps) + index on category.
    """

    def __init__(self, path: str | os.PathLike[str], *, read_only: bool = False) -> None:
        self._path = str(path)
        self._read_only = read_only
        self._generation: int = 0
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if read_only:
            self._conn, self._temp_dir = open_read_only_sqlite(path)
            self._conn.row_factory = sqlite3.Row
            self._meta_cols = self._entry_columns()
            return
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so callers from different host threads
        # thread can share the connection; access stays serialized by the
        # GIL for our tiny workloads.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._meta_cols = self._entry_columns()

    @property
    def read_only(self) -> bool:
        return self._read_only

    def _entry_columns(self) -> set[str]:
        return {str(r[1]) for r in self._conn.execute("PRAGMA table_info(entries)").fetchall()}

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                key               TEXT PRIMARY KEY,
                title             TEXT NOT NULL,
                summary           TEXT NOT NULL,
                category          TEXT NOT NULL,
                body              TEXT NOT NULL,
                created_at        REAL,
                updated_at        REAL,
                superseded_by     TEXT,
                approx_timestamps INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_entries_category
                ON entries(category);
            """
        )
        cols = self._entry_columns()
        migrations = (
            ("created_at", "ALTER TABLE entries ADD COLUMN created_at REAL"),
            ("updated_at", "ALTER TABLE entries ADD COLUMN updated_at REAL"),
            ("superseded_by", "ALTER TABLE entries ADD COLUMN superseded_by TEXT"),
            (
                "approx_timestamps",
                "ALTER TABLE entries ADD COLUMN approx_timestamps INTEGER NOT NULL DEFAULT 0",
            ),
        )
        for col, ddl in migrations:
            if col not in cols:
                self._conn.execute(ddl)
        self._conn.commit()

    def list(self) -> builtins.list[Entry]:
        cur = self._conn.execute(
            "SELECT key, title, summary, category FROM entries ORDER BY category ASC, title ASC"
        )
        return [(r["key"], r["title"], r["summary"], r["category"]) for r in cur.fetchall()]

    def get(self, key: str) -> str | None:
        cur = self._conn.execute("SELECT body FROM entries WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["body"] if row else None

    def bodies(self) -> dict[str, str]:
        return {r["key"]: r["body"] for r in self._conn.execute("SELECT key, body FROM entries")}

    def search(self, query: str) -> builtins.list[str]:
        q = (query or "").strip()
        if not q:
            return []
        escaped_q = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{escaped_q}%"
        cur = self._conn.execute(
            "SELECT key FROM entries "
            "WHERE title LIKE ? ESCAPE '\\' "
            "OR summary LIKE ? ESCAPE '\\' "
            "OR body LIKE ? ESCAPE '\\' "
            "ORDER BY category ASC, title ASC",
            (like, like, like),
        )
        return [r["key"] for r in cur.fetchall()]

    def fingerprint(self) -> tuple[int, int, int]:
        """Cheap change token combining in-process writes and SQLite's data_version."""
        row = self._conn.execute("PRAGMA data_version").fetchone()
        data_version = int(row[0]) if row else 0
        return (self._generation, self._conn.total_changes, data_version)

    @staticmethod
    def _row_meta(row: sqlite3.Row) -> EntryMeta:
        return EntryMeta(
            created_at=float(row["created_at"]) if row["created_at"] is not None else None,
            updated_at=float(row["updated_at"]) if row["updated_at"] is not None else None,
            superseded_by=str(row["superseded_by"]) if row["superseded_by"] is not None else None,
            approx_timestamps=bool(row["approx_timestamps"]),
        )

    def get_meta(self, key: str) -> EntryMeta | None:
        if "created_at" not in self._meta_cols:
            return EntryMeta() if self.get(key) is not None else None
        row = self._conn.execute(
            "SELECT created_at, updated_at, superseded_by, approx_timestamps "
            "FROM entries WHERE key = ?",
            (key,),
        ).fetchone()
        return self._row_meta(row) if row is not None else None

    def all_meta(self) -> dict[str, EntryMeta]:
        if "created_at" not in self._meta_cols:
            return {key: EntryMeta() for key, *_rest in self.list()}
        rows = self._conn.execute(
            "SELECT key, created_at, updated_at, superseded_by, approx_timestamps FROM entries"
        )
        return {str(r["key"]): self._row_meta(r) for r in rows}

    def set_superseded_by(
        self, key: str, superseded_by: str | None, *, updated_at: float | None = None
    ) -> bool:
        ts = updated_at if updated_at is not None else time.time()
        cur = self._conn.execute(
            "UPDATE entries SET superseded_by = ?, updated_at = ? WHERE key = ?",
            (superseded_by, ts, key),
        )
        self._conn.commit()
        if cur.rowcount > 0:
            self._generation += 1
            return True
        return False

    def backfill_timestamps(self, timestamps: Mapping[str, tuple[float, float, bool]]) -> int:
        updated = 0
        for key, (created_at, updated_at, approx) in timestamps.items():
            cur = self._conn.execute(
                "UPDATE entries SET "
                "  created_at = COALESCE(created_at, ?), "
                "  updated_at = COALESCE(updated_at, ?), "
                "  approx_timestamps = CASE "
                "    WHEN created_at IS NULL OR updated_at IS NULL THEN ? "
                "    ELSE approx_timestamps END "
                "WHERE key = ? AND (created_at IS NULL OR updated_at IS NULL)",
                (created_at, updated_at, int(bool(approx)), key),
            )
            updated += cur.rowcount
        if updated:
            self._conn.commit()
            self._generation += 1
        return updated

    def put(
        self,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
        body: str,
    ) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO entries "
            "  (key, title, summary, category, body, created_at, updated_at, "
            "   superseded_by, approx_timestamps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0) "
            "ON CONFLICT(key) DO UPDATE SET "
            "  title=excluded.title, summary=excluded.summary, "
            "  category=excluded.category, body=excluded.body, "
            "  created_at=COALESCE(entries.created_at, excluded.created_at), "
            "  updated_at=excluded.updated_at, "
            "  approx_timestamps=CASE WHEN entries.created_at IS NULL THEN 0 "
            "    ELSE entries.approx_timestamps END",
            (key, title, summary, category, body, now, now),
        )
        self._conn.commit()
        self._generation += 1

    def remove(self, key: str) -> None:
        self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
        self._conn.commit()
        self._generation += 1

    def close(self) -> None:
        close_sqlite(self._conn, self._temp_dir)
        self._temp_dir = None
