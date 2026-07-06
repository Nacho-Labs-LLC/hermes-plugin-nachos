"""SqliteStore — default local store driver (stdlib sqlite3 only)."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import List, Optional

from .base import Entry, MemoryStore


class SqliteStore(MemoryStore):
    """Single-table sqlite store. Zero external deps.

    Schema: entries(key PK, title, summary, category, body) + index on
    category. search() is a LIKE substring scan over title+summary+body;
    semantic ranking is the Scorer's job, not the store's.
    """

    def __init__(self, path):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the provider's background prefetch
        # thread can share the connection; access stays serialized by the
        # GIL for our tiny workloads.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                key      TEXT PRIMARY KEY,
                title    TEXT NOT NULL,
                summary  TEXT NOT NULL,
                category TEXT NOT NULL,
                body     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_category
                ON entries(category);
            """
        )
        self._conn.commit()

    def list(self) -> List[Entry]:
        cur = self._conn.execute(
            "SELECT key, title, summary, category FROM entries "
            "ORDER BY category ASC, title ASC"
        )
        return [
            (r["key"], r["title"], r["summary"], r["category"])
            for r in cur.fetchall()
        ]

    def get(self, key: str) -> Optional[str]:
        cur = self._conn.execute(
            "SELECT body FROM entries WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return row["body"] if row else None

    def search(self, query: str) -> List[str]:
        q = (query or "").strip()
        if not q:
            return []
        like = f"%{q}%"
        cur = self._conn.execute(
            "SELECT key FROM entries "
            "WHERE title LIKE ? OR summary LIKE ? OR body LIKE ? "
            "ORDER BY category ASC, title ASC",
            (like, like, like),
        )
        return [r["key"] for r in cur.fetchall()]

    def put(
        self,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
        body: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO entries (key, title, summary, category, body) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "  title=excluded.title, summary=excluded.summary, "
            "  category=excluded.category, body=excluded.body",
            (key, title, summary, category, body),
        )
        self._conn.commit()

    def remove(self, key: str) -> None:
        self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
        self._conn.commit()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()
