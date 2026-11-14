"""Sidecar index: entry vectors + recall usage, one SQLite file per store.

Lives next to the store (``memory.db`` -> ``memory.index.db``, and the same
for ``memory.md``) so it works identically for both store drivers and never
touches the hand-editable flat file. Everything here is derived or
telemetry: deleting the file loses usage history and forces a re-embed, but
never loses a memory.

Vectors are L2-normalized float32 blobs, so cosine similarity is a dot
product. Search is brute force in pure Python, which is fine for the few
thousand entries a curated memory realistically holds.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sqlite3
import time
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def index_path_for(store_path: Path) -> Path:
    store_path = Path(store_path)
    return store_path.with_name(f"{store_path.stem}.index.db")


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


def _normalize(vec: Iterable[float]) -> list[float]:
    values = [float(x) for x in vec]
    norm = math.sqrt(sum(x * x for x in values))
    return [x / norm for x in values] if norm else values


def _pack(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return arr.tolist()


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class MemoryIndex:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                key          TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                model        TEXT NOT NULL,
                vec          BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS usage (
                key          TEXT PRIMARY KEY,
                first_seen   REAL NOT NULL,
                recall_count INTEGER NOT NULL DEFAULT 0,
                hit_count    INTEGER NOT NULL DEFAULT 0,
                last_used    REAL
            );
            CREATE TABLE IF NOT EXISTS changes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            REAL NOT NULL,
                key           TEXT NOT NULL,
                action        TEXT NOT NULL,
                session_id    TEXT,
                prev_title    TEXT,
                prev_summary  TEXT,
                prev_category TEXT,
                prev_body     TEXT
            );
            CREATE INDEX IF NOT EXISTS changes_key ON changes (key);
            CREATE TABLE IF NOT EXISTS inbox (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts      REAL NOT NULL,
                session_id      TEXT,
                title           TEXT NOT NULL,
                summary         TEXT NOT NULL,
                category        TEXT NOT NULL,
                body            TEXT NOT NULL,
                reason          TEXT,
                duplicates_json TEXT NOT NULL DEFAULT '[]',
                status          TEXT NOT NULL DEFAULT 'pending'
            );
            """
        )
        self._conn.commit()
        self._cache: tuple[str, dict[str, list[float]]] | None = None

    # -- vectors -----------------------------------------------------------

    def stale_keys(self, docs: dict[str, str], model: str) -> list[str]:
        """Keys whose text changed, were never embedded, or used another model."""
        known = {
            key: (h, m)
            for key, h, m in self._conn.execute("SELECT key, content_hash, model FROM vectors")
        }
        return [key for key, text in docs.items() if known.get(key) != (content_hash(text), model)]

    def upsert(self, key: str, text: str, model: str, vec: list[float]) -> None:
        self._conn.execute(
            "INSERT INTO vectors (key, content_hash, model, vec) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET content_hash=excluded.content_hash, "
            "model=excluded.model, vec=excluded.vec",
            (key, content_hash(text), model, _pack(_normalize(vec))),
        )
        self._conn.commit()
        self._cache = None

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
        self._conn.execute("DELETE FROM usage WHERE key = ?", (key,))
        self._conn.commit()
        self._cache = None

    def delete_vector(self, key: str) -> None:
        """Drop only the entry's vector so lazy sync re-embeds it."""
        self._conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
        self._conn.commit()
        self._cache = None

    def retain(self, keys: Iterable[str]) -> int:
        """Drop vector and usage rows for entries no longer in the store.

        The ``changes`` log is left alone so removed entries stay revertible.
        """
        live = set(keys)
        indexed = {k for (k,) in self._conn.execute("SELECT key FROM vectors")}
        tracked = {k for (k,) in self._conn.execute("SELECT key FROM usage")}
        gone = (indexed | tracked) - live
        for key in gone:
            self._conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            self._conn.execute("DELETE FROM usage WHERE key = ?", (key,))
        self._conn.commit()
        if gone & indexed:
            self._cache = None
        return len(gone)

    def vectors(self, model: str) -> dict[str, list[float]]:
        if self._cache is None or self._cache[0] != model:
            rows = self._conn.execute("SELECT key, vec FROM vectors WHERE model = ?", (model,))
            self._cache = (model, {key: _unpack(blob) for key, blob in rows})
        return self._cache[1]

    def search(
        self, query_vec: list[float], model: str, top_k: int = 10, exclude: str | None = None
    ) -> list[tuple[str, float]]:
        q = _normalize(query_vec)
        scored = [(key, dot(q, vec)) for key, vec in self.vectors(model).items() if key != exclude]
        scored.sort(key=lambda item: -item[1])
        return scored[:top_k]

    def count(self, model: str | None = None) -> int:
        if model is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
        return int(
            self._conn.execute("SELECT COUNT(*) FROM vectors WHERE model = ?", (model,)).fetchone()[
                0
            ]
        )

    # -- usage -------------------------------------------------------------

    def record_seen(self, keys: Iterable[str]) -> None:
        now = time.time()
        self._conn.executemany(
            "INSERT OR IGNORE INTO usage (key, first_seen) VALUES (?, ?)",
            [(k, now) for k in keys],
        )
        self._conn.commit()

    def _bump(self, keys: Iterable[str], column: str) -> None:
        now = time.time()
        for key in keys:
            self._conn.execute(
                "INSERT OR IGNORE INTO usage (key, first_seen) VALUES (?, ?)", (key, now)
            )
            self._conn.execute(
                f"UPDATE usage SET {column} = {column} + 1, last_used = ? WHERE key = ?",
                (now, key),
            )
        self._conn.commit()

    def record_recall(self, key: str) -> None:
        """The agent fetched this entry's body by key."""
        self._bump([key], "recall_count")

    def record_hits(self, keys: Iterable[str]) -> None:
        """This entry was returned by a query."""
        self._bump(keys, "hit_count")

    def usage(self) -> dict[str, dict[str, float]]:
        rows = self._conn.execute(
            "SELECT key, first_seen, recall_count, hit_count, last_used FROM usage"
        )
        return {
            key: {
                "first_seen": first,
                "recall_count": recalls,
                "hit_count": hits,
                "last_used": last,
            }
            for key, first, recalls, hits, last in rows
        }

    # -- change log --------------------------------------------------------

    def record_change(
        self,
        key: str,
        action: str,
        session_id: str = "",
        prev: tuple[str, str, str, str] | None = None,
    ) -> int:
        """Log a change. ``prev`` is (title, summary, category, body) as it was
        BEFORE the change, or None when the entry did not exist."""
        title, summary, category, body = prev if prev is not None else (None,) * 4
        cur = self._conn.execute(
            "INSERT INTO changes (ts, key, action, session_id, prev_title, "
            "prev_summary, prev_category, prev_body) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), key, action, session_id or "", title, summary, category, body),
        )
        self._conn.commit()
        change_id = int(cur.lastrowid or 0)
        if change_id % 100 == 0:
            self.trim_changes()
        return change_id

    _CHANGE_COLS = (
        "id, ts, key, action, session_id, prev_title, prev_summary, prev_category, prev_body"
    )

    @staticmethod
    def _change_row(row: Any) -> dict[str, Any]:
        names = (
            "id",
            "ts",
            "key",
            "action",
            "session_id",
            "prev_title",
            "prev_summary",
            "prev_category",
            "prev_body",
        )
        return dict(zip(names, row))

    def changes(self, key: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Recent changes, newest first, optionally for one key."""
        sql = f"SELECT {self._CHANGE_COLS} FROM changes"
        params: list[str | int] = []
        if key is not None:
            sql += " WHERE key = ?"
            params.append(key)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [self._change_row(r) for r in self._conn.execute(sql, params)]

    def get_change(self, change_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            f"SELECT {self._CHANGE_COLS} FROM changes WHERE id = ?",
            (change_id,),
        ).fetchone()
        return self._change_row(row) if row else None

    def count_changes(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM changes").fetchone()[0])

    def trim_changes(self, keep: int = 1000) -> int:
        """Keep only the newest ``keep`` change rows; returns rows dropped."""
        cur = self._conn.execute(
            "DELETE FROM changes WHERE id <= "
            "(SELECT id FROM changes ORDER BY id DESC LIMIT 1 OFFSET ?)",
            (max(0, int(keep)),),
        )
        self._conn.commit()
        return cur.rowcount

    # -- review inbox (suggested memories awaiting approval) -----------------

    _INBOX_COLS = (
        "id",
        "created_ts",
        "session_id",
        "title",
        "summary",
        "category",
        "body",
        "reason",
        "duplicates_json",
        "status",
    )

    def _inbox_row(self, row: Any) -> dict[str, Any]:
        d = dict(zip(self._INBOX_COLS, row))
        try:
            d["duplicates"] = json.loads(d.pop("duplicates_json") or "[]")
        except ValueError:
            d["duplicates"] = []
        return d

    def add_inbox(
        self,
        *,
        session_id: str,
        title: str,
        summary: str,
        category: str,
        body: str,
        reason: str = "",
        duplicates: list | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO inbox (created_ts, session_id, title, summary, category, body, "
            "reason, duplicates_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                session_id or "",
                title,
                summary,
                category,
                body,
                reason or "",
                json.dumps(duplicates or []),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def list_inbox(self, status: str | None = "pending") -> list[dict[str, Any]]:
        sql = f"SELECT {', '.join(self._INBOX_COLS)} FROM inbox"
        params: list[str] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        return [self._inbox_row(r) for r in self._conn.execute(sql + " ORDER BY id", params)]

    def get_inbox(self, inbox_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            f"SELECT {', '.join(self._INBOX_COLS)} FROM inbox WHERE id = ?",
            (inbox_id,),
        ).fetchone()
        return self._inbox_row(row) if row else None

    def count_inbox(self, status: str = "pending") -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM inbox WHERE status = ?", (status,)).fetchone()[
                0
            ]
        )

    def set_inbox_status(self, inbox_id: int, status: str) -> None:
        self._conn.execute("UPDATE inbox SET status = ? WHERE id = ?", (status, inbox_id))
        self._conn.commit()

    def trim_inbox(self, max_age_days: float = 30, now: float | None = None) -> int:
        """Drop approved/rejected rows older than ``max_age_days``."""
        cutoff = (now if now is not None else time.time()) - max_age_days * 86400
        cur = self._conn.execute(
            "DELETE FROM inbox WHERE status != 'pending' AND created_ts < ?", (cutoff,)
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()
