"""Usage telemetry, change log, review inbox and provenance tables of the sidecar index.

A mixin over ``MemoryIndex``'s connection: these tables hold state keyed by entry (not vectors),
so they share nothing with the vector store beyond the SQLite file.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from typing import Any

from .index_schema import INBOX_COLS


class RecordTables:
    _conn: sqlite3.Connection
    _path: str
    _has_prev_superseded: bool
    _inbox_cols: tuple[str, ...]

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
        prev_superseded_by: str | None = None,
    ) -> int:
        """Log a change. ``prev`` is (title, summary, category, body) as it was
        BEFORE the change, or None when the entry did not exist."""
        title, summary, category, body = prev if prev is not None else (None,) * 4
        if self._path != ":memory:":
            self._conn.execute("PRAGMA synchronous=FULL")
        try:
            cur = self._conn.execute(
                "INSERT INTO changes (ts, key, action, session_id, prev_title, "
                "prev_summary, prev_category, prev_body, prev_superseded_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    key,
                    action,
                    session_id or "",
                    title,
                    summary,
                    category,
                    body,
                    prev_superseded_by,
                ),
            )
            self._conn.commit()
        finally:
            if self._path != ":memory:":
                self._conn.execute("PRAGMA synchronous=NORMAL")
        change_id = int(cur.lastrowid or 0)
        if change_id % 100 == 0:
            self.trim_changes()
        return change_id

    _CHANGE_COLS = (
        "id, ts, key, action, session_id, prev_title, prev_summary, "
        "prev_category, prev_body, prev_superseded_by"
    )
    _LEGACY_CHANGE_COLS = (
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
            "prev_superseded_by",
        )
        out = dict(zip(names, row))
        out.setdefault("prev_superseded_by", None)
        return out

    def _change_cols(self) -> str:
        return self._CHANGE_COLS if self._has_prev_superseded else self._LEGACY_CHANGE_COLS

    def changes(self, key: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Recent changes, newest first, optionally for one key."""
        sql = f"SELECT {self._change_cols()} FROM changes"
        params: list[str | int] = []
        if key is not None:
            sql += " WHERE key = ?"
            params.append(key)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [self._change_row(r) for r in self._conn.execute(sql, params)]

    def get_change(self, change_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            f"SELECT {self._change_cols()} FROM changes WHERE id = ?",
            (change_id,),
        ).fetchone()
        return self._change_row(row) if row else None

    def change_timestamps(self) -> dict[str, tuple[float, float]]:
        """Return ``{key: (earliest_ts, latest_ts)}`` from the change log."""
        rows = self._conn.execute("SELECT key, MIN(ts), MAX(ts) FROM changes GROUP BY key")
        return {str(k): (float(mn), float(mx)) for k, mn, mx in rows if mn is not None}

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

    def _inbox_row(self, row: Any) -> dict[str, Any]:
        d: dict[str, Any] = dict.fromkeys(INBOX_COLS)
        d.update(zip(self._inbox_cols, row))
        d["grounded"] = None if d["grounded"] is None else bool(d["grounded"])
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
        quote: str = "",
        grounded: bool | None = None,
        checkpoint_id: str | None = None,
        message_index: int | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO inbox (created_ts, session_id, title, summary, category, body, "
            "reason, duplicates_json, quote, grounded, checkpoint_id, message_index) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                session_id or "",
                title,
                summary,
                category,
                body,
                reason or "",
                json.dumps(duplicates or []),
                quote or "",
                None if grounded is None else int(grounded),
                checkpoint_id,
                message_index,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def list_inbox(self, status: str | None = "pending") -> list[dict[str, Any]]:
        sql = f"SELECT {', '.join(self._inbox_cols)} FROM inbox"
        params: list[str] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        return [self._inbox_row(r) for r in self._conn.execute(sql + " ORDER BY id", params)]

    def get_inbox(self, inbox_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            f"SELECT {', '.join(self._inbox_cols)} FROM inbox WHERE id = ?",
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

    # -- provenance (where a saved entry came from) --------------------------

    def set_provenance(
        self,
        key: str,
        *,
        session_id: str = "",
        captured_at: float | None = None,
        source: str = "manual",
        checkpoint_id: str | None = None,
        message_index: int | None = None,
        quote: str = "",
        grounded: bool | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO provenance (key, session_id, captured_at, source, "
            "checkpoint_id, message_index, quote, grounded) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                session_id or "",
                captured_at if captured_at is not None else time.time(),
                source,
                checkpoint_id,
                message_index,
                quote or None,
                None if grounded is None else int(grounded),
            ),
        )
        self._conn.commit()

    def get_provenance_many(self, keys: Iterable[str]) -> dict[str, dict[str, Any]]:
        """``{key: provenance}`` for the keys that have any (empty on files without the table)."""
        wanted = list(dict.fromkeys(keys))
        if not wanted or "provenance" not in self._provenance_tables():
            return {}
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(wanted), 500):
            batch = wanted[i : i + 500]
            marks = ", ".join("?" * len(batch))
            rows = self._conn.execute(
                "SELECT key, session_id, captured_at, source, checkpoint_id, message_index, "
                f"quote, grounded FROM provenance WHERE key IN ({marks})",
                batch,
            )
            for key, sid, at, source, cp, idx, quote, grounded in rows:
                out[str(key)] = {
                    "session_id": sid or "",
                    "captured_at": float(at),
                    "source": source,
                    "checkpoint_id": cp,
                    "message_index": idx,
                    "quote": quote or "",
                    "grounded": None if grounded is None else bool(grounded),
                }
        return out

    def get_provenance(self, key: str) -> dict[str, Any] | None:
        return self.get_provenance_many([key]).get(key)

    def delete_provenance(self, key: str) -> None:
        self._conn.execute("DELETE FROM provenance WHERE key = ?", (key,))
        self._conn.commit()

    def _provenance_tables(self) -> set[str]:
        return {
            str(r[0])
            for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'provenance'"
            )
        }
