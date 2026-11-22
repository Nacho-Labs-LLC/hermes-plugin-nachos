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
import random
import shutil
import sqlite3
import statistics
import tempfile
import time
from array import array
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .calibration import CALIBRATION_MIN_HEADS, blend_profiles, prior_profile

_SQLITE_MAGIC = b"SQLite format 3\x00"
_SNAPSHOT_ATTEMPTS = 4  # initial attempt + 3 retries


_CALIBRATION_MIN_HEADS = CALIBRATION_MIN_HEADS
_CALIBRATION_DRIFT = 0.2  # re-sample once the head count moves this far from the last sample
_CALIBRATION_PAIRS = 4000
_PROFILE_PERCENTILES = (50, 75, 80, 85, 90, 95, 99, 99.5)


def _pair_profile(vectors: list[list[float]]) -> dict[str, float]:
    """Percentiles, mean and std of cosine similarity over a fixed-seed sample of vector pairs."""
    rnd = random.Random(0)  # noqa: S311 - deterministic sampling, not security
    n = len(vectors)
    seen: set[tuple[int, int]] = set()
    sims: list[float] = []
    target = min(_CALIBRATION_PAIRS, n * (n - 1) // 2)
    while len(sims) < target:
        a, b = rnd.sample(range(n), 2)
        pair = (a, b) if a < b else (b, a)
        if pair in seen:
            continue
        seen.add(pair)
        sims.append(dot(vectors[a], vectors[b]))
    sims.sort()
    last = len(sims) - 1
    out = {f"p{p:g}": sims[min(last, round(p / 100 * last))] for p in _PROFILE_PERCENTILES}
    out["mean"] = statistics.fmean(sims)
    out["std"] = statistics.pstdev(sims)
    return out


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


def _needs_wal_snapshot(path: Path) -> bool:
    """True when opening ``path`` directly with ``mode=ro`` could create/mutate WAL sidecars."""
    if any(os.path.lexists(f"{path}{suffix}") for suffix in ("-wal", "-shm")):
        return True
    with path.open("rb") as fh:
        header = fh.read(20)
    return (
        len(header) == 20 and header[:16] == _SQLITE_MAGIC and (header[18] == 2 or header[19] == 2)
    )


def _stat_sig(path: Path, *, required: bool = False) -> tuple[int, int] | None:
    if not required and not os.path.lexists(path):
        return None
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def _copy_wal_snapshot(resolved: Path) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Copy a WAL database and its ``-wal`` sidecar into a private temp dir consistently.

    Records ``(size, mtime_ns)`` of both files before and after copying and retries if either
    changes mid-copy (for example due to a concurrent write or checkpoint). Fails closed if the
    sources do not stabilise.
    """
    wal_src = Path(f"{resolved}-wal")
    for _ in range(_SNAPSHOT_ATTEMPTS):
        before = (_stat_sig(resolved, required=True), _stat_sig(wal_src))
        temp_dir = tempfile.TemporaryDirectory(prefix="nachos-ro-")
        try:
            target = Path(temp_dir.name) / resolved.name
            shutil.copyfile(resolved, target)
            if before[1] is not None:
                shutil.copyfile(wal_src, Path(f"{target}-wal"))
            after = (_stat_sig(resolved, required=True), _stat_sig(wal_src))
            if after == before:
                return target, temp_dir
        except FileNotFoundError:
            if not os.path.lexists(resolved) or os.path.lexists(wal_src):
                temp_dir.cleanup()
                raise
        except Exception:
            temp_dir.cleanup()
            raise
        temp_dir.cleanup()
    raise OSError(
        f"SQLite WAL snapshot did not stabilise after {_SNAPSHOT_ATTEMPTS} attempts: {resolved}"
    )


def _verify_sqlite_copy(conn: sqlite3.Connection, source: Path) -> None:
    rows = [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]
    if rows != ["ok"]:
        detail = "; ".join(rows) or "empty quick_check result"
        raise sqlite3.DatabaseError(f"SQLite snapshot failed quick_check for {source}: {detail}")


def open_read_only_sqlite(
    path: str | os.PathLike[str],
) -> tuple[sqlite3.Connection, tempfile.TemporaryDirectory[str] | None]:
    """Open a SQLite file read-only without creating, checkpointing, or modifying source files.

    Rollback-mode databases are opened directly with ``mode=ro`` and ``PRAGMA query_only = ON``.
    WAL-mode databases (header version 2 or existing ``-wal``/``-shm`` sidecars) are copied along
    with any ``-wal`` file into a private temporary directory with stability retries and verified
    via ``PRAGMA quick_check`` so SQLite can rebuild its transient ``-shm`` index there without
    touching the original files or directory.
    """
    resolved = Path(path).resolve()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    target = resolved
    try:
        if _needs_wal_snapshot(resolved):
            target, temp_dir = _copy_wal_snapshot(resolved)
        conn = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, check_same_thread=False)
        try:
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA schema_version").fetchone()
            if temp_dir is not None:
                _verify_sqlite_copy(conn, resolved)
        except Exception:
            conn.close()
            raise
    except Exception:
        if temp_dir is not None:
            temp_dir.cleanup()
        raise
    return conn, temp_dir


def close_sqlite(
    conn: sqlite3.Connection, temp_dir: tempfile.TemporaryDirectory[str] | None
) -> None:
    """Close a SQLite connection and clean up any read-only snapshot temp directory."""
    with contextlib.suppress(Exception):
        conn.close()
    if temp_dir is not None:
        with contextlib.suppress(Exception):
            temp_dir.cleanup()


class MemoryIndex:
    def __init__(self, path: str | os.PathLike[str], *, read_only: bool = False) -> None:
        self._path = str(path)
        self._read_only = read_only
        self._cache: tuple[str, dict[str, list[float]], dict[str, list[list[float]]]] | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if read_only:
            self._conn, self._temp_dir = open_read_only_sqlite(path)
            self._chunked = "chunk_no" in self._table_columns("vectors")
            self._has_prev_superseded = "prev_superseded_by" in self._table_columns("changes")
            self._inbox_cols = self._existing_inbox_cols()
            return
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                key          TEXT NOT NULL,
                chunk_no     INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL,
                model        TEXT NOT NULL,
                vec          BLOB NOT NULL,
                PRIMARY KEY (key, chunk_no, model)
            );
            CREATE TABLE IF NOT EXISTS usage (
                key          TEXT PRIMARY KEY,
                first_seen   REAL NOT NULL,
                recall_count INTEGER NOT NULL DEFAULT 0,
                hit_count    INTEGER NOT NULL DEFAULT 0,
                last_used    REAL
            );
            CREATE TABLE IF NOT EXISTS changes (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                 REAL NOT NULL,
                key                TEXT NOT NULL,
                action             TEXT NOT NULL,
                session_id         TEXT,
                prev_title         TEXT,
                prev_summary       TEXT,
                prev_category      TEXT,
                prev_body          TEXT,
                prev_superseded_by TEXT
            );
            CREATE INDEX IF NOT EXISTS changes_key ON changes (key);
            CREATE TABLE IF NOT EXISTS calibration (
                model      TEXT PRIMARY KEY,
                head_count INTEGER NOT NULL,
                profile    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS links (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                pos INTEGER NOT NULL,
                PRIMARY KEY (src, dst)
            );
            CREATE INDEX IF NOT EXISTS links_dst ON links (dst);
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
            CREATE TABLE IF NOT EXISTS provenance (
                key           TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL DEFAULT '',
                captured_at   REAL NOT NULL,
                source        TEXT NOT NULL DEFAULT 'manual',
                checkpoint_id TEXT,
                message_index INTEGER,
                quote         TEXT,
                grounded      INTEGER
            );
            """
        )
        self._conn.commit()
        self._migrate_vectors_table()
        self._migrate_changes_table()
        self._migrate_inbox_table()
        self._inbox_cols = self._existing_inbox_cols()
        self._chunked = True
        self._has_prev_superseded = True

    @property
    def read_only(self) -> bool:
        return self._read_only

    def _table_columns(self, table: str) -> set[str]:
        return {str(row[1]) for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _migrate_changes_table(self) -> None:
        cols = self._table_columns("changes")
        if cols and "prev_superseded_by" not in cols:
            self._conn.execute("ALTER TABLE changes ADD COLUMN prev_superseded_by TEXT")
            self._conn.commit()

    def _migrate_inbox_table(self) -> None:
        cols = self._table_columns("inbox")
        for name, decl in self._INBOX_EXTRA:
            if name not in cols:
                self._conn.execute(f"ALTER TABLE inbox ADD COLUMN {name} {decl}")
        self._conn.commit()

    def _existing_inbox_cols(self) -> tuple[str, ...]:
        have = self._table_columns("inbox")
        return tuple(c for c in self._INBOX_COLS if c in have)

    def _migrate_vectors_table(self) -> None:
        cols = self._table_columns("vectors")
        if cols and "chunk_no" in cols:
            self._migrate_vectors_primary_key()
            return
        if not cols:
            return
        # Pre-chunking schema had PRIMARY KEY (key) without chunk_no. Vectors are derived
        # data, so recreating the table lets lazy sync rebuild chunked vectors automatically.
        self._conn.executescript(
            """
            DROP TABLE vectors;
            CREATE TABLE vectors (
                key          TEXT NOT NULL,
                chunk_no     INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL,
                model        TEXT NOT NULL,
                vec          BLOB NOT NULL,
                PRIMARY KEY (key, chunk_no, model)
            );
            """
        )
        self._conn.commit()

    def _migrate_vectors_primary_key(self) -> None:
        """Add ``model`` to the primary key so a model switch keeps the old vectors until done.

        Existing rows are copied over unchanged; the table is rebuilt only for databases that
        still have the ``(key, chunk_no)`` key.
        """
        pk_columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(vectors)") if row[5]
        }
        if "model" in pk_columns:
            return
        self._conn.executescript(
            """
            ALTER TABLE vectors RENAME TO vectors_old;
            CREATE TABLE vectors (
                key          TEXT NOT NULL,
                chunk_no     INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL,
                model        TEXT NOT NULL,
                vec          BLOB NOT NULL,
                PRIMARY KEY (key, chunk_no, model)
            );
            INSERT INTO vectors SELECT key, chunk_no, content_hash, model, vec FROM vectors_old;
            DROP TABLE vectors_old;
            """
        )
        self._conn.commit()

    # -- vectors -----------------------------------------------------------

    def stale_keys(self, docs: Mapping[str, str | list[str]], model: str) -> list[str]:
        """Keys whose chunks changed, shrank/grew, or have no vectors from ``model`` yet.

        Vectors of other models are ignored (and kept), so after a model switch every entry is
        stale until it is re-embedded.
        """
        known: dict[str, dict[int, tuple[str, str]]] = {}
        for key, chunk_no, h, m in self._conn.execute(
            "SELECT key, chunk_no, content_hash, model FROM vectors WHERE model = ?", (model,)
        ):
            known.setdefault(key, {})[int(chunk_no)] = (h, m)
        stale: list[str] = []
        for key, value in docs.items():
            chunks = [value] if isinstance(value, str) else value
            expected = {i: (content_hash(text), model) for i, text in enumerate(chunks)}
            if known.get(key) != expected:
                stale.append(key)
        return stale

    def upsert_chunks(
        self, key: str, texts: list[str], model: str, vecs: list[list[float]]
    ) -> None:
        """Store all chunk vectors (``chunk_no = 0..n-1``) for ``key``, pruning extra chunks."""
        if len(texts) != len(vecs):
            raise ValueError(f"expected {len(texts)} vectors for {key!r}, got {len(vecs)}")
        self._conn.execute(
            "DELETE FROM vectors WHERE key = ? AND model = ? AND chunk_no >= ?",
            (key, model, len(texts)),
        )
        for chunk_no, (text, vec) in enumerate(zip(texts, vecs)):
            self._conn.execute(
                "INSERT INTO vectors (key, chunk_no, content_hash, model, vec) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(key, chunk_no, model) DO UPDATE SET "
                "content_hash=excluded.content_hash, vec=excluded.vec",
                (key, chunk_no, content_hash(text), model, _pack(_normalize(vec))),
            )
        self._conn.commit()
        self._cache = None

    def upsert(self, key: str, text: str, model: str, vec: list[float]) -> None:
        self.upsert_chunks(key, [text], model, [vec])

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
        self._conn.execute("DELETE FROM usage WHERE key = ?", (key,))
        self.delete_provenance(key)
        self._conn.commit()
        self._cache = None

    def delete_vector(self, key: str) -> None:
        """Drop all of an entry's vectors so lazy sync re-embeds it."""
        self._conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
        self._conn.commit()
        self._cache = None

    def retain(self, keys: Iterable[str]) -> int:
        """Drop vector and usage rows for entries no longer in the store.

        The ``changes`` log is left alone so removed entries stay revertible.
        """
        live = set(keys)
        indexed = {k for (k,) in self._conn.execute("SELECT DISTINCT key FROM vectors")}
        tracked = {k for (k,) in self._conn.execute("SELECT key FROM usage")}
        sourced = {k for (k,) in self._conn.execute("SELECT key FROM provenance")}
        gone = (indexed | tracked | sourced) - live
        for key in gone:
            self._conn.execute("DELETE FROM vectors WHERE key = ?", (key,))
            self._conn.execute("DELETE FROM usage WHERE key = ?", (key,))
            self._conn.execute("DELETE FROM provenance WHERE key = ?", (key,))
        self._conn.commit()
        if gone & indexed:
            self._cache = None
        return len(gone)

    def _load_vectors(
        self, model: str
    ) -> tuple[dict[str, list[float]], dict[str, list[list[float]]]]:
        if self._cache is not None and self._cache[0] == model:
            return self._cache[1], self._cache[2]
        heads: dict[str, list[float]] = {}
        all_vecs: dict[str, list[list[float]]] = {}
        if self._chunked:
            rows = self._conn.execute(
                "SELECT key, chunk_no, vec FROM vectors WHERE model = ? ORDER BY key, chunk_no",
                (model,),
            )
            for key, chunk_no, blob in rows:
                vec = _unpack(blob)
                if int(chunk_no) == 0:
                    heads[key] = vec
                all_vecs.setdefault(key, []).append(vec)
        else:
            rows = self._conn.execute("SELECT key, vec FROM vectors WHERE model = ?", (model,))
            for key, blob in rows:
                vec = _unpack(blob)
                heads[key] = vec
                all_vecs[key] = [vec]
        self._cache = (model, heads, all_vecs)
        return heads, all_vecs

    def vectors(self, model: str) -> dict[str, list[float]]:
        """Head vectors (``chunk_no = 0``) for ``model`` (used for duplicate detection)."""
        return self._load_vectors(model)[0]

    def search(
        self,
        query_vec: list[float],
        model: str,
        top_k: int = 10,
        exclude: str | None = None,
        *,
        head_only: bool = False,
        chunk_penalty: float = 0.0,
    ) -> list[tuple[str, float]]:
        return [
            (key, score)
            for key, score, _chunk in self.search_chunks(
                query_vec,
                model,
                top_k,
                exclude,
                head_only=head_only,
                chunk_penalty=chunk_penalty,
            )
        ]

    def search_chunks(
        self,
        query_vec: list[float],
        model: str,
        top_k: int = 10,
        exclude: str | None = None,
        *,
        head_only: bool = False,
        chunk_penalty: float = 0.0,
    ) -> list[tuple[str, float, int]]:
        """(key, score, best chunk_no) by descending score.

        An entry scores its head vector, or its best body chunk minus ``chunk_penalty`` when
        that is higher: body chunks are narrower evidence than the head, so a small penalty
        keeps a chunk from outranking a head with an equal score.
        """
        q = _normalize(query_vec)
        heads, all_vecs = self._load_vectors(model)
        scored: list[tuple[str, float, int]] = []
        if head_only:
            scored = [(key, dot(q, vec), 0) for key, vec in heads.items() if key != exclude]
        else:
            for key, vecs in all_vecs.items():
                if key == exclude or not vecs:
                    continue
                best, best_chunk = dot(q, vecs[0]), 0
                for chunk_no in range(1, len(vecs)):
                    sim = dot(q, vecs[chunk_no]) - chunk_penalty
                    if sim > best:
                        best, best_chunk = sim, chunk_no
                scored.append((key, best, best_chunk))
        scored.sort(key=lambda item: -item[1])
        return scored[:top_k]

    def count(self, model: str | None = None) -> int:
        """Number of distinct indexed entries (for ``model`` if given)."""
        if model is None:
            return int(self._conn.execute("SELECT COUNT(DISTINCT key) FROM vectors").fetchone()[0])
        return int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT key) FROM vectors WHERE model = ?", (model,)
            ).fetchone()[0]
        )

    def vector_count(self, model: str | None = None) -> int:
        """Total stored chunk vectors (for ``model`` if given)."""
        if model is None:
            return int(self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
        return int(
            self._conn.execute("SELECT COUNT(*) FROM vectors WHERE model = ?", (model,)).fetchone()[
                0
            ]
        )

    def model_counts(self) -> dict[str, int]:
        """Distinct indexed entries per embedding model (the configured one and any kept old ones)."""
        rows = self._conn.execute("SELECT model, COUNT(DISTINCT key) FROM vectors GROUP BY model")
        return {str(model): int(count) for model, count in rows}

    def purge_other_models(self, keep: str) -> int:
        """Drop vectors and calibration of every model except ``keep``; returns rows removed.

        Called once ``keep`` covers every entry, so the old model's vectors stay available (and
        a switch back is instant) until the re-embed has finished.
        """
        cursor = self._conn.execute("DELETE FROM vectors WHERE model != ?", (keep,))
        removed = max(cursor.rowcount, 0)
        self._conn.execute("DELETE FROM calibration WHERE model != ?", (keep,))
        self._conn.commit()
        self._cache = None
        return removed

    # -- links -------------------------------------------------------------

    def set_links(self, src: str, dsts: list[str]) -> None:
        """Replace ``src``'s outgoing links (``dsts`` in body order)."""
        self._conn.execute("DELETE FROM links WHERE src = ?", (src,))
        self._conn.executemany(
            "INSERT OR IGNORE INTO links (src, dst, pos) VALUES (?, ?, ?)",
            [(src, dst, pos) for pos, dst in enumerate(dsts)],
        )
        self._conn.commit()

    def delete_links(self, src: str) -> None:
        """Drop ``src``'s outgoing links; links *to* it stay and read as broken."""
        self._conn.execute("DELETE FROM links WHERE src = ?", (src,))
        self._conn.commit()

    def replace_links(self, forward: Mapping[str, list[str]]) -> bool:
        """Make the stored links equal ``forward``; True when anything changed."""
        current = self.all_links()
        wanted = {src: dsts for src, dsts in forward.items() if dsts}
        changed = [src for src in set(current) | set(wanted) if current.get(src) != wanted.get(src)]
        for src in changed:
            self.set_links(src, wanted.get(src, []))
        return bool(changed)

    def all_links(self) -> dict[str, list[str]]:
        """``{src: [dst, ...]}`` for every stored link, in body order."""
        out: dict[str, list[str]] = {}
        for src, dst in self._conn.execute("SELECT src, dst FROM links ORDER BY src, pos"):
            out.setdefault(src, []).append(dst)
        return out

    def links_from(self, src: str) -> list[str]:
        rows = self._conn.execute("SELECT dst FROM links WHERE src = ? ORDER BY pos", (src,))
        return [str(dst) for (dst,) in rows]

    def links_to(self, dst: str) -> list[str]:
        rows = self._conn.execute("SELECT src FROM links WHERE dst = ? ORDER BY src", (dst,))
        return [str(src) for (src,) in rows]

    # -- similarity calibration -------------------------------------------

    def similarity_profile(self, model: str, *, refresh: bool = False) -> dict[str, float] | None:
        """Random-pair cosine profile for ``model``, using offline priors during cold start.

        Cosine ranges differ a lot between embedding models (random pairs sit near 0.05 for
        MiniLM and near 0.50 for gte-modernbert), so absolute cutoffs are not portable. Below
        ``_CALIBRATION_MIN_HEADS`` entries this falls back to the model's offline prior (or None
        for unknown models); above that it samples the store's own head vectors (cached in the
        index, resampled after ``_CALIBRATION_DRIFT`` drift) and blends smoothly from the prior
        to the empirical store distribution.
        """
        heads = self._load_vectors(model)[0]
        count = len(heads)
        prior = prior_profile(model)
        if count < _CALIBRATION_MIN_HEADS:
            return prior
        row = None
        with contextlib.suppress(sqlite3.OperationalError):  # read-only copy of an old index
            row = self._conn.execute(
                "SELECT head_count, profile FROM calibration WHERE model = ?", (model,)
            ).fetchone()
        if (
            not refresh
            and row is not None
            and abs(count - int(row[0])) <= _CALIBRATION_DRIFT * int(row[0])
        ):
            empirical: dict[str, float] = json.loads(row[1])
            return blend_profiles(prior, empirical, count)
        empirical = _pair_profile(list(heads.values()))
        if not self._read_only:
            self._conn.execute(
                "INSERT INTO calibration (model, head_count, profile) VALUES (?, ?, ?) "
                "ON CONFLICT(model) DO UPDATE SET head_count=excluded.head_count, "
                "profile=excluded.profile",
                (model, count, json.dumps(empirical)),
            )
            self._conn.commit()
        return blend_profiles(prior, empirical, count)

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
        "quote",
        "grounded",
        "checkpoint_id",
        "message_index",
    )
    #: Columns added after the first release; ``_migrate_inbox_table`` adds them to old files.
    _INBOX_EXTRA = (
        ("quote", "TEXT"),
        ("grounded", "INTEGER"),
        ("checkpoint_id", "TEXT"),
        ("message_index", "INTEGER"),
    )

    def _inbox_row(self, row: Any) -> dict[str, Any]:
        d: dict[str, Any] = dict.fromkeys(self._INBOX_COLS)
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

    def close(self) -> None:
        if not self._read_only and self._path != ":memory:":
            with contextlib.suppress(Exception):
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        close_sqlite(self._conn, self._temp_dir)
        self._temp_dir = None
