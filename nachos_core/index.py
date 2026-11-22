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
import sqlite3
import statistics
import tempfile
from array import array
from collections.abc import Iterable, Mapping
from pathlib import Path

from .calibration import CALIBRATION_MIN_HEADS, blend_profiles, prior_profile
from .index_records import RecordTables
from .index_schema import create_and_migrate, existing_inbox_cols, table_columns
from .sqlite_util import close_sqlite, open_read_only_sqlite

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


class MemoryIndex(RecordTables):
    def __init__(self, path: str | os.PathLike[str], *, read_only: bool = False) -> None:
        self._path = str(path)
        self._read_only = read_only
        self._cache: tuple[str, dict[str, list[float]], dict[str, list[list[float]]]] | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if read_only:
            self._conn, self._temp_dir = open_read_only_sqlite(path)
            # An index written by an older release may lack these columns.
            self._chunked = "chunk_no" in table_columns(self._conn, "vectors")
            self._has_prev_superseded = "prev_superseded_by" in table_columns(self._conn, "changes")
            self._inbox_cols = existing_inbox_cols(self._conn)
            return
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        create_and_migrate(self._conn)
        self._inbox_cols = existing_inbox_cols(self._conn)
        self._chunked = True
        self._has_prev_superseded = True

    @property
    def read_only(self) -> bool:
        return self._read_only

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
        ``CALIBRATION_MIN_HEADS`` entries this falls back to the model's offline prior (or None
        for unknown models); above that it samples the store's own head vectors (cached in the
        index, resampled after ``_CALIBRATION_DRIFT`` drift) and blends smoothly from the prior
        to the empirical store distribution.
        """
        heads = self._load_vectors(model)[0]
        count = len(heads)
        prior = prior_profile(model)
        if count < CALIBRATION_MIN_HEADS:
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

    def close(self) -> None:
        if not self._read_only and self._path != ":memory:":
            with contextlib.suppress(Exception):
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        close_sqlite(self._conn, self._temp_dir)
        self._temp_dir = None
