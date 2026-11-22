"""MemoryService — the harness-agnostic memory API behind the Hermes adapter.

Owns one or two *scopes*, each a store plus its sidecar index:

* ``profile`` — always present; private to the active Hermes profile.
* ``shared``  — optional; one store every profile on the machine can read and
  write explicitly (``scope="shared"``). Shown in the manifest under
  ``shared/<category>``.

On top of the stores it adds:

* **Hybrid recall** — lexical term matching fused with embedding similarity
  (reciprocal-rank fusion). Without an embedder it is lexical only.
* **Duplicate flagging** — a put reports existing entries that look like the
  same memory (embedding similarity, or token overlap when no embedder), so
  the agent can merge instead of accumulating near-copies. It flags; it does
  not block.
* **Change log** — every create/update/remove/revert is recorded with the
  previous content so it can be undone (``revert``).
* **Usage telemetry** — first-seen, recall and query-hit counts per entry, so
  hygiene can find memories nobody ever uses.
* **Redaction** — an injected ``redact(text) -> text`` applied to everything
  written. The Hermes adapter supplies Hermes' own secret redactor.

Embedding is lazy: vectors are computed on put and reconciled (hand edits,
model changes) on the next query. An embedder error
disables semantic matching for the rest of the process and is reported in
``status()``; memory keeps working lexically.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

from .chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNKS_PER_ENTRY,
    chunk_body,
    embed_chunks,
    embed_text,
)
from .embeddings import DOCUMENT, QUERY, Embedder
from .index import MemoryIndex, dot
from .lexical import LexicalIndex, jaccard
from .quality import looks_like_update, low_signal_reason
from .store.base import Entry, EntryMeta, MemoryStore
from .store.md_store import slugify

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "MAX_CHUNKS_PER_ENTRY",
    "PROFILE",
    "SHARED",
    "Draft",
    "MemoryService",
    "Scope",
    "chunk_body",
    "embed_chunks",
    "embed_text",
]

logger = logging.getLogger(__name__)

PROFILE = "profile"
SHARED = "shared"

_RRF_K = 60
# Hybrid recall only: a lexical hit must match at least this IDF-weighted share of the query's
# terms to take part in fusion. Weak partial matches ("service", "internal") are noise next to
# a semantic list, and RRF would otherwise let them displace or double-count real answers.
# Model-independent, unlike an absolute cosine floor (which only suited MiniLM's score range);
# 0.35-0.45 is a plateau on bench/ (see docs/benchmark.md).
_MIN_LEXICAL_COVERAGE = 0.4
_LEXICAL_DUP_THRESHOLD = 0.6
# Recency tie-break tolerance on RRF scores: covers exact ties and symmetric (0,1)/(1,0)
# hybrid rank swaps (1/60 + 1/62 - 2/61 ~= 8.8e-6) without overriding wider score gaps.
_RECENCY_TIE_EPS = 1e-5


def _normalize_supersedes(supersedes: str | list[str] | None, self_key: str) -> list[str]:
    if not supersedes:
        return []
    raw = [supersedes] if isinstance(supersedes, str) else list(supersedes)
    out: list[str] = []
    for item in raw:
        k = str(item or "").strip()
        if k and k != self_key and k not in out:
            out.append(k)
    return out


def _apply_recency_tiebreak(
    items: list[tuple[tuple[str, str], dict[str, Any]]],
    metas_by_scope: dict[str, dict[str, EntryMeta]],
) -> list[tuple[tuple[str, str], dict[str, Any]]]:
    """Within near-equal fused score clusters (same match count), prefer newer updated_at."""
    if len(items) < 2:
        return items

    def _updated(entry: tuple[tuple[str, str], dict[str, Any]]) -> float:
        meta = metas_by_scope[entry[0][0]].get(entry[0][1])
        return meta.updated_at if meta is not None and meta.updated_at is not None else 0.0

    out: list[tuple[tuple[str, str], dict[str, Any]]] = []
    i = 0
    while i < len(items):
        lead_score = items[i][1]["score"]
        lead_matches = len(items[i][1]["match"])
        j = i + 1
        while (
            j < len(items)
            and len(items[j][1]["match"]) == lead_matches
            and lead_score - items[j][1]["score"] <= _RECENCY_TIE_EPS
        ):
            j += 1
        cluster = items[i:j]
        if len(cluster) > 1:
            cluster.sort(key=lambda entry: -_updated(entry))
        out.extend(cluster)
        i = j
    return out


@dataclass
class Draft:
    """A normalised, gated prospective write (see ``MemoryService.prepare``)."""

    title: str
    summary: str
    category: str
    body: str
    key: str
    scope: str
    error: str | None = None


@dataclass
class Scope:
    name: str
    store: MemoryStore
    index: MemoryIndex

    def close(self) -> None:
        for part in (self.store, self.index):
            close = getattr(part, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()


class MemoryService:
    def __init__(
        self,
        scopes: OrderedDict[str, Scope],
        *,
        embedder: Embedder | None = None,
        duplicate_threshold: float = 0.65,
        redact: Callable[[str], str] | None = None,
    ):
        if PROFILE not in scopes:
            raise ValueError("a 'profile' scope is required")
        self.scopes = scopes
        self.embedder = embedder
        self.duplicate_threshold = duplicate_threshold
        self.redact = redact or (lambda text: text)
        self.embedder_error: str | None = None
        self.session_id: str = ""
        self._seen_keys: dict[str, frozenset] = {}
        self._lexical_cache: dict[str, tuple[object, LexicalIndex]] = {}
        for scope in self.scopes.values():
            self._backfill_timestamps(scope)

    @staticmethod
    def _backfill_timestamps(scope: Scope) -> None:
        """Populate missing created_at/updated_at from change log, usage first_seen, or now."""
        if scope.store.read_only or scope.index.read_only:
            return
        metas = scope.store.all_meta()
        missing = [k for k, m in metas.items() if m.created_at is None or m.updated_at is None]
        if not missing:
            return
        change_ts = scope.index.change_timestamps()
        usage = scope.index.usage()
        now = time.time()
        to_fill: dict[str, tuple[float, float, bool]] = {}
        for key in missing:
            if key in change_ts:
                created_at, updated_at = change_ts[key]
            elif key in usage and usage[key].get("first_seen"):
                created_at = updated_at = float(usage[key]["first_seen"])
            else:
                created_at = updated_at = now
            to_fill[key] = (created_at, updated_at, True)
        scope.store.backfill_timestamps(to_fill)

    @property
    def profile(self) -> Scope:
        return self.scopes[PROFILE]

    def close(self) -> None:
        self._lexical_cache.clear()
        for scope in self.scopes.values():
            scope.close()
        if self.embedder is not None:
            with contextlib.suppress(Exception):
                self.embedder.close()

    # -- embeddings --------------------------------------------------------

    @property
    def semantic_enabled(self) -> bool:
        return self.embedder is not None and self.embedder_error is None

    def _active_embedder(self) -> Embedder | None:
        """The embedder when semantic search is usable right now, else None."""
        return self.embedder if self.semantic_enabled else None

    def _embed(self, texts: list[str], input_type: str = DOCUMENT) -> list[list[float]] | None:
        embedder = self._active_embedder()
        if embedder is None or not texts:
            return None
        try:
            return embedder.embed(texts, input_type)
        except Exception as e:  # any backend failure degrades to lexical
            self.embedder_error = f"{type(e).__name__}: {e}"[:300]
            logger.warning(
                "Nachos embeddings disabled for this session (%s); recall falls "
                "back to lexical matching.",
                self.embedder_error,
            )
            return None

    def sync_index(self, scope: Scope) -> int:
        """Embed entries that are new, edited, or embedded with another model."""
        entries = scope.store.list()
        scope.index.retain(key for key, *_ in entries)
        embedder = self._active_embedder()
        if embedder is None:
            return 0
        embedder.warm_up()  # may refine model_id (e.g. CLI pooling/prefix) before it is compared
        bodies = scope.store.bodies()
        docs = {
            key: embed_chunks(title, summary, bodies.get(key, ""))
            for key, title, summary, _category in entries
        }
        stale = scope.index.stale_keys(docs, embedder.model_id)
        if not stale:
            return 0
        flat_texts = [text for key in stale for text in docs[key]]
        vectors = self._embed(flat_texts, DOCUMENT)
        if vectors is None:
            return 0
        offset = 0
        for key in stale:
            chunks = docs[key]
            key_vecs = vectors[offset : offset + len(chunks)]
            offset += len(chunks)
            scope.index.upsert_chunks(key, chunks, embedder.model_id, key_vecs)
        return len(stale)

    # -- manifest ----------------------------------------------------------

    def manifest_entries(self, *, include_superseded: bool = False) -> list[Entry]:
        rows: list[Entry] = []
        for scope in self.scopes.values():
            entries = scope.store.list()
            metas = scope.store.all_meta()
            if not include_superseded:
                entries = [
                    e for e in entries if not (metas.get(e[0]) and metas[e[0]].superseded_by)
                ]
            keys = frozenset(e[0] for e in entries)
            if self._seen_keys.get(scope.name) != keys:
                scope.index.record_seen(keys)
                self._seen_keys[scope.name] = keys
            if scope.name == PROFILE:
                rows.extend(entries)
            else:
                rows.extend((k, t, s, f"{scope.name}/{c}") for k, t, s, c in entries)
        return rows

    def superseded_count(self) -> int:
        """Total number of superseded entries hidden from the default manifest."""
        return sum(
            1
            for scope in self.scopes.values()
            for meta in scope.store.all_meta().values()
            if meta.superseded_by
        )

    # -- recall ------------------------------------------------------------

    def _scopes_for(self, scope: str | None) -> list[Scope]:
        if scope:
            if scope not in self.scopes:
                raise KeyError(scope)
            return [self.scopes[scope]]
        return list(self.scopes.values())

    def get(self, key: str, scope: str | None = None) -> tuple[str, str] | None:
        """Return (scope_name, body) for an exact key, profile scope first."""
        for s in self._scopes_for(scope):
            body = s.store.get(key)
            if body is not None:
                s.index.record_recall(key)
                return s.name, body
        return None

    def get_entry(self, key: str, scope: str | None = None) -> dict[str, Any] | None:
        """Return full dict for an exact key, including a superseded_by pointer when superseded."""
        for s in self._scopes_for(scope):
            body = s.store.get(key)
            if body is None:
                continue
            s.index.record_recall(key)
            meta = s.store.get_meta(key) or EntryMeta()
            out: dict[str, Any] = {"key": key, "scope": s.name, "body": body}
            if meta.created_at is not None:
                out["created_at"] = meta.created_at
            if meta.updated_at is not None:
                out["updated_at"] = meta.updated_at
            if meta.superseded_by:
                out["superseded_by"] = meta.superseded_by
                out["note"] = (
                    f"This entry has been superseded by '{meta.superseded_by}'. "
                    "Fetch that key for current information."
                )
            return out
        return None

    def _invalidate_lexical(self, scope_name: str) -> None:
        self._lexical_cache.pop(scope_name, None)

    def _update_lexical_after_put(
        self,
        scope: Scope,
        fp_before: object,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
    ) -> None:
        cached = self._lexical_cache.get(scope.name)
        if fp_before is None or cached is None or cached[0] != fp_before or "\n" in summary:
            self._invalidate_lexical(scope.name)
            return
        stored_body = scope.store.get(key)
        fp_after = scope.store.fingerprint()
        if stored_body is None or fp_after is None:
            self._invalidate_lexical(scope.name)
            return
        cached[1].upsert(key, title=title, summary=summary, category=category, body=stored_body)
        self._lexical_cache[scope.name] = (fp_after, cached[1])

    def _lexical_index(self, scope: Scope) -> LexicalIndex:
        fp = scope.store.fingerprint()
        if fp is not None:
            cached = self._lexical_cache.get(scope.name)
            if cached is not None and cached[0] == fp:
                return cached[1]
        index = LexicalIndex.build(scope.store.list(), scope.store.bodies())
        if fp is not None:
            self._lexical_cache[scope.name] = (fp, index)
        return index

    def _lexical(self, scope: Scope, query: str, min_coverage: float = 0.0) -> list[str]:
        return self._lexical_index(scope).search(query, min_coverage=min_coverage)

    def _select_ranked(
        self,
        fused: dict[tuple[str, str], dict[str, Any]],
        *,
        limit: int,
        include_superseded: bool,
    ) -> list[tuple[tuple[str, str], dict[str, Any]]]:
        """Filter superseded entries (unless requested) and apply recency tie-breaking."""
        if not fused:
            return []
        scope_names = {scope_name for scope_name, _key in fused}
        metas_by_scope = {name: self.scopes[name].store.all_meta() for name in scope_names}
        items = [
            item
            for item in fused.items()
            if include_superseded
            or not (
                (meta := metas_by_scope[item[0][0]].get(item[0][1])) is not None
                and meta.superseded_by
            )
        ]
        # Ties go to entries matched by both lexical and semantic, then newer updated_at on near-ties.
        items.sort(key=lambda item: (-item[1]["score"], -len(item[1]["match"])))
        items = _apply_recency_tiebreak(items, metas_by_scope)
        return items[:limit]

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        scope: str | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        targets = self._scopes_for(scope)
        qvec = self._query_vector(targets, query)
        fused: dict[tuple[str, str], dict[str, Any]] = {}
        for s in targets:
            for key, rank, how, similarity in self._ranked_hits(s, query, qvec, limit):
                row = fused.setdefault((s.name, key), {"score": 0.0, "match": []})
                row["score"] += 1.0 / (_RRF_K + rank)
                row["match"].append(how)
                if similarity is not None:
                    row["similarity"] = round(similarity, 3)

        ranked = self._select_ranked(fused, limit=limit, include_superseded=include_superseded)
        return self._hydrate(ranked)

    def _query_vector(self, targets: list[Scope], query: str) -> list[float] | None:
        """Embedding of ``query`` (after syncing the target indexes), or None if lexical-only."""
        if not self.semantic_enabled:
            return None
        for s in targets:
            self.sync_index(s)
        vectors = self._embed([query], QUERY)
        return vectors[0] if vectors else None

    def _ranked_hits(
        self, s: Scope, query: str, qvec: list[float] | None, limit: int
    ) -> Iterator[tuple[str, int, str, float | None]]:
        """(key, rank, how, similarity) hits for one scope, lexical then semantic."""
        # With an embedder, lexical sits half a step below semantic so a
        # lexical-only rank-0 hit does not tie a semantic-only rank-0 hit.
        offset = 1 if qvec is not None else 0
        min_coverage = _MIN_LEXICAL_COVERAGE if qvec is not None else 0.0
        for rank, key in enumerate(self._lexical(s, query, min_coverage)[: limit * 3]):
            yield key, rank + offset, "lexical", None
        embedder = self._active_embedder()
        if qvec is not None and embedder is not None:
            hits = s.index.search(qvec, embedder.model_id, top_k=limit * 3)
            for rank, (key, sim) in enumerate(hits):
                yield key, rank, "semantic", sim

    def _hydrate(
        self, ranked: list[tuple[tuple[str, str], dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Turn fused ranking rows into result dicts, recording a hit per returned entry."""
        results = []
        hits_by_scope: dict[str, list[str]] = {}
        for (scope_name, key), row in ranked:
            s = self.scopes[scope_name]
            body = s.store.get(key)
            if body is None:
                continue
            hits_by_scope.setdefault(scope_name, []).append(key)
            result = {
                "key": key,
                "scope": scope_name,
                "body": body,
                "match": "+".join(row["match"]),
            }
            if "similarity" in row:
                result["similarity"] = row["similarity"]
            meta = s.store.get_meta(key)
            if meta is not None and meta.superseded_by:
                result["superseded_by"] = meta.superseded_by
            results.append(result)
        for scope_name, keys in hits_by_scope.items():
            self.scopes[scope_name].index.record_hits(keys)
        return results

    # -- duplicates --------------------------------------------------------

    def find_similar(
        self,
        title: str,
        summary: str,
        *,
        exclude: tuple[str, str] | None = None,
        vector: list[float] | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Existing entries that look like the same memory, most similar first."""
        found: dict[tuple[str, str], dict[str, Any]] = {}
        label = f"{title} {summary}"
        for s in self.scopes.values():
            index = self._lexical_index(s)
            metas = s.store.all_meta()
            excluded_key = exclude[1] if exclude and exclude[0] == s.name else None
            for key, doc_title, score in index.similar_labels(
                label, threshold=_LEXICAL_DUP_THRESHOLD, exclude=excluded_key
            ):
                if metas.get(key) and metas[key].superseded_by:
                    continue
                found[(s.name, key)] = {
                    "key": key,
                    "scope": s.name,
                    "title": doc_title,
                    "similarity": round(score, 3),
                    "match": "lexical",
                }
            embedder = self._active_embedder()
            if vector is not None and embedder is not None:
                for key, sim in s.index.search(
                    vector, embedder.model_id, top_k=limit + 1, head_only=True
                ):
                    if (
                        (s.name, key) == exclude
                        or sim < self.duplicate_threshold
                        or (metas.get(key) and metas[key].superseded_by)
                    ):
                        continue
                    prev = found.get((s.name, key))
                    if prev is None or sim > prev["similarity"]:
                        doc = index.docs.get(key)
                        found[(s.name, key)] = {
                            "key": key,
                            "scope": s.name,
                            "title": doc.title if doc is not None else key,
                            "similarity": round(sim, 3),
                            "match": "semantic",
                        }
        return sorted(found.values(), key=lambda d: -d["similarity"])[:limit]

    def duplicate_pairs(self, *, sync: bool = True) -> list[dict[str, Any]]:
        """Every pair above the duplicate threshold; ``sync=False`` never updates the index.

        Offline lexical audits use ``sync=False`` with no embedder. Normal callers
        retain lazy reconciliation of vectors and removed-entry telemetry.
        """
        pairs: list[dict[str, Any]] = []
        if sync:
            for s in self.scopes.values():
                self.sync_index(s)
        flat: list[tuple[str, str, str, str]] = []
        for s in self.scopes.values():
            metas = s.store.all_meta()
            flat.extend(
                (s.name, k, t, sm)
                for k, t, sm, _c in s.store.list()
                if not (metas.get(k) and metas[k].superseded_by)
            )
        vecs: dict[tuple[str, str], list[float]] = {}
        embedder = self._active_embedder()
        if embedder is not None:
            for s in self.scopes.values():
                for key, vec in s.index.vectors(embedder.model_id).items():
                    vecs[(s.name, key)] = vec
        for i, (sa, ka, ta, sma) in enumerate(flat):
            for sb, kb, tb, smb in flat[i + 1 :]:
                va, vb = vecs.get((sa, ka)), vecs.get((sb, kb))
                if va is not None and vb is not None:
                    sim, how = dot(va, vb), "semantic"
                    if sim < self.duplicate_threshold:
                        continue
                else:
                    sim, how = jaccard(f"{ta} {sma}", f"{tb} {smb}"), "lexical"
                    if sim < _LEXICAL_DUP_THRESHOLD:
                        continue
                pairs.append(
                    {
                        "a": {"scope": sa, "key": ka, "title": ta},
                        "b": {"scope": sb, "key": kb, "title": tb},
                        "similarity": round(sim, 3),
                        "match": how,
                    }
                )
        pairs.sort(key=lambda p: -p["similarity"])
        return pairs

    # -- change log --------------------------------------------------------

    @staticmethod
    def _snapshot(scope: Scope, key: str) -> tuple[str, str, str, str] | None:
        """(title, summary, category, body) for ``key``, or None if absent."""
        body = scope.store.get(key)
        if body is None:
            return None
        for k, title, summary, category in scope.store.list():
            if k == key:
                return title, summary, category, body
        return None

    @staticmethod
    def _same(prev: tuple[str, str, str, str] | None, new: tuple[str, str, str, str]) -> bool:
        if prev is None:
            return False
        if prev == new:
            return True
        # The flat-file store stores the summary as the first line of the body.
        title, summary, category, body = new
        led = summary + ("\n\n" + body if body else "")
        return prev == (title, summary, category, led)

    def _log(
        self,
        scope: Scope,
        key: str,
        action: str,
        prev: tuple[str, str, str, str] | None,
        prev_superseded_by: str | None = None,
    ) -> None:
        try:
            scope.index.record_change(
                key, action, self.session_id, prev, prev_superseded_by=prev_superseded_by
            )
        except Exception as e:  # the log must never block a write
            logger.warning("Nachos change log write failed: %s", e)

    def history(
        self, key: str | None = None, scope: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Recent memory changes across scopes, newest first."""
        rows: list[dict[str, Any]] = []
        for s in self._scopes_for(scope):
            for c in s.index.changes(key=key, limit=limit):
                preview = " ".join(str(c["prev_summary"] or "").split())
                if len(preview) > 60:
                    preview = preview[:59] + "…"
                rows.append(
                    {
                        "scope": s.name,
                        "id": c["id"],
                        "ts": c["ts"],
                        "key": c["key"],
                        "action": c["action"],
                        "session_id": c["session_id"],
                        "preview": preview,
                    }
                )
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return rows[:limit]

    def revert(self, change_id: int, scope: str = PROFILE) -> dict[str, Any]:
        """Restore the state an entry had before change ``change_id``."""
        if scope not in self.scopes:
            return {"error": f"Unknown or disabled scope {scope!r}; available: {list(self.scopes)}"}
        target = self.scopes[scope]
        try:
            change = target.index.get_change(int(change_id))
        except (TypeError, ValueError):
            change = None
        if change is None:
            return {"error": f"No change {change_id!r} in scope {scope!r}."}
        key = change["key"]
        current = self._snapshot(target, key)
        cur_meta = target.store.get_meta(key)
        cur_superseded = cur_meta.superseded_by if cur_meta is not None else None
        prev_superseded = change.get("prev_superseded_by")
        if change["action"] == "supersede" and current is not None:
            target.store.set_superseded_by(key, prev_superseded)
            restored = "restored"
        elif change["prev_title"] is None:
            # Entry did not exist before the change: undoing means removing it.
            if current is not None:
                target.store.remove(key)
            target.index.delete_vector(key)
            restored = "removed"
        else:
            target.store.put(
                key,
                title=change["prev_title"],
                summary=change["prev_summary"] or "",
                category=change["prev_category"] or "general",
                body=change["prev_body"] or "",
            )
            if prev_superseded != (target.store.get_meta(key) or EntryMeta()).superseded_by:
                target.store.set_superseded_by(key, prev_superseded)
            target.index.delete_vector(key)
            target.index.record_seen([key])
            restored = "restored"
        self._invalidate_lexical(target.name)
        self._log(target, key, "revert", current, prev_superseded_by=cur_superseded)
        return {
            "reverted": change["id"],
            "key": key,
            "scope": scope,
            "undid": change["action"],
            "result": restored,
            "title": change["prev_title"] or key,
        }

    # -- writes ------------------------------------------------------------

    def prepare(
        self,
        *,
        title: str,
        body: str,
        summary: str = "",
        category: str = "general",
        scope: str = PROFILE,
    ) -> Draft:
        """Normalise and gate a prospective write without touching any store.

        The single source of truth for what ``put`` accepts: redaction, summary
        defaulting, key derivation and the low-signal gates all live here, so
        dry-runs (``tools/import_memories.py``) cannot drift from real writes.
        ``Draft.error`` is set when the write would be rejected.
        """
        if scope not in self.scopes:
            return Draft(
                "",
                "",
                "",
                "",
                "",
                scope,
                f"Unknown or disabled scope {scope!r}; available: {list(self.scopes)}",
            )
        title = self.redact((title or "").strip())
        body = self.redact((body or "").strip())
        summary = self.redact((summary or "").strip())
        if not summary:
            summary = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        category = (category or "general").strip() or "general"
        draft = Draft(title, summary, category, body, slugify(title) if title else "", scope)
        draft.error = self._gate(draft)
        return draft

    @staticmethod
    def _gate(draft: Draft) -> str | None:
        if not draft.title:
            return "title is required"
        reason = low_signal_reason(draft.body)
        if reason:
            return f"body is low-signal ({reason}); write real content, not a placeholder."
        reason = low_signal_reason(draft.summary)
        if reason:
            return (
                f"summary is low-signal ({reason}); this is the ONLY "
                "text the agent sees in the manifest for this entry "
                "without recalling it — write a real, keyword-dense "
                "summary."
            )
        return None

    def _apply_supersedes(self, scope: Scope, new_key: str, targets: list[str]) -> list[str]:
        applied: list[str] = []
        for old_key in targets:
            prev = self._snapshot(scope, old_key)
            if prev is None:
                continue
            old_meta = scope.store.get_meta(old_key)
            old_sup = old_meta.superseded_by if old_meta is not None else None
            self._log(scope, old_key, "supersede", prev, prev_superseded_by=old_sup)
            if scope.store.set_superseded_by(old_key, new_key):
                applied.append(old_key)
        if applied:
            self._invalidate_lexical(scope.name)
        return applied

    def _suggest_supersedes(
        self,
        scope: Scope,
        new_text: str,
        similar: list[dict[str, Any]],
        already_superseded: set[str],
    ) -> list[str]:
        suggested: list[str] = []
        for dup in similar:
            if dup["scope"] != scope.name or dup["key"] in already_superseded:
                continue
            snap = self._snapshot(scope, dup["key"])
            if snap is None:
                continue
            old_text = f"{snap[0]}\n{snap[1]}\n{snap[3]}"
            if looks_like_update(old_text, new_text):
                suggested.append(dup["key"])
        return suggested

    def put(
        self,
        *,
        title: str,
        body: str,
        summary: str = "",
        category: str = "general",
        scope: str = PROFILE,
        supersedes: str | list[str] | None = None,
    ) -> dict[str, Any]:
        draft = self.prepare(
            title=title, body=body, summary=summary, category=category, scope=scope
        )
        if draft.error:
            return {"error": draft.error}
        title, summary, category, body, key = (
            draft.title,
            draft.summary,
            draft.category,
            draft.body,
            draft.key,
        )
        target = self.scopes[scope]
        is_update = target.store.get(key) is not None

        chunks = embed_chunks(title, summary, body)
        chunk_vectors: list[list[float]] | None = None
        vector: list[float] | None = None
        if self.semantic_enabled:
            for s in self.scopes.values():
                self.sync_index(s)
            chunk_vectors = self._embed(chunks, DOCUMENT)
            vector = chunk_vectors[0] if chunk_vectors else None
        similar = self.find_similar(title, summary, exclude=(scope, key), vector=vector)

        prev = self._snapshot(target, key)
        prev_meta = target.store.get_meta(key)
        if not self._same(prev, (title, summary, category, body)):
            self._log(
                target,
                key,
                "update" if prev else "create",
                prev,
                prev_superseded_by=prev_meta.superseded_by if prev_meta else None,
            )
        fp_before = target.store.fingerprint()
        target.store.put(key, title=title, summary=summary, category=category, body=body)
        self._update_lexical_after_put(
            target, fp_before, key, title=title, summary=summary, category=category
        )
        embedder = self._active_embedder()
        if chunk_vectors is not None and embedder is not None:
            target.index.upsert_chunks(key, chunks, embedder.model_id, chunk_vectors)
        target.index.record_seen([key])

        applied = self._apply_supersedes(target, key, _normalize_supersedes(supersedes, key))
        suggested = (
            self._suggest_supersedes(target, f"{title}\n{summary}\n{body}", similar, set(applied))
            if similar
            else []
        )

        result: dict[str, Any] = {
            "stored": key,
            "title": title,
            "category": category,
            "scope": scope,
            "updated": is_update,
        }
        if applied:
            result["superseded"] = applied
        if similar:
            result["possible_duplicates"] = similar
            if suggested:
                result["suggested_supersedes"] = suggested
                result["note"] = (
                    "Similar entries already exist and this looks like an update. If it replaces "
                    f"them, re-put with supersedes={suggested!r} (or merge and remove the old entry)."
                )
            else:
                result["note"] = (
                    "Similar entries already exist. If this is the same memory, merge "
                    "the content into one entry (put with the existing title) and "
                    "remove the other."
                )
        return result

    def remove(self, key: str, scope: str | None = None) -> str | None:
        """Remove ``key``; returns the scope it was removed from, or None."""
        for s in self._scopes_for(scope):
            if s.store.get(key) is not None:
                meta = s.store.get_meta(key)
                self._log(
                    s,
                    key,
                    "remove",
                    self._snapshot(s, key),
                    prev_superseded_by=meta.superseded_by if meta else None,
                )
                s.store.remove(key)
                self._invalidate_lexical(s.name)
                s.index.delete(key)
                return s.name
        return None

    # -- hygiene -----------------------------------------------------------

    def unused(self, older_than_days: float, now: float | None = None) -> list[dict[str, Any]]:
        """Entries older than N days (first seen) never recalled or returned."""
        now = now if now is not None else time.time()
        cutoff = now - older_than_days * 86400
        rows: list[dict[str, Any]] = []
        for s in self.scopes.values():
            usage = s.index.usage()
            metas = s.store.all_meta()
            for key, title, _summary, category in s.store.list():
                if metas.get(key) and metas[key].superseded_by:
                    continue
                u = usage.get(key)
                if u is None or u["first_seen"] > cutoff:
                    continue
                if u["recall_count"] or u["hit_count"]:
                    continue
                rows.append(
                    {
                        "scope": s.name,
                        "key": key,
                        "title": title,
                        "category": category,
                        "age_days": round((now - u["first_seen"]) / 86400, 1),
                    }
                )
        rows.sort(key=lambda r: -r["age_days"])
        return rows

    def superseded(
        self, older_than_days: float = 30, now: float | None = None
    ) -> list[dict[str, Any]]:
        """Superseded entries older than N days (candidates for removal)."""
        now = now if now is not None else time.time()
        cutoff = now - older_than_days * 86400
        rows: list[dict[str, Any]] = []
        for s in self.scopes.values():
            metas = s.store.all_meta()
            for key, title, _summary, category in s.store.list():
                meta = metas.get(key)
                if meta is None or not meta.superseded_by:
                    continue
                ts = meta.updated_at if meta.updated_at is not None else meta.created_at
                if ts is None or ts > cutoff:
                    continue
                rows.append(
                    {
                        "scope": s.name,
                        "key": key,
                        "title": title,
                        "category": category,
                        "superseded_by": meta.superseded_by,
                        "age_days": round((now - ts) / 86400, 1),
                    }
                )
        rows.sort(key=lambda r: (-r["age_days"], r["key"]))
        return rows

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "embeddings": self.embedder.model_id if self.embedder else "off",
            "embeddings_error": self.embedder_error,
            "scopes": {},
        }
        for s in self.scopes.values():
            entries = s.store.list()
            usage = s.index.usage()
            metas = s.store.all_meta()
            out["scopes"][s.name] = {
                "entries": len(entries),
                "superseded": sum(1 for m in metas.values() if m.superseded_by),
                "approx_timestamps": sum(1 for m in metas.values() if m.approx_timestamps),
                "changes": s.index.count_changes(),
                "indexed": s.index.count(self.embedder.model_id) if self.embedder else 0,
                "never_used": sum(
                    1
                    for k, *_ in entries
                    if not (metas.get(k) and metas[k].superseded_by)
                    and not (
                        usage.get(k, {}).get("recall_count") or usage.get(k, {}).get("hit_count")
                    )
                ),
            }
        return out
