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
from .index import MemoryIndex
from .lexical import LexicalIndex
from .quality import low_signal_reason
from .store.base import Entry, EntryMeta, MemoryStore
from .store.md_store import slugify
from .validity import (
    RECENCY_MIN_GAP_SECONDS,
    RECENCY_TIE_EPS,
    backfill_scope_timestamps,
    entry_payload,
    find_duplicate_pairs,
    find_similar_entries,
    format_put_result,
    history_rows,
    normalize_supersedes,
    possible_updates_for,
    redirect_hits,
    restore_from_change,
    scope_status,
    select_ranked,
    superseded_entries,
    unused_entries,
)

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
_RECENCY_TIE_EPS = RECENCY_TIE_EPS
_RECENCY_MIN_GAP_SECONDS = RECENCY_MIN_GAP_SECONDS


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
        self._meta_cache: dict[str, tuple[object, dict[str, EntryMeta]]] = {}
        for scope in self.scopes.values():
            backfill_scope_timestamps(scope.store, scope.index)

    @property
    def profile(self) -> Scope:
        return self.scopes[PROFILE]

    def close(self) -> None:
        self._lexical_cache.clear()
        self._meta_cache.clear()
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
            metas = self._scope_meta(scope)
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
            for meta in self._scope_meta(scope).values()
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
            return entry_payload(key, s.name, body, s.store.get_meta(key) or EntryMeta())
        return None

    def _invalidate_lexical(self, scope_name: str) -> None:
        self._lexical_cache.pop(scope_name, None)
        self._meta_cache.pop(scope_name, None)

    def _scope_meta(self, scope: Scope) -> dict[str, EntryMeta]:
        fp = scope.store.fingerprint()
        if fp is not None:
            cached = self._meta_cache.get(scope.name)
            if cached is not None and cached[0] == fp:
                return cached[1]
        metas = scope.store.all_meta()
        if fp is not None:
            self._meta_cache[scope.name] = (fp, metas)
        return metas

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
        meta_cached = self._meta_cache.get(scope.name)
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
        if meta_cached is not None and meta_cached[0] == fp_before:
            new_meta = scope.store.get_meta(key)
            if new_meta is not None:
                meta_cached[1][key] = new_meta
                self._meta_cache[scope.name] = (fp_after, meta_cached[1])
            else:
                self._meta_cache.pop(scope.name, None)
        else:
            self._meta_cache.pop(scope.name, None)

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
        scope_names = {scope_name for scope_name, _key in fused}
        metas_by_scope = {name: self._scope_meta(self.scopes[name]) for name in scope_names}
        return select_ranked(
            fused,
            metas_by_scope,
            limit=limit,
            include_superseded=include_superseded,
            eps=_RECENCY_TIE_EPS,
            min_gap_seconds=_RECENCY_MIN_GAP_SECONDS,
        )

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
            for key, rank, how, similarity in self._ranked_hits(
                s, query, qvec, limit, include_superseded=include_superseded
            ):
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
        self,
        s: Scope,
        query: str,
        qvec: list[float] | None,
        limit: int,
        *,
        include_superseded: bool = False,
    ) -> Iterator[tuple[str, int, str, float | None]]:
        """(key, rank, how, similarity) hits for one scope, lexical then semantic."""
        # With an embedder, lexical sits half a step below semantic so a
        # lexical-only rank-0 hit does not tie a semantic-only rank-0 hit.
        offset = 1 if qvec is not None else 0
        min_coverage = _MIN_LEXICAL_COVERAGE if qvec is not None else 0.0
        metas = None if include_superseded else self._scope_meta(s)
        lex_hits = redirect_hits(
            ((k, None) for k in self._lexical(s, query, min_coverage)), metas, limit=limit * 3
        )
        for rank, (key, _sim) in enumerate(lex_hits):
            yield key, rank + offset, "lexical", None
        embedder = self._active_embedder()
        if qvec is not None and embedder is not None:
            extra = 0 if metas is None else sum(1 for m in metas.values() if m.superseded_by)
            raw_sem = s.index.search(qvec, embedder.model_id, top_k=limit * 3 + extra)
            for rank, (key, sim) in enumerate(redirect_hits(raw_sem, metas, limit=limit * 3)):
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
            meta = self._scope_meta(s).get(key)
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
        embedder = self._active_embedder()
        scope_map = {
            s.name: (self._lexical_index(s), s.index, self._scope_meta(s))
            for s in self.scopes.values()
        }
        return find_similar_entries(
            scope_map,
            title,
            summary,
            exclude=exclude,
            vector=vector,
            model_id=embedder.model_id if embedder is not None else None,
            duplicate_threshold=self.duplicate_threshold,
            lexical_threshold=_LEXICAL_DUP_THRESHOLD,
            limit=limit,
        )

    def duplicate_pairs(self, *, sync: bool = True) -> list[dict[str, Any]]:
        """Every pair above the duplicate threshold; ``sync=False`` never updates the index.

        Offline lexical audits use ``sync=False`` with no embedder. Normal callers
        retain lazy reconciliation of vectors and removed-entry telemetry.
        """
        if sync:
            for s in self.scopes.values():
                self.sync_index(s)
        embedder = self._active_embedder()
        scope_map = {s.name: (s.store, s.index, self._scope_meta(s)) for s in self.scopes.values()}
        return find_duplicate_pairs(
            scope_map,
            model_id=embedder.model_id if embedder is not None else None,
            duplicate_threshold=self.duplicate_threshold,
            lexical_threshold=_LEXICAL_DUP_THRESHOLD,
        )

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
        return history_rows(
            {s.name: s.index for s in self._scopes_for(scope)}, key=key, limit=limit
        )

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
        restored = restore_from_change(target.store, target.index, change, current is not None)
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

        applied = self._apply_supersedes(target, key, normalize_supersedes(supersedes, key))
        possible = possible_updates_for(
            target.store,
            self._lexical_index(target),
            target.index,
            self._scope_meta(target),
            model_id=embedder.model_id if (vector is not None and embedder is not None) else None,
            vector=vector,
            self_key=key,
            title=title,
            summary=summary,
            body=body,
            already_superseded=set(applied),
            semantic_threshold=max(0.45, self.duplicate_threshold - 0.17),
        )

        return format_put_result(
            key=key,
            title=title,
            category=category,
            scope=scope,
            is_update=is_update,
            applied_supersedes=applied,
            similar=similar,
            possible_updates=possible,
        )

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
        scope_map = {s.name: (s.store, s.index, self._scope_meta(s)) for s in self.scopes.values()}
        return unused_entries(scope_map, older_than_days=older_than_days, now=now)

    def superseded(
        self, older_than_days: float = 30, now: float | None = None
    ) -> list[dict[str, Any]]:
        """Superseded entries older than N days (candidates for removal)."""
        scope_map = {s.name: (s.store, self._scope_meta(s)) for s in self.scopes.values()}
        return superseded_entries(scope_map, older_than_days=older_than_days, now=now)

    def status(self) -> dict[str, Any]:
        model_id = self.embedder.model_id if self.embedder else None
        return {
            "embeddings": model_id or "off",
            "embeddings_error": self.embedder_error,
            "scopes": {
                s.name: scope_status(s.store, s.index, self._scope_meta(s), model_id)
                for s in self.scopes.values()
            },
        }
