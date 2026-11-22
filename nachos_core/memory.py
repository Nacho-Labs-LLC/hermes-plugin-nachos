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
from collections.abc import Sequence
from typing import Any, Callable

from .chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNKS_PER_ENTRY,
    chunk_body,
    embed_chunks,
    embed_text,
)
from .embeddings import Embedder
from .graph import ScopeGraph
from .lexical import LexicalIndex
from .linked import (
    link_fields,
    link_hints,
    link_report,
    merged_link_counts,
)
from .links import resolve_successor
from .provenance import MANUAL, MAX_QUOTE_CHARS, source_view
from .ranking import (
    DEFAULT_BLEND,
    update_threshold_for,
)
from .recall import CHUNK_PENALTY, Recall, RecallTuning
from .reindex import DEFAULT_BUDGET_S, SemanticRuntime
from .scope import PROFILE, SHARED, Scope, ScopeCaches, select_scopes
from .store.base import Entry, EntryMeta
from .validity import (
    backfill_scope_timestamps,
    entry_payload,
    format_put_result,
    history_rows,
    normalize_supersedes,
    possible_updates_for,
    restore_from_change,
    scope_status,
    superseded_entries,
    suspicious_supersedes,
    unused_entries,
)
from .writes import Draft, prepare_draft, same_content, snapshot

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


class MemoryService:
    def __init__(
        self,
        scopes: OrderedDict[str, Scope],
        *,
        embedder: Embedder | None = None,
        duplicate_threshold: float = 0.65,
        redact: Callable[[str], str] | None = None,
        chunk_penalty: float = CHUNK_PENALTY,
        similarity_floor: float | str | None = None,
        calibrate_duplicates: bool = False,
        calibrate_updates: bool = True,
        rerank_candidates: int = 25,
        rerank_mode: str = DEFAULT_BLEND,
        reindex_budget_s: float = DEFAULT_BUDGET_S,
        rerank_borderline: bool = False,
        related_per_hit: int = 3,
        related_max: int = 8,
    ):
        if PROFILE not in scopes:
            raise ValueError("a 'profile' scope is required")
        self.scopes = scopes
        self.duplicate_threshold = duplicate_threshold
        self.redact = redact or (lambda text: text)
        #: Subtracted from body-chunk similarities so a chunk must beat the head by this much.
        self.chunk_penalty = chunk_penalty
        #: None = no floor; a float = absolute cosine; a profile key such as "p95" = that
        #: percentile of the model's own random-pair cosine (see MemoryIndex.similarity_profile).
        self.similarity_floor = similarity_floor
        #: Raise ``duplicate_threshold`` to the model's random-pair ``DUP_PERCENTILE`` when higher.
        #: Off by default so an explicitly configured threshold is used exactly as given.
        self.calibrate_duplicates = calibrate_duplicates
        #: Raise the semantic ``possible_updates`` floor to the model's calibrated percentile.
        self.calibrate_updates = calibrate_updates
        #: Fused results sent to the cross-encoder, when the embedder offers one.
        self.rerank_candidates = rerank_candidates
        self.rerank_mode = rerank_mode
        #: When True (rerank="borderline"), skip full reranking and only verify borderline queries.
        self.rerank_borderline = rerank_borderline
        #: ``related`` entries attached to each result (0 disables), and over the whole result list.
        self.related_per_hit = related_per_hit
        self.related_max = related_max
        self._caches = ScopeCaches()
        self._semantic = SemanticRuntime(embedder, reindex_budget_s)
        self._graphs = {
            name: ScopeGraph(
                name,
                s.store,
                s.index,
                lambda s=s: self._caches.meta(s),  # type: ignore[misc]
                lambda s=s: self._caches.titles(s),  # type: ignore[misc]
                lambda s=s: self._caches.lexical(s),  # type: ignore[misc]
            )
            for name, s in scopes.items()
        }
        self._recall = Recall(self.scopes, self._graphs, self._caches, self._semantic, self._tuning)
        self.session_id: str = ""
        self._seen_keys: dict[str, frozenset] = {}
        for scope in self.scopes.values():
            backfill_scope_timestamps(scope.store, scope.index)

    @property
    def profile(self) -> Scope:
        return self.scopes[PROFILE]

    def close(self) -> None:
        self._caches.clear()
        self._graphs.clear()
        for scope in self.scopes.values():
            scope.close()
        if self.embedder is not None:
            with contextlib.suppress(Exception):
                self.embedder.close()

    # -- embeddings --------------------------------------------------------

    @property
    def embedder(self) -> Embedder | None:
        return self._semantic.embedder

    @embedder.setter
    def embedder(self, embedder: Embedder | None) -> None:
        self._semantic.embedder = embedder

    @property
    def embedder_error(self) -> str | None:
        return self._semantic.error

    @property
    def semantic_enabled(self) -> bool:
        return self._semantic.enabled

    @property
    def reindexing(self) -> bool:
        """True while a model switch is still re-embedding entries (recall is lexical meanwhile)."""
        return self._semantic.reindexing

    @property
    def reindex_budget_s(self) -> float:
        return self._semantic.reindex_budget_s

    def sync_index(self, scope: Scope, *, budget_s: float | None = None) -> int:
        """Embed entries that are new, edited, or embedded with another model (see ``reindex``)."""
        return self._semantic.sync(scope, budget_s=budget_s)

    def reindex(self, *, scope: str | None = None) -> dict[str, Any]:
        """Re-embed every stale entry now (no time budget); returns the model status.

        For the explicit ``python -m nachos_hermes.models reindex`` command; normal recall
        advances the work in bounded steps instead.
        """
        for s in select_scopes(self.scopes, scope):
            self._semantic.sync(s, budget_s=float("inf"))
        return self.model_status()

    def model_status(self) -> dict[str, Any]:
        """Configured embedding model, per-scope coverage, kept old models and calibration."""
        return self._semantic.status(self.scopes)

    # -- manifest ----------------------------------------------------------

    def manifest_entries(self, *, include_superseded: bool = False) -> list[Entry]:
        rows: list[Entry] = []
        for scope in self.scopes.values():
            entries = scope.store.list()
            metas = self._caches.meta(scope)
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
            for meta in self._caches.meta(scope).values()
            if meta.superseded_by
        )

    # -- recall ------------------------------------------------------------

    @property
    def _lexical_cache(self) -> dict[str, tuple[object, LexicalIndex]]:
        return self._caches.lexical_cache

    def _scopes(self, scope: str | None) -> list[Scope]:
        return select_scopes(self.scopes, scope)

    def get(self, key: str, scope: str | None = None) -> tuple[str, str] | None:
        """Return (scope_name, body) for an exact key, profile scope first."""
        for s in self._scopes(scope):
            body = s.store.get(key)
            if body is not None:
                s.index.record_recall(key)
                return s.name, body
        return None

    def get_entry(self, key: str, scope: str | None = None) -> dict[str, Any] | None:
        """Return full dict for an exact key, including a superseded_by pointer when superseded."""
        for s in self._scopes(scope):
            body = s.store.get(key)
            if body is None:
                continue
            s.index.record_recall(key)
            payload = entry_payload(key, s.name, body, s.store.get_meta(key) or EntryMeta())
            source = source_view(s.provenance([key]).get(key), evidence=True)
            if source:
                payload["source"] = source
            payload.update(link_fields(self._graphs[s.name], key))
            return payload
        return None

    def _lexical(self, scope: Scope, query: str, min_coverage: float = 0.0) -> list[str]:
        return self._caches.lexical(scope).search(query, min_coverage=min_coverage)

    def search(
        self,
        query: str = "",
        *,
        queries: Sequence[str] | None = None,
        limit: int = 10,
        scope: str | None = None,
        include_superseded: bool = False,
        entity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid recall; ``entity`` keeps only entries that mention it (query may be empty)."""
        return self._recall.search(
            query,
            queries=queries,
            limit=limit,
            scope=scope,
            include_superseded=include_superseded,
            entity=entity,
        )

    def search_with_confidence(
        self,
        query: str = "",
        *,
        queries: Sequence[str] | None = None,
        limit: int = 10,
        scope: str | None = None,
        include_superseded: bool = False,
        entity: str | None = None,
        mode: str = "on",
    ) -> dict[str, Any]:
        """Search and attach a per-query confidence assessment + explicit abstention signal."""
        return self._recall.search_with_confidence(
            query,
            queries=queries,
            limit=limit,
            scope=scope,
            include_superseded=include_superseded,
            entity=entity,
            mode=mode,
        )

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
        return self._recall.find_similar(
            title, summary, exclude=exclude, vector=vector, limit=limit
        )

    def duplicate_pairs(self, *, sync: bool = True) -> list[dict[str, Any]]:
        """Every pair above the duplicate threshold; ``sync=False`` never updates the index.

        Offline lexical audits use ``sync=False`` with no embedder. Normal callers
        retain lazy reconciliation of vectors and removed-entry telemetry.
        """
        return self._recall.duplicate_pairs(sync=sync)

    def _duplicate_threshold_for(self, scope: Scope, model_id: str) -> float:
        return self._recall.duplicate_threshold_for_scope(scope, model_id)

    def _tuning(self) -> RecallTuning:
        return RecallTuning(
            duplicate_threshold=self.duplicate_threshold,
            calibrate_duplicates=self.calibrate_duplicates,
            chunk_penalty=self.chunk_penalty,
            similarity_floor=self.similarity_floor,
            rerank_candidates=self.rerank_candidates,
            rerank_mode=self.rerank_mode,
            rerank_borderline=self.rerank_borderline,
            related_per_hit=self.related_per_hit,
            related_max=self.related_max,
        )

    # -- change log --------------------------------------------------------

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
        return history_rows({s.name: s.index for s in self._scopes(scope)}, key=key, limit=limit)

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
        current = snapshot(target.store, key)
        cur_meta = target.store.get_meta(key)
        cur_superseded = cur_meta.superseded_by if cur_meta is not None else None
        fp_before = target.store.fingerprint()
        restored = restore_from_change(target.store, target.index, change, current is not None)
        self._caches.invalidate(target.name)
        self._graphs[target.name].after_write(fp_before, key)
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
        key: str | None = None,
    ) -> Draft:
        """Normalise and gate a prospective write without touching any store (see ``writes``)."""
        return prepare_draft(
            self.scopes,
            self.redact,
            title=title,
            body=body,
            summary=summary,
            category=category,
            scope=scope,
            key=key,
        )

    def _apply_supersedes(self, scope: Scope, new_key: str, targets: list[str]) -> list[str]:
        applied: list[str] = []
        for old_key in targets:
            prev = snapshot(scope.store, old_key)
            if prev is None:
                continue
            old_meta = scope.store.get_meta(old_key)
            old_sup = old_meta.superseded_by if old_meta is not None else None
            self._log(scope, old_key, "supersede", prev, prev_superseded_by=old_sup)
            if scope.store.set_superseded_by(old_key, new_key):
                applied.append(old_key)
        if applied:
            self._caches.invalidate(scope.name)
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
        provenance: dict[str, Any] | None = None,
        key: str | None = None,
    ) -> dict[str, Any]:
        """Write an entry; ``key`` updates an existing key and ``provenance`` overrides manual."""
        draft = self.prepare(
            title=title, body=body, summary=summary, category=category, scope=scope, key=key
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
            if not self.reindexing:  # mid-switch writes are picked up by the re-embed
                chunk_vectors = self._semantic.embed(chunks)
                vector = chunk_vectors[0] if chunk_vectors else None
        similar = self.find_similar(title, summary, exclude=(scope, key), vector=vector)

        prev = snapshot(target.store, key)
        prev_meta = target.store.get_meta(key)
        unchanged = prev is not None and same_content(prev, (title, summary, category, body))
        if not unchanged:
            self._log(
                target,
                key,
                "update" if prev else "create",
                prev,
                prev_superseded_by=prev_meta.superseded_by if prev_meta else None,
            )
        fp_before = target.store.fingerprint()
        target.store.put(key, title=title, summary=summary, category=category, body=body)
        self._caches.refresh_after_put(
            target, fp_before, key, title=title, summary=summary, category=category
        )
        self._graphs[scope].after_write(fp_before, key, summary)
        embedder = self._semantic.active()
        if chunk_vectors is not None and embedder is not None:
            target.index.upsert_chunks(key, chunks, embedder.model_id, chunk_vectors)
        target.index.record_seen([key])
        self._record_provenance(target, key, provenance, unchanged=unchanged)

        applied = self._apply_supersedes(target, key, normalize_supersedes(supersedes, key))
        base_update_threshold = max(0.45, self.duplicate_threshold - 0.17)
        sem_threshold = (
            update_threshold_for(
                target.index,
                embedder.model_id,
                base_update_threshold,
                calibrate=self.calibrate_updates,
                similarity_floor=self.similarity_floor,
            )
            if vector is not None and embedder is not None
            else base_update_threshold
        )
        possible = possible_updates_for(
            target.store,
            self._caches.lexical(target),
            target.index,
            self._caches.meta(target),
            model_id=embedder.model_id if (vector is not None and embedder is not None) else None,
            vector=vector,
            self_key=key,
            title=title,
            summary=summary,
            body=body,
            already_superseded=set(applied),
            semantic_threshold=sem_threshold,
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
            **link_hints(
                self._graphs[target.name],
                key,
                title,
                summary,
                body,
                skip=set(applied),
                vector=vector,
                model_id=embedder.model_id if embedder is not None else None,
            ),
        )

    def _record_provenance(
        self, scope: Scope, key: str, provenance: dict[str, Any] | None, *, unchanged: bool
    ) -> None:
        """Store where ``key`` came from; never blocks a write.

        A manual re-save of identical content keeps the existing record (it may hold a capture
        quote); any other save replaces it, since the content it described has changed.
        """
        try:
            if provenance is None:
                if unchanged and scope.index.get_provenance(key) is not None:
                    return
                provenance = {"source": MANUAL, "session_id": self.session_id}
            row = dict(provenance)
            row["quote"] = self.redact(str(row.get("quote") or ""))[:MAX_QUOTE_CHARS]
            scope.index.set_provenance(key, **row)
        except Exception as e:
            logger.warning("Nachos provenance write failed: %s", e)

    def remove(self, key: str, scope: str | None = None) -> str | None:
        """Remove ``key``; returns the scope it was removed from, or None."""
        for s in self._scopes(scope):
            if s.store.get(key) is not None:
                meta = s.store.get_meta(key)
                self._log(
                    s,
                    key,
                    "remove",
                    snapshot(s.store, key),
                    prev_superseded_by=meta.superseded_by if meta else None,
                )
                fp_before = s.store.fingerprint()
                s.store.remove(key)
                self._caches.invalidate(s.name)
                s.index.delete(key)
                self._graphs[s.name].after_write(fp_before, key)
                return s.name
        return None

    # -- links -------------------------------------------------------------

    def link_counts(self) -> dict[str, int]:
        """Linked-neighbour counts per active entry (only entries that have any)."""
        return merged_link_counts(self._graphs)

    def link_report(self, scope: str = PROFILE) -> dict[str, Any]:
        """Broken links, orphan hubs and unlinked entries sharing entities (read-only, offline).

        Computed straight from the store, so it works on a read-only index and ignores the
        sidecar ``links`` table.
        """
        return link_report(self._graphs[scope])

    def link_supersede(self, old_key: str, new_key: str, scope: str = PROFILE) -> dict[str, Any]:
        """Mark ``old_key`` superseded by ``new_key`` (logged, so ``revert`` undoes it)."""
        target = self.scopes[scope]
        old_meta = target.store.get_meta(old_key)
        error = None
        if old_key == new_key:
            error = "an entry cannot supersede itself"
        elif target.store.get(old_key) is None or target.store.get(new_key) is None:
            error = "both entries must exist"
        elif old_meta is not None and old_meta.superseded_by:
            error = f"{old_key!r} is already superseded by {old_meta.superseded_by!r}"
        elif resolve_successor(new_key, self._caches.meta(target)) in (None, old_key):
            # new_key's chain must end at an active entry that is not old_key (no cycles)
            error = f"{new_key!r} is superseded back to {old_key!r} or has no active successor"
        if error:
            return {"error": error}
        return {"superseded": self._apply_supersedes(target, new_key, [old_key])}

    def unlink_supersede(
        self, key: str, expect: str | None = None, scope: str = PROFILE
    ) -> dict[str, Any]:
        """Clear ``key``'s supersede link (optionally only if it points at ``expect``)."""
        target = self.scopes[scope]
        meta = target.store.get_meta(key)
        current = meta.superseded_by if meta is not None else None
        if not current:
            return {"error": f"{key!r} is not superseded"}
        if expect is not None and current != expect:
            return {"error": f"{key!r} is superseded by {current!r}, not {expect!r}"}
        prev = snapshot(target.store, key)
        self._log(target, key, "supersede", prev, prev_superseded_by=current)
        fp_before = target.store.fingerprint()
        target.store.set_superseded_by(key, None)
        self._caches.invalidate(target.name)
        self._graphs[target.name].after_write(fp_before, key)
        return {"unlinked": key, "was": current}

    # -- hygiene -----------------------------------------------------------

    def unused(self, older_than_days: float, now: float | None = None) -> list[dict[str, Any]]:
        """Entries older than N days (first seen) never recalled or returned."""
        scope_map = {s.name: (s.store, s.index, self._caches.meta(s)) for s in self.scopes.values()}
        return unused_entries(scope_map, older_than_days=older_than_days, now=now)

    def superseded(
        self, older_than_days: float = 30, now: float | None = None
    ) -> list[dict[str, Any]]:
        """Superseded entries older than N days (candidates for removal)."""
        scope_map = {s.name: (s.store, self._caches.meta(s)) for s in self.scopes.values()}
        return superseded_entries(scope_map, older_than_days=older_than_days, now=now)

    def suspicious_supersedes(self) -> list[dict[str, Any]]:
        """Superseded links whose old and new entries differ in a facet qualifier."""
        scope_map = {s.name: (s.store, self._caches.meta(s)) for s in self.scopes.values()}
        return suspicious_supersedes(scope_map)

    def status(self) -> dict[str, Any]:
        model_id = self.embedder.model_id if self.embedder else None
        return {
            "embeddings": model_id or "off",
            "embeddings_error": self._semantic.error,
            "model_state": self.model_status()["state"],
            "scopes": {
                s.name: scope_status(s.store, s.index, self._caches.meta(s), model_id)
                for s in self.scopes.values()
            },
        }
