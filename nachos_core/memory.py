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
from collections.abc import Iterator, Sequence
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
from .graph import ScopeGraph
from .index import MemoryIndex
from .lexical import LexicalIndex
from .linked import (
    attach_related,
    entity_matches,
    entity_rows,
    link_fields,
    link_hints,
    link_report,
    merged_link_counts,
)
from .provenance import MANUAL, MAX_QUOTE_CHARS, source_view
from .quality import low_signal_reason
from .ranking import (
    DEFAULT_BLEND,
    build_confidence_response,
    duplicate_threshold_for,
    find_duplicate_pairs,
    find_similar_entries,
    fuse_multi_query,
    hydrate_ranked,
    normalize_queries,
    rerank_rows,
    scope_ranked_hits,
    update_threshold_for,
)
from .reindex import DEFAULT_BUDGET_S, embed_stale, finish, scope_models
from .store.base import Entry, EntryMeta, MemoryStore
from .store.md_store import slugify
from .support import refresh_after_put, same_content, snapshot
from .validity import (
    RECENCY_MIN_GAP_SECONDS,
    RECENCY_TIE_EPS,
    backfill_scope_timestamps,
    entry_payload,
    format_put_result,
    history_rows,
    normalize_supersedes,
    possible_updates_for,
    resolve_successor,
    restore_from_change,
    scope_status,
    select_ranked,
    superseded_entries,
    suspicious_supersedes,
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
# Body chunks score their cosine minus this, so a chunk must beat the head by a margin; 0.05 won
# paraphrase hit@1 back without hurting long-body on bench/ (0.12 starts to; see docs/benchmark.md).
CHUNK_PENALTY = 0.05
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
        self.embedder = embedder
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
        self._graphs = {
            name: ScopeGraph(
                name,
                s.store,
                s.index,
                lambda s=s: self._scope_meta(s),  # type: ignore[misc]
                lambda s=s: self._titles(s),  # type: ignore[misc]
                lambda s=s: self._lexical_index(s),  # type: ignore[misc]
            )
            for name, s in scopes.items()
        }
        self._rerank_error: str | None = None
        #: Seconds one recall or write may spend advancing a bulk re-embed (see ``reindex``).
        self.reindex_budget_s = reindex_budget_s
        #: Stale entries left per scope after the last sync; non-zero = semantic recall is paused.
        self._reindex_remaining: dict[str, int] = {}
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
        self._graphs.clear()
        for scope in self.scopes.values():
            scope.close()
        if self.embedder is not None:
            with contextlib.suppress(Exception):
                self.embedder.close()

    # -- embeddings --------------------------------------------------------

    @property
    def semantic_enabled(self) -> bool:
        return self.embedder is not None and self.embedder_error is None

    @property
    def reindexing(self) -> bool:
        """True while a model switch is still re-embedding entries (recall is lexical meanwhile)."""
        return any(self._reindex_remaining.values())

    def _active_embedder(self) -> Embedder | None:
        """The embedder when semantic search is usable right now, else None.

        None also while a model switch is mid-way: vectors of two models must not be mixed, so
        recall, duplicate checks and update hints fall back to lexical matching until done.
        """
        return self.embedder if self.semantic_enabled and not self.reindexing else None

    def _embed(self, texts: list[str], input_type: str = DOCUMENT) -> list[list[float]] | None:
        embedder = self.embedder if self.semantic_enabled else None
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

    def sync_index(self, scope: Scope, *, budget_s: float | None = None) -> int:
        """Embed entries that are new, edited, or embedded with another model.

        A handful of stale entries are embedded inline. More than ``reindex.BULK_ENTRIES``
        (a model switch, a first run on an existing store) is re-embedded in small batches
        under ``budget_s`` (default ``reindex_budget_s``) per call: see :mod:`nachos_core.reindex`.
        """
        entries = scope.store.list()
        scope.index.retain(key for key, *_ in entries)
        embedder = self.embedder if self.semantic_enabled else None
        if embedder is None:
            return 0
        embedder.warm_up()  # may refine model_id (e.g. CLI pooling/prefix) before it is compared
        bodies = scope.store.bodies()
        docs = {
            key: embed_chunks(title, summary, bodies.get(key, ""))
            for key, title, summary, _category in entries
        }
        stale = scope.index.stale_keys(docs, embedder.model_id)
        done = embed_stale(
            lambda texts: self._embed(texts, DOCUMENT),
            scope.index,
            embedder.model_id,
            docs,
            stale,
            budget_s=self.reindex_budget_s if budget_s is None else budget_s,
            bulk=self._reindex_remaining.get(scope.name, 0) > 0,
        )
        remaining = len(stale) - done
        self._reindex_remaining[scope.name] = remaining
        if remaining == 0:
            finish(scope.index, embedder.model_id)
        return done

    def reindex(self, *, scope: str | None = None) -> dict[str, Any]:
        """Re-embed every stale entry now (no time budget); returns the model status.

        For the explicit ``python -m nachos_hermes.models reindex`` command; normal recall
        advances the work in bounded steps instead.
        """
        for s in self._scopes_for(scope):
            self.sync_index(s, budget_s=float("inf"))
        return self.model_status()

    def model_status(self) -> dict[str, Any]:
        """Configured embedding model, per-scope coverage, kept old models and calibration."""
        embedder = self.embedder
        model_id = embedder.model_id if embedder else None
        if embedder is None:
            state = "off"
        elif self.embedder_error:
            state = "error"
        else:
            state = "reindexing" if self.reindexing else "ready"
        return {
            "model": model_id,
            "state": state,
            "error": self.embedder_error,
            "scopes": {
                s.name: scope_models(
                    s.index,
                    model_id,
                    len(s.store.list()),
                    self._reindex_remaining.get(s.name, 0),
                )
                for s in self.scopes.values()
            },
        }

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
            payload = entry_payload(key, s.name, body, s.store.get_meta(key) or EntryMeta())
            source = source_view(self._provenance(s, [key]).get(key), evidence=True)
            if source:
                payload["source"] = source
            payload.update(link_fields(self._graphs[s.name], key))
            return payload
        return None

    @staticmethod
    def _provenance(scope: Scope, keys: list[str]) -> dict[str, dict[str, Any]]:
        try:
            return scope.index.get_provenance_many(keys)
        except Exception as e:  # provenance is advisory; never break a read
            logger.debug("Nachos provenance read failed: %s", e)
            return {}

    def _titles(self, s: Scope) -> dict[str, str]:
        return {key: doc.title for key, doc in self._lexical_index(s).docs.items()}

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
        query: str = "",
        *,
        queries: Sequence[str] | None = None,
        limit: int = 10,
        scope: str | None = None,
        include_superseded: bool = False,
        entity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid recall; ``entity`` keeps only entries that mention it (query may be empty)."""
        results, _query_list, _stats, _chunks = self._run_search(
            query,
            queries=queries,
            limit=limit,
            scope=scope,
            include_superseded=include_superseded,
            entity=entity,
        )
        return results

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
        entity_clean = (entity or "").strip()
        results, query_list, stats, chunks = self._run_search(
            query,
            queries=queries,
            limit=limit,
            scope=scope,
            include_superseded=include_superseded,
            entity=entity_clean or None,
            collect_stats=True,
        )
        if not query_list and entity_clean:
            confident = bool(results)
            payload: dict[str, Any] = {
                "query": "",
                "entity": entity_clean,
                "confident": confident,
                "confidence": "high" if confident else "none",
                "confidence_score": 1.0 if confident else 0.0,
                "signals": {"entity": entity_clean},
                "count": len(results),
                "results": results,
            }
            if not confident:
                payload["no_confident_match"] = True
                payload["note"] = "No matching memory found."
            return payload
        payload, error = build_confidence_response(
            query_list,
            results,
            stats,
            self.scopes,
            chunks=chunks,
            lexical_fn=self._lexical_index,
            meta_fn=self._scope_meta,
            embedder=self._active_embedder(),
            rerank_borderline=self.rerank_borderline and self._reranking(),
            mode=mode,
        )
        if entity_clean:
            payload["entity"] = entity_clean
        if error is not None:
            self._rerank_error = error
        return payload

    def _run_search(
        self,
        query: str = "",
        *,
        queries: Sequence[str] | None = None,
        limit: int = 10,
        scope: str | None = None,
        include_superseded: bool = False,
        entity: str | None = None,
        collect_stats: bool = False,
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
        list[dict[str, list[float]]],
        dict[tuple[str, str], int],
    ]:
        query_list = normalize_queries(query, queries)
        entity_clean = (entity or "").strip()
        targets = self._scopes_for(scope)
        allowed = (
            entity_matches([self._graphs[t.name] for t in targets], entity_clean)
            if entity_clean
            else None
        )
        if not query_list:
            if allowed is None:
                return [], [], [], {}
            rows = entity_rows(allowed, self._graphs, include_superseded=include_superseded)
            hydrated = hydrate_ranked(rows[:limit], self.scopes, self._scope_meta, self._provenance)
            return self._related(hydrated), [], [], {}
        qvecs = self._query_vectors(targets, query_list)
        depth = limit if allowed is None else limit * 5
        fused, stats_list = fuse_multi_query(
            query_list,
            qvecs,
            targets,
            self._ranked_hits,
            limit=depth,
            include_superseded=include_superseded,
            collect_stats=collect_stats,
            rrf_k=_RRF_K,
        )
        if allowed is not None:
            fused = {k: v for k, v in fused.items() if k in allowed}
        reranking = self._reranking() and not self.rerank_borderline
        pool_limit = max(limit, self.rerank_candidates) if reranking else limit
        ranked = self._select_ranked(fused, limit=pool_limit, include_superseded=include_superseded)
        if reranking:
            ranked = self._rerank(query_list, ranked, limit)
        chunks = {pair: int(row.get("chunk", 0)) for pair, row in ranked}
        results = self._related(
            hydrate_ranked(ranked, self.scopes, self._scope_meta, self._provenance)
        )
        return results, query_list, stats_list, chunks

    def _related(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return attach_related(results, self._graphs, self.related_per_hit, self.related_max)

    def _reranking(self) -> bool:
        embedder = self._active_embedder()
        return embedder is not None and embedder.supports_rerank and self._rerank_error is None

    def _rerank(
        self,
        queries: Sequence[str],
        ranked: list[tuple[tuple[str, str], dict[str, Any]]],
        limit: int,
    ) -> list[tuple[tuple[str, str], dict[str, Any]]]:
        """Reorder the top fused candidates with the embedder's cross-encoder (best effort)."""
        embedder = self._active_embedder()
        if embedder is None:
            return ranked[:limit]
        out, error = rerank_rows(
            embedder,
            queries,
            ranked,
            self.scopes,
            limit=limit,
            rerank_candidates=self.rerank_candidates,
            rerank_mode=self.rerank_mode,
        )
        if error is not None:
            self._rerank_error = error
            logger.warning("Nachos reranking disabled for this session (%s).", self._rerank_error)
        return out

    def _query_vectors(
        self, targets: list[Scope], query_list: list[str]
    ) -> list[list[float]] | None:
        """Batched query embeddings (after syncing target indexes), or None if lexical-only."""
        if not self.semantic_enabled or not query_list:
            return None
        for s in targets:
            self.sync_index(s)
        if self.reindexing:
            return None  # mid-switch: lexical only until every entry has the new model's vectors
        vectors = self._embed(query_list, QUERY)
        return vectors if vectors and len(vectors) == len(query_list) else None

    def _ranked_hits(
        self,
        s: Scope,
        query: str,
        qvec: list[float] | None,
        limit: int,
        *,
        include_superseded: bool = False,
        stats: dict[str, list[float]] | None = None,
    ) -> Iterator[tuple[str, int, str, float | None, int]]:
        """(key, rank, how, similarity, best chunk_no) hits for one scope, lexical then semantic."""
        embedder = self._active_embedder()
        return scope_ranked_hits(
            self._lexical_index(s),
            s.index,
            None if include_superseded else self._scope_meta(s),
            query,
            qvec,
            embedder.model_id if embedder is not None else None,
            limit=limit,
            min_lexical_coverage=_MIN_LEXICAL_COVERAGE,
            chunk_penalty=self.chunk_penalty,
            similarity_floor=self.similarity_floor,
            stats=stats,
        )

    def _duplicate_threshold_for(self, s: Scope, model_id: str) -> float:
        return duplicate_threshold_for(
            s.index, model_id, self.duplicate_threshold, calibrate=self.calibrate_duplicates
        )

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
        model_id = embedder.model_id if embedder is not None else None
        scope_map = {
            s.name: (
                self._lexical_index(s),
                s.index,
                self._scope_meta(s),
                self._duplicate_threshold_for(s, model_id)
                if model_id is not None
                else self.duplicate_threshold,
            )
            for s in self.scopes.values()
        }
        return find_similar_entries(
            scope_map,
            title,
            summary,
            exclude=exclude,
            vector=vector,
            model_id=model_id,
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
        model_id = embedder.model_id if embedder is not None else None
        threshold = self.duplicate_threshold
        if model_id is not None:
            for s in self.scopes.values():
                threshold = max(threshold, self._duplicate_threshold_for(s, model_id))
        scope_map = {s.name: (s.store, s.index, self._scope_meta(s)) for s in self.scopes.values()}
        return find_duplicate_pairs(
            scope_map,
            model_id=model_id,
            duplicate_threshold=threshold,
            lexical_threshold=_LEXICAL_DUP_THRESHOLD,
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
        current = snapshot(target.store, key)
        cur_meta = target.store.get_meta(key)
        cur_superseded = cur_meta.superseded_by if cur_meta is not None else None
        fp_before = target.store.fingerprint()
        restored = restore_from_change(target.store, target.index, change, current is not None)
        self._invalidate_lexical(target.name)
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
        draft = Draft(
            title, summary, category, body, key or (slugify(title) if title else ""), scope
        )
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
            prev = snapshot(scope.store, old_key)
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
                chunk_vectors = self._embed(chunks, DOCUMENT)
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
        if not refresh_after_put(
            self._lexical_cache,
            self._meta_cache,
            target.store,
            scope,
            fp_before,
            key,
            title=title,
            summary=summary,
            category=category,
        ):
            self._invalidate_lexical(scope)
        self._graphs[scope].after_write(fp_before, key, summary)
        embedder = self._active_embedder()
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
        for s in self._scopes_for(scope):
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
                self._invalidate_lexical(s.name)
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
        elif resolve_successor(new_key, self._scope_meta(target)) in (None, old_key):
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
        self._invalidate_lexical(target.name)
        self._graphs[target.name].after_write(fp_before, key)
        return {"unlinked": key, "was": current}

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

    def suspicious_supersedes(self) -> list[dict[str, Any]]:
        """Superseded links whose old and new entries differ in a facet qualifier."""
        scope_map = {s.name: (s.store, self._scope_meta(s)) for s in self.scopes.values()}
        return suspicious_supersedes(scope_map)

    def status(self) -> dict[str, Any]:
        model_id = self.embedder.model_id if self.embedder else None
        return {
            "embeddings": model_id or "off",
            "embeddings_error": self.embedder_error,
            "model_state": self.model_status()["state"],
            "scopes": {
                s.name: scope_status(s.store, s.index, self._scope_meta(s), model_id)
                for s in self.scopes.values()
            },
        }
