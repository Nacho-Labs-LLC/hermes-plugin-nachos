"""Read side of ``MemoryService``: hybrid search, confidence, and duplicate/similar lookups.

Stateless apart from the reranker's failure flag; everything it reads (scopes, graphs, caches,
the embedder) is handed in by the service, and the live tunables come from ``tuning()`` so a
setting changed after construction is honoured on the next call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .embeddings import QUERY
from .graph import ScopeGraph
from .linked import attach_related, entity_matches, entity_rows
from .ranking import (
    build_confidence_response,
    duplicate_threshold_for,
    find_duplicate_pairs,
    find_similar_entries,
    fuse_multi_query,
    hydrate_ranked,
    normalize_queries,
    rerank_rows,
    scope_ranked_hits,
)
from .reindex import SemanticRuntime
from .scope import Scope, ScopeCaches, select_scopes
from .validity import RECENCY_MIN_GAP_SECONDS, RECENCY_TIE_EPS, select_ranked

logger = logging.getLogger(__name__)

_RRF_K = 60
# Body chunks score their cosine minus this, so a chunk must beat the head by a margin: 0.05 wins
# paraphrase hit@1 back without hurting long-body queries, 0.12 starts to.
CHUNK_PENALTY = 0.05
# Hybrid recall only: a lexical hit must match at least this IDF-weighted share of the query's
# terms to take part in fusion. Weak partial matches ("service", "internal") are noise next to a
# semantic list, and RRF would otherwise let them displace or double-count real answers. Unlike
# an absolute cosine floor (which only suited MiniLM's score range) it is model-independent;
# 0.35-0.45 is a plateau on the bench fixture.
MIN_LEXICAL_COVERAGE = 0.4
_LEXICAL_DUP_THRESHOLD = 0.6


@dataclass(frozen=True)
class RecallTuning:
    """The ``MemoryService`` settings recall depends on, read fresh for every call."""

    duplicate_threshold: float
    calibrate_duplicates: bool
    chunk_penalty: float
    #: None = no floor; a float = absolute cosine; a profile key such as "p95" = that percentile
    #: of the model's own random-pair cosine (see ``MemoryIndex.similarity_profile``).
    similarity_floor: float | str | None
    rerank_candidates: int
    rerank_mode: str
    #: rerank="borderline": skip full reranking and only verify borderline queries.
    rerank_borderline: bool
    related_per_hit: int
    related_max: int


class Recall:
    def __init__(
        self,
        scopes: Mapping[str, Scope],
        graphs: Mapping[str, ScopeGraph],
        caches: ScopeCaches,
        semantic: SemanticRuntime,
        tuning: Callable[[], RecallTuning],
    ) -> None:
        self.scopes = scopes
        self._graphs = graphs
        self._caches = caches
        self._semantic = semantic
        self._settings = tuning
        self._rerank_error: str | None = None

    @property
    def tuning(self) -> RecallTuning:
        return self._settings()

    def _select_ranked(
        self,
        fused: dict[tuple[str, str], dict[str, Any]],
        *,
        limit: int,
        include_superseded: bool,
    ) -> list[tuple[tuple[str, str], dict[str, Any]]]:
        """Filter superseded entries (unless requested) and apply recency tie-breaking."""
        scope_names = {scope_name for scope_name, _key in fused}
        metas_by_scope = {name: self._caches.meta(self.scopes[name]) for name in scope_names}
        return select_ranked(
            fused,
            metas_by_scope,
            limit=limit,
            include_superseded=include_superseded,
            eps=RECENCY_TIE_EPS,
            min_gap_seconds=RECENCY_MIN_GAP_SECONDS,
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
            lexical_fn=self._caches.lexical,
            meta_fn=self._caches.meta,
            embedder=self._semantic.active(),
            rerank_borderline=self.tuning.rerank_borderline and self._reranking(),
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
        targets = select_scopes(self.scopes, scope)
        allowed = (
            entity_matches([self._graphs[t.name] for t in targets], entity_clean)
            if entity_clean
            else None
        )
        if not query_list:
            if allowed is None:
                return [], [], [], {}
            rows = entity_rows(allowed, self._graphs, include_superseded=include_superseded)
            hydrated = hydrate_ranked(
                rows[:limit], self.scopes, self._caches.meta, Scope.provenance
            )
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
        reranking = self._reranking() and not self.tuning.rerank_borderline
        pool_limit = max(limit, self.tuning.rerank_candidates) if reranking else limit
        ranked = self._select_ranked(fused, limit=pool_limit, include_superseded=include_superseded)
        if reranking:
            ranked = self._rerank(query_list, ranked, limit)
        chunks = {pair: int(row.get("chunk", 0)) for pair, row in ranked}
        results = self._related(
            hydrate_ranked(ranked, self.scopes, self._caches.meta, Scope.provenance)
        )
        return results, query_list, stats_list, chunks

    def _related(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return attach_related(
            results, self._graphs, self.tuning.related_per_hit, self.tuning.related_max
        )

    def _reranking(self) -> bool:
        embedder = self._semantic.active()
        return embedder is not None and embedder.supports_rerank and self._rerank_error is None

    def _rerank(
        self,
        queries: Sequence[str],
        ranked: list[tuple[tuple[str, str], dict[str, Any]]],
        limit: int,
    ) -> list[tuple[tuple[str, str], dict[str, Any]]]:
        """Reorder the top fused candidates with the embedder's cross-encoder (best effort)."""
        embedder = self._semantic.active()
        if embedder is None:
            return ranked[:limit]
        out, error = rerank_rows(
            embedder,
            queries,
            ranked,
            self.scopes,
            limit=limit,
            rerank_candidates=self.tuning.rerank_candidates,
            rerank_mode=self.tuning.rerank_mode,
        )
        if error is not None:
            self._rerank_error = error
            logger.warning("Nachos reranking disabled for this session (%s).", self._rerank_error)
        return out

    def _query_vectors(
        self, targets: list[Scope], query_list: list[str]
    ) -> list[list[float]] | None:
        """Batched query embeddings (after syncing target indexes), or None if lexical-only."""
        if not self._semantic.enabled or not query_list:
            return None
        for s in targets:
            self._semantic.sync(s)
        if self._semantic.reindexing:
            return None  # mid-switch: lexical only until every entry has the new model's vectors
        vectors = self._semantic.embed(query_list, QUERY)
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
        embedder = self._semantic.active()
        return scope_ranked_hits(
            self._caches.lexical(s),
            s.index,
            None if include_superseded else self._caches.meta(s),
            query,
            qvec,
            embedder.model_id if embedder is not None else None,
            limit=limit,
            min_lexical_coverage=MIN_LEXICAL_COVERAGE,
            chunk_penalty=self.tuning.chunk_penalty,
            similarity_floor=self.tuning.similarity_floor,
            stats=stats,
        )

    def duplicate_threshold_for_scope(self, s: Scope, model_id: str) -> float:
        return duplicate_threshold_for(
            s.index,
            model_id,
            self.tuning.duplicate_threshold,
            calibrate=self.tuning.calibrate_duplicates,
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
        embedder = self._semantic.active()
        model_id = embedder.model_id if embedder is not None else None
        scope_map = {
            s.name: (
                self._caches.lexical(s),
                s.index,
                self._caches.meta(s),
                self.duplicate_threshold_for_scope(s, model_id)
                if model_id is not None
                else self.tuning.duplicate_threshold,
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
                self._semantic.sync(s)
        embedder = self._semantic.active()
        model_id = embedder.model_id if embedder is not None else None
        threshold = self.tuning.duplicate_threshold
        if model_id is not None:
            for s in self.scopes.values():
                threshold = max(threshold, self.duplicate_threshold_for_scope(s, model_id))
        scope_map = {s.name: (s.store, s.index, self._caches.meta(s)) for s in self.scopes.values()}
        return find_duplicate_pairs(
            scope_map,
            model_id=model_id,
            duplicate_threshold=threshold,
            lexical_threshold=_LEXICAL_DUP_THRESHOLD,
        )
