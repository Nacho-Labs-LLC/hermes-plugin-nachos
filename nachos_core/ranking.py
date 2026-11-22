"""Pure helpers for the final ranking stage: rerank passages and order blending.

No store, index or embedder access; ``MemoryService`` supplies the data.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from .calibration import assess_confidence, format_search_response, is_borderline
from .chunking import embed_chunks
from .index import dot
from .lexical import jaccard
from .links import resolve_successor
from .provenance import source_view
from .validity import redirect_hits

if TYPE_CHECKING:
    from .embeddings import Embedder
    from .index import MemoryIndex
    from .lexical import LexicalIndex
    from .store.base import EntryMeta, MemoryStore

T = TypeVar("T")

#: Characters of body sent to the cross-encoder per candidate (its own limit is in tokens).
PASSAGE_CHARS = 1500
#: Random-pair cosine percentile used to calibrate ``duplicate_threshold`` and ``possible_updates``.
DUP_PERCENTILE = "p99"
UPDATE_PERCENTILE = "p99"
#: Max query variants fused in one multi-query search call.
MAX_SEARCH_QUERIES = 4
#: Top fused candidates re-scored per variant during multi-query reranking (keeps total pairs small).
MULTI_QUERY_RERANK_CANDIDATES = 5
#: Top results re-scored when optional borderline cross-encoder verification triggers.
BORDERLINE_VERIFY_CANDIDATES = 5


def normalize_queries(
    query: str = "",
    queries: Sequence[str] | None = None,
    *,
    max_queries: int = MAX_SEARCH_QUERIES,
) -> list[str]:
    """Combine ``query`` and optional ``queries`` into up to ``max_queries`` unique non-blank strings."""
    raw: list[str] = []
    if query:
        raw.append(query)
    if queries:
        raw.extend(queries)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        low = cleaned.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(cleaned)
        if len(out) >= max_queries:
            break
    return out


def rerank_passage(title: str, summary: str, body: str, chunk_no: int = 0) -> str:
    """Text a cross-encoder reads for one candidate: title, summary and the best-matching body part.

    ``chunk_no`` is the entry's best-scoring vector (0 = the head, so the start of the body).
    Chunk texts are recomputed from the body rather than stored; the split is deterministic.
    """
    chunks = embed_chunks(title, summary, body)
    # Chunks carry a title prefix line; drop it. Chunk 0 is the head, i.e. the body start.
    in_range = 0 < chunk_no < len(chunks)
    excerpt = chunks[chunk_no].split("\n", 1)[-1] if in_range else body
    return f"{title}\n{summary}\n{excerpt}"[:PASSAGE_CHARS]


#: Default final-order blend: fused rank contributes 1/(1 + rank/60) (RRF scaled so rank 0 = 1)
#: and the cross-encoder contributes its 0..1 relevance score, equally weighted.
DEFAULT_BLEND = "sum:1"


def blend_order(keys: list[T], rerank_scores: list[float], mode: str = DEFAULT_BLEND) -> list[T]:
    """Reorder ``keys`` (fused order) using ``rerank_scores`` (aligned, higher is better).

    ``rerank``: pure cross-encoder order, fused rank breaks ties. ``sum:<w>``: fused-rank
    contribution plus ``w`` times the rerank score, so a confident reranker can overturn the
    fused order while a lukewarm one only breaks near-ties. ``blend:<k>``: reciprocal-rank
    fusion of the rerank rank and the fused rank with constant ``k``.
    """
    if len(keys) != len(rerank_scores):
        raise ValueError("one rerank score per key is required")
    name, _, arg = mode.partition(":")
    if name == "sum":
        weight = float(arg or 1)
        totals = [60.0 / (60 + i) + weight * rerank_scores[i] for i in range(len(keys))]
        return [keys[i] for i in sorted(range(len(keys)), key=lambda i: (-totals[i], i))]
    by_score = sorted(range(len(keys)), key=lambda i: (-rerank_scores[i], i))
    if name == "rerank":
        return [keys[i] for i in by_score]
    k = float(arg or 10)
    rerank_rank = {i: r for r, i in enumerate(by_score)}
    blended = {i: 1.0 / (k + rerank_rank[i]) + 1.0 / (k + i) for i in range(len(keys))}
    return [keys[i] for i in sorted(range(len(keys)), key=lambda i: (-blended[i], i))]


def floor_for(index: MemoryIndex, model_id: str, setting: float | str | None) -> float:
    """Minimum cosine for a semantic hit: none, an absolute value, or a calibrated percentile."""
    if setting is None:
        return -1.0
    if isinstance(setting, (int, float)):
        return float(setting)
    profile = index.similarity_profile(model_id)
    return profile[setting] if profile is not None and setting in profile else -1.0


def duplicate_threshold_for(
    index: MemoryIndex,
    model_id: str,
    base_threshold: float,
    *,
    calibrate: bool,
) -> float:
    """``base_threshold``, raised to the model's random-pair ``DUP_PERCENTILE`` when calibrating."""
    if not calibrate:
        return base_threshold
    profile = index.similarity_profile(model_id)
    return max(base_threshold, profile[DUP_PERCENTILE]) if profile is not None else base_threshold


def update_threshold_for(
    index: MemoryIndex,
    model_id: str,
    base_threshold: float,
    *,
    calibrate: bool = True,
    similarity_floor: float | str | None = None,
) -> float:
    """Minimum head-vector cosine for ``possible_updates``, raised to the calibrated floor."""
    if not calibrate:
        return base_threshold
    profile = index.similarity_profile(model_id)
    if profile is None:
        return base_threshold
    return max(
        base_threshold,
        profile[UPDATE_PERCENTILE],
        floor_for(index, model_id, similarity_floor),
    )


def _build_passages(items: Sequence[tuple[str, str, int]], scopes: Mapping[str, Any]) -> list[str]:
    labels = {
        name: {k: (t, sm) for k, t, sm, _c in scopes[name].store.list()}
        for name in {scope_name for scope_name, _key, _ch in items}
    }
    passages: list[str] = []
    for scope_name, key, chunk_no in items:
        title, summary = labels[scope_name].get(key, (key, ""))
        body = scopes[scope_name].store.get(key) or ""
        passages.append(rerank_passage(title, summary, body, chunk_no))
    return passages


def _max_rerank_scores(
    embedder: Embedder, queries: Sequence[str], passages: list[str]
) -> tuple[list[float] | None, str | None]:
    best: list[float] | None = None
    for q_text in queries:
        try:
            scores = embedder.rerank(q_text, passages)
        except Exception as e:  # a reranker failure never costs the fused result
            return None, f"{type(e).__name__}: {e}"[:300]
        if scores is None:
            return None, None
        best = list(scores) if best is None else [max(a, b) for a, b in zip(best, scores)]
    return best, None


def rerank_rows(
    embedder: Embedder,
    query: str | Sequence[str],
    ranked: list[tuple[tuple[str, str], dict[str, Any]]],
    scopes: Mapping[str, Any],
    *,
    limit: int,
    rerank_candidates: int,
    rerank_mode: str,
) -> tuple[list[tuple[tuple[str, str], dict[str, Any]]], str | None]:
    """Reorder the top fused candidates with ``embedder``'s cross-encoder; returns ``(rows, error)``.

    For multi-query searches (``len(queries) > 1``), re-scores only the top
    ``MULTI_QUERY_RERANK_CANDIDATES`` fused entries across each variant and takes the max
    cross-encoder score per candidate so total pairs stay bounded (<= 20 for 4 variants).
    """
    queries = [query] if isinstance(query, str) else [q for q in query if q]
    if not queries:
        return ranked[:limit], None
    is_multi = len(queries) > 1
    cap = MULTI_QUERY_RERANK_CANDIDATES if is_multi else max(limit, rerank_candidates)
    candidates = ranked[:cap]
    if len(candidates) < 2:
        return ranked[:limit], None
    items = [(sc, k, int(row.get("chunk", 0))) for (sc, k), row in candidates]
    scores, error = _max_rerank_scores(embedder, queries, _build_passages(items, scopes))
    if error is not None or scores is None:
        return ranked[:limit], error
    order = blend_order(list(range(len(candidates))), scores, rerank_mode)
    reranked = [
        (candidates[i][0], {**candidates[i][1], "rerank": round(scores[i], 3)}) for i in order
    ]
    if is_multi and len(reranked) < limit:
        reranked.extend(ranked[len(candidates) : limit])
    return reranked[:limit], None


def scope_ranked_hits(
    lexical_index: LexicalIndex,
    index: MemoryIndex,
    metas: Mapping[str, EntryMeta] | None,
    query: str,
    qvec: list[float] | None,
    model_id: str | None,
    *,
    limit: int,
    min_lexical_coverage: float,
    chunk_penalty: float,
    similarity_floor: float | str | None,
    stats: dict[str, list[float]] | None = None,
) -> Iterator[tuple[str, int, str, float | None, int]]:
    """Yield ``(key, rank, how, similarity, chunk)`` for one scope, redirecting superseded hits."""
    offset = 1 if qvec is not None else 0
    coverage = min_lexical_coverage if qvec is not None else 0.0
    lex_scored = lexical_index.search_scored(query, min_coverage=coverage)
    lex_hits = redirect_hits(((k, sc, 0) for k, sc in lex_scored), metas, limit=limit * 3)
    if stats is not None:
        stats["lex"].extend(sc for _k, sc, _ch in lex_hits[:5] if sc is not None)
    for rank, (key, _sim, _chunk) in enumerate(lex_hits):
        yield key, rank + offset, "lexical", None, 0
    if qvec is not None and model_id is not None:
        extra = 0 if metas is None else sum(1 for m in metas.values() if m.superseded_by)
        raw_sem = index.search_chunks(
            qvec, model_id, top_k=limit * 3 + extra, chunk_penalty=chunk_penalty
        )
        floor = floor_for(index, model_id, similarity_floor)
        above = ((k, sim, ch) for k, sim, ch in raw_sem if sim >= floor)
        sem_hits = redirect_hits(above, metas, limit=limit * 3)
        if stats is not None:
            stats["sem"].extend(sim for _k, sim, _ch in sem_hits[:5] if sim is not None)
        for rank, (key, sim, chunk) in enumerate(sem_hits):
            yield key, rank, "semantic", sim, chunk


def _top_coverage(
    lexical_index: LexicalIndex,
    metas: Mapping[str, EntryMeta],
    query: str,
    key: str,
) -> float:
    best = lexical_index.oov_coverage(query, key)
    for cand_key, meta in metas.items():
        if meta.superseded_by and resolve_successor(cand_key, metas) == key:
            best = max(best, lexical_index.oov_coverage(query, cand_key))
    return best


def _assess_one_query(
    query: str,
    top: Mapping[str, Any],
    stats: Mapping[str, list[float]],
    *,
    top_lexical: LexicalIndex,
    top_metas: Mapping[str, EntryMeta],
    profile: Mapping[str, float] | None,
) -> dict[str, Any]:
    lex_cov = _top_coverage(top_lexical, top_metas, query, top["key"])
    lex_scores = sorted(stats.get("lex") or (), reverse=True)
    if len(lex_scores) >= 2 and lex_scores[0] > 0:
        lex_margin = (lex_scores[0] - lex_scores[min(4, len(lex_scores) - 1)]) / lex_scores[0]
    else:
        lex_margin = 1.0 if lex_scores else 0.0
    sem_scores = sorted(stats.get("sem") or (), reverse=True)
    sem_top1 = sem_scores[0] if sem_scores else None
    sem_topk = sem_scores[min(4, len(sem_scores) - 1)] if len(sem_scores) >= 2 else None
    return assess_confidence(
        has_results=True,
        lexical_coverage=lex_cov,
        lexical_margin=lex_margin,
        semantic_top1=sem_top1,
        semantic_topk=sem_topk,
        profile=profile if sem_top1 is not None else None,
        agreement="lexical+semantic" in str(top.get("match", "")),
        rerank_score=top.get("rerank"),
    )


def evaluate_search_confidence(
    queries: Sequence[str],
    results: list[dict[str, Any]],
    stats_per_query: Sequence[Mapping[str, list[float]]],
    *,
    top_lexical: LexicalIndex | None,
    top_index: MemoryIndex | None,
    top_metas: Mapping[str, EntryMeta] | None,
    model_id: str | None,
    mode: str = "on",
) -> dict[str, Any]:
    """Compute per-query confidence (best across query variants) and build the response."""
    primary = queries[0] if queries else ""
    if not results or not queries or top_lexical is None or top_index is None:
        payload = format_search_response(
            primary,
            results,
            assess_confidence(has_results=False, lexical_coverage=0.0),
            mode=mode,
        )
    else:
        top = results[0]
        has_sem = any(st.get("sem") for st in stats_per_query)
        profile = top_index.similarity_profile(model_id) if (model_id and has_sem) else None
        assessments = [
            _assess_one_query(
                q_text,
                top,
                q_stats,
                top_lexical=top_lexical,
                top_metas=top_metas or {},
                profile=profile,
            )
            for q_text, q_stats in zip(queries, stats_per_query)
        ]
        best = max(assessments, key=lambda a: float(a["confidence_score"]))
        payload = format_search_response(primary, results, best, mode=mode)
    if len(queries) > 1:
        payload["queries"] = list(queries)
    return payload


def verify_borderline_confidence(
    embedder: Embedder,
    queries: Sequence[str],
    results: list[dict[str, Any]],
    stats_per_query: Sequence[Mapping[str, list[float]]],
    payload: dict[str, Any],
    scopes: Mapping[str, Any],
    *,
    chunks: Mapping[tuple[str, str], int] | None = None,
    top_lexical: LexicalIndex | None,
    top_index: MemoryIndex | None,
    top_metas: Mapping[str, EntryMeta] | None,
    model_id: str | None,
    mode: str = "on",
) -> tuple[dict[str, Any], str | None]:
    """Re-score only borderline queries with the cross-encoder on the top few results."""
    has_sem = any(st.get("sem") for st in stats_per_query)
    if not results or not is_borderline(payload, semantic=has_sem):
        return payload, None
    top_n = results[:BORDERLINE_VERIFY_CANDIDATES]
    chunk_map = chunks or {}
    items = [
        (str(r["scope"]), str(r["key"]), int(chunk_map.get((str(r["scope"]), str(r["key"])), 0)))
        for r in top_n
    ]
    scores, error = _max_rerank_scores(embedder, queries, _build_passages(items, scopes))
    if error is not None or scores is None:
        return payload, error
    for row, sc in zip(top_n, scores):
        row["rerank"] = round(sc, 3)
    updated = evaluate_search_confidence(
        queries,
        results,
        stats_per_query,
        top_lexical=top_lexical,
        top_index=top_index,
        top_metas=top_metas,
        model_id=model_id,
        mode=mode,
    )
    updated["signals"]["verified"] = True
    return updated, None


def build_confidence_response(
    queries: Sequence[str],
    results: list[dict[str, Any]],
    stats_per_query: Sequence[Mapping[str, list[float]]],
    scopes: Mapping[str, Any],
    *,
    chunks: Mapping[tuple[str, str], int],
    lexical_fn: Any,
    meta_fn: Any,
    embedder: Embedder | None,
    rerank_borderline: bool,
    mode: str = "on",
) -> tuple[dict[str, Any], str | None]:
    """Assemble search confidence and optionally verify borderline hits with the cross-encoder."""
    top_scope = scopes[results[0]["scope"]] if results else None
    top_lex = lexical_fn(top_scope) if top_scope is not None else None
    top_idx = top_scope.index if top_scope is not None else None
    top_metas = meta_fn(top_scope) if top_scope is not None else None
    model_id = embedder.model_id if embedder is not None else None
    payload = evaluate_search_confidence(
        queries,
        results,
        stats_per_query,
        top_lexical=top_lex,
        top_index=top_idx,
        top_metas=top_metas,
        model_id=model_id,
        mode=mode,
    )
    if rerank_borderline and embedder is not None and results:
        return verify_borderline_confidence(
            embedder,
            queries,
            results,
            stats_per_query,
            payload,
            scopes,
            chunks=chunks,
            top_lexical=top_lex,
            top_index=top_idx,
            top_metas=top_metas,
            model_id=model_id,
            mode=mode,
        )
    return payload, None


def fuse_scope_hits(
    fused: dict[tuple[str, str], dict[str, Any]],
    scope_name: str,
    hits: Iterator[tuple[str, int, str, float | None, int]],
    *,
    rrf_k: int,
) -> None:
    """Accumulate one scope's ranked hits into the RRF ``fused`` map (keeping best similarity)."""
    for key, rank, how, similarity, chunk in hits:
        row = fused.setdefault((scope_name, key), {"score": 0.0, "match": []})
        row["score"] += 1.0 / (rrf_k + rank)
        if how not in row["match"]:
            row["match"].append(how)
        if similarity is not None:
            rounded = round(similarity, 3)
            if "similarity" not in row or rounded > row["similarity"]:
                row["similarity"] = rounded
                row["chunk"] = chunk


def fuse_multi_query(
    query_list: Sequence[str],
    qvecs: Sequence[list[float]] | None,
    targets: Sequence[Any],
    ranked_hits_fn: Any,
    *,
    limit: int,
    include_superseded: bool,
    collect_stats: bool,
    rrf_k: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, list[float]]]]:
    """Run ``ranked_hits_fn`` across all queries and scopes, fusing into one RRF map."""
    fused: dict[tuple[str, str], dict[str, Any]] = {}
    stats_list: list[dict[str, list[float]]] = []
    for idx, q_text in enumerate(query_list):
        qvec = qvecs[idx] if qvecs is not None else None
        q_stats: dict[str, list[float]] = {"lex": [], "sem": []}
        for s in targets:
            fuse_scope_hits(
                fused,
                s.name,
                ranked_hits_fn(
                    s,
                    q_text,
                    qvec,
                    limit,
                    include_superseded=include_superseded,
                    stats=q_stats if collect_stats else None,
                ),
                rrf_k=rrf_k,
            )
        stats_list.append(q_stats)
    return fused, stats_list


def hydrate_ranked(
    ranked: list[tuple[tuple[str, str], dict[str, Any]]],
    scopes: Mapping[str, Any],
    scope_meta: Any,
    provenance_fn: Any = None,
) -> list[dict[str, Any]]:
    """Turn fused ranking rows into result dicts, recording a hit per returned entry."""
    results: list[dict[str, Any]] = []
    hits_by_scope: dict[str, list[str]] = {}
    for (scope_name, key), row in ranked:
        s = scopes[scope_name]
        body = s.store.get(key)
        if body is None:
            continue
        hits_by_scope.setdefault(scope_name, []).append(key)
        result: dict[str, Any] = {
            "key": key,
            "scope": scope_name,
            "body": body,
            "match": "+".join(sorted(row["match"])),
        }
        if "similarity" in row:
            result["similarity"] = row["similarity"]
        if "rerank" in row:
            result["rerank"] = row["rerank"]
        meta = scope_meta(s).get(key)
        if meta is not None and meta.superseded_by:
            result["superseded_by"] = meta.superseded_by
        results.append(result)
    for scope_name, keys in hits_by_scope.items():
        scopes[scope_name].index.record_hits(keys)
    if provenance_fn is not None:
        sources = {n: provenance_fn(scopes[n], ks) for n, ks in hits_by_scope.items()}
        for result in results:
            source = source_view(sources.get(result["scope"], {}).get(result["key"]))
            if source:
                result["source"] = source
    return results


def find_similar_entries(
    scopes: Mapping[str, tuple[LexicalIndex, MemoryIndex, Mapping[str, EntryMeta], float]],
    title: str,
    summary: str,
    *,
    exclude: tuple[str, str] | None,
    vector: list[float] | None,
    model_id: str | None,
    lexical_threshold: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Existing active entries that look like the same memory, most similar first."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    label = f"{title} {summary}"
    for scope_name, (lex, index, metas, duplicate_threshold) in scopes.items():
        excluded_key = exclude[1] if exclude and exclude[0] == scope_name else None
        for key, doc_title, score in lex.similar_labels(
            label, threshold=lexical_threshold, exclude=excluded_key
        ):
            if metas.get(key) and metas[key].superseded_by:
                continue
            found[(scope_name, key)] = {
                "key": key,
                "scope": scope_name,
                "title": doc_title,
                "similarity": round(score, 3),
                "match": "lexical",
            }
        if vector is not None and model_id is not None:
            for key, sim in index.search(vector, model_id, top_k=limit + 1, head_only=True):
                if (
                    (scope_name, key) == exclude
                    or sim < duplicate_threshold
                    or (metas.get(key) and metas[key].superseded_by)
                ):
                    continue
                prev = found.get((scope_name, key))
                if prev is None or sim > prev["similarity"]:
                    doc = lex.docs.get(key)
                    found[(scope_name, key)] = {
                        "key": key,
                        "scope": scope_name,
                        "title": doc.title if doc is not None else key,
                        "similarity": round(sim, 3),
                        "match": "semantic",
                    }
    return sorted(found.values(), key=lambda d: -d["similarity"])[:limit]


def find_duplicate_pairs(
    scopes: Mapping[str, tuple[MemoryStore, MemoryIndex, Mapping[str, EntryMeta]]],
    *,
    model_id: str | None,
    duplicate_threshold: float,
    lexical_threshold: float,
) -> list[dict[str, Any]]:
    """Compute all active near-duplicate entry pairs across scopes."""
    flat: list[tuple[str, str, str, str]] = []
    vecs: dict[tuple[str, str], list[float]] = {}
    for scope_name, (store, index, metas) in scopes.items():
        flat.extend(
            (scope_name, k, t, sm)
            for k, t, sm, _c in store.list()
            if not (metas.get(k) and metas[k].superseded_by)
        )
        if model_id is not None:
            for key, vec in index.vectors(model_id).items():
                vecs[(scope_name, key)] = vec
    pairs: list[dict[str, Any]] = []
    for i, (sa, ka, ta, sma) in enumerate(flat):
        for sb, kb, tb, smb in flat[i + 1 :]:
            va, vb = vecs.get((sa, ka)), vecs.get((sb, kb))
            if va is not None and vb is not None:
                sim, how = dot(va, vb), "semantic"
                if sim < duplicate_threshold:
                    continue
            else:
                sim, how = jaccard(f"{ta} {sma}", f"{tb} {smb}"), "lexical"
                if sim < lexical_threshold:
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
