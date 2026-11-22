"""Pure helpers for the final ranking stage: rerank passages and order blending.

No store, index or embedder access; ``MemoryService`` supplies the data.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from .calibration import assess_confidence, format_search_response
from .chunking import embed_chunks
from .index import dot
from .lexical import jaccard
from .validity import redirect_hits, resolve_successor

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


def rerank_rows(
    embedder: Embedder,
    query: str,
    ranked: list[tuple[tuple[str, str], dict[str, Any]]],
    scopes: Mapping[str, Any],
    *,
    limit: int,
    rerank_candidates: int,
    rerank_mode: str,
) -> tuple[list[tuple[tuple[str, str], dict[str, Any]]], str | None]:
    """Reorder the top fused candidates with ``embedder``'s cross-encoder; returns ``(rows, error)``."""
    candidates = ranked[: max(limit, rerank_candidates)]
    if len(candidates) < 2:
        return ranked[:limit], None
    labels = {
        name: {k: (t, sm) for k, t, sm, _c in scopes[name].store.list()}
        for name in {scope_name for (scope_name, _key), _row in candidates}
    }
    passages: list[str] = []
    for (scope_name, key), row in candidates:
        title, summary = labels[scope_name].get(key, (key, ""))
        body = scopes[scope_name].store.get(key) or ""
        passages.append(rerank_passage(title, summary, body, int(row.get("chunk", 0))))
    try:
        scores = embedder.rerank(query, passages)
    except Exception as e:  # a reranker failure never costs the fused result
        return ranked[:limit], f"{type(e).__name__}: {e}"[:300]
    if scores is None:
        return ranked[:limit], None
    order = blend_order(list(range(len(candidates))), scores, rerank_mode)
    out = [
        (candidates[i][0], {**candidates[i][1], "rerank": round(scores[i], 3)})
        for i in order[:limit]
    ]
    return out, None


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


def evaluate_search_confidence(
    query: str,
    results: list[dict[str, Any]],
    stats: Mapping[str, list[float]],
    *,
    top_lexical: LexicalIndex | None,
    top_index: MemoryIndex | None,
    top_metas: Mapping[str, EntryMeta] | None,
    model_id: str | None,
) -> dict[str, Any]:
    """Compute per-query confidence and assemble the structured search response."""
    if not results or top_lexical is None or top_index is None:
        return format_search_response(
            query, results, assess_confidence(has_results=False, lexical_coverage=0.0)
        )
    top = results[0]
    lex_cov = _top_coverage(top_lexical, top_metas or {}, query, top["key"])
    lex_scores = sorted(stats.get("lex") or (), reverse=True)
    if len(lex_scores) >= 2 and lex_scores[0] > 0:
        lex_margin = (lex_scores[0] - lex_scores[min(4, len(lex_scores) - 1)]) / lex_scores[0]
    else:
        lex_margin = 1.0 if lex_scores else 0.0
    sem_scores = sorted(stats.get("sem") or (), reverse=True)
    sem_top1 = sem_scores[0] if sem_scores else None
    sem_topk = sem_scores[min(4, len(sem_scores) - 1)] if len(sem_scores) >= 2 else None
    profile = (
        top_index.similarity_profile(model_id) if (model_id and sem_top1 is not None) else None
    )
    assessment = assess_confidence(
        has_results=True,
        lexical_coverage=lex_cov,
        lexical_margin=lex_margin,
        semantic_top1=sem_top1,
        semantic_topk=sem_topk,
        profile=profile,
        agreement="lexical+semantic" in str(top.get("match", "")),
        rerank_score=top.get("rerank"),
    )
    return format_search_response(query, results, assessment)


def fuse_scope_hits(
    fused: dict[tuple[str, str], dict[str, Any]],
    scope_name: str,
    hits: Iterator[tuple[str, int, str, float | None, int]],
    *,
    rrf_k: int,
) -> None:
    """Accumulate one scope's ranked hits into the RRF ``fused`` map."""
    for key, rank, how, similarity, chunk in hits:
        row = fused.setdefault((scope_name, key), {"score": 0.0, "match": []})
        row["score"] += 1.0 / (rrf_k + rank)
        row["match"].append(how)
        if similarity is not None:
            row["similarity"] = round(similarity, 3)
            row["chunk"] = chunk


def hydrate_ranked(
    ranked: list[tuple[tuple[str, str], dict[str, Any]]],
    scopes: Mapping[str, Any],
    scope_meta: Any,
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
            "match": "+".join(row["match"]),
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
