"""Pure helpers for the final ranking stage: rerank passages and order blending.

No store, index or embedder access; ``MemoryService`` supplies the data.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from .chunking import embed_chunks
from .validity import redirect_hits

if TYPE_CHECKING:
    from .embeddings import Embedder
    from .index import MemoryIndex
    from .lexical import LexicalIndex
    from .store.base import EntryMeta

T = TypeVar("T")

#: Characters of body sent to the cross-encoder per candidate (its own limit is in tokens).
PASSAGE_CHARS = 1500
#: Random-pair cosine percentile used to calibrate ``duplicate_threshold``.
DUP_PERCENTILE = "p99"


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
) -> Iterator[tuple[str, int, str, float | None, int]]:
    """Yield ``(key, rank, how, similarity, chunk)`` for one scope, redirecting superseded hits."""
    offset = 1 if qvec is not None else 0
    coverage = min_lexical_coverage if qvec is not None else 0.0
    lex_hits = redirect_hits(
        ((k, None, 0) for k in lexical_index.search(query, min_coverage=coverage)),
        metas,
        limit=limit * 3,
    )
    for rank, (key, _sim, _chunk) in enumerate(lex_hits):
        yield key, rank + offset, "lexical", None, 0
    if qvec is not None and model_id is not None:
        extra = 0 if metas is None else sum(1 for m in metas.values() if m.superseded_by)
        raw_sem = index.search_chunks(
            qvec, model_id, top_k=limit * 3 + extra, chunk_penalty=chunk_penalty
        )
        floor = floor_for(index, model_id, similarity_floor)
        above = ((k, sim, ch) for k, sim, ch in raw_sem if sim >= floor)
        for rank, (key, sim, chunk) in enumerate(redirect_hits(above, metas, limit=limit * 3)):
            yield key, rank, "semantic", sim, chunk
