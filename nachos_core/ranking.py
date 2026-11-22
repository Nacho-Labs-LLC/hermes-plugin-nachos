"""Pure helpers for the final ranking stage: rerank passages and order blending.

No store, index or embedder access; ``MemoryService`` supplies the data.
"""

from __future__ import annotations

from typing import TypeVar

from .chunking import embed_chunks

T = TypeVar("T")

#: Characters of body sent to the cross-encoder per candidate (its own limit is in tokens).
PASSAGE_CHARS = 1500


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
