"""Model-quality diagnostics, separate from MemoryService fusion (stdlib only).

For one embedding setup it reports, on a query category (default ``paraphrase``):

* the cosine distribution (p10/p50/p90) of (query, expected entry) pairs and of
  (query, random entry) pairs,
* vector-only ranking quality: pure cosine search over all entries, no lexical
  list, no fusion, no similarity floor,
* duplicate-threshold behaviour: entry/entry cosine for near-duplicate sibling
  pairs versus all other pairs, with sibling recall and the false-flag rate at
  ``--threshold`` (before) and at ``max(--threshold, model p99)`` (after calibration).

    python -m bench.diagnose --command ./nachos-embeddings --model Xenova/bge-small-en-v1.5 --dtype q8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any

from bench.recall import (
    DATA_DIR,
    hit_at,
    load_dataset,
    mrr_at,
    percentile,
    rank_of,
    resolve_embedder,
)
from nachos_core.index import _normalize, _pair_profile, dot
from nachos_core.lexical import jaccard
from nachos_core.memory import embed_text
from nachos_core.store.md_store import slugify


def _spread(values: list[float]) -> list[float]:
    return [round(percentile(values, p), 3) for p in (10, 50, 90)]


def query_pairs(sims: list[list[float]], queries: list[dict], keys: list[str]) -> dict[str, Any]:
    """Cosine spread of expected vs random entries, and vector-only ranks, per query."""
    rnd = random.Random(1)  # noqa: S311 - reproducible sampling, not security
    pos: list[float] = []
    neg: list[float] = []
    ranks: list[int | None] = []
    for q, row in zip(queries, sims):
        expected = {keys.index(k) for k in q["expect"]}
        pos.append(max(row[i] for i in expected))
        others = [i for i in range(len(keys)) if i not in expected]
        neg += [row[i] for i in rnd.sample(others, min(20, len(others)))]
        top = [keys[i] for i in sorted(range(len(keys)), key=lambda i: -row[i])[:10]]
        ranks.append(rank_of(q["expect"], top, q.get("match", "any")))
    return {
        "n": len(queries),
        "expected_p10_p50_p90": _spread(pos),
        "random_p10_p50_p90": _spread(neg),
        "random_p99": round(percentile(neg, 99), 3),
        "vector_only": {
            "hit@1": round(hit_at(ranks, 1), 3),
            "hit@5": round(hit_at(ranks, 5), 3),
            "mrr@10": round(mrr_at(ranks, 10), 3),
        },
    }


def duplicate_pairs(
    vectors: list[list[float]], memories: list[dict], queries: list[dict], threshold: float
) -> dict[str, Any]:
    """Entry/entry cosine for near-duplicate siblings versus all other pairs."""
    keys = [slugify(m["title"]) for m in memories]
    near = sorted({keys.index(q["expect"][0]) for q in queries if q["category"] == "near-dup"})
    siblings = {
        (a, b)
        for a in near
        for b in near
        if a < b and jaccard(memories[a]["title"], memories[b]["title"]) >= 0.3
    }
    sib = [dot(vectors[a], vectors[b]) for a, b in siblings]
    rest = [
        dot(vectors[a], vectors[b])
        for a in range(len(keys))
        for b in range(a + 1, len(keys))
        if (a, b) not in siblings
    ]
    profile = _pair_profile(vectors)
    calibrated = max(threshold, profile["p99"])

    def rates(limit: float) -> dict[str, float]:
        return {
            "threshold": round(limit, 3),
            "sibling_recall": round(sum(s >= limit for s in sib) / len(sib), 3),
            "other_false_flag_rate": round(sum(s >= limit for s in rest) / len(rest), 4),
        }

    return {
        "sibling_pairs": len(sib),
        "sibling_p10_p50_p90": _spread(sib),
        "other_pairs_p50_p90_p99": [round(percentile(rest, p), 3) for p in (50, 90, 99)],
        "before": rates(threshold),
        "after_calibration": rates(calibrated),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench.diagnose", description=__doc__.split("\n\n")[0])
    ap.add_argument("--embeddings", default="nachos")
    ap.add_argument("--model", default="")
    ap.add_argument("--dtype", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--document-prefix", default="")
    ap.add_argument("--command", default="nachos-embeddings")
    ap.add_argument("--no-types", action="store_true")
    ap.add_argument("--category", default="paraphrase")
    ap.add_argument("--threshold", type=float, default=0.65, help="duplicate_threshold to evaluate")
    opts = ap.parse_args(argv)
    embedder, reason = resolve_embedder(opts.embeddings, opts)
    if embedder is None:
        print(f"skipped: {reason or 'no embedder (off)'}", file=sys.stderr)
        return 1
    memories, queries = load_dataset(DATA_DIR)
    keys = [slugify(m["title"]) for m in memories]
    try:
        docs = [embed_text(m["title"], m["summary"], m["body"]) for m in memories]
        vectors = [_normalize(v) for v in embedder.embed(docs, "document")]
        sel = [q for q in queries if q["category"] == opts.category]
        qvecs = [_normalize(v) for v in embedder.embed([q["query"] for q in sel], "query")]
        sims = [[dot(qv, d) for d in vectors] for qv in qvecs]
        report = {
            "model_id": embedder.model_id,
            "category": opts.category,
            "queries": query_pairs(sims, sel, keys),
            "duplicates": duplicate_pairs(vectors, memories, queries, opts.threshold),
        }
    finally:
        embedder.close()
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
