"""Safe, incremental re-embedding after an embedding model change.

Changing ``embedding_model`` (or dtype, pooling, ...) makes every stored vector
belong to a different vector space. Instead of re-embedding everything inside
the next recall:

* **Old vectors stay** until the new model covers every entry (the index keeps
  one row per ``(entry, chunk, model)``), so an interrupted switch loses nothing
  and switching back is instant.
* **Work is bounded**: more than ``BULK_ENTRIES`` stale entries are re-embedded
  in small, time-sized batches under a per-call budget, so no single recall or write
  waits for the whole index.
* **Recall stays available**: while a scope is not fully re-embedded the service
  answers from lexical matching alone; mixing partial new-model coverage into
  the ranking would be misleading.
* **Thresholds follow the model**: when the last entry is done the old model's
  vectors and calibration are dropped and the new model's similarity percentiles
  are recomputed (duplicate and update floors depend on them).

Stdlib only; no Hermes imports.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .index import MemoryIndex

logger = logging.getLogger(__name__)

#: At most this many stale entries are re-embedded inline (the normal incremental case:
#: a few new or edited entries). More is a bulk re-embed and runs in bounded steps.
BULK_ENTRIES = 24
#: Chunk texts in the first embedding call of a bulk step; later calls are sized to the time left.
FIRST_BATCH_TEXTS = 2
#: Upper bound on chunk texts per embedding call.
MAX_BATCH_TEXTS = 32
#: Seconds one recall/write spends advancing a bulk re-embed (at least one small batch always runs).
DEFAULT_BUDGET_S = 0.25

Embed = Callable[[list[str]], "list[list[float]] | None"]


def _take(keys: list[str], docs: Mapping[str, list[str]], texts: int) -> list[str]:
    """Leading keys whose chunk texts add up to about ``texts`` (always at least one key)."""
    taken: list[str] = []
    count = 0
    for key in keys:
        if taken and count + len(docs[key]) > texts:
            break
        taken.append(key)
        count += len(docs[key])
    return taken


def embed_stale(
    embed: Embed,
    index: MemoryIndex,
    model_id: str,
    docs: Mapping[str, list[str]],
    stale: list[str],
    *,
    budget_s: float,
    bulk: bool = False,
) -> int:
    """Embed stale entries and store their vectors; returns how many were completed.

    Up to ``BULK_ENTRIES`` run in one call unless ``bulk`` (a switch already under way keeps
    its budget to the end). Larger sets start with a small batch and size each
    next one to the time left in ``budget_s`` (``float("inf")`` runs to completion). Stops early, without
    raising, when ``embed`` returns None (the backend failed).
    """
    bulk = bulk or len(stale) > BULK_ENTRIES
    allowance = FIRST_BATCH_TEXTS  # chunk texts for the next call (bulk only)
    deadline = time.monotonic() + budget_s
    done = 0
    while done < len(stale):
        keys = stale[done:] if not bulk else _take(stale[done:], docs, allowance)
        texts = [text for key in keys for text in docs[key]]
        began = time.monotonic()
        vectors = embed(texts)
        if vectors is None:
            break
        offset = 0
        for key in keys:
            chunks = docs[key]
            index.upsert_chunks(key, chunks, model_id, vectors[offset : offset + len(chunks)])
            offset += len(chunks)
        done += len(keys)
        if not bulk:
            continue
        now = time.monotonic()
        if now >= deadline:
            break
        # Entries differ a lot in size (long bodies are many chunks), so time per text, not entry.
        per_text = max((now - began) / len(texts), 1e-4)
        room = min(deadline - now, MAX_BATCH_TEXTS * per_text)  # finite even for an inf budget
        allowance = max(1, min(MAX_BATCH_TEXTS, int(room / per_text)))
    return done


def finish(index: MemoryIndex, model_id: str) -> int:
    """Complete a switch: drop other models' vectors and recalibrate ``model_id``.

    Returns the number of old vector rows removed (0 when there was nothing to clean up).
    """
    if all(model == model_id for model in index.model_counts()):
        return 0
    removed = index.purge_other_models(model_id)
    index.similarity_profile(model_id, refresh=True)
    logger.info(
        "Nachos re-embedding finished for %s: removed %d old vectors, recalibrated thresholds.",
        model_id,
        removed,
    )
    return removed


def scope_models(
    index: MemoryIndex,
    model_id: str | None,
    entries: int,
    remaining: int,
) -> dict[str, Any]:
    """Per-scope embedding status: coverage by the configured model, kept old models, calibration."""
    counts = index.model_counts()
    current = counts.get(model_id, 0) if model_id else 0
    profile = index.similarity_profile(model_id) if model_id else None
    return {
        "entries": entries,
        "embedded": min(current, entries),
        "pending": max(0, entries - current) if model_id else 0,
        "reindexing": remaining > 0,
        "other_models": {m: n for m, n in sorted(counts.items()) if m != model_id},
        "calibration": (
            {k: round(v, 3) for k, v in profile.items() if k in ("p50", "p75", "p99")}
            if profile
            else None
        ),
    }
