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

``SemanticRuntime`` owns the embedder and this bookkeeping for ``MemoryService``.

Stdlib only; no Hermes imports.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from .chunking import embed_chunks
from .embeddings import DOCUMENT, Embedder

if TYPE_CHECKING:
    from .index import MemoryIndex
    from .scope import Scope

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


class SemanticRuntime:
    """The embedder, its failure state and per-scope re-embed progress.

    An embedder error disables semantic matching for the rest of the process (``error``); callers
    then fall back to lexical matching. ``remaining`` holds the stale entries left per scope after
    the last sync: non-zero means a model switch is mid-way and semantic recall is paused.
    """

    def __init__(self, embedder: Embedder | None, reindex_budget_s: float = DEFAULT_BUDGET_S):
        self.embedder = embedder
        self.error: str | None = None
        #: Seconds one recall or write may spend advancing a bulk re-embed.
        self.reindex_budget_s = reindex_budget_s
        self.remaining: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        return self.embedder is not None and self.error is None

    @property
    def reindexing(self) -> bool:
        """True while a model switch is still re-embedding entries (recall is lexical meanwhile)."""
        return any(self.remaining.values())

    def active(self) -> Embedder | None:
        """The embedder when semantic search is usable right now, else None.

        None also while a model switch is mid-way: vectors of two models must not be mixed, so
        recall, duplicate checks and update hints fall back to lexical matching until done.
        """
        return self.embedder if self.enabled and not self.reindexing else None

    def embed(self, texts: list[str], input_type: str = DOCUMENT) -> list[list[float]] | None:
        embedder = self.embedder if self.enabled else None
        if embedder is None or not texts:
            return None
        try:
            return embedder.embed(texts, input_type)
        except Exception as e:  # any backend failure degrades to lexical
            self.error = f"{type(e).__name__}: {e}"[:300]
            logger.warning(
                "Nachos embeddings disabled for this session (%s); recall falls "
                "back to lexical matching.",
                self.error,
            )
            return None

    def sync(self, scope: Scope, *, budget_s: float | None = None) -> int:
        """Embed entries that are new, edited, or embedded with another model.

        A handful of stale entries are embedded inline. More than ``BULK_ENTRIES`` (a model
        switch, a first run on an existing store) is re-embedded in small batches under
        ``budget_s`` (default ``reindex_budget_s``) per call. Also drops index rows of entries
        that left the store.
        """
        entries = scope.store.list()
        scope.index.retain(key for key, *_ in entries)
        embedder = self.embedder if self.enabled else None
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
            self.embed,
            scope.index,
            embedder.model_id,
            docs,
            stale,
            budget_s=self.reindex_budget_s if budget_s is None else budget_s,
            bulk=self.remaining.get(scope.name, 0) > 0,
        )
        remaining = len(stale) - done
        self.remaining[scope.name] = remaining
        if remaining == 0:
            finish(scope.index, embedder.model_id)
        return done

    def status(self, scopes: Mapping[str, Scope]) -> dict[str, Any]:
        """Configured embedding model, per-scope coverage, kept old models and calibration."""
        embedder = self.embedder
        model_id = embedder.model_id if embedder else None
        if embedder is None:
            state = "off"
        elif self.error:
            state = "error"
        else:
            state = "reindexing" if self.reindexing else "ready"
        return {
            "model": model_id,
            "state": state,
            "error": self.error,
            "scopes": {
                s.name: scope_models(
                    s.index, model_id, len(s.store.list()), self.remaining.get(s.name, 0)
                )
                for s in scopes.values()
            },
        }
