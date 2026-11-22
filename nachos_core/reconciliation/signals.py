"""Collect audit signals from a read-only MemoryService for the findings generator."""

from __future__ import annotations

from ..lexical import LexicalIndex
from ..memory import PROFILE, MemoryService
from ..validity import LEXICAL_UPDATE_THRESHOLD
from .findings import AuditSignals


def collect_signals(service: MemoryService) -> AuditSignals:
    """Duplicate and related-record pairs (lexical), suspicious supersede links and per-entry age/link metadata.

    Never syncs the index or embeds: the offline audit stays deterministic and write-free.
    """
    scope = service.scopes[PROFILE]
    metas = {}
    for key, *_rest in scope.store.list():
        meta = scope.store.get_meta(key)
        metas[key] = (
            meta.updated_at if meta is not None else None,
            meta.superseded_by if meta is not None else None,
        )
    pairs = {
        frozenset((p["a"]["key"], p["b"]["key"])): p for p in service.duplicate_pairs(sync=False)
    }
    lexical = LexicalIndex.build(scope.store.list(), scope.store.bodies())
    for key, title, summary, _category in scope.store.list():
        for other, sim, _terms in lexical.similar_documents(
            title=title,
            summary=summary,
            body=scope.store.get(key) or "",
            threshold=LEXICAL_UPDATE_THRESHOLD,
            exclude={key},
        ):
            pairs.setdefault(
                frozenset((key, other)),
                {
                    "a": {"scope": PROFILE, "key": min(key, other)},
                    "b": {"scope": PROFILE, "key": max(key, other)},
                    "similarity": round(sim, 3),
                    "match": "related",
                },
            )
    ordered = sorted(pairs.values(), key=lambda p: (-p["similarity"], p["a"]["key"], p["b"]["key"]))
    return AuditSignals(
        duplicate_pairs=ordered,
        suspicious_supersedes=service.suspicious_supersedes(),
        metas=metas,
    )
