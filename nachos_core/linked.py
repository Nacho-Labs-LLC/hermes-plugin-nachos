"""Link- and entity-aware recall helpers behind ``MemoryService`` (kept out of memory.py).

Everything takes ``ScopeGraph`` objects (one per scope, with store, sidecar index and lazy
metadata/title lookups), so these functions never import the service.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .graph import ScopeGraph, describe
from .links import (
    EntityIndex,
    broken_links,
    lexical_similarities,
    orphan_hubs,
    parse_links,
    suggest_links,
    unlinked_related,
)

Ref = tuple[str, str]  # (scope name, key)
Fused = dict[Ref, dict[str, Any]]


def link_fields(view: ScopeGraph, key: str) -> dict[str, Any]:
    """``links`` / ``backlinks`` (``{key, title}`` rows) and ``broken_links`` for an entry."""
    near = view.neighbors(key)
    titles = view.titles()
    out: dict[str, Any] = {}
    if near["links"]:
        out["links"] = describe(near["links"], titles)
    if near["backlinks"]:
        out["backlinks"] = describe(near["backlinks"], titles)
    if near["broken"]:
        out["broken_links"] = near["broken"]
    return out


def _word_matches(view: ScopeGraph, word: str) -> list[str]:
    pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
    titles = view.titles()
    return sorted(
        key
        for key, body in view.store.bodies().items()
        if pattern.search(body) or pattern.search(titles.get(key, ""))
    )


def entity_matches(views: list[ScopeGraph], entity: str) -> set[Ref]:
    """Entries mentioning ``entity``: the entity index, else a whole-word text match."""
    found: set[Ref] = set()
    for view in views:
        keys = view.entities().entries_for(entity) or _word_matches(view, entity)
        found.update((view.name, k) for k in keys)
    return found


def entity_rows(
    allowed: set[Ref], views: Mapping[str, ScopeGraph], *, include_superseded: bool
) -> list[tuple[Ref, dict[str, Any]]]:
    """Fused-style rows for an entity lookup without a query: active entries, by scope and key."""
    rows: list[tuple[Ref, dict[str, Any]]] = []
    for ref in sorted(allowed):
        meta = views[ref[0]].metas().get(ref[1])
        if meta is not None and meta.superseded_by and not include_superseded:
            continue
        rows.append((ref, {"score": 0.0, "match": ["entity"]}))
    return rows


def attach_related(
    results: list[dict[str, Any]], views: Mapping[str, ScopeGraph], per_hit: int, total: int
) -> list[dict[str, Any]]:
    """Add up to ``per_hit`` one-hop neighbours to each result, ``total`` overall, never a hit.

    Order is deterministic: outgoing links in body order, then backlinks by key. Results are
    walked in rank order and each neighbour appears once, under its best-ranked hit.
    """
    if per_hit <= 0 or total <= 0:
        return results
    used = {(r["scope"], r["key"]) for r in results}
    budget = total
    for result in results:
        if budget <= 0:
            break
        view = views[result["scope"]]
        near = view.neighbors(result["key"])
        candidates = [(k, "links") for k in near["links"]]
        candidates += [(k, "linked_from") for k in near["backlinks"]]
        titles = view.titles() if candidates else {}
        related: list[dict[str, str]] = []
        for key, how in candidates:
            if len(related) >= per_hit or budget <= 0:
                break
            if (view.name, key) in used:
                continue
            used.add((view.name, key))
            related.append({"key": key, "title": titles.get(key, key), "relation": how})
            budget -= 1
        if related:
            result["related"] = related
    return results


def link_hints(
    view: ScopeGraph,
    key: str,
    title: str,
    summary: str,
    body: str,
    *,
    skip: set[str],
    vector: list[float] | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """``suggested_links`` and ``broken_links`` for a freshly written entry.

    With a head ``vector`` its similarity to every other entry joins the entity rules
    (see ``suggest_links``).
    """
    targets = parse_links(body)
    resolved = {view.resolve(t) for t in targets}
    metas = view.metas()
    active = {k for k, m in metas.items() if not m.superseded_by}
    embedded: list[tuple[str, float]] = []
    if vector is not None and model_id is not None:
        found = view.index.search(vector, model_id, top_k=10**9, exclude=key, head_only=True)
        embedded = [(k, sim) for k, sim in found if k in active]
    suggested = suggest_links(
        self_key=key,
        title=title,
        summary=summary,
        body=body,
        index=view.entities(),
        titles=view.titles(),
        active=active,
        skip=skip | {r for r in resolved if r},
        lexical=lexical_similarities(key, view.lexical(), active),
        embedded=embedded,
    )
    return {"suggested_links": suggested, "broken_links": [t for t in targets if t not in metas]}


def link_report(view: ScopeGraph) -> dict[str, Any]:
    """Broken links, orphan hubs and unlinked entries sharing entities, straight from the store."""
    entries = view.store.list()
    bodies = view.store.bodies()
    forward = {k: parse_links(b) for k, b in bodies.items()}
    active = {k for k, m in view.metas().items() if not m.superseded_by}
    titles = {key: title for key, title, _summary, _category in entries}
    index = EntityIndex.build(entries, bodies)
    return {
        "broken_links": [b for b in broken_links(forward, view.metas()) if b["key"] in active],
        "orphan_hubs": orphan_hubs(index, titles, forward, view.metas(), active),
        "unlinked_related": unlinked_related(index, forward, view.metas(), active),
    }


def merged_link_counts(graphs: Mapping[str, ScopeGraph]) -> dict[str, int]:
    """Linked-neighbour counts per active entry across scopes (only entries that have any)."""
    counts: dict[str, int] = {}
    for graph in graphs.values():
        for key, n in graph.link_counts().items():
            counts[key] = max(counts.get(key, 0), n)
    return counts
