"""Linked memory: explicit ``[[key]]`` links and link suggestions from shared entities.

Pure functions only (no store, index or embedder), so the same code serves the live
``MemoryService`` and the offline report-only ``nachos-maintain``.

* **Links.** ``[[key]]`` or ``[[Some Title]]`` (also ``[[key|label]]``) in an entry body
  names another entry of the same scope. Code spans and fenced blocks are ignored. A link
  to an entry that was superseded resolves to its active successor; a link to nothing is
  *broken*, which is tolerated and reported.
* **Suggestions.** Entities (see ``entities``) are *strong* or *weak*; a suggestion needs one
  strong shared entity or two weak ones, and entities found in many entries are ignored as
  generic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .entities import (
    LINK_RE,
    EntityIndex,
    extract_entities,
    lead_name,
    name_vocabulary,
    names_in,
)
from .lexical import LexicalIndex
from .store.base import EntryMeta
from .store.md_store import slugify

__all__ = [
    "EntityIndex",
    "broken_links",
    "extract_entities",
    "name_vocabulary",
    "orphan_hubs",
    "parse_links",
    "resolve_successor",
    "resolved_edges",
    "suggest_links",
    "unlinked_related",
]

_CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)

STRONG_KINDS = frozenset({"id", "host", "path", "handle", "name"})
_STRONG_WEIGHT = 2.0
_WEAK_WEIGHT = 1.0
_MIN_PAIR_SCORE = 4.0  # two strong shared entities, or one strong plus two weak
_MAX_SUGGESTIONS = 3
_MIN_NEIGHBORS = 20  # fewer entries than this give no meaningful similarity spread
# Similarity z-scores (see _semantic_rows): linkable alone / with a shared entity. Word overlap
# alone is a single z-score; with embeddings the word and meaning z-scores are summed.
_Z_ALONE_LEXICAL, _Z_ENTITY_LEXICAL = 4.75, 3.5
_Z_ALONE_COMBINED, _Z_ENTITY_COMBINED = 6.5, 4.5
_MAX_SHARED_SHOWN = 3
_MAX_CHAIN = 32


# -- explicit links ----------------------------------------------------------


def parse_links(body: str) -> list[str]:
    """Target keys of ``[[...]]`` references in ``body``: ordered, deduplicated, code skipped."""
    seen: dict[str, None] = {}
    for raw in LINK_RE.findall(_CODE_RE.sub(" ", body or "")):
        key = slugify(raw)
        if key != "untitled":
            seen.setdefault(key)
    return list(seen)


def resolve_successor(key: str, metas: Mapping[str, EntryMeta]) -> str | None:
    """Follow ``superseded_by`` from ``key`` to its active entry; None if broken or cyclic."""
    seen = {key}
    current = key
    for _ in range(_MAX_CHAIN):
        meta = metas.get(current)
        if meta is None:
            return None
        nxt = meta.superseded_by
        if not nxt:
            return current
        if nxt in seen:
            return None
        seen.add(nxt)
        current = nxt
    return None


def broken_links(
    forward: Mapping[str, list[str]], metas: Mapping[str, EntryMeta]
) -> list[dict[str, str]]:
    """``{"key", "target"}`` for every link whose target does not exist (superseded is fine)."""
    return [
        {"key": src, "target": dst}
        for src in sorted(forward)
        for dst in forward[src]
        if dst not in metas
    ]


# -- suggestions and reports -------------------------------------------------


def _weight(kind: str) -> float:
    return _STRONG_WEIGHT if kind in STRONG_KINDS else _WEAK_WEIGHT


def _scored(shared: dict[str, str], idf: Mapping[str, float]) -> float:
    return sum(_weight(kind) * idf.get(entity, 1.0) for entity, kind in shared.items())


def _passes(shared: Mapping[str, str]) -> bool:
    """One strong shared entity, or two independent weak ones (``ines`` inside ``ask ines`` is one)."""
    if any(kind in STRONG_KINDS for kind in shared.values()):
        return True
    weak = list(shared)
    independent = [e for e in weak if not any(e != o and e in o.split() for o in weak)]
    return len(independent) >= 2


def _shared_with(
    ents: Mapping[str, str],
    index: EntityIndex,
    active: set[str],
    skip: set[str],
    cap: int,
) -> tuple[dict[str, dict[str, str]], dict[str, float]]:
    shared: dict[str, dict[str, str]] = {}
    idf: dict[str, float] = {}
    for entity, kind in ents.items():
        owners = index.postings.get(entity, set()) & active
        if not owners or len(owners) > cap:
            continue
        idf[entity] = 1.0 + math.log(max(len(active), 1) / len(owners)) / 4.0
        for key in owners - skip:
            shared.setdefault(key, {})[entity] = kind
    return shared, idf


def _name_hits(
    *,
    text: str,
    title: str,
    self_key: str,
    index: EntityIndex,
    titles: Mapping[str, str],
    active: set[str],
    cap: int,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Entries tied to the new text by a project name: ``(shared rows, name -> entry key)``.

    The entry that owns a name (its title's first word) is a hit when the name is also used
    by at least one other entry, so a one-off mention never counts. Entries whose titles
    start with the same name as ``title`` are peers.
    """
    shared: dict[str, dict[str, str]] = {}
    vocab = name_vocabulary({k: t for k, t in titles.items() if k in active}, index.lowercase)
    for name in names_in(text, vocab):
        owner = vocab[name]
        others = index.postings.get(name, set()) & active - {owner, self_key}
        if owner in active and owner != self_key and others:
            shared.setdefault(owner, {})[name] = "name"
    lead = lead_name(title, index.lowercase)
    peers = [k for k, t in titles.items() if k in active and lead_name(t, index.lowercase) == lead]
    if lead and len(peers) <= cap:
        for peer in peers:
            if peer != self_key:
                shared.setdefault(peer, {})[lead] = "name"
    return shared, vocab


def lexical_similarities(
    key: str, lexical: LexicalIndex, active: set[str]
) -> list[tuple[str, float]]:
    """IDF-weighted cosine between ``key``'s words and every other active entry's words."""
    doc = lexical.docs.get(key)
    if doc is None or not active:
        return []
    n = max(len(lexical.docs), 1)

    def idf(term: str) -> float:
        return math.log(n / max(lexical.doc_freq.get(term, 1), 1))

    weights = {t: idf(t) for t in doc.doc_words}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    overlap: dict[str, float] = {}
    for term, w in weights.items():
        for other in lexical.postings.get(term, ()):
            if other != key and other in active:
                overlap[other] = overlap.get(other, 0.0) + w * w
    out = []
    for other in sorted(active - {key}):
        shared = overlap.get(other, 0.0)
        theirs = lexical.docs[other].doc_words if shared else ()
        denom = norm * (math.sqrt(sum(idf(t) ** 2 for t in theirs)) or 1.0)
        out.append((other, shared / denom if shared else 0.0))
    return out


def _zscores(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    values = [v for _k, v in pairs]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
    return {k: (v - mean) / sd for k, v in pairs}


def _semantic_rows(
    lexical: Sequence[tuple[str, float]], embedded: Sequence[tuple[str, float]]
) -> dict[str, tuple[float, str]]:
    """``{key: (z, band)}`` for entries whose similarity stands out from the rest of memory.

    Similarities are judged against the spread of this entry's own similarities to everything
    else (z-scores), not absolute cosines, because ranges differ a lot between models and
    between wording and meaning. Word overlap (IDF-weighted cosine) always counts; with an
    embedding the two z-scores are added. At ``high`` an entry is linkable on its own; at
    ``mid`` it also needs a shared entity.
    """
    if len(lexical) < _MIN_NEIGHBORS:
        return {}
    z = _zscores(lexical)
    alone, assisted = _Z_ALONE_LEXICAL, _Z_ENTITY_LEXICAL
    if len(embedded) >= _MIN_NEIGHBORS:
        zemb = _zscores(embedded)
        z = {k: v + zemb.get(k, 0.0) for k, v in z.items()}
        alone, assisted = _Z_ALONE_COMBINED, _Z_ENTITY_COMBINED
    return {k: (v, "high" if v >= alone else "mid") for k, v in z.items() if v >= assisted}


def _reason(shared: Mapping[str, str], sem: tuple[float, str] | None, meaning: bool) -> str:
    parts = []
    if sem is not None:
        parts.append("similar " + ("meaning and wording" if meaning else "wording"))
    if shared:
        parts.append("shares " + ", ".join(sorted(shared)[:_MAX_SHARED_SHOWN]))
    return " and ".join(parts)


def suggest_links(
    *,
    self_key: str,
    title: str = "",
    summary: str,
    body: str,
    index: EntityIndex,
    titles: Mapping[str, str],
    active: set[str],
    skip: Iterable[str] = (),
    lexical: Sequence[tuple[str, float]] = (),
    embedded: Sequence[tuple[str, float]] = (),
    limit: int = _MAX_SUGGESTIONS,
) -> list[dict[str, Any]]:
    """Existing active entries worth linking to the new one, best first, at most ``limit``.

    Two signals, both precision-first. *Entities*: one strong shared entity (identifier,
    machine host, path, handle, project name) or two independent weak ones. *Similarity*
    (``lexical`` word-overlap and, with embeddings, ``embedded`` cosine similarities to every
    other entry): similarity far above the entry's usual spread (z-score), or somewhat above
    it together with any shared non-generic entity.
    ``skip`` holds keys not worth suggesting (already linked, possible updates, duplicates).
    Each row carries a short ``reason``.
    """
    text = f"{summary}\n{body}"
    skipped = set(skip) | {self_key}
    cap = index.generic_cap(len(active))
    shared, idf = _shared_with(extract_entities(text), index, active, skipped, cap)
    named, _vocab = _name_hits(
        text=text,
        title=title,
        self_key=self_key,
        index=index,
        titles=titles,
        active=active,
        cap=cap,
    )
    for key, found in named.items():
        if key not in skipped:
            shared.setdefault(key, {}).update(found)
            idf.update({name: 1.5 for name in found})
    semantic = {k: v for k, v in _semantic_rows(lexical, embedded).items() if k in active}
    rows = []
    for key in set(shared) | set(semantic):
        if key in skipped:
            continue
        found, sem = shared.get(key, {}), semantic.get(key)
        entity_ok = _passes(found)
        if not (entity_ok or (sem and (sem[1] == "high" or found))):
            continue
        score = (3.0 + _scored(found, idf) if entity_ok else 0.0) + (sem[0] + 1 if sem else 0.0)
        rows.append((-score, key, found, sem))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        {
            "key": key,
            "title": titles.get(key, key),
            "shared": sorted(found)[:_MAX_SHARED_SHOWN],
            "reason": _reason(found, sem, bool(embedded)),
        }
        for _score, key, found, sem in rows[:limit]
    ]


def resolved_edges(
    forward: Mapping[str, list[str]], metas: Mapping[str, EntryMeta]
) -> set[tuple[str, str]]:
    """Directed ``(source, target)`` links with superseded endpoints moved to their successors."""
    edges: set[tuple[str, str]] = set()
    for src, dsts in forward.items():
        s = resolve_successor(src, metas) or src
        for dst in dsts:
            d = resolve_successor(dst, metas)
            if d is not None and d != s:
                edges.add((s, d))
    return edges


def _linked_pairs(
    forward: Mapping[str, list[str]], metas: Mapping[str, EntryMeta]
) -> set[tuple[str, str]]:
    return {(min(s, d), max(s, d)) for s, d in resolved_edges(forward, metas)}


def unlinked_related(
    index: EntityIndex,
    forward: Mapping[str, list[str]],
    metas: Mapping[str, EntryMeta],
    active: set[str],
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Active entry pairs that strongly share entities but have no link either way."""
    cap = index.generic_cap(len(active))
    pair_shared: dict[tuple[str, str], dict[str, str]] = {}
    idf: dict[str, float] = {}
    for entity, owners in index.postings.items():
        live = sorted(owners & active)
        if len(live) < 2 or len(live) > cap:
            continue
        idf[entity] = 1.0 + math.log(max(len(active), 1) / len(live)) / 4.0
        kind = index.by_key[live[0]][entity]
        for i, a in enumerate(live):
            for b in live[i + 1 :]:
                pair_shared.setdefault((a, b), {})[entity] = kind
    linked = _linked_pairs(forward, metas)
    rows = [
        (-_scored(found, idf), pair, found)
        for pair, found in pair_shared.items()
        if pair not in linked and _scored(found, {}) >= _MIN_PAIR_SCORE
    ]
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        {"a": a, "b": b, "shared": sorted(found)[:_MAX_SHARED_SHOWN]}
        for _score, (a, b), found in rows[:limit]
    ]


def orphan_hubs(
    index: EntityIndex,
    titles: Mapping[str, str],
    forward: Mapping[str, list[str]],
    metas: Mapping[str, EntryMeta],
    active: set[str],
    *,
    min_mentions: int = 2,
) -> list[dict[str, Any]]:
    """Entries named by a project name in several other entries that nothing links to."""
    vocab = name_vocabulary({k: t for k, t in titles.items() if k in active}, index.lowercase)
    linked_to = {d for _s, d in resolved_edges(forward, metas)}
    owner_of = {owner: name for name, owner in vocab.items()}
    rows = []
    for owner, name in sorted(owner_of.items()):
        mentions = sorted(k for k in index.postings.get(name, set()) & active if k != owner)
        if len(mentions) >= min_mentions and owner not in linked_to:
            rows.append({"key": owner, "name": name, "mentioned_by": mentions})
    return rows
