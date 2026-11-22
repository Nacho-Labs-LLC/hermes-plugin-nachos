"""Linked memory: explicit ``[[key]]`` links and cheap, LLM-free entity extraction.

Pure functions only (no store, index or embedder), so the same code serves the live
``MemoryService`` and the offline report-only ``nachos-maintain``.

* **Links.** ``[[key]]`` or ``[[Some Title]]`` (also ``[[key|label]]``) in an entry body
  names another entry of the same scope. Code spans and fenced blocks are ignored. A link
  to an entry that was superseded resolves to its active successor; a link to nothing is
  *broken*, which is tolerated and reported.
* **Entities.** Identifier-like terms (``db-prim-01``, ``notif_queue``), hostnames, paths,
  ``@handles``, CamelCase names, capitalised multiword names and mid-sentence capitalised
  words are extracted with regexes into an entity -> entries index. Project names already
  used as titles (a unique, capitalised first word of a title) count as entities too.
  Entities are *strong* (identifier, host, path, handle, CamelCase, multiword, project
  name) or *weak* (a lone capitalised word); suggestions need one strong shared entity or
  two weak ones, and entities found in many entries are ignored as generic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

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

_LINK_RE = re.compile(r"\[\[([^\[\]\n|]{1,120})(?:\|[^\[\]\n]*)?\]\]")
_CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)

_HANDLE_RE = re.compile(r"(?<![\w@.])@([A-Za-z][\w-]{2,})(?![\w/-])")
_HOST_RE = re.compile(
    r"\b(?:[a-z0-9][a-z0-9-]*\.)+(?:internal|io|com|net|org|dev|local|cloud|ai|app|co|eu|de)\b"
)
_PATH_RE = re.compile(r"(?<![\w/.:-])(?:/?[\w.-]{3,}(?:/[\w.-]{2,})+)")
_IDENT_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b|\b[A-Z][a-z]+[A-Z]{2,}\b")
_MULTI_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:[ \t][A-Z][a-z]{2,})+\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-z]{3,}\b")
_LOWER_RE = re.compile(r"(?<![A-Za-z])[a-z]{5,}\b")

_SENTENCE_END = ".!?:#>*-|(\n"
_SHORT_PREFIX_STOP = frozenset(
    [
        "no",
        "re",
        "co",
        "non",
        "pre",
        "un",
        "in",
        "on",
        "up",
        "to",
        "of",
        "by",
        "is",
        "at",
        "as",
        "it",
        "pro",
        "sub",
        "off",
        "per",
        "out",
    ]
)
_COMMON_CAPS = frozenset(
    [
        "the",
        "this",
        "that",
        "these",
        "those",
        "there",
        "then",
        "they",
        "them",
        "their",
        "use",
        "used",
        "using",
        "never",
        "always",
        "not",
        "only",
        "new",
        "all",
        "any",
        "each",
        "every",
        "if",
        "when",
        "for",
        "from",
        "with",
        "and",
        "but",
        "see",
        "note",
        "notes",
        "after",
        "before",
        "also",
        "here",
        "what",
        "why",
        "how",
        "who",
        "which",
        "where",
        "yes",
        "please",
        "default",
        "example",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "free",
        "pro",
        "business",
        "enterprise",
        "day",
        "week",
        "month",
        "year",
        "daily",
        "weekly",
        "nothing",
        "something",
        "none",
        "some",
        "more",
        "most",
        "other",
        "another",
    ]
)
_PATH_STOP = frozenset({"and/or", "input/output", "client/server", "read/write"})

# Instance-level kinds (a specific host, id, path, person or project) are strong; topical kinds
# (product names, capitalised words, bare domains) are weak and need a second shared entity.
STRONG_KINDS = frozenset({"id", "host", "path", "handle", "name"})
_STRONG_WEIGHT = 2.0
_WEAK_WEIGHT = 1.0
_MIN_PAIR_SCORE = 4.0  # two strong shared entities, or one strong plus two weak
_MAX_SUGGESTIONS = 5
_MAX_SHARED_SHOWN = 3
_MAX_CHAIN = 32


# -- explicit links ----------------------------------------------------------


def parse_links(body: str) -> list[str]:
    """Target keys of ``[[...]]`` references in ``body``: ordered, deduplicated, code skipped."""
    seen: dict[str, None] = {}
    for raw in _LINK_RE.findall(_CODE_RE.sub(" ", body or "")):
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


# -- entities ----------------------------------------------------------------


def _clean(text: str) -> str:
    return _LINK_RE.sub(" ", text or "")


def _host_kind(host: str) -> str:
    """Machine names (``db-1.internal``, ``a.b.example.io``) are strong; bare domains are not."""
    private = host.endswith((".internal", ".local"))
    return "host" if private or host.count(".") >= 2 else "domain"


def _add(found: dict[str, str], term: str, kind: str) -> None:
    found.setdefault(term, kind)


def _is_identifier(term: str) -> bool:
    parts = re.split(r"[-_]", term)
    if len(term) < 5 or len(parts) < 2:
        return False
    if any(c.isdigit() for c in term) or "_" in term or len(parts) >= 3:
        return True
    return len(parts[0]) <= 3 and parts[0] not in _SHORT_PREFIX_STOP and len(parts[1]) >= 3


def _sentence_initial(text: str, start: int) -> bool:
    prev = text[:start].rstrip(" \t")
    return not prev or prev[-1] in _SENTENCE_END


def _multiword(text: str, found: dict[str, str]) -> None:
    for m in _MULTI_RE.finditer(text):
        words = m.group(0).split()
        while words and words[0].lower() in _COMMON_CAPS:
            words.pop(0)
        if len(words) >= 2 and not all(w.lower() in _COMMON_CAPS for w in words):
            _add(found, " ".join(words).lower(), "multi")


def _proper(text: str, found: dict[str, str]) -> None:
    for m in _PROPER_RE.finditer(text):
        word = m.group(0)
        if word.lower() in _COMMON_CAPS or _sentence_initial(text, m.start()):
            continue
        _add(found, word.lower(), "proper")


def extract_entities(text: str) -> dict[str, str]:
    """``{entity: kind}`` for entity-like terms in ``text`` (entities are lowercased)."""
    text = _clean(text)
    found: dict[str, str] = {}
    for m in _HANDLE_RE.finditer(text):
        _add(found, "@" + m.group(1).lower(), "handle")
    host_spans = []
    for m in _HOST_RE.finditer(text):
        host_spans.append(m.span())
        _add(found, m.group(0), _host_kind(m.group(0)))
    for m in _PATH_RE.finditer(text):
        term = m.group(0).rstrip(".")
        if term.lower() not in _PATH_STOP and any(c.isalpha() for c in term):
            _add(found, term.lower(), "path")
    for m in _IDENT_RE.finditer(text):
        inside_host = any(lo <= m.start() and m.end() <= hi for lo, hi in host_spans)
        if _is_identifier(m.group(0)) and not inside_host:
            _add(found, m.group(0), "id")
    for m in _CAMEL_RE.finditer(text):
        _add(found, m.group(0).lower(), "camel")
    _multiword(text, found)
    _proper(text, found)
    return found


def _lead_name(title: str, seen: Mapping[str, int]) -> str | None:
    """The capitalised, never-lowercase first word of ``title`` (a likely project name)."""
    words = (title or "").split()
    word = words[0] if words else ""
    if len(word) < 5 or not word.isalpha() or not word[0].isupper():
        return None
    lower = word.lower()
    return None if lower in _COMMON_CAPS or seen.get(lower) else lower


def name_vocabulary(
    titles: Mapping[str, str], lowercase_words: Mapping[str, int] | None = None
) -> dict[str, str]:
    """``{lowercased first word: key}`` for titles led by a unique capitalised project name.

    A word also seen in lowercase in entry text (``production``, ``staging``) is a common
    word, not a name, so it is excluded.
    """
    firsts: dict[str, list[str]] = {}
    for key, title in titles.items():
        name = _lead_name(title, lowercase_words or {})
        if name:
            firsts.setdefault(name, []).append(key)
    return {name: keys[0] for name, keys in firsts.items() if len(keys) == 1}


def _names_in(text: str, vocab: Mapping[str, str]) -> dict[str, str]:
    """Project names (see ``name_vocabulary``) mentioned as capitalised words in ``text``."""
    if not vocab:
        return {}
    return {w.lower(): "name" for w in _PROPER_RE.findall(_clean(text)) if w.lower() in vocab}


class EntityIndex:
    """entity -> entries, with each entry's own entities; supports incremental updates."""

    def __init__(self) -> None:
        self.by_key: dict[str, dict[str, str]] = {}
        self.postings: dict[str, set[str]] = {}
        #: lowercase words of 5+ letters -> number of entries using them in lowercase
        self.lowercase: dict[str, int] = {}
        self._lower_of: dict[str, set[str]] = {}

    @classmethod
    def build(
        cls, entries: Iterable[tuple[str, str, str, str]], bodies: Mapping[str, str]
    ) -> EntityIndex:
        index = cls()
        for key, _title, summary, _category in entries:
            index.upsert(key, summary, bodies.get(key, ""))
        return index

    def upsert(self, key: str, summary: str, body: str) -> None:
        self.remove(key)
        text = f"{summary}\n{body}"
        found = extract_entities(text)
        lower = set(_LOWER_RE.findall(text))
        self._lower_of[key] = lower
        for word in lower:
            self.lowercase[word] = self.lowercase.get(word, 0) + 1
        self.by_key[key] = found
        for entity in found:
            self.postings.setdefault(entity, set()).add(key)

    def remove(self, key: str) -> None:
        for word in self._lower_of.pop(key, ()):
            self.lowercase[word] -= 1
        for entity in self.by_key.pop(key, {}):
            keys = self.postings.get(entity)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self.postings[entity]

    def entries_for(self, entity: str, active: set[str] | None = None) -> list[str]:
        keys = self.postings.get((entity or "").strip().lower(), set())
        return sorted(keys if active is None else keys & active)

    def generic_cap(self, active_count: int) -> int:
        """Entities in more active entries than this are generic and never drive suggestions."""
        return max(4, round(0.06 * active_count))


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
    limit: int = _MAX_SUGGESTIONS,
) -> list[dict[str, Any]]:
    """Existing active entries the new text shares entities with, best first (precision-first).

    ``skip`` holds keys not worth suggesting (already linked, or listed as possible updates).
    """
    text = f"{summary}\n{body}"
    ents = extract_entities(text)
    vocab = name_vocabulary({k: t for k, t in titles.items() if k in active}, index.lowercase)
    ents.update(_names_in(text, vocab))
    skipped = set(skip) | {self_key}
    cap = index.generic_cap(len(active))
    shared, idf = _shared_with(ents, index, active, skipped, cap)
    for name in _names_in(text, vocab):  # the entry that owns a project name is a direct hit
        owner = vocab[name]
        if owner in active and owner not in skipped:
            shared.setdefault(owner, {})[name] = "name"
            idf.setdefault(name, 1.5)
    lead = _lead_name(title, index.lowercase)
    if lead:  # entries whose titles start with the same project name are facets of one thing
        peers = [
            k for k, t in titles.items() if k in active and _lead_name(t, index.lowercase) == lead
        ]
        if len(peers) <= cap:
            for peer in peers:
                if peer not in skipped:
                    shared.setdefault(peer, {})[lead] = "name"
                    idf.setdefault(lead, 1.5)
    rows = [(-_scored(found, idf), key, found) for key, found in shared.items() if _passes(found)]
    rows.sort(key=lambda r: (r[0], r[1]))
    return [
        {"key": key, "title": titles.get(key, key), "shared": sorted(found)[:_MAX_SHARED_SHOWN]}
        for _score, key, found in rows[:limit]
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
