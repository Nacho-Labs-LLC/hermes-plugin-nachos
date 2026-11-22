"""Cheap, LLM-free entity extraction and the entity -> entries index.

Identifier-like terms (``db-prim-01``, ``notif_queue``), hostnames, paths, ``@handles``,
CamelCase names, capitalised multiword names and mid-sentence capitalised words are extracted
with regexes. Project names already used as titles (a unique, capitalised first word of a
title) count as entities too. Entities are *strong* (identifier, host, path, handle, CamelCase,
multiword, project name) or *weak* (a lone capitalised word). Pure functions and one in-memory
index: no store, embedder or Hermes access.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# Wiki-link syntax, stripped before extraction so link targets are not read as entities;
# ``links.parse_links`` reuses it.
LINK_RE = re.compile(r"\[\[([^\[\]\n|]{1,120})(?:\|[^\[\]\n]*)?\]\]")

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
        "access",
        "after",
        "alert",
        "alerts",
        "all",
        "also",
        "always",
        "and",
        "annual",
        "another",
        "any",
        "april",
        "august",
        "backup",
        "backups",
        "before",
        "budget",
        "business",
        "but",
        "checklist",
        "customer",
        "customers",
        "daily",
        "database",
        "databases",
        "day",
        "december",
        "default",
        "delivery",
        "deploy",
        "deployment",
        "deployments",
        "deploys",
        "development",
        "each",
        "enterprise",
        "every",
        "example",
        "february",
        "for",
        "free",
        "friday",
        "from",
        "guide",
        "here",
        "how",
        "if",
        "incident",
        "incidents",
        "january",
        "july",
        "june",
        "limit",
        "limits",
        "local",
        "march",
        "may",
        "meeting",
        "meetings",
        "migration",
        "migrations",
        "monday",
        "monitoring",
        "month",
        "monthly",
        "more",
        "most",
        "never",
        "new",
        "none",
        "not",
        "note",
        "notes",
        "nothing",
        "november",
        "october",
        "office",
        "only",
        "order",
        "orders",
        "other",
        "payment",
        "payments",
        "pipeline",
        "pipelines",
        "plan",
        "plans",
        "please",
        "policies",
        "policy",
        "prices",
        "pricing",
        "pro",
        "process",
        "product",
        "production",
        "products",
        "project",
        "projects",
        "quarterly",
        "release",
        "releases",
        "report",
        "reports",
        "restore",
        "review",
        "reviews",
        "rule",
        "rules",
        "saturday",
        "search",
        "security",
        "see",
        "september",
        "server",
        "servers",
        "service",
        "services",
        "setup",
        "shipping",
        "some",
        "something",
        "staging",
        "status",
        "storage",
        "sunday",
        "support",
        "system",
        "systems",
        "team",
        "teams",
        "template",
        "templates",
        "test",
        "testing",
        "tests",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "thursday",
        "tuesday",
        "use",
        "used",
        "using",
        "wednesday",
        "week",
        "weekly",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "workflow",
        "workflows",
        "year",
        "yes",
    ]
)
_PATH_STOP = frozenset({"and/or", "input/output", "client/server", "read/write"})

# Instance-level kinds (a specific host, id, path, person or project) are strong; topical kinds
# (product names, capitalised words, bare domains) are weak and need a second shared entity.


# -- entities ----------------------------------------------------------------


def _clean(text: str) -> str:
    return LINK_RE.sub(" ", text or "")


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


def lead_name(title: str, seen: Mapping[str, int]) -> str | None:
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
        name = lead_name(title, lowercase_words or {})
        if name:
            firsts.setdefault(name, []).append(key)
    return {name: keys[0] for name, keys in firsts.items() if len(keys) == 1}


def names_in(text: str, vocab: Mapping[str, str]) -> dict[str, str]:
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
