"""Prefetch scorer seam — rank candidate entries against a query.

Tier 2 (prefetch) uses a Scorer to pick the most relevant entries to
inject each turn. The seam lets the store's coarse search() results be
ranked by different strategies:

  * LexicalScorer (default) — hand-rolled TF-IDF, zero external deps.
  * SemanticScorer          — phase 2, delegates to an embedding MCP.
                              NOT implemented here; get_scorer('semantic')
                              raises until the adapter lands.

The Scorer ranks over Entry tuples (key, title, summary, category); it
scores the concatenation of title+summary as the searchable text. (Body
is intentionally excluded from ranking text to keep the hot path light —
the store already substring-matched bodies in search(); prefetch ranks
the *labels* the manifest shows, which is what keeps recall aligned with
what the agent can see.)
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List

from .store.base import Entry

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class Scorer(ABC):
    """Ranks candidate entries against a query, best first."""

    @abstractmethod
    def rank(
        self,
        query: str,
        candidates: List[Entry],
        top_n: int = 5,
    ) -> List[str]:
        """Return up to ``top_n`` entry keys ordered best-first."""


class LexicalScorer(Scorer):
    """TF-IDF ranking over the candidate corpus, pure Python.

    IDF is computed across the candidate set each call (N is small — the
    prefetch candidate list, not the whole store), so rarer query terms
    weigh more. Score = sum over query terms of tf(term, doc) * idf(term).
    """

    def rank(
        self,
        query: str,
        candidates: List[Entry],
        top_n: int = 5,
    ) -> List[str]:
        q_terms = set(_tokenize(query))
        if not q_terms or not candidates:
            return []

        # doc text per candidate = title + summary
        docs: List[List[str]] = []
        for (_key, title, summary, _cat) in candidates:
            docs.append(_tokenize(f"{title} {summary}"))

        n_docs = len(docs)
        # document frequency for query terms only
        df: Dict[str, int] = {}
        for term in q_terms:
            df[term] = sum(1 for d in docs if term in d)

        # idf: log((N+1)/(df+1)) + 1  (smoothed, always positive)
        idf: Dict[str, float] = {
            term: math.log((n_docs + 1) / (df[term] + 1)) + 1.0
            for term in q_terms
        }

        scored: List[tuple] = []  # (score, orig_index, key)
        for idx, ((key, _t, _s, _c), tokens) in enumerate(
            zip(candidates, docs)
        ):
            if not tokens:
                continue
            tf = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in q_terms:
                if term in tf:
                    score += (tf[term] / doc_len) * idf[term]
            if score > 0:
                scored.append((score, idx, key))

        # sort by score desc, stable on original order for ties
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [key for (_score, _idx, key) in scored[:top_n]]


def get_scorer(name: str, *, semantic_backend: str = "nachos") -> Scorer:
    """Factory: return a scorer by name.

    'lexical' (default) is the zero-dep TF-IDF scorer. 'semantic' returns a
    SemanticScorer over the given backend ('nachos'|'sentence-transformers'|
    'openai'); it lazily loads its backend and falls back to lexical at
    ranking time if the backend is unavailable, so selecting it never
    hard-fails the hot path.
    """
    n = (name or "lexical").strip().lower()
    if n == "lexical":
        return LexicalScorer()
    if n == "semantic":
        # imported lazily to keep the module import graph dependency-free
        from .semantic import SemanticScorer
        return SemanticScorer(backend=semantic_backend)
    raise ValueError(f"Unknown scorer {name!r}. Use 'lexical' or 'semantic'.")
