"""Lexical helpers for recall: tokenising, stemming, Jaccard overlap and IDF scoring.

Pure functions over strings; no store access. ``MemoryService`` supplies the corpus.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping

WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]+")

# Function words ignored by the summary-quality gate.
STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "it",
        "this",
        "that",
        "with",
        "as",
        "by",
        "at",
        "be",
        "are",
        "was",
        "were",
        "not",
        "no",
        "but",
        "if",
        "so",
        "do",
        "does",
        "did",
    }
)

# Query-time stopwords: the quality set plus question/pronoun words.
QUERY_STOPWORDS = STOPWORDS | frozenset(
    {
        "from",
        "into",
        "about",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "can",
        "will",
        "just",
        "than",
        "then",
        "there",
        "their",
        "they",
        "them",
        "our",
        "you",
        "your",
    }
)


def stem(word: str) -> str:
    """Fold simple plurals so 'editors' and 'editor' compare equal."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text: str) -> set[str]:
    return {stem(w) for w in WORD_RE.findall((text or "").lower()) if w not in QUERY_STOPWORDS}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def query_terms(query: str) -> tuple[set[str], list[str]]:
    """(raw words, sorted stemmed terms) of a query, minus stopwords and short words."""
    raw = {w for w in WORD_RE.findall(query.lower()) if w not in QUERY_STOPWORDS and len(w) >= 3}
    return raw, sorted({stem(w) for w in raw})


def idf_weights(terms: Iterable[str], doc_words: Mapping[str, set[str]]) -> dict[str, float]:
    """Inverse-document-frequency weight for each term present in at least one document."""
    n_docs = len(doc_words) or 1
    weights: dict[str, float] = {}
    for t in terms:
        df = sum(1 for words in doc_words.values() if t in words)
        if df:
            weights[t] = math.log(1 + n_docs / df)
    return weights


def score_document(
    weights: Mapping[str, float], words: set[str], label: set[str], phrase_hit: bool
) -> float:
    """Weighted term hits (label hits count extra), plus a bonus for the whole phrase."""
    score = sum(w for t, w in weights.items() if t in words)
    score += sum(0.5 * w for t, w in weights.items() if t in label)
    if score and phrase_hit:
        score += sum(weights.values())
    return score
