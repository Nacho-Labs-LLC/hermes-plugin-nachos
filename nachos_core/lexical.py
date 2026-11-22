"""Lexical helpers for recall: tokenising, stemming, Jaccard overlap and IDF scoring.

Pure functions over strings; no store access. ``MemoryService`` supplies the corpus.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

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


def idf_weights(
    terms: Iterable[str], doc_words: Mapping[str, AbstractSet[str]]
) -> dict[str, float]:
    """Inverse-document-frequency weight for each term present in at least one document."""
    n_docs = len(doc_words) or 1
    weights: dict[str, float] = {}
    for t in terms:
        df = sum(1 for words in doc_words.values() if t in words)
        if df:
            weights[t] = math.log(1 + n_docs / df)
    return weights


def score_document(
    weights: Mapping[str, float],
    words: AbstractSet[str],
    label: AbstractSet[str],
    phrase_hit: bool,
) -> float:
    """Weighted term hits (label hits count extra), plus a bonus for the whole phrase."""
    word_score = 0.0
    label_score = 0.0
    for t, w in weights.items():
        if t in words:
            word_score += w
        if t in label:
            label_score += 0.5 * w
    score = word_score + label_score
    if score and phrase_hit:
        score += sum(weights.values())
    return score


@dataclass(frozen=True)
class DocumentTerms:
    """Cached tokenised/stemmed term sets and lowercased text for one entry."""

    title_words: frozenset[str]
    summary_words: frozenset[str]
    body_words: frozenset[str]
    label_words: frozenset[str]
    doc_words: frozenset[str]
    haystack: str


class LexicalIndex:
    """In-memory per-scope lexical index: per-doc term sets, postings and doc frequencies."""

    def __init__(
        self,
        docs: dict[str, DocumentTerms],
        postings: dict[str, frozenset[str]],
        doc_freq: dict[str, int],
    ) -> None:
        self.docs = docs
        self.postings = postings
        self.doc_freq = doc_freq

    @classmethod
    def build(
        cls,
        entries: Iterable[tuple[str, str, str, str]],
        bodies: Mapping[str, str],
    ) -> LexicalIndex:
        docs: dict[str, DocumentTerms] = {}
        postings_mut: dict[str, set[str]] = {}
        for key, title, summary, _category in entries:
            body = bodies.get(key) or ""
            title_w = frozenset(tokens(title))
            summary_w = frozenset(tokens(summary))
            body_w = frozenset(tokens(body))
            label_w = title_w | summary_w
            doc_w = label_w | body_w
            docs[key] = DocumentTerms(
                title_words=title_w,
                summary_words=summary_w,
                body_words=body_w,
                label_words=label_w,
                doc_words=doc_w,
                haystack=f"{title}\n{summary}\n{body}".lower(),
            )
            for term in doc_w:
                postings_mut.setdefault(term, set()).add(key)
        postings = {term: frozenset(keys) for term, keys in postings_mut.items()}
        doc_freq = {term: len(keys) for term, keys in postings.items()}
        return cls(docs, postings, doc_freq)

    def idf(self, terms: Iterable[str]) -> dict[str, float]:
        """Inverse-document-frequency weight for each term present in at least one document."""
        n_docs = len(self.docs) or 1
        weights: dict[str, float] = {}
        for t in terms:
            df = self.doc_freq.get(t, 0)
            if df:
                weights[t] = math.log(1 + n_docs / df)
        return weights

    def _matched_keys(
        self,
        weights: Mapping[str, float],
        terms: list[str],
        raw: set[str],
        stripped: str,
    ) -> set[str]:
        matched: set[str] = set()
        for t in weights:
            matched.update(self.postings[t])
        # Stemming only breaks literal substring matching for -ies -> -y plurals.
        if not any(t.endswith("y") for t in weights):
            return matched
        probes = tuple(dict.fromkeys((*weights, *terms, *raw, stripped)))
        return {k for k in matched if any(p in self.docs[k].haystack for p in probes)}

    def search(self, query: str) -> list[str]:
        """Rank matching document keys by IDF-weighted term overlap and phrase bonus."""
        raw, terms = query_terms(query)
        if not terms:
            return []
        weights = self.idf(terms)
        if not weights:
            return []
        needle = query.lower()
        matched = self._matched_keys(weights, terms, raw, needle.strip())
        scored: list[tuple[float, str]] = []
        for key in matched:
            doc = self.docs[key]
            score = score_document(weights, doc.doc_words, doc.label_words, needle in doc.haystack)
            if score > 0:
                scored.append((score, key))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [key for _score, key in scored]
