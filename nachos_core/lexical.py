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

    title: str
    summary: str
    category: str
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
        postings: dict[str, set[str]],
        label_postings: dict[str, set[str]],
        doc_freq: dict[str, int],
    ) -> None:
        self.docs = docs
        self.postings = postings
        self.label_postings = label_postings
        self.doc_freq = doc_freq

    @staticmethod
    def _discard_terms(mapping: dict[str, set[str]], key: str, terms: AbstractSet[str]) -> None:
        for term in terms:
            bucket = mapping.get(term)
            if bucket is None:
                continue
            bucket.discard(key)
            if not bucket:
                del mapping[term]

    def remove(self, key: str) -> None:
        """Remove ``key`` from the cached term sets, postings and document frequencies."""
        old = self.docs.pop(key, None)
        if old is None:
            return
        self._discard_terms(self.postings, key, old.doc_words)
        for term in old.doc_words:
            bucket = self.postings.get(term)
            if bucket:
                self.doc_freq[term] = len(bucket)
            else:
                self.doc_freq.pop(term, None)
        self._discard_terms(self.label_postings, key, old.label_words)

    def upsert(
        self,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
        body: str,
    ) -> None:
        """Insert or replace one entry in the cached term sets and postings."""
        if key in self.docs:
            self.remove(key)
        title_w = frozenset(tokens(title))
        summary_w = frozenset(tokens(summary))
        body_w = frozenset(tokens(body))
        label_w = title_w | summary_w
        doc_w = label_w | body_w
        self.docs[key] = DocumentTerms(
            title=title,
            summary=summary,
            category=category,
            title_words=title_w,
            summary_words=summary_w,
            body_words=body_w,
            label_words=label_w,
            doc_words=doc_w,
            haystack=f"{title}\n{summary}\n{body}".lower(),
        )
        for term in doc_w:
            bucket = self.postings.setdefault(term, set())
            bucket.add(key)
            self.doc_freq[term] = len(bucket)
        for term in label_w:
            self.label_postings.setdefault(term, set()).add(key)

    @classmethod
    def build(
        cls,
        entries: Iterable[tuple[str, str, str, str]],
        bodies: Mapping[str, str],
    ) -> LexicalIndex:
        index = cls({}, {}, {}, {})
        for key, title, summary, category in entries:
            index.upsert(
                key,
                title=title,
                summary=summary,
                category=category,
                body=bodies.get(key) or "",
            )
        return index

    def similar_labels(
        self,
        label: str,
        *,
        threshold: float,
        exclude: str | None = None,
    ) -> list[tuple[str, str, float]]:
        """Return ``(key, title, score)`` for entries with label Jaccard >= ``threshold``."""
        query_words = tokens(label)
        if not query_words:
            return []
        n_query = len(query_words)
        # Pigeonhole prefix filter: a match needs at least min_overlap shared terms,
        # so it must hit at least one of the (n_query - min_overlap + 1) rarest terms.
        min_overlap = max(1, math.ceil(threshold * n_query))
        if min_overlap > 1 and (min_overlap - 1) / n_query >= threshold:
            min_overlap -= 1
        prefix_len = max(1, n_query - min_overlap + 1)
        by_freq = sorted(query_words, key=lambda t: len(self.label_postings.get(t, ())))
        candidates: set[str] = set()
        for term in by_freq[:prefix_len]:
            bucket = self.label_postings.get(term)
            if bucket:
                candidates.update(bucket)
        if exclude is not None:
            candidates.discard(exclude)
        matches: list[tuple[str, str, float]] = []
        for key in candidates:
            doc = self.docs[key]
            overlap = len(query_words & doc.label_words)
            score = overlap / (n_query + len(doc.label_words) - overlap)
            if score >= threshold:
                matches.append((key, doc.title, score))
        matches.sort(
            key=lambda item: (self.docs[item[0]].category, self.docs[item[0]].title, item[0])
        )
        return matches

    def _term_idf(self, term: str, n_docs: int) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + n_docs / df) if df else 0.0

    def _doc_norm_sq(self, doc: DocumentTerms, n_docs: int, idf_cache: dict[str, float]) -> float:
        total = 0.0
        for t in doc.doc_words:
            w = idf_cache.get(t)
            if w is None:
                w = self._term_idf(t, n_docs)
                idf_cache[t] = w
            tw = w * (2.0 if t in doc.label_words else 1.0)
            total += tw * tw
        return total

    def similar_documents(
        self,
        *,
        title: str,
        summary: str,
        body: str,
        threshold: float,
        exclude: AbstractSet[str] = frozenset(),
        limit: int = 12,
    ) -> list[tuple[str, float, tuple[str, ...]]]:
        """Return ``(key, cosine, top_shared_terms)`` using IDF-weighted document cosine."""
        label_w = frozenset(tokens(f"{title} {summary}"))
        doc_w = label_w | frozenset(tokens(body))
        q_idf = self.idf(doc_w) if (doc_w and self.docs) else {}
        if not q_idf:
            return []
        q_weights = {t: w * (2.0 if t in label_w else 1.0) for t, w in q_idf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_weights.values()))
        candidates = {
            key for term in q_weights for key in self.postings.get(term, ()) if key not in exclude
        }
        if not q_norm or not candidates:
            return []
        n_docs = len(self.docs)
        idf_cache = dict(q_idf)
        matches: list[tuple[str, float, tuple[str, ...]]] = []
        for key in candidates:
            doc = self.docs[key]
            shared = [t for t in q_weights if t in doc.doc_words]
            d_sq = self._doc_norm_sq(doc, n_docs, idf_cache) if shared else 0.0
            if d_sq <= 0.0:
                continue
            dot_prod = sum(
                q_weights[t] * q_idf[t] * (2.0 if t in doc.label_words else 1.0) for t in shared
            )
            sim = dot_prod / (q_norm * math.sqrt(d_sq))
            if sim >= threshold:
                shared.sort(key=lambda t: (-q_idf[t], t))
                matches.append((key, sim, tuple(shared[:3])))
        matches.sort(key=lambda item: (-item[1], item[0]))
        return matches[:limit]

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

    def search_scored(self, query: str, min_coverage: float = 0.0) -> list[tuple[str, float]]:
        """Rank matching ``(key, score)`` pairs by IDF-weighted term overlap and phrase bonus."""
        raw, terms = query_terms(query)
        if not terms:
            return []
        weights = self.idf(terms)
        if not weights:
            return []
        needle = query.lower()
        total_weight = sum(weights.values()) or 1.0
        matched = self._matched_keys(weights, terms, raw, needle.strip())
        scored: list[tuple[float, str]] = []
        for key in matched:
            doc = self.docs[key]
            if min_coverage > 0.0:
                covered = sum(w for t, w in weights.items() if t in doc.doc_words)
                if covered / total_weight < min_coverage:
                    continue
            score = score_document(weights, doc.doc_words, doc.label_words, needle in doc.haystack)
            if score > 0:
                scored.append((score, key))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(key, score) for score, key in scored]

    def search(self, query: str, min_coverage: float = 0.0) -> list[str]:
        """Rank matching document keys by IDF-weighted term overlap and phrase bonus."""
        return [key for key, _score in self.search_scored(query, min_coverage=min_coverage)]

    def oov_coverage(self, query: str, key: str) -> float:
        """IDF-weighted fraction of ALL query terms (OOV terms weighted at max IDF) present in ``key``."""
        _raw, terms = query_terms(query)
        doc = self.docs.get(key)
        if not terms or doc is None:
            return 0.0
        n_docs = len(self.docs) or 1
        max_idf = math.log(1.0 + n_docs)
        matched = 0.0
        total = 0.0
        for term in terms:
            df = self.doc_freq.get(term, 0)
            w = math.log(1.0 + n_docs / df) if df else max_idf
            total += w
            if term in doc.doc_words:
                matched += w
        return matched / total if total else 0.0
