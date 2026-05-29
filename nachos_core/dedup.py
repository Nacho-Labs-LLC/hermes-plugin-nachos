"""Fact deduplication — incoming facts vs existing facts.

Design principle: memory maintenance is dedup/merge, not deletion.
Facts are permanent — we only consolidate when the same fact is
observed again. Two facts are "the same" when their normalized
(subject, predicate, object) match exactly. Confidence merges via a
weighted average biased toward the higher value.

This file is pure logic. Storage is the adapter's problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from .types import MemoryFact


_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Lowercase, trim, collapse whitespace. Used for matching only."""
    return _WS_RE.sub(" ", (s or "").strip().lower())


def is_exact_match(a: MemoryFact, b: MemoryFact) -> bool:
    """Same (subject, predicate, object) after normalization."""
    return (
        _normalize(a.subject) == _normalize(b.subject)
        and _normalize(a.predicate) == _normalize(b.predicate)
        and _normalize(a.object) == _normalize(b.object)
    )


def merge_fact(existing: MemoryFact, incoming: MemoryFact) -> MemoryFact:
    """Merge incoming into existing. Returns the merged fact.

    Confidence: weighted average + 0.05 nudge, capped at 1.0. The +0.05
    rewards re-observation — seeing the same fact twice should bump
    confidence even if both observations were modest.

    Kind: incoming wins if non-default. Source session: keep existing's.
    """
    e_conf = existing.confidence if existing.confidence is not None else 0.5
    i_conf = incoming.confidence if incoming.confidence is not None else 0.5
    merged_conf = min(1.0, (e_conf + i_conf) / 2 + 0.05)

    kind = existing.kind
    if incoming.kind and incoming.kind != "general":
        kind = incoming.kind

    return MemoryFact(
        subject=existing.subject,
        predicate=existing.predicate,
        object=existing.object,
        confidence=merged_conf,
        kind=kind,
        source_session=existing.source_session,
        extracted_at=existing.extracted_at,
    )


@dataclass
class DedupResult:
    to_insert: List[MemoryFact] = field(default_factory=list)
    to_update: List[Tuple[MemoryFact, MemoryFact]] = field(default_factory=list)
    # ↑ (existing, merged_replacement)


def deduplicate_facts(incoming: List[MemoryFact],
                      existing: List[MemoryFact]) -> DedupResult:
    """Split incoming facts into to-insert vs to-update against existing."""
    result = DedupResult()
    for fact in incoming:
        match = next((e for e in existing if is_exact_match(e, fact)), None)
        if match is not None:
            result.to_update.append((match, merge_fact(match, fact)))
        else:
            result.to_insert.append(fact)
    return result
