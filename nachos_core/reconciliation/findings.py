"""Deterministic findings and proposals from audit signals; no LLM, no network, no writes.

Turns what the offline audit already knows into reviewable material inside a packet:

* duplicate pair        -> ``merge`` (merged text drafted by line union, duplicate disabled)
* newer fact over older -> ``supersede`` (likely outdated)
* suspicious supersedes -> ``unlink``
* low-quality record    -> ``rewrite`` (summary from the body) or ``delete`` (placeholder body)
* same facet, differing facts, no update cue -> ``choose-one`` (low confidence; the newer
  record is the default survivor, the alternative is named in the rationale)

Every proposal starts with no assessments, so it cannot be accepted until a human verifies
preservation in the viewer. ``confidence`` and ``evidence`` are heuristics, never proof.
An optional ``drafter`` may replace the merged text; it is only ever supplied explicitly.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..quality import (
    fact_change,
    flag_reason,
    has_update_cue,
    low_signal_reason,
    qualifier_difference,
    temporal_progression,
)
from .hashing import file_sha256
from .models import timestamp

MAX_PROPOSALS = 200
_SUMMARY_CHARS = 200
_WORD = re.compile(r"\w+")
_RANK = {"high": 0, "medium": 1, "low": 2}

Drafter = Callable[[dict[str, str], dict[str, str]], "dict[str, str] | None"]


@dataclass(frozen=True)
class AuditSignals:
    """Audit output the generator consumes (all plain data)."""

    duplicate_pairs: list[dict[str, Any]] = field(default_factory=list)
    suspicious_supersedes: list[dict[str, Any]] = field(default_factory=list)
    # key -> (updated_at or None, superseded_by or None)
    metas: Mapping[str, tuple[float | None, str | None]] = field(default_factory=dict)


@dataclass
class Generated:
    findings: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


def _digest_id(prefix: str, *parts: str) -> str:
    return f"{prefix}-" + file_sha256("\0".join(parts).encode())[:16]


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def _text(record: dict[str, str]) -> str:
    return f"{record['title']}\n{record['summary']}\n{record['body']}"


def _excerpt(record: dict[str, str]) -> str:
    return record["summary"] or record["title"]


def _observation(source_ids: list[str], claim: str, detail: str, at: str) -> dict[str, Any]:
    return {
        "kind": "source_content",
        "status": "unverified",
        "source_ids": source_ids,
        "claim": claim,
        "observation": detail,
        "observed_at": at,
    }


class _Builder:
    def __init__(self, inventory: dict[str, Any], signals: AuditSignals, at: str) -> None:
        self.at = at
        self.signals = signals
        self.by_key: dict[str, dict[str, Any]] = {
            s["before"]["key"]: s for s in inventory["sources"] if s["kind"] == "memory_record"
        }
        self.out = Generated()
        self.claimed: set[str] = set()

    # -- shared plumbing ---------------------------------------------------

    def record(self, key: str) -> dict[str, str]:
        record: dict[str, str] = self.by_key[key]["before"]
        return record

    def updated(self, key: str) -> float | None:
        return self.signals.metas.get(key, (None, None))[0]

    def superseded(self, key: str) -> bool:
        return self.signals.metas.get(key, (None, None))[1] is not None

    def newer_first(self, a: str, b: str) -> tuple[str, str] | None:
        """(newer, older) by updated_at; None when the order cannot be established."""
        ua, ub = self.updated(a), self.updated(b)
        if ua is None or ub is None or ua == ub:
            return None
        return (a, b) if ua > ub else (b, a)

    def finding(self, kind: str, keys: list[str], rationale: str, claim: str) -> dict[str, Any]:
        ids = [self.by_key[k]["id"] for k in keys]
        return {
            "id": _digest_id("find", kind, *sorted(keys)),
            "type": kind,
            "source_ids": ids,
            "excerpts": [
                {"source_id": self.by_key[k]["id"], "text": _excerpt(self.record(k))} for k in keys
            ],
            "rationale": rationale,
            "evidence": [
                _observation(ids, claim, "Raised by the offline audit; unverified.", self.at)
            ],
            "limitations": [
                "Heuristic over stored text only; no usage or freshness evidence was consulted.",
                "Confidence is a ranking aid, not evidence. Verify against full bodies.",
            ],
        }

    def change(self, key: str, operation: str, after: Any) -> dict[str, Any]:
        source = self.by_key[key]
        return {
            "source_id": source["id"],
            "source_sha256": source["sha256"],
            "operation": operation,
            "before": source["before"],
            "after": after,
        }

    def add(
        self,
        kind: str,
        confidence: str,
        finding: dict[str, Any],
        changes: list[dict[str, Any]],
    ) -> None:
        touched = {c["source_id"] for c in changes}
        if touched & self.claimed:
            self.out.skipped.append({"finding": finding["id"], "reason": "source already claimed"})
            return
        if len(self.out.proposals) >= MAX_PROPOSALS:
            self.out.skipped.append({"finding": finding["id"], "reason": "proposal limit"})
            return
        self.claimed |= touched
        self.out.findings.append(finding)
        self.out.proposals.append(
            {
                "id": _digest_id("prop", kind, *sorted(touched)),
                "revision": 1,
                "kind": kind,
                "confidence": confidence,
                "finding_ids": [finding["id"]],
                "depends_on": [],
                "changes": changes,
                "assessments": [],
            }
        )

    def flag_only(self, finding: dict[str, Any]) -> None:
        self.out.findings.append(finding)

    # -- pairs: duplicate, outdated, conflict ------------------------------

    def pairs(self, drafter: Drafter | None) -> None:
        for pair in self.signals.duplicate_pairs:
            a, b = pair["a"]["key"], pair["b"]["key"]
            if a not in self.by_key or b not in self.by_key:
                continue
            if self.superseded(a) or self.superseded(b):
                continue
            self.pair(a, b, pair, drafter)

    def pair(self, a: str, b: str, pair: dict[str, Any], drafter: Drafter | None) -> None:
        similarity = float(pair.get("similarity", 0.0))
        ra, rb = self.record(a), self.record(b)
        if qualifier_difference(ra["title"], _text(ra), rb["title"], _text(rb)) is not None:
            return  # distinct facets (region/team/...), not a duplicate or a conflict
        order = self.newer_first(a, b)
        newer, older = order if order else (min(a, b), max(a, b))
        old_text, new_text = _text(self.record(older)), _text(self.record(newer))
        changed = fact_change(old_text, new_text)
        cue = has_update_cue(new_text) or temporal_progression(
            self.record(older)["title"], self.record(newer)["title"]
        )
        if changed is None and not cue:
            if pair.get("match") == "related":
                return  # related but not duplicate, changed or superseding: nothing to propose
            self.duplicate(newer, older, similarity, drafter)
        elif order is None:
            self.flag_only(
                self.finding(
                    "conflict",
                    sorted([a, b]),
                    f"Records '{a}' and '{b}' describe the same facet with different facts "
                    f"({changed[0]!r} vs {changed[1]!r}) and their age order is unknown."
                    if changed
                    else f"Records '{a}' and '{b}' overlap and may be versions of one fact.",
                    "Overlapping records with differing facts; no timestamps to order them.",
                )
            )
        elif cue:
            self.outdated(newer, older, changed)
        else:
            self.conflict(newer, older, changed)

    def duplicate(self, newer: str, older: str, similarity: float, drafter: Drafter | None) -> None:
        survivor, other = self.record(newer), self.record(older)
        merged = _merge(survivor, other)
        if drafter is not None:
            drafted = drafter(survivor, other)
            if drafted is not None:
                merged = {**survivor, **drafted, "key": survivor["key"]}
        confidence = (
            "high"
            if similarity >= 0.9 or _norm(survivor["title"]) == _norm(other["title"])
            else "medium"
            if similarity >= 0.75
            else "low"
        )
        finding = self.finding(
            "duplicate",
            [newer, older],
            f"'{older}' and '{newer}' look like the same fact (similarity {similarity:.2f}). "
            f"Proposed: keep '{newer}' (more recent), fold in any lines only '{older}' has, "
            f"and retire '{older}'.",
            f"Near-duplicate records (similarity {similarity:.2f}).",
        )
        changes = []
        if merged != survivor:
            changes.append(self.change(newer, "edit", merged))
        changes.append(self.change(older, "disable", None))
        self.add("merge", confidence, finding, changes)

    def outdated(self, newer: str, older: str, changed: tuple[str, str] | None) -> None:
        detail = f" ({changed[0]!r} -> {changed[1]!r})" if changed else ""
        finding = self.finding(
            "stale",
            [newer, older],
            f"'{newer}' reads as a newer version of '{older}'{detail}; the older record is "
            "probably outdated. Proposed: mark it superseded by the newer one (reversible).",
            "Newer record carries an update cue or changed fact over an older one.",
        )
        confidence = "high" if changed else "medium"
        self.add(
            "supersede",
            confidence,
            finding,
            [self.change(older, "supersede", {"superseded_by": newer})],
        )

    def conflict(self, newer: str, older: str, changed: tuple[str, str] | None) -> None:
        detail = f" ({changed[0]!r} vs {changed[1]!r})" if changed else ""
        finding = self.finding(
            "conflict",
            [newer, older],
            f"'{newer}' and '{older}' state different facts for the same facet{detail} with no "
            f"update cue. Default proposal: the newer '{newer}' wins and '{older}' is marked "
            f"superseded; the alternative is to keep '{older}' (reject this proposal).",
            "Same-facet records with differing facts and no update cue.",
        )
        self.add(
            "choose-one",
            "low",
            finding,
            [self.change(older, "supersede", {"superseded_by": newer})],
        )

    # -- links -------------------------------------------------------------

    def links(self) -> None:
        for row in self.signals.suspicious_supersedes:
            old, new = row["old_key"], row["new_key"]
            if old not in self.by_key or new not in self.by_key:
                continue
            finding = self.finding(
                "unsupported",
                [old, new],
                f"'{old}' is marked superseded by '{new}', but they differ in a facet "
                f"qualifier ({row['reason']}); they may be distinct facts. Proposed: remove "
                "the link so both are active again (reversible).",
                f"Supersede link looks unsupported: {row['reason']}.",
            )
            self.add(
                "unlink", "medium", finding, [self.change(old, "unlink", {"superseded_by": new})]
            )

    # -- quality -----------------------------------------------------------

    def quality(self) -> None:
        for key in sorted(self.by_key):
            if self.superseded(key) or self.by_key[key]["id"] in self.claimed:
                continue
            record = self.record(key)
            body_reason = low_signal_reason(record["body"])
            if body_reason:
                finding = self.finding(
                    "unsupported",
                    [key],
                    f"The body of '{key}' is low-signal ({body_reason}); there is nothing "
                    "worth keeping. Proposed: retire it (reversible).",
                    f"Placeholder-like body ({body_reason}).",
                )
                self.add("delete", "medium", finding, [self.change(key, "disable", None)])
                continue
            reason = flag_reason(record["title"], record["summary"], record["body"])
            if reason is None:
                continue
            finding = self.finding(
                "clarification",
                [key],
                f"Summary of '{key}' flagged: {reason}. A hint, not evidence of falsehood.",
                f"Heuristic flag: {reason}",
            )
            summary = _draft_summary(record)
            if summary is None:
                self.flag_only(finding)
                continue
            confidence = "low" if reason.startswith("no shared") else "medium"
            self.add(
                "rewrite",
                confidence,
                finding,
                [self.change(key, "edit", {**record, "summary": summary})],
            )


def _merge(survivor: dict[str, str], other: dict[str, str]) -> dict[str, str]:
    """Survivor plus any non-empty line of the other body it does not already contain."""
    have = _norm(survivor["body"])
    extra = [
        line.strip()
        for line in other["body"].splitlines()
        if line.strip() and _norm(line) and _norm(line) not in have
    ]
    if not extra:
        return dict(survivor)
    body = survivor["body"].rstrip() + "\n" + "\n".join(extra)
    return {**survivor, "body": body}


def _draft_summary(record: dict[str, str]) -> str | None:
    """First substantive body line as the summary, only if it passes the same checks."""
    for line in record["body"].splitlines():
        candidate = line.strip().lstrip("#-*> ").strip()
        if len(_WORD.findall(candidate)) < 4 or _norm(candidate) == _norm(record["title"]):
            continue
        if len(candidate) > _SUMMARY_CHARS:
            candidate = candidate[:_SUMMARY_CHARS].rsplit(" ", 1)[0].rstrip(",;:")
        if (
            candidate != record["summary"]
            and flag_reason(record["title"], candidate, record["body"]) is None
        ):
            return candidate
        return None
    return None


def generate_findings(
    inventory: dict[str, Any],
    signals: AuditSignals,
    *,
    observed_at: str,
    drafter: Drafter | None = None,
) -> Generated:
    """Findings plus proposals for a validated inventory; deterministic for equal inputs."""
    timestamp(observed_at)
    builder = _Builder(inventory, signals, observed_at)
    builder.pairs(drafter)
    builder.links()
    builder.quality()
    builder.out.proposals.sort(key=lambda p: (_RANK[p["confidence"]], p["id"]))
    return builder.out
