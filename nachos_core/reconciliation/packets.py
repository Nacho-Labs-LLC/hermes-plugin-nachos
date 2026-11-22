"""Bounded views, optional heuristic findings and sealed packets; never proposes.

Nothing here approves, disables, deletes, calls a network/LLM, or upgrades
inventory coverage. Proposals come only from explicit caller input. A finder runs
only when passed explicitly and its output is validated like any other input.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from ..quality import flag_reason
from .codec import _forbidden, _pairs
from .hashing import MAX_BYTES, file_sha256
from .models import SCHEMAS, seal_document, timestamp, validate_document

MAX_FINDINGS = 1000
_HEADING = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
_EXCERPT_CHARS = 200


def _text_of(source: dict[str, Any]) -> str:
    before = source["before"]
    return str(before["body"] if isinstance(before, dict) else before)


def _split(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    # Split on "\n" only: other line breaks (CR, FF, U+2028) must stay inside the text so
    # section excerpts remain exact substrings of the source.
    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            sections.append((match.group(1), []))
        else:
            sections[-1][1].append(line)
    return [s for i, s in enumerate(sections) if i or s[1] or len(sections) == 1]


def _view(source: dict[str, Any], limits: tuple[int, int, int]) -> dict[str, Any]:
    max_sections, max_chars, budget = limits
    sections: list[dict[str, Any]] = []
    parts = _split(_text_of(source))
    for index, (heading, lines) in enumerate(parts):
        body = "\n".join(lines).strip("\n")
        if len(sections) >= max_sections or budget <= 0:
            break
        allowed = min(max_chars, budget)
        excerpt = body[:allowed]
        budget -= len(excerpt)
        sections.append(
            {
                "index": index,
                "heading": heading[:max_chars],
                "excerpt": excerpt,
                "char_count": len(body),
                "truncated": len(excerpt) < len(body),
            }
        )
    return {
        "source_id": source["id"],
        "sections": sections,
        "omitted_sections": len(parts) - len(sections),
    }


def bounded_views(
    inventory: dict[str, Any],
    *,
    max_sections_per_source: int = 64,
    max_section_chars: int = 2000,
    max_total_chars: int = 200_000,
) -> list[dict[str, Any]]:
    """Pure deterministic per-source section excerpts with explicit truncation flags."""
    remaining = max_total_chars
    views = []
    for source in inventory["sources"]:
        view = _view(source, (max_sections_per_source, max_section_chars, remaining))
        remaining -= sum(len(s["excerpt"]) for s in view["sections"])
        views.append(view)
    return views


def _excerpt(record: dict[str, str]) -> str:
    for name in ("summary", "title"):
        if record[name]:
            return record[name]
    return record["body"][:_EXCERPT_CHARS]


def _heuristic(source: dict[str, Any], observed_at: str) -> dict[str, Any] | None:
    record = source["before"]
    reason = flag_reason(record["title"], record["summary"], record["body"])
    excerpt = _excerpt(record)
    if not reason or not excerpt:
        return None
    sid = source["id"]
    return {
        "id": "heur-" + file_sha256(f"{sid}\0{reason}".encode())[:16],
        "type": "clarification",
        "source_ids": [sid],
        "excerpts": [{"source_id": sid, "text": excerpt}],
        "rationale": f"Heuristic review hint: {reason}. A hint, not evidence of falsehood.",
        "evidence": [
            {
                "kind": "source_content",
                "status": "unverified",
                "source_ids": [sid],
                "claim": f"Heuristic flag: {reason}",
                "observation": "Raised by the quality heuristic over stored text; unverified.",
                "observed_at": observed_at,
            }
        ],
        "limitations": [
            "Heuristic only; no usage or freshness evidence was consulted.",
        ],
    }


def heuristic_findings(inventory: dict[str, Any], *, observed_at: str) -> list[dict[str, Any]]:
    """Explicit clarification hints for memory records; never proposes changes.

    ``observed_at`` (ISO 8601 with timezone) is required: the caller states when the
    heuristic was run; no placeholder time is ever substituted.
    """
    timestamp(observed_at)
    found = (
        _heuristic(source, observed_at)
        for source in inventory["sources"]
        if source["kind"] == "memory_record"
    )
    return [item for item in found if item is not None]


def load_findings_json(raw: bytes) -> list[dict[str, Any]]:
    """Strictly parse a bounded JSON array of finding objects (hostile input)."""
    if type(raw) is not bytes or len(raw) > MAX_BYTES:
        raise ValueError("findings must be bounded bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_float=_forbidden,
            parse_constant=_forbidden,
        )
    except (json.JSONDecodeError, UnicodeError, RecursionError, TypeError) as exc:
        raise ValueError("invalid findings JSON") from exc
    if type(value) is not list or len(value) > MAX_FINDINGS:
        raise ValueError("findings must be a bounded JSON array")
    if any(type(item) is not dict for item in value):
        raise ValueError("each finding must be an object")
    return value


def build_packet(
    inventory: dict[str, Any],
    *,
    findings: Sequence[dict[str, Any]] = (),
    proposals: Sequence[dict[str, Any]] = (),
    finder: Callable[[list[dict[str, Any]]], Sequence[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Seal inventory + explicit findings/proposals; validation rejects bad excerpts."""
    checked = validate_document(inventory)
    if checked["schema"] != SCHEMAS["inventory"]:
        raise ValueError("expected an inventory")
    gathered = list(findings)
    if finder is not None:
        gathered.extend(finder(bounded_views(checked)))
    return seal_document(
        {
            "schema": SCHEMAS["packet"],
            "profile": checked["profile"],
            "inventory": checked,
            "findings": gathered,
            "proposals": list(proposals),
        }
    )
