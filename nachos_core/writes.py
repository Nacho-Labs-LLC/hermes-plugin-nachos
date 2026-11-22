"""Pure helpers of the write path: normalising and gating a prospective write, and change-log
snapshots. ``MemoryService.put`` is the only caller that touches stores."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Callable

from .quality import low_signal_reason
from .store.base import MemoryStore
from .store.md_store import slugify

Snapshot = tuple[str, str, str, str]  # (title, summary, category, body)


@dataclass
class Draft:
    """A normalised, gated prospective write (see ``MemoryService.prepare``)."""

    title: str
    summary: str
    category: str
    body: str
    key: str
    scope: str
    error: str | None = None


def prepare_draft(
    scope_names: Collection[str],
    redact: Callable[[str], str],
    *,
    title: str,
    body: str,
    summary: str = "",
    category: str = "general",
    scope: str,
    key: str | None = None,
) -> Draft:
    """Normalise and gate a write without touching any store.

    The single source of truth for what ``put`` accepts: redaction, summary defaulting, key
    derivation and the low-signal gates all live here, so dry-runs (``tools/import_memories.py``)
    cannot drift from real writes. ``Draft.error`` is set when the write would be rejected.
    """
    if scope not in scope_names:
        return Draft(
            "",
            "",
            "",
            "",
            "",
            scope,
            f"Unknown or disabled scope {scope!r}; available: {list(scope_names)}",
        )
    title = redact((title or "").strip())
    body = redact((body or "").strip())
    summary = redact((summary or "").strip())
    if not summary:
        summary = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    category = (category or "general").strip() or "general"
    draft = Draft(title, summary, category, body, key or (slugify(title) if title else ""), scope)
    draft.error = _gate(draft)
    return draft


def _gate(draft: Draft) -> str | None:
    if not draft.title:
        return "title is required"
    reason = low_signal_reason(draft.body)
    if reason:
        return f"body is low-signal ({reason}); write real content, not a placeholder."
    reason = low_signal_reason(draft.summary)
    if reason:
        return (
            f"summary is low-signal ({reason}); this is the ONLY "
            "text the agent sees in the manifest for this entry "
            "without recalling it — write a real, keyword-dense "
            "summary."
        )
    return None


def snapshot(store: MemoryStore, key: str) -> Snapshot | None:
    """(title, summary, category, body) for ``key``, or None if absent."""
    body = store.get(key)
    if body is None:
        return None
    for k, title, summary, category in store.list():
        if k == key:
            return title, summary, category, body
    return None


def same_content(prev: Snapshot | None, new: Snapshot) -> bool:
    if prev is None:
        return False
    if prev == new:
        return True
    # The flat-file store stores the summary as the first line of the body.
    title, summary, category, body = new
    led = summary + ("\n\n" + body if body else "")
    return prev == (title, summary, category, led)
