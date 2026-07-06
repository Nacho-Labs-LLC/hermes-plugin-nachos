"""TOC manifest — the always-on, never-truncating table of contents.

Tier 1 of the memory-manifest design. Renders EVERY entry as one line,
grouped by category:

    # Memory Manifest
    ## <category>
    - <title> — <summary>            (prefetched entries marked ►)

HARD INVARIANT: render_toc never drops an entry. It scales with entry
COUNT (one line each), not entry body size — that's what removes the
char ceiling. If a soft char budget is given and the full render exceeds
it, summaries are SHORTENED (and, as a last resort, dropped, leaving the
title) but no entry line is ever omitted. The agent must always be able
to see that an entry exists so it can recall it.

This module is intentionally separate from the legacy manifest.py (which
belongs to the old extraction design and is removed in phase 2).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import List, Optional, Set

from .store.base import Entry

_HEADER = "# Memory Manifest"
_HINT = "(Index only — full entry via memory recall. \u25ba = prefetched this turn.)"
_MARK = "\u25ba "   # ► prefetched
_ELLIPSIS = "\u2026"


def build_toc(entries: Iterable[Entry]) -> List[Entry]:
    """Normalize + stable-sort entries by (category, title)."""
    rows = [
        (e[0], e[1], e[2], e[3])
        for e in entries
    ]
    rows.sort(key=lambda t: (t[3].lower(), t[1].lower()))
    return rows


def render_toc(
    entries: Iterable[Entry],
    *,
    prefetched: Optional[Set[str]] = None,
    char_budget: Optional[int] = None,
) -> str:
    """Render the full manifest. Never omits an entry.

    prefetched: keys to mark with ► (entries whose bodies are also being
                injected this turn via the prefetch tier).
    char_budget: soft target. If exceeded, summaries are progressively
                shortened, then dropped to title-only — entries are NEVER
                removed.
    """
    rows = build_toc(entries)
    if not rows:
        return ""
    marks = prefetched or set()

    def _render(summary_cap: Optional[int]) -> str:
        lines: List[str] = [_HEADER, _HINT]
        current_cat = None
        for (key, title, summary, category) in rows:
            if category != current_cat:
                lines.append("")
                lines.append(f"## {category}")
                current_cat = category
            mark = _MARK if key in marks else ""
            summ = summary or ""
            if summary_cap is not None and len(summ) > summary_cap:
                summ = summ[: max(0, summary_cap - 1)].rstrip() + _ELLIPSIS
            if summ:
                lines.append(f"- {mark}{title} \u2014 {summ}")
            else:
                lines.append(f"- {mark}{title}")
        return "\n".join(lines)

    full = _render(None)
    if char_budget is None or len(full) <= char_budget:
        return full

    # Over budget: progressively shorten summaries. Try a descending set
    # of caps; entries are never dropped, only their summaries shrink.
    for cap in (120, 90, 60, 40, 24, 0):
        candidate = _render(cap)
        if len(candidate) <= char_budget:
            return candidate
    # Even title-only (cap=0) exceeds budget — return it anyway. The
    # invariant (never omit an entry) outranks the soft budget.
    return _render(0)
