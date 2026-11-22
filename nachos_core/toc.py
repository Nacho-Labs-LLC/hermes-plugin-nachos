"""TOC manifest — the always-on, never-truncating table of contents.

The always-on manifest. Renders EVERY entry as one line,
grouped by category:

    # Memory Manifest
    ## <category>
    - <title> — <summary>

HARD INVARIANT: render_toc never drops an entry. It scales with entry
COUNT (one line each), not entry body size — that's what removes the
char ceiling. If a soft char budget is given and the full render exceeds
it, summaries are SHORTENED (and, as a last resort, dropped, leaving the
title) but no entry line is ever omitted. The agent must always be able
to see that an entry exists so it can recall it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .store.base import Entry

_HEADER = "# Memory Manifest"
_HINT = "(Index only — full entry via memory recall.)"
_ELLIPSIS = "\u2026"


def build_toc(entries: Iterable[Entry]) -> list[Entry]:
    """Normalize + stable-sort entries by (category, title)."""
    rows = [(e[0], e[1], e[2], e[3]) for e in entries]
    rows.sort(key=lambda t: (t[3].lower(), t[1].lower()))
    return rows


def render_toc(
    entries: Iterable[Entry],
    *,
    char_budget: int | None = None,
    superseded_count: int = 0,
    link_counts: Mapping[str, int] | None = None,
) -> str:
    """Render the full manifest. Never omits an entry.

    char_budget: soft target. If exceeded, summaries are progressively
                shortened, then dropped to title-only — entries are NEVER
                removed.
    link_counts: ``{key: linked entries}``. Entries with links get a ``[n links]`` suffix, but
                only when the annotated manifest still fits ``char_budget`` without
                shortening anything more than the plain one needs; otherwise it is left out.
    """
    rows = build_toc(entries)
    footer = (
        f"{superseded_count} superseded {'entry' if superseded_count == 1 else 'entries'} hidden."
        if superseded_count > 0
        else ""
    )
    if not rows:
        return f"{_HEADER}\n{_HINT}\n\n{footer}" if footer else ""

    plain, plain_cap = _fit(rows, footer, char_budget, None)
    counts = {k: n for k, n in (link_counts or {}).items() if n > 0}
    if not counts or not any(row[0] in counts for row in rows):
        return plain
    annotated, cap = _fit(rows, footer, char_budget, counts)
    fits = char_budget is None or len(annotated) <= char_budget
    return annotated if fits and cap == plain_cap else plain


def _fit(
    rows: list[Entry], footer: str, char_budget: int | None, counts: Mapping[str, int] | None
) -> tuple[str, int | None]:
    """Render at the longest summary cap that fits ``char_budget``; returns (text, cap used)."""

    def _render(summary_cap: int | None) -> str:
        lines: list[str] = [_HEADER, _HINT]
        current_cat = None
        for key, title, summary, category in rows:
            if category != current_cat:
                lines.append("")
                lines.append(f"## {category}")
                current_cat = category
            summ = summary or ""
            if summary_cap is not None and len(summ) > summary_cap:
                summ = summ[: max(0, summary_cap - 1)].rstrip() + _ELLIPSIS
            n = (counts or {}).get(key, 0)
            tag = f" [{n} link{'s' if n != 1 else ''}]" if n else ""
            lines.append(f"- {title} \u2014 {summ}{tag}" if summ else f"- {title}{tag}")
        if footer:
            lines.append("")
            lines.append(footer)
        return "\n".join(lines)

    full = _render(None)
    if char_budget is None or len(full) <= char_budget:
        return full, None
    # Over budget: progressively shorten summaries. Try a descending set
    # of caps; entries are never dropped, only their summaries shrink.
    for cap in (120, 90, 60, 40, 24, 0):
        candidate = _render(cap)
        if len(candidate) <= char_budget:
            return candidate, cap
    # Even title-only (cap=0) exceeds budget — return it anyway. The
    # invariant (never omit an entry) outranks the soft budget.
    return _render(0), 0
