"""MDStore — flatfile titled-markdown store driver.

On-disk format (hand-editable and grep-able is a HARD requirement):

    # <Category>

    ## <Entry Title>
    <first line = the summary>
    <rest of the body...>

    ## <Another Title>
    ...

    # <Another Category>
    ...

Rules the parser enforces (tolerant of hand edits):
  * A top-level ``# Heading`` opens a category. Text before the first
    category falls under category "general".
  * A ``## Heading`` opens an entry; the heading text is the title.
  * ``key`` = slug(title). Titles should be unique; on collision the
    later entry wins on write (matching sqlite upsert semantics).
  * ``summary`` = the FIRST non-blank line of the entry body.
  * ``body`` = everything under the ## heading (including the summary
    line), trimmed.

Writing re-serializes the whole file grouped by category. Hand edits and
programmatic edits round-trip through the same representation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import Entry, MemoryStore

_H1_RE = re.compile(r"^#\s+(.*\S)\s*$")
_H2_RE = re.compile(r"^##\s+(.*\S)\s*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_DEFAULT_CATEGORY = "general"


def slugify(title: str) -> str:
    s = _SLUG_RE.sub("-", (title or "").strip().lower()).strip("-")
    return s or "untitled"


def _first_line(body: str) -> str:
    for line in (body or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


class _Record:
    __slots__ = ("body", "category", "key", "summary", "title")

    def __init__(self, key, title, summary, category, body):
        self.key = key
        self.title = title
        self.summary = summary
        self.category = category
        self.body = body


class MDStore(MemoryStore):
    def __init__(self, path):
        self._path = Path(str(path))
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # -- parsing -----------------------------------------------------------

    def _read(self) -> Dict[str, _Record]:
        """Parse the file into {key: _Record}. Missing file -> empty."""
        records: Dict[str, _Record] = {}
        if not self._path.exists():
            return records
        text = self._path.read_text(encoding="utf-8")

        category = _DEFAULT_CATEGORY
        title: Optional[str] = None
        body_lines: List[str] = []

        def _flush():
            if title is None:
                return
            body = "\n".join(body_lines).strip()
            key = slugify(title)
            records[key] = _Record(
                key=key,
                title=title,
                summary=_first_line(body),
                category=category,
                body=body,
            )

        for line in text.splitlines():
            m1 = _H1_RE.match(line)
            if m1:
                _flush()
                title = None
                body_lines = []
                category = m1.group(1).strip()
                continue
            m2 = _H2_RE.match(line)
            if m2:
                _flush()
                title = m2.group(1).strip()
                body_lines = []
                continue
            if title is not None:
                body_lines.append(line)
        _flush()
        return records

    # -- serialization -----------------------------------------------------

    def _write(self, records: Dict[str, _Record]) -> None:
        # group by category, ordered (category, title)
        by_cat: Dict[str, List[_Record]] = {}
        for rec in records.values():
            by_cat.setdefault(rec.category, []).append(rec)

        out: List[str] = []
        for cat in sorted(by_cat.keys()):
            out.append(f"# {cat}")
            out.append("")
            for rec in sorted(by_cat[cat], key=lambda r: r.title.lower()):
                out.append(f"## {rec.title}")
                body = rec.body.strip()
                if body:
                    out.append(body)
                out.append("")
        text = "\n".join(out).rstrip() + "\n"
        self._path.write_text(text, encoding="utf-8")

    # -- interface ---------------------------------------------------------

    def list(self) -> List[Entry]:
        records = self._read()
        rows: List[Entry] = [
            (r.key, r.title, r.summary, r.category)
            for r in records.values()
        ]
        rows.sort(key=lambda t: (t[3].lower(), t[1].lower()))
        return rows

    def get(self, key: str) -> Optional[str]:
        rec = self._read().get(key)
        return rec.body if rec else None

    def search(self, query: str) -> List[str]:
        q = (query or "").strip().lower()
        if not q:
            return []
        hits: List[Tuple[str, str]] = []  # (key, sort_key)
        for r in self._read().values():
            hay = f"{r.title}\n{r.summary}\n{r.body}".lower()
            if q in hay:
                sort_key = r.category.lower() + "\x00" + r.title.lower()
                hits.append((r.key, sort_key))
        hits.sort(key=lambda t: t[1])
        return [k for k, _ in hits]

    def put(
        self,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
        body: str,
    ) -> None:
        records = self._read()
        # summary is derived from the first body line on read; ensure the
        # body leads with the given summary so round-trips are stable.
        body = (body or "").strip()
        if summary and _first_line(body) != summary.strip():
            body = summary.strip() + ("\n\n" + body if body else "")
        records[key] = _Record(
            key=key,
            title=title,
            summary=summary or _first_line(body),
            category=category or _DEFAULT_CATEGORY,
            body=body,
        )
        self._write(records)

    def remove(self, key: str) -> None:
        records = self._read()
        if key in records:
            del records[key]
            self._write(records)
