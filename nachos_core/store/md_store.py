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

import builtins
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from .base import Entry, EntryMeta, MemoryStore

_H1_RE = re.compile(r"^#\s+(.*\S)\s*$")
_H2_RE = re.compile(r"^##\s+(.*\S)\s*$")
_META_RE = re.compile(r"^<!--\s*(.+?)\s*-->$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_DEFAULT_CATEGORY = "general"


def _format_ts(ts: float, approx: bool = False) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    iso = (
        dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if dt.microsecond
        else dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    return f"~{iso}" if approx else iso


def _parse_ts(raw: str) -> tuple[float | None, bool]:
    val = raw.strip()
    approx = val.startswith("~")
    if approx:
        val = val[1:].strip()
    if val.endswith("Z"):
        val = val[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(val)
    except ValueError:
        return None, approx
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp(), approx


def _parse_meta_comment(line: str) -> EntryMeta | None:
    match = _META_RE.match(line.strip())
    if not match:
        return None
    fields: dict[str, str] = {}
    for part in match.group(1).split("|"):
        k, sep, v = part.partition(":")
        if sep:
            fields[k.strip().lower()] = v.strip()
    if not any(k in fields for k in ("created", "updated", "superseded_by")):
        return None
    created_at, approx_c = _parse_ts(fields["created"]) if "created" in fields else (None, False)
    updated_at, approx_u = _parse_ts(fields["updated"]) if "updated" in fields else (None, False)
    superseded_by = fields.get("superseded_by") or None
    return EntryMeta(
        created_at=created_at,
        updated_at=updated_at,
        superseded_by=superseded_by,
        approx_timestamps=approx_c or approx_u,
    )


def _format_meta_comment(meta: EntryMeta) -> str | None:
    parts: list[str] = []
    if meta.created_at is not None:
        parts.append(f"created: {_format_ts(meta.created_at, meta.approx_timestamps)}")
    if meta.updated_at is not None:
        parts.append(f"updated: {_format_ts(meta.updated_at, meta.approx_timestamps)}")
    if meta.superseded_by:
        parts.append(f"superseded_by: {meta.superseded_by}")
    return f"<!-- {' | '.join(parts)} -->" if parts else None


def slugify(title: str) -> str:
    s = _SLUG_RE.sub("-", (title or "").strip().lower()).strip("-")
    return s or "untitled"


def _first_line(body: str) -> str:
    for line in (body or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


class _Record:
    __slots__ = ("body", "category", "key", "meta", "summary", "title")

    def __init__(
        self,
        key: str,
        title: str,
        summary: str,
        category: str,
        body: str,
        meta: EntryMeta | None = None,
    ) -> None:
        self.key = key
        self.title = title
        self.summary = summary
        self.category = category
        self.body = body
        self.meta = meta or EntryMeta()


class MDStore(MemoryStore):
    def __init__(self, path: str | os.PathLike[str], *, read_only: bool = False) -> None:
        self._path = Path(str(path))
        self._read_only = read_only
        if not read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._generation: int = 0
        self._parsed_cache: tuple[tuple[int, int, int, int, int], dict[str, _Record]] | None = None

    @property
    def read_only(self) -> bool:
        return self._read_only

    # -- parsing -----------------------------------------------------------

    @staticmethod
    def _parse(text: str) -> dict[str, _Record]:
        records: dict[str, _Record] = {}
        category = _DEFAULT_CATEGORY
        title: str | None = None
        body_lines: list[str] = []
        meta: EntryMeta | None = None

        def _flush() -> None:
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
                meta=meta,
            )

        for line in text.splitlines():
            m1 = _H1_RE.match(line)
            if m1:
                _flush()
                title, body_lines, meta = None, [], None
                category = m1.group(1).strip()
                continue
            m2 = _H2_RE.match(line)
            if m2:
                _flush()
                title, body_lines, meta = m2.group(1).strip(), [], None
                continue
            if title is not None:
                if meta is None and not any(ln.strip() for ln in body_lines):
                    parsed_meta = _parse_meta_comment(line)
                    if parsed_meta is not None:
                        meta = parsed_meta
                        continue
                body_lines.append(line)
        _flush()
        return records

    def _read(self) -> dict[str, _Record]:
        """Parse the file into {key: _Record}, cached by fingerprint. Missing file -> empty."""
        fp = self.fingerprint()
        if self._parsed_cache is not None and self._parsed_cache[0] == fp:
            return self._parsed_cache[1]
        if fp[1] == -1:
            records: dict[str, _Record] = {}
        else:
            try:
                records = self._parse(self._path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return {}
        self._parsed_cache = (fp, records)
        return records

    # -- serialization -----------------------------------------------------

    def _write(self, records: dict[str, _Record]) -> None:
        # group by category, ordered (category, title)
        by_cat: dict[str, list[_Record]] = {}
        for rec in records.values():
            by_cat.setdefault(rec.category, []).append(rec)

        out: list[str] = []
        for cat in sorted(by_cat.keys()):
            out.append(f"# {cat}")
            out.append("")
            for rec in sorted(by_cat[cat], key=lambda r: r.title.lower()):
                out.append(f"## {rec.title}")
                meta_line = _format_meta_comment(rec.meta)
                if meta_line:
                    out.append(meta_line)
                body = rec.body.strip()
                if body:
                    out.append(body)
                out.append("")
        text = "\n".join(out).rstrip() + "\n"
        self._path.write_text(text, encoding="utf-8")
        self._generation += 1
        self._parsed_cache = None

    # -- interface ---------------------------------------------------------

    def fingerprint(self) -> tuple[int, int, int, int, int]:
        """Cheap change token combining in-process writes and file stat metadata."""
        try:
            st = self._path.stat()
        except OSError:
            return (self._generation, -1, -1, -1, -1)
        return (self._generation, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def list(self) -> builtins.list[Entry]:
        records = self._read()
        rows: list[Entry] = [(r.key, r.title, r.summary, r.category) for r in records.values()]
        rows.sort(key=lambda t: (t[3].lower(), t[1].lower()))
        return rows

    def get(self, key: str) -> str | None:
        rec = self._read().get(key)
        return rec.body if rec else None

    def bodies(self) -> dict[str, str]:
        return {key: rec.body for key, rec in self._read().items()}

    def search(self, query: str) -> builtins.list[str]:
        q = (query or "").strip().lower()
        if not q:
            return []
        hits: list[tuple[str, str]] = []  # (key, sort_key)
        for r in self._read().values():
            hay = f"{r.title}\n{r.summary}\n{r.body}".lower()
            if q in hay:
                sort_key = r.category.lower() + "\x00" + r.title.lower()
                hits.append((r.key, sort_key))
        hits.sort(key=lambda t: t[1])
        return [k for k, _ in hits]

    def get_meta(self, key: str) -> EntryMeta | None:
        rec = self._read().get(key)
        return rec.meta if rec else None

    def all_meta(self) -> dict[str, EntryMeta]:
        return {key: rec.meta for key, rec in self._read().items()}

    def set_superseded_by(
        self, key: str, superseded_by: str | None, *, updated_at: float | None = None
    ) -> bool:
        records = dict(self._read())
        rec = records.get(key)
        if rec is None:
            return False
        ts = updated_at if updated_at is not None else time.time()
        new_meta = EntryMeta(
            created_at=rec.meta.created_at if rec.meta.created_at is not None else ts,
            updated_at=ts,
            superseded_by=superseded_by,
            approx_timestamps=rec.meta.approx_timestamps,
        )
        records[key] = _Record(
            key=rec.key,
            title=rec.title,
            summary=rec.summary,
            category=rec.category,
            body=rec.body,
            meta=new_meta,
        )
        self._write(records)
        return True

    def backfill_timestamps(self, timestamps: Mapping[str, tuple[float, float, bool]]) -> int:
        records = dict(self._read())
        updated = 0
        for key, (created_at, updated_at, approx) in timestamps.items():
            rec = records.get(key)
            if rec is None:
                continue
            if rec.meta.created_at is not None and rec.meta.updated_at is not None:
                continue
            records[key] = _Record(
                key=rec.key,
                title=rec.title,
                summary=rec.summary,
                category=rec.category,
                body=rec.body,
                meta=EntryMeta(
                    created_at=rec.meta.created_at
                    if rec.meta.created_at is not None
                    else created_at,
                    updated_at=rec.meta.updated_at
                    if rec.meta.updated_at is not None
                    else updated_at,
                    superseded_by=rec.meta.superseded_by,
                    approx_timestamps=approx,
                ),
            )
            updated += 1
        if updated:
            self._write(records)
        return updated

    def put(
        self,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
        body: str,
    ) -> None:
        if "\n" in title or "\r" in title:
            raise ValueError("Markdown injection detected: newlines not allowed in title")
        if "\n" in category or "\r" in category:
            raise ValueError("Markdown injection detected: newlines not allowed in category")

        body = (body or "").strip()
        records = dict(self._read())
        # summary is derived from the first body line on read; ensure the
        # body leads with the given summary so round-trips are stable.
        if summary and _first_line(body) != summary.strip():
            body = summary.strip() + ("\n\n" + body if body else "")

        for line in body.splitlines():
            if _H1_RE.match(line) or _H2_RE.match(line):
                raise ValueError("Markdown injection detected: headers not allowed in body")

        now = time.time()
        existing = records.get(key)
        if existing is not None and existing.meta.created_at is not None:
            meta = EntryMeta(
                created_at=existing.meta.created_at,
                updated_at=now,
                superseded_by=existing.meta.superseded_by,
                approx_timestamps=existing.meta.approx_timestamps,
            )
        else:
            meta = EntryMeta(
                created_at=now,
                updated_at=now,
                superseded_by=existing.meta.superseded_by if existing is not None else None,
                approx_timestamps=False,
            )

        records[key] = _Record(
            key=key,
            title=title,
            summary=summary or _first_line(body),
            category=category or _DEFAULT_CATEGORY,
            body=body,
            meta=meta,
        )
        self._write(records)

    def remove(self, key: str) -> None:
        records = dict(self._read())
        if key in records:
            del records[key]
            self._write(records)
