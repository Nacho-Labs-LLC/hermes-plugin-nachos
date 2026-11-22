"""Bounded read of current memory/skill/reference sources; never writes or discovers.

Only explicit labels (fixed memory store label plus the caller's exact skill and
reference allowlist) are read, through the pinned no-follow SourceBoundary. A
SQLite store is read through the shared WAL-safe read-only reader
(nachos_core.index.open_read_only_sqlite: direct mode=ro for rollback databases, a
private verified copy for WAL ones), after the pinned boundary has confirmed the label
is a regular file; only a fixed authorizer-restricted SELECT ever runs. Flatfile memory is parsed from captured bytes by a pure reimplementation
of store/md_store.py parsing. Store content (keys, titles, bodies) is hostile:
it never selects a path, locator or command. Duplicate record keys are rejected
(md_store would silently keep the later one, which would hide a record).
"""

from __future__ import annotations

import errno
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..index import close_sqlite, open_read_only_sqlite
from ..store.md_store import slugify
from .boundary import AcquisitionError, SourceBoundary
from .hashing import canonical_json, file_sha256, memory_sha256

SQLITE_LABEL = "nachos/memory.db"
FLATFILE_LABEL = "nachos/memory.md"
MAX_LABELS = 256
MAX_RECORDS = 1024
MAX_FILE_BYTES = 512 * 1024
MAX_RECORD_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_SQLITE_BYTES = 8 * 1024 * 1024
_FIELDS = ("key", "title", "summary", "category", "body")
_QUERY = "SELECT key, title, summary, category, body FROM entries ORDER BY key LIMIT ?"
_H1_RE = re.compile(r"^#\s+(.*\S)\s*$")
_H2_RE = re.compile(r"^##\s+(.*\S)\s*$")


@dataclass(frozen=True)
class SourceSpec:
    store: str
    skills: tuple[str, ...] = ()
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class CurrentSource:
    identity: str
    kind: str
    role: str
    locator: str
    sha256: str
    before: str | dict[str, str]
    byte_size: int
    char_size: int


def _check_structure(spec: SourceSpec) -> str:
    """Spec shape only (no per-label policy); returns the fixed memory-store label."""
    if spec.store not in ("sqlite", "flatfile"):
        raise AcquisitionError("invalid_store")
    labels = (*spec.skills, *spec.references)
    if (
        type(spec.skills) is not tuple
        or type(spec.references) is not tuple
        or len(spec.skills) > MAX_LABELS
        or len(spec.references) > MAX_LABELS
        or len(set(labels)) != len(labels)
    ):
        raise AcquisitionError("invalid_spec")
    memory = SQLITE_LABEL if spec.store == "sqlite" else FLATFILE_LABEL
    if memory in labels:
        raise AcquisitionError("invalid_spec")
    return memory


def _check_spec(boundary: SourceBoundary, spec: SourceSpec) -> str:
    memory = _check_structure(spec)
    for label in (memory, *spec.skills, *spec.references):
        boundary.require_label(label)
    return memory


def _text(value: Any) -> str:
    if type(value) is not str or "\0" in value:
        raise AcquisitionError("invalid_encoding")
    return value


def _decode(raw: bytes) -> str:
    try:
        return _text(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise AcquisitionError("invalid_encoding") from exc


class _Budget:
    """Total JSON-visible bytes retained across all sources."""

    def __init__(self) -> None:
        self.used = 0

    def spend(self, amount: int) -> None:
        self.used += amount
        if self.used > MAX_TOTAL_BYTES:
            raise AcquisitionError("size_limit")


def _file_source(
    boundary: SourceBoundary, label: str, kind: str, role: str, prefix: str, budget: _Budget
) -> CurrentSource:
    raw = boundary.read_file(label, max_bytes=MAX_FILE_BYTES).data
    text = _decode(raw)
    budget.spend(len(json.dumps(text, ensure_ascii=False).encode("utf-8")))
    return CurrentSource(
        f"{prefix}:{label}", kind, role, label, file_sha256(raw), text, len(raw), len(text)
    )


def _record_source(record: dict[str, str], label: str, budget: _Budget) -> CurrentSource:
    for name in _FIELDS:
        _text(record[name])
    raw = canonical_json(record)
    if len(raw) > MAX_RECORD_BYTES:
        raise AcquisitionError("size_limit")
    budget.spend(len(raw))
    text = raw.decode("utf-8")
    return CurrentSource(
        f"memory:{record['key']}",
        "memory_record",
        "primary",
        label,
        memory_sha256(record),
        dict(record),
        len(raw),
        len(text),
    )


# -- SQLite: detached bytes parsed from a private scratch copy only ---------


def _authorizer(action: int, arg1: str | None, arg2: str | None, *_rest: Any) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 == "entries" and arg2 in _FIELDS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _entries_rows(path: Path) -> list[tuple[Any, ...]]:
    """Read entries through the shared WAL-safe reader (a private copy for WAL databases)."""
    try:
        conn, temp_dir = open_read_only_sqlite(path)
    except (sqlite3.Error, OSError) as exc:
        raise AcquisitionError("invalid_sqlite") from exc
    try:
        # `entries` must be a real table (a view could fabricate rows); reading the
        # schema row executes nothing. Then lock down to the fixed SELECT.
        kinds = conn.execute(
            "SELECT type FROM sqlite_master"
            " WHERE name = 'entries' AND sql NOT LIKE 'CREATE VIRTUAL%'"
        )
        if kinds.fetchall() != [("table",)]:
            raise AcquisitionError("invalid_sqlite")
        conn.set_authorizer(_authorizer)
        return conn.execute(_QUERY, (MAX_RECORDS + 1,)).fetchall()
    except sqlite3.Error as exc:
        raise AcquisitionError("invalid_sqlite") from exc
    finally:
        close_sqlite(conn, temp_dir)


def _sqlite_records(boundary: SourceBoundary) -> list[dict[str, str]]:
    # The pinned open verifies the label is a regular, unaliased file below a symlink-free
    # root; the reader then snapshots it (WAL included) without writing to the source.
    with boundary.open_source(SQLITE_LABEL) as source:
        if source.check_regular().st_size > MAX_SQLITE_BYTES:
            raise AcquisitionError("size_limit")
        path = boundary.root.joinpath(*SQLITE_LABEL.split("/"))
        rows = _entries_rows(path)
    if len(rows) > MAX_RECORDS:
        raise AcquisitionError("record_limit")
    records = []
    for row in rows:
        if len(row) != len(_FIELDS) or any(type(value) is not str for value in row):
            raise AcquisitionError("invalid_sqlite")
        records.append(dict(zip(_FIELDS, row)))
    return records


# -- Flatfile: pure parse matching store/md_store.py semantics --------------


def _first_line(body: str) -> str:
    for line in body.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _flat_record(title: str, category: str, lines: list[str]) -> dict[str, str]:
    body = "\n".join(lines).strip()
    return {
        "key": slugify(title),
        "title": title,
        "summary": _first_line(body),
        "category": category,
        "body": body,
    }


def _append(records: list[dict[str, str]], record: dict[str, str], limit: int) -> None:
    if len(records) >= limit:
        raise AcquisitionError("record_limit")
    records.append(record)


def parse_flatfile(text: str, *, max_records: int | None = None) -> list[dict[str, str]]:
    """Parse md_store's titled-markdown format without constructing MDStore.

    The record count is capped DURING parsing (default ``MAX_RECORDS``): exceeding it
    raises ``AcquisitionError("record_limit")`` before more records are built.
    """
    limit = MAX_RECORDS if max_records is None else max_records
    records: list[dict[str, str]] = []
    category = "general"
    title: str | None = None
    lines: list[str] = []
    # Path.read_text applies universal newlines before splitlines().
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        match1, match2 = _H1_RE.match(line), _H2_RE.match(line)
        if (match1 or match2) and title is not None:
            _append(records, _flat_record(title, category, lines), limit)
        if match1:
            title, lines, category = None, [], match1.group(1).strip()
        elif match2:
            title, lines = match2.group(1).strip(), []
        elif title is not None:
            lines.append(line)
    if title is not None:
        _append(records, _flat_record(title, category, lines), limit)
    return records


def _flat_records(boundary: SourceBoundary) -> list[dict[str, str]]:
    raw = boundary.read_file(FLATFILE_LABEL, max_bytes=MAX_FILE_BYTES * 4).data
    return parse_flatfile(_decode(raw))


def _memory_sources(
    boundary: SourceBoundary, spec: SourceSpec, budget: _Budget
) -> list[CurrentSource]:
    if spec.store == "sqlite":
        records = _sqlite_records(boundary)
        label = SQLITE_LABEL
    else:
        records = _flat_records(boundary)
        label = FLATFILE_LABEL
    if len(records) > MAX_RECORDS:
        raise AcquisitionError("record_limit")
    keys = [record["key"] for record in records]
    if len(set(keys)) != len(keys):
        raise AcquisitionError("duplicate_key")
    return [_record_source(record, label, budget) for record in records]


def _reason_for(exc: AcquisitionError) -> str:
    """Stable per-label reason; a file that does not exist is ``source_not_found``."""
    code = str(exc)
    cause = exc.__cause__
    gone = isinstance(cause, OSError) and cause.errno in (errno.ENOENT, errno.ENOTDIR)
    return "source_not_found" if code == "source_unavailable" and gone else code


def _label_sources(
    boundary: SourceBoundary,
    spec: SourceSpec,
    label: str,
    memory: str,
    budget: _Budget,
) -> list[CurrentSource]:
    boundary.require_label(label)
    if label == memory:
        return _memory_sources(boundary, spec, budget)
    if label in spec.skills:
        return [_file_source(boundary, label, "skill_file", "primary", "skill", budget)]
    return [_file_source(boundary, label, "reference_file", "reference", "reference", budget)]


def read_each_source(
    boundary: SourceBoundary, spec: SourceSpec
) -> tuple[dict[str, CurrentSource], dict[str, str]]:
    """Read every explicit label independently.

    Returns ``(sources_by_identity, failures_by_label)``: a label whose own read fails
    (stable reason code, never content) is reported in ``failures_by_label`` and does not
    hide the other labels. ``source_not_found`` means the file itself does not exist
    .
    Structurally invalid specs (``invalid_store``/``invalid_spec``) still raise.
    """
    memory = _check_structure(spec)
    budget = _Budget()
    found: list[CurrentSource] = []
    failures: dict[str, str] = {}
    for label in (memory, *spec.skills, *spec.references):
        spent = budget.used
        try:
            found += _label_sources(boundary, spec, label, memory, budget)
        except AcquisitionError as exc:
            budget.used = spent
            failures[label] = _reason_for(exc)
    return {item.identity: item for item in found}, failures


def read_current_sources(boundary: SourceBoundary, spec: SourceSpec) -> dict[str, CurrentSource]:
    """Read exactly the memory store and the named skill/reference files.

    Raises AcquisitionError (stable reason, no content) for anything missing,
    unreadable, oversized or malformed; creates nothing.
    """
    _check_spec(boundary, spec)
    found, failures = read_each_source(
        boundary,
        spec,
    )
    for reason in failures.values():
        raise AcquisitionError("source_unavailable" if reason == "source_not_found" else reason)
    return found
