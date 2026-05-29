"""Hermes adapter for the Nachos memory layer.

This is the ONLY file in nachos-memory that touches hermes_*. It wraps
Hermes' built-in memory file (~/.hermes/<profile>/memory.txt) into the
MemoryReadProtocol the manifest builder expects.

Read-side only. The manifest never writes — it just summarizes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HermesMemoryReader:
    """Wraps Hermes' built-in memory + extracted facts as MemoryReadProtocol."""

    def __init__(self, hermes_home: Optional[Path] = None):
        self.hermes_home = hermes_home or _resolve_hermes_home()
        self._fact_store = None  # JsonlFactStore — set by plugin after init

    def set_fact_store(self, fact_store) -> None:
        """Attach an extracted-fact store so fact_counts include extractions.

        Called by the plugin after both pieces are constructed. Decouples
        adapter init from extractor init.
        """
        self._fact_store = fact_store

    # -- MemoryReadProtocol -------------------------------------------------

    def list_entries(self, kind: Optional[str] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Return entries from memory.txt + user.txt parsed as records.

        Hermes stores memory as plain text with `§` separators (see
        agent/memory_*.py). Each chunk between separators is one entry.
        Entries don't carry an explicit 'kind' — we infer it from
        content shape (preference vs fact vs note).
        """
        entries: List[Dict[str, Any]] = []
        memories_dir = _resolve_memories_dir(self.hermes_home)
        for path, target in [
            (memories_dir / "MEMORY.md", "memory"),
            (memories_dir / "USER.md", "user"),
        ]:
            if not path.exists():
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.debug("Could not read %s: %s", path, e)
                continue
            for chunk in _split_chunks(raw):
                inferred_kind = _infer_kind(chunk, target)
                if kind and inferred_kind != kind:
                    continue
                entries.append({
                    "content": chunk,
                    "kind": inferred_kind,
                    "tags": [],
                    "target": target,
                    "created_at": "",  # Hermes memory doesn't track per-entry timestamps
                })
                if len(entries) >= limit:
                    return entries
        return entries

    def list_recent_topics(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Pull recent session titles from Hermes' session DB.

        Best-effort. Returns [] if the DB isn't available or the schema
        changed; manifest builder catches and continues.
        """
        try:
            import sqlite3
            db = self.hermes_home / "state.db"
            if not db.exists():
                return []
            con = sqlite3.connect(str(db))
            try:
                # Hermes session table schema is { id, started_at, title?, ... }
                # — schema varies; try with title first, fall back to id.
                try:
                    cur = con.execute(
                        "SELECT title, started_at FROM sessions "
                        "WHERE title IS NOT NULL AND title != '' "
                        "ORDER BY started_at DESC LIMIT ?",
                        (limit,),
                    )
                    rows = cur.fetchall()
                except sqlite3.OperationalError:
                    cur = con.execute(
                        "SELECT id, started_at FROM sessions "
                        "ORDER BY started_at DESC LIMIT ?",
                        (limit,),
                    )
                    rows = cur.fetchall()
            finally:
                con.close()
            return [
                {
                    "topic": str(title) if title else "",
                    "session_age": _format_age(started_at),
                    "session_id": "",
                }
                for title, started_at in rows
                if title
            ]
        except Exception as e:
            logger.debug("list_recent_topics failed: %s", e)
            return []

    def fact_counts_by_kind(self) -> Dict[str, int]:
        """Group counts across:
          1. Inferred kinds from raw memory entries (heuristic)
          2. Extracted MemoryFact rows from the fact store, if attached

        Extracted-fact counts are namespaced as 'extracted:<kind>' so
        the manifest renders them distinctly from heuristic-classified
        memory entries. Both sets are useful — the heuristic gives you
        a sense of the raw memory file, the extracted facts give you
        the structured triples.
        """
        counts: Dict[str, int] = {}
        for entry in self.list_entries(limit=1000):
            k = entry["kind"]
            counts[k] = counts.get(k, 0) + 1

        if self._fact_store is not None:
            try:
                for fact in self._fact_store.list_all():
                    key = f"extracted:{fact.kind}"
                    counts[key] = counts.get(key, 0) + 1
            except Exception as e:
                logger.debug("Could not load fact counts: %s", e)

        return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_hermes_home() -> Path:
    """Find the active HERMES_HOME — prefer the official helper if present."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore
        return get_hermes_home()
    except Exception:
        from os import environ
        return Path(environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def _resolve_memories_dir(hermes_home: Path) -> Path:
    """Resolve the memories directory, preferring Hermes' official helper.

    Falling back on the literal `<home>/memories` path is fine for current
    Hermes layout but will silently desync if upstream renames. Importing
    from tools.memory_tool keeps us tethered to the canonical location.
    """
    try:
        from tools.memory_tool import get_memory_dir  # type: ignore
        return get_memory_dir()
    except Exception:
        return hermes_home / "memories"


_CHUNK_SEP_RE = re.compile(r"\n\s*§\s*\n")


def _split_chunks(raw: str) -> List[str]:
    return [c.strip() for c in _CHUNK_SEP_RE.split(raw) if c.strip()]


_PREF_HEURISTICS = (
    "prefers", "preference", "favorite", "default", "always", "never",
)
_PROJECT_HEURISTICS = ("repo", "project", "directory", "/dev/", "github.com")
_ENV_HEURISTICS = ("os ", "macos", "linux", "windows", "bash", "zsh",
                   "python", "node", "version")


def _infer_kind(content: str, target: str) -> str:
    """Heuristic kind classification — good enough for manifest grouping.

    The user.txt file is treated as 'preference' by default since
    that's its purpose. memory.txt entries get scanned for hints.
    """
    if target == "user":
        return "preference"
    lower = content.lower()
    if any(h in lower for h in _PREF_HEURISTICS):
        return "preference"
    if any(h in lower for h in _ENV_HEURISTICS):
        return "environment"
    if any(h in lower for h in _PROJECT_HEURISTICS):
        return "project"
    return "general"


def _format_age(started_at) -> str:
    """Convert sessions.db started_at to a rough relative age string."""
    try:
        from datetime import datetime, timezone
        if isinstance(started_at, (int, float)):
            then = datetime.fromtimestamp(started_at, tz=timezone.utc)
        else:
            # ISO string fallback
            then = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - then
        if delta.days >= 7:
            return f"{delta.days // 7}w ago"
        if delta.days >= 1:
            return f"{delta.days}d ago"
        if delta.seconds >= 3600:
            return f"{delta.seconds // 3600}h ago"
        return "recent"
    except Exception:
        return "recent"
