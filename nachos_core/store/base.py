"""Store interface (ABC) + the Entry tuple contract.

All drivers implement the same 5 methods. The interface is deliberately
synchronous and local — see package docstring.
"""

from __future__ import annotations

import builtins
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

# (key, title, summary, category)
Entry = tuple[str, str, str, str]


@dataclass(frozen=True)
class EntryMeta:
    """Per-entry timestamp and validity metadata."""

    created_at: float | None = None
    updated_at: float | None = None
    superseded_by: str | None = None
    approx_timestamps: bool = False


class MemoryStore(ABC):
    """Synchronous, local store of memory entries.

    An "entry" has: key, title, summary, category, body. Only the first
    four (the Entry tuple) ever feed the always-on manifest; the body is
    fetched on demand via get().
    """

    @abstractmethod
    def list(self) -> builtins.list[Entry]:
        """Return ALL entries as (key, title, summary, category) tuples.

        Must be cheap — the manifest calls this every turn and
        renders every entry. Ordering is (category, title) for stable
        manifest output.
        """

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Return the full body for ``key``, or None if absent."""

    @abstractmethod
    def search(self, query: str) -> builtins.list[str]:
        """Return candidate keys matching ``query`` (substring/lexical).

        Used by nachos_memory_recall's query mode for fuzzy lookup.
        Empty query -> empty list.
        """

    @abstractmethod
    def put(
        self,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
        body: str,
    ) -> None:
        """Insert or replace the entry identified by ``key``."""

    @abstractmethod
    def remove(self, key: str) -> None:
        """Delete the entry identified by ``key`` (no-op if absent)."""

    def bodies(self) -> dict[str, str]:
        """Return {key: body} for every entry in one pass.

        Default implementation calls get() per entry; drivers override it
        when a bulk read is cheaper (e.g. the flat file parses once).
        """
        out: dict[str, str] = {}
        for key, *_rest in self.list():
            body = self.get(key)
            if body is not None:
                out[key] = body
        return out

    def fingerprint(self) -> object:
        """Cheap token that changes whenever the store's contents change.

        Used to validate in-memory caches (including against external edits).
        Returns ``None`` if the driver cannot fingerprint its backing storage,
        which disables cross-query caching for that driver.
        """
        return None

    @property
    def read_only(self) -> bool:
        """True when this store was opened read-only and must never be mutated."""
        return False

    def get_meta(self, key: str) -> EntryMeta | None:
        """Return timestamp and validity metadata for ``key``, or None if absent."""
        return EntryMeta() if self.get(key) is not None else None

    def all_meta(self) -> dict[str, EntryMeta]:
        """Return ``{key: EntryMeta}`` for every entry in one pass."""
        return {key: EntryMeta() for key, *_rest in self.list()}

    def set_superseded_by(
        self, key: str, superseded_by: str | None, *, updated_at: float | None = None
    ) -> bool:
        """Record (or clear) ``superseded_by`` on ``key`` without deleting it."""
        del key, superseded_by, updated_at
        return False

    def backfill_timestamps(self, timestamps: Mapping[str, tuple[float, float, bool]]) -> int:
        """Populate missing ``(created_at, updated_at, approx)`` metadata on existing entries."""
        del timestamps
        return 0
