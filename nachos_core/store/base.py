"""Store interface (ABC) + the Entry tuple contract.

All drivers implement the same 5 methods. The interface is deliberately
synchronous and local — see package docstring.
"""

from __future__ import annotations

import builtins
from abc import ABC, abstractmethod

# (key, title, summary, category)
Entry = tuple[str, str, str, str]


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
