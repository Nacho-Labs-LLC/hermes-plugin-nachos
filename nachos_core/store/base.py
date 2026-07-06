"""Store interface (ABC) + the Entry tuple contract.

All drivers implement the same 5 methods. The interface is deliberately
synchronous and local — see package docstring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

# (key, title, summary, category)
Entry = Tuple[str, str, str, str]


class MemoryStore(ABC):
    """Synchronous, local store of memory entries.

    An "entry" has: key, title, summary, category, body. Only the first
    four (the Entry tuple) ever feed the always-on manifest; the body is
    fetched on demand via get().
    """

    @abstractmethod
    def list(self) -> List[Entry]:
        """Return ALL entries as (key, title, summary, category) tuples.

        Must be cheap — the manifest tier calls this every turn and
        renders every entry. Ordering is (category, title) for stable
        manifest output.
        """

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Return the full body for ``key``, or None if absent."""

    @abstractmethod
    def search(self, query: str) -> List[str]:
        """Return candidate keys matching ``query`` (substring/lexical).

        This is coarse candidate generation for the prefetch tier — the
        Scorer does the fine ranking. Empty query -> empty list.
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
