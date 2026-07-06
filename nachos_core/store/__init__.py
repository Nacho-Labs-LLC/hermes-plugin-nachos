"""Nachos memory store seam — 5-method synchronous local interface.

The manifest / prefetch / recall tiers sit ABOVE this interface and do
not care which driver is underneath. Two drivers ship in v1:

  * SqliteStore (default) — stdlib sqlite3, fast indexed local store.
  * MDStore                — flatfile titled-markdown, hand-editable.

HARD DESIGN CONSTRAINT: the interface is synchronous and local. It
cannot express a network/async backend by design — anyone wanting
cloud/vector memory uses a different Hermes MemoryProvider. This keeps
the "list every entry cheaply" manifest requirement always satisfiable.

Entry shape everywhere: a 4-tuple (key, title, summary, category).
  key      — stable identifier (slug of title for MDStore, PK for sqlite)
  title    — short human label, shown in the manifest
  summary  — one keyword-dense line, shown in the manifest
  category — grouping axis for the manifest (e.g. 'user', 'deposco')
  body     — full content (NOT in the tuple; fetched via get(key))
"""

from __future__ import annotations

from .base import Entry, MemoryStore
from .md_store import MDStore
from .sqlite_store import SqliteStore

__all__ = ["Entry", "MDStore", "MemoryStore", "SqliteStore", "get_store"]


def get_store(kind: str, path):
    """Factory: return a store driver by name.

    kind: 'sqlite' (default) | 'flatfile'
    path: filesystem path (sqlite db file / markdown file).
    """
    k = (kind or "sqlite").strip().lower()
    if k in ("sqlite", "sqlite3", "db"):
        return SqliteStore(path)
    if k in ("flatfile", "md", "markdown"):
        return MDStore(path)
    raise ValueError(
        f"Unknown store kind {kind!r}. Use 'sqlite' or 'flatfile'."
    )
