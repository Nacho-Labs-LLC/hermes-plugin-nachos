# nachos_core — harness-agnostic memory layer.
#
# Pure Python, stdlib only. Does NOT import from hermes_* / agent.*.
# All host wiring lives in the plugin (plugins/memory/nachos, etc.).
from .quality import is_redundant_summary, low_signal_reason, shares_vocabulary
from .snapshots import (
    Snapshot,
    SnapshotMeta,
    SnapshotStore,
)
from .store import Entry, MDStore, MemoryStore, SqliteStore, get_store
from .toc import build_toc, render_toc

__all__ = [
    "Entry",
    "MDStore",
    # store seam
    "MemoryStore",
    # snapshots
    "Snapshot",
    "SnapshotMeta",
    "SnapshotStore",
    "SqliteStore",
    # manifest
    "build_toc",
    "get_store",
    # quality
    "is_redundant_summary",
    "low_signal_reason",
    "render_toc",
    "shares_vocabulary",
]
