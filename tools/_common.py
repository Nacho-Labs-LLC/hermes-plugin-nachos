"""Shared helpers for the maintenance CLIs in this directory.

Importing this module puts the repository root on ``sys.path`` so the tools
can import ``nachos_core`` / ``nachos_hermes`` when run as plain scripts.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.store import get_store


def default_store_path(store_kind: str) -> Path:
    from hermes_constants import get_hermes_home

    base = Path(get_hermes_home()) / "nachos"
    return base / ("memory.md" if store_kind == "flatfile" else "memory.db")


def open_service(
    store_kind: str, path: Path, *, embedder=None, redact: Callable[[str], str] | None = None
) -> MemoryService:
    """Open the profile scope at ``path`` (store plus sidecar index)."""
    scope = Scope(PROFILE, get_store(store_kind, path), MemoryIndex(index_path_for(path)))
    return MemoryService(OrderedDict([(PROFILE, scope)]), embedder=embedder, redact=redact)
