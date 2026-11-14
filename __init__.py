"""Catalog-install wrapper for the Nachos memory provider."""

import sys
from pathlib import Path

_PLUGIN_ROOT = str(Path(__file__).resolve().parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

# Export the provider class so Hermes' directory-discovery heuristics recognize
# the memory-provider contract before importing this wrapper. An aliased
# register function alone is invisible to those scans.
from nachos_hermes.memory_provider import NachosMemoryProvider
from nachos_hermes.memory_provider import register as _register_memory

__all__ = ["NachosMemoryProvider", "register"]


def register(ctx):
    """Register the Nachos memory provider from a directory install."""
    _register_memory(ctx)
