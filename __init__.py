"""Catalog-install wrapper for the Nachos Memory and Context plugin suite."""

import sys
from pathlib import Path

_PLUGIN_ROOT = str(Path(__file__).resolve().parent)
if _PLUGIN_ROOT not in sys.path:
    sys.path.insert(0, _PLUGIN_ROOT)

# Export the provider classes so Hermes' directory-discovery heuristics recognize
# both contracts before importing this wrapper. Aliased register functions alone
# are invisible to those scans.
from nachos_hermes.context_engine import NachosContextEngine  # noqa: E402
from nachos_hermes.context_engine import register as _register_context  # noqa: E402
from nachos_hermes.memory_provider import NachosMemoryProvider  # noqa: E402
from nachos_hermes.memory_provider import register as _register_memory  # noqa: E402

__all__ = ["NachosContextEngine", "NachosMemoryProvider", "register"]


def register(ctx):
    """Register both supported Nachos components from a directory install."""
    _register_memory(ctx)
    _register_context(ctx)
