"""Catalog-install wrapper for the Nachos Memory and Context plugin suite."""

from nachos_hermes.context_engine import register as _register_context
from nachos_hermes.memory_provider import register as _register_memory


def register(ctx):
    """Register both supported Nachos components from a directory install."""
    _register_memory(ctx)
    _register_context(ctx)
