"""Helpers for embedding-model settings.

Stdlib only. There are no model presets: the model is whatever ``embedding_model`` names
(blank = the backend's default, MiniLM for ``nachos-embeddings``).
"""

from __future__ import annotations

#: ``embedding_*`` prefix value that switches the model's built-in prefix off.
NO_PREFIX = "none"


def prefix_argument(value: str) -> str | None:
    """CLI value for a configured prefix: None = leave to the model's profile, ``""`` = disable.

    Blank means "use the model's built-in prefix"; the word ``none`` means "no prefix".
    """
    if value == "":
        return None
    return "" if value.strip().lower() == NO_PREFIX else value
