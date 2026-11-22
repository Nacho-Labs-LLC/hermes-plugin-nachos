"""Embedding model presets and how configured settings resolve to a concrete model.

Stdlib only. A *preset* is a named (model, dtype) bundle for the ``nachos``
backend, chosen from the recall benchmark (see docs/benchmark.md,
"Choosing a model"). Explicit settings always beat a preset:

* ``embedding_model`` set  -> that model; the preset's dtype is *not* applied
  (the model may not ship that file), only ``embedding_dtype`` is.
* ``embedding_dtype`` set  -> overrides the preset's dtype.
* neither set              -> the preset's model and dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

#: ``embedding_*`` prefix value that switches the model's built-in prefix off.
NO_PREFIX = "none"


@dataclass(frozen=True)
class Preset:
    name: str
    model: str
    dtype: str
    dimension: int
    download_mb: int
    summary: str


PRESETS: dict[str, Preset] = {
    "small": Preset(
        "small",
        "Xenova/all-MiniLM-L6-v2",
        "q8",
        384,
        23,
        "Smallest and fastest; the lightweight default.",
    ),
    "balanced": Preset(
        "balanced",
        "Snowflake/snowflake-arctic-embed-s",
        "q8",
        384,
        34,
        "Retrieval-tuned 384-dim model; same index size as small, better on natural questions.",
    ),
    "quality": Preset(
        "quality",
        "Alibaba-NLP/gte-modernbert-base",
        "q8",
        768,
        150,
        "Best recall in the benchmark; about 2x slower to embed and a 768-dim index.",
    ),
}

#: Valid ``embedding_preset`` values (blank = no preset).
PRESET_CHOICES = ("", *PRESETS)


def resolve_model(preset: str, model: str, dtype: str) -> tuple[str, str]:
    """Effective (model, dtype) for the nachos backend; explicit settings win over the preset."""
    chosen = PRESETS.get((preset or "").strip().lower())
    if chosen is None:
        return model, dtype
    if model:
        return model, dtype
    return chosen.model, dtype or chosen.dtype


def prefix_argument(value: str) -> str | None:
    """CLI value for a configured prefix: None = leave to the model's profile, ``""`` = disable.

    Blank means "use the model's built-in prefix"; the word ``none`` means "no prefix".
    """
    if value == "":
        return None
    return "" if value.strip().lower() == NO_PREFIX else value
