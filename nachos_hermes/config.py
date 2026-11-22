"""Nachos settings for the Hermes adapter: defaults, setup schema, host paths, redactor.

Everything here is import-safe outside Hermes: host modules (``agent.redact``,
``hermes_constants``, ``hermes_cli``) are imported lazily and failures degrade to
sensible fallbacks.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

from nachos_core.embeddings import EMBEDDING_BACKENDS, Embedder, make_embedder

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "store": "sqlite",
    "manifest_char_budget": 1200,
    "embeddings": "auto",
    "embedding_model": "",
    "embedding_dtype": "",
    "embedding_pooling": "",
    "embedding_max_length": 0,
    "embedding_base_url": "",
    "embedding_api_key_env": "",
    "embedding_query_prefix": "",
    "embedding_document_prefix": "",
    # MiniLM: paraphrases score 0.68-0.73, distinct entries <=0.22; flagging is non-blocking, so false positives are cheap.
    "duplicate_threshold": 0.65,
    "duplicate_calibration": "auto",
    "similarity_floor": "off",
    "rerank": "off",
    "rerank_model": "",
    "rerank_dtype": "",
    "rerank_max_length": 0,
    "rerank_candidates": 25,
    "reindex_budget_ms": 250,
    "shared_store": "off",
    "shared_store_path": "",
    "redact_secrets": "on",
    "checkpoint_max_age_days": 30,
    "capture": "review",
}

_VERBATIM_KEYS = frozenset({"embedding_query_prefix", "embedding_document_prefix"})
# String settings with a closed set of values; anything else is ignored.
_CHOICES: dict[str, tuple[str, ...]] = {
    "capture": ("review", "off"),
    "embedding_pooling": ("", "mean", "cls"),
    "rerank": ("off", "nachos"),
    "similarity_floor": ("off", "auto"),
    "duplicate_calibration": ("auto", "off"),
}

CONFIG_SCHEMA: list[dict[str, Any]] = [
    {
        "key": "store",
        "description": "Durable memory store format.",
        "choices": ["sqlite", "flatfile"],
    },
    {
        "key": "manifest_char_budget",
        "type": "int",
        "description": "Soft character target for the always-on manifest (entries are never dropped).",
    },
    {
        "key": "embeddings",
        "choices": list(EMBEDDING_BACKENDS),
        "description": "Semantic recall backend. auto = nachos-embeddings CLI, then "
        "sentence-transformers, then an OpenAI-compatible URL if set; off = lexical only.",
    },
    {
        "key": "embedding_model",
        "description": "Embedding model id (backend-specific; blank = backend default, MiniLM for nachos-embeddings). "
        "For the nachos backend any Hugging Face ONNX encoder or a local directory path works.",
    },
    {
        "key": "embedding_dtype",
        "description": "Model precision for the nachos backend, e.g. q8 (about 3x faster, cosine ~0.99 vs fp32). Blank = backend default. Changing it re-embeds.",
    },
    {
        "key": "embedding_pooling",
        "choices": ["", "mean", "cls"],
        "description": "Override the nachos model's pooling (blank = the model's built-in profile; "
        "bge, arctic and gte-modernbert use cls). Changing it re-embeds.",
    },
    {
        "key": "embedding_max_length",
        "type": "int",
        "description": "Truncate inputs to this many tokens for the nachos backend (0 = model default; "
        "can only lower the tokenizer limit). Changing it re-embeds.",
    },
    {
        "key": "embedding_base_url",
        "description": "OpenAI-compatible base URL, e.g. http://localhost:11434/v1 for Ollama.",
    },
    {
        "key": "embedding_api_key_env",
        "description": "Name of the environment variable holding the embeddings API key (the key itself is never stored).",
    },
    {
        "key": "embedding_query_prefix",
        "description": "Text prepended to search queries before embedding (e.g. 'query: ' for E5, "
        "'search_query: ' for nomic). Leading/trailing spaces are kept. For the nachos backend "
        "blank keeps the model's built-in prefix and the word 'none' disables it; openai and "
        "sentence-transformers have no built-ins (blank = no prefix).",
    },
    {
        "key": "embedding_document_prefix",
        "description": "Text prepended to stored entries before embedding (e.g. 'passage: ' for E5, "
        "'search_document: ' for nomic); same blank / 'none' rules as the query prefix. "
        "Changing it re-embeds.",
    },
    {
        "key": "duplicate_threshold",
        "type": "float",
        "description": "Embedding similarity (0-1) at which a new entry is flagged as a likely duplicate.",
    },
    {
        "key": "duplicate_calibration",
        "choices": ["auto", "off"],
        "description": "auto raises duplicate_threshold to the embedding model's own 99th "
        "percentile of random-pair similarity when that is higher (cosine ranges differ a lot "
        "between models). A duplicate_threshold other than the default 0.65 is always used "
        "as given; off uses duplicate_threshold exactly for every model.",
    },
    {
        "key": "similarity_floor",
        "choices": ["off", "auto"],
        "description": "auto drops semantic hits below the model's own 75th percentile of "
        "random-pair similarity (calibrated from the stored vectors).",
    },
    {
        "key": "rerank",
        "choices": ["off", "nachos"],
        "description": "nachos = re-score the top fused results with a cross-encoder served by "
        "the nachos-embeddings CLI (needs the nachos embeddings backend and a CLI with "
        "--rerank-model; otherwise ignored). Adds roughly 0.3 s per recall.",
    },
    {
        "key": "rerank_model",
        "description": "Cross-encoder model id (blank = jinaai/jina-reranker-v1-tiny-en).",
    },
    {
        "key": "rerank_dtype",
        "description": "Cross-encoder precision, e.g. q8 (blank = CLI default).",
    },
    {
        "key": "rerank_max_length",
        "type": "int",
        "description": "Cross-encoder input length in tokens (0 = the model's default: 1024 for "
        "the jina reranker, 512 otherwise).",
    },
    {
        "key": "rerank_candidates",
        "type": "int",
        "description": "How many top fused results the reranker re-scores.",
    },
    {
        "key": "reindex_budget_ms",
        "type": "int",
        "description": "After an embedding model change, how long one recall or write may spend "
        "re-embedding entries (milliseconds; at least one small batch always runs). Recall is "
        "lexical until every entry is re-embedded; `python -m nachos_hermes.models reindex` "
        "finishes it at once.",
    },
    {
        "key": "shared_store",
        "choices": ["off", "on"],
        "description": "Enable a memory scope shared by every Hermes profile on this machine.",
    },
    {
        "key": "shared_store_path",
        "description": "Directory for the shared store (blank = <hermes root>/nachos-shared).",
    },
    {
        "key": "redact_secrets",
        "choices": ["on", "off"],
        "description": "Run Hermes' secret redactor over everything written to memory.",
    },
    {
        "key": "checkpoint_max_age_days",
        "type": "int",
        "description": "Delete pre-compression transcript checkpoints older than this (0 = keep forever).",
    },
    {
        "key": "capture",
        "choices": ["review", "off"],
        "description": "review = at session end, queue the transcript and suggest memories in a "
        "review inbox (/nachos-memory-inbox). Uses an auxiliary LLM: set "
        "auxiliary.nachos_capture.model in config.yaml to a cheap model, or it may "
        "fall back to your main model.",
    },
]


def hermes_redactor() -> Callable[[str], str] | None:
    """Hermes' own secret redactor, forced on (memory is a persistence boundary)."""
    try:
        from agent.redact import redact_sensitive_text
    except Exception:
        return None

    def _redact(text: str) -> str:
        if not text:
            return text
        try:
            # file_read=True swaps secrets for a non-reusable sentinel rather
            # than a head/tail mask that looks like a real truncated key.
            redacted: str = redact_sensitive_text(text, force=True, file_read=True)
            return redacted
        except TypeError:  # older Hermes without file_read
            redacted_plain: str = redact_sensitive_text(text, force=True)
            return redacted_plain
        except Exception as e:  # never block a write on the redactor
            logger.debug("Nachos redaction failed: %s", e)
            return text

    return _redact


def hermes_root(hermes_home: Path) -> Path:
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root())
    except Exception:
        return hermes_home.parent.parent if hermes_home.parent.name == "profiles" else hermes_home


def shared_dir(cfg: dict[str, Any], hermes_home: Path) -> Path:
    """Shared-store directory for a config (explicit path, else <hermes root>/nachos-shared)."""
    if cfg.get("shared_store_path"):
        return Path(cfg["shared_store_path"]).expanduser()
    return hermes_root(hermes_home) / "nachos-shared"


def store_file(directory: Path, store_kind: str) -> Path:
    return directory / ("memory.md" if store_kind == "flatfile" else "memory.db")


def apply_config(cfg: dict[str, Any], values: dict[str, Any]) -> None:
    """Merge valid ``values`` into ``cfg`` in place, coercing to each default's type."""
    for key, default in DEFAULTS.items():
        value = values.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(default, float) and isinstance(value, (int, float)):
            if 0 < float(value) <= 1:
                cfg[key] = float(value)
        elif isinstance(default, int) and isinstance(value, (int, float)):
            cfg[key] = int(value)
        elif isinstance(default, str) and isinstance(value, str):
            if key not in _VERBATIM_KEYS:  # prefixes keep their meaningful spaces
                value = value.strip()
            if key in _CHOICES and value not in _CHOICES[key]:
                continue
            cfg[key] = value


def _default_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _shared_sources(home: Path) -> list[Any]:
    """Raw config dicts that may set the shared store: legacy config.yaml, then profile JSON."""
    sources: list[Any] = []
    try:  # legacy nachos.memory.* keys in config.yaml (local file read)
        from hermes_cli.config import cfg_get, load_config

        legacy = load_config()
        sources.append(
            {
                k: cfg_get(legacy, "nachos", "memory", k)
                for k in ("shared_store", "shared_store_path")
            }
        )
    except Exception:
        logger.debug("Nachos legacy shared-store config unavailable", exc_info=True)
    with contextlib.suppress(OSError, ValueError):
        sources.append(json.loads((home / "nachos" / "config.json").read_text(encoding="utf-8")))
    return sources


def backup_paths() -> list[str]:
    """Shared-store directory when it lives outside HERMES_HOME. No init, no network."""
    try:
        home = _default_home()
        cfg: dict[str, Any] = dict(DEFAULTS)
        for values in _shared_sources(home):  # later sources (profile config) win
            if isinstance(values, dict):
                for key in ("shared_store", "shared_store_path"):
                    if isinstance(values.get(key), str):
                        cfg[key] = values[key].strip()
        if cfg["shared_store"] != "on":
            return []
        shared = shared_dir(cfg, home)
        try:
            shared.resolve().relative_to(home.resolve())
            return []
        except ValueError:
            return [str(shared)]
    except Exception as e:
        logger.debug("Nachos backup_paths failed: %s", e)
        return []


#: Hermes ``auxiliary.<task>.model`` LLM settings Nachos reads (Hermes semantics, unchanged).
AUXILIARY_TASKS: dict[str, str] = {
    "nachos_capture": "session-end memory suggestions (capture: review)",
    "nachos_summary": "replacement summaries (tools/correct_summaries.py)",
}


def make_embedder_from_config(cfg: dict[str, Any]) -> Embedder | None:
    """The embedder a resolved config asks for (``None`` = lexical only); may raise."""
    return make_embedder(
        cfg["embeddings"],
        model=cfg["embedding_model"],
        dtype=cfg["embedding_dtype"],
        base_url=cfg["embedding_base_url"],
        api_key_env=cfg["embedding_api_key_env"],
        query_prefix=cfg["embedding_query_prefix"],
        document_prefix=cfg["embedding_document_prefix"],
        rerank=cfg["rerank"] == "nachos",
        rerank_model=cfg["rerank_model"],
        rerank_dtype=cfg["rerank_dtype"],
        pooling=cfg["embedding_pooling"],
        max_length=max(0, int(cfg["embedding_max_length"])),
        rerank_max_length=max(0, int(cfg["rerank_max_length"])),
    )


def profile_config(hermes_home: Path) -> dict[str, Any]:
    """Defaults overlaid with ``<home>/nachos/config.json`` (what ``hermes memory setup`` wrote)."""
    cfg: dict[str, Any] = dict(DEFAULTS)
    with contextlib.suppress(OSError, ValueError):
        values = json.loads((hermes_home / "nachos" / "config.json").read_text(encoding="utf-8"))
        if isinstance(values, dict):
            apply_config(cfg, values)
    return cfg


def auxiliary_models() -> dict[str, str | None]:
    """Configured auxiliary LLM per Nachos task, when Hermes' config is readable (else ``None``)."""
    models: dict[str, str | None] = dict.fromkeys(AUXILIARY_TASKS)
    try:
        from hermes_cli.config import cfg_get, load_config

        loaded = load_config()
        for task in AUXILIARY_TASKS:
            value = cfg_get(loaded, "auxiliary", task, "model")
            models[task] = value if isinstance(value, str) and value else None
    except Exception:
        logger.debug("Hermes config unavailable for auxiliary models", exc_info=True)
    return models
