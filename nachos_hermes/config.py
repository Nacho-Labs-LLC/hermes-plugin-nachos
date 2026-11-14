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

from nachos_core.embeddings import EMBEDDING_BACKENDS

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "store": "sqlite",
    "manifest_char_budget": 1200,
    "embeddings": "auto",
    "embedding_model": "",
    "embedding_dtype": "",
    "embedding_base_url": "",
    "embedding_api_key_env": "",
    # MiniLM: paraphrases score 0.68-0.73, distinct entries <=0.22; flagging is non-blocking, so false positives are cheap.
    "duplicate_threshold": 0.65,
    "shared_store": "off",
    "shared_store_path": "",
    "redact_secrets": "on",
    "checkpoint_max_age_days": 30,
    "capture": "review",
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
        "description": "Embedding model id (backend-specific; blank = backend default).",
    },
    {
        "key": "embedding_dtype",
        "description": "Model precision for the nachos backend, e.g. q8 (about 3x faster, cosine ~0.99 vs fp32). Blank = backend default. Changing it re-embeds.",
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
        "key": "duplicate_threshold",
        "type": "float",
        "description": "Embedding similarity (0-1) at which a new entry is flagged as a likely duplicate.",
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
            value = value.strip()
            if key == "capture" and value not in ("review", "off"):
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
