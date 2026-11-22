"""Model settings, status and re-embedding for a Nachos profile.

    python -m nachos_hermes.models list    [--profile-home H] [--json]
    python -m nachos_hermes.models status  --profile-home H [--json]
    python -m nachos_hermes.models reindex --profile-home H [--json]

``list`` and ``status`` only read: no model is loaded, nothing is written, no network.
``reindex`` re-embeds every entry with the configured model now (what recall otherwise does
in bounded steps after a model change) and does load the model.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any

from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.store import MDStore, SqliteStore, get_store

from .config import (
    AUXILIARY_TASKS,
    DEFAULTS,
    auxiliary_models,
    make_embedder_from_config,
    profile_config,
    store_file,
)

#: Every embedding / rerank / re-embed setting (none is a secret: the API key is only an env var name).
_MODEL_KEYS = tuple(
    key
    for key in DEFAULTS
    if key == "embeddings" or key.startswith(("embedding_", "rerank", "reindex_"))
)


def effective_embedding(cfg: dict[str, Any]) -> dict[str, str]:
    """The nachos model the config names (blank = the CLI's default)."""
    return {
        "model": cfg["embedding_model"] or "(backend default)",
        "dtype": cfg["embedding_dtype"] or "(backend default)",
    }


def list_report(home: Path | None) -> dict[str, Any]:
    """The profile's model settings and the auxiliary LLMs, in one structure."""
    cfg = profile_config(home) if home is not None else profile_config(Path("/nonexistent"))
    return {
        "profile_home": str(home) if home is not None else None,
        "settings": {key: cfg[key] for key in _MODEL_KEYS},
        "effective_embedding": effective_embedding(cfg),
        "auxiliary_llm": {
            task: {"purpose": purpose, "model": model}
            for (task, purpose), model in zip(AUXILIARY_TASKS.items(), auxiliary_models().values())
        },
    }


def _open_read_only(home: Path, store_kind: str) -> tuple[Any, MemoryIndex, Path]:
    path = home / "nachos" / store_file(Path("."), store_kind).name
    if not path.is_file():
        raise ValueError(f"no store at {path}")
    store = (
        SqliteStore(path, read_only=True)
        if store_kind == "sqlite"
        else MDStore(path, read_only=True)
    )
    index_path = index_path_for(path)
    index = (
        MemoryIndex(index_path, read_only=True) if index_path.exists() else MemoryIndex(":memory:")
    )
    return store, index, path


def status_report(home: Path, store_kind: str) -> dict[str, Any]:
    """Vectors per model in the profile's index versus what the config asks for. Read-only."""
    cfg = profile_config(home)
    store, index, path = _open_read_only(home, store_kind)
    try:
        entries = len(store.list())
        counts = index.model_counts()
    finally:
        getattr(store, "close", lambda: None)()
        index.close()
    wanted = effective_embedding(cfg)
    prefix = f"nachos:{cfg['embedding_model'] or 'default'}"
    # The exact id also carries pooling/prefix details only the running CLI reports, so match
    # the family: nachos:<model>[:<dtype>...].
    current = {
        m: n
        for m, n in counts.items()
        if cfg["embeddings"] != "off"
        and m.startswith(prefix)
        and (not cfg["embedding_dtype"] or cfg["embedding_dtype"] in m)
    }
    covered = max(current.values(), default=0)
    return {
        "store_path": str(path),
        "entries": entries,
        "configured": wanted,
        "embeddings": cfg["embeddings"],
        "models_in_index": counts,
        "configured_model_vectors": current,
        "coverage": round(min(covered, entries) / entries, 3) if entries else 1.0,
        "needs_reindex": cfg["embeddings"] != "off" and covered < entries,
        "stale_models": sorted(m for m in counts if m not in current),
    }


def _human_list(report: dict[str, Any]) -> str:
    lines = [f"Settings for {report['profile_home'] or 'defaults'}:"]
    lines += [f"  {key}: {json.dumps(value)}" for key, value in report["settings"].items()]
    eff = report["effective_embedding"]
    lines += ["", f"Effective nachos embedding model: {eff['model']} (dtype {eff['dtype']})"]
    lines.append("")
    lines.append(
        "Auxiliary LLMs (Hermes config.yaml, auxiliary.<task>.model; not embedding models):"
    )
    for task, info in report["auxiliary_llm"].items():
        lines.append(f"  auxiliary.{task}.model: {info['model'] or '(unset)'}  # {info['purpose']}")
    return "\n".join(lines)


def _human_status(report: dict[str, Any]) -> str:
    cfg = report["configured"]
    lines = [
        f"{report['store_path']}: {report['entries']} entries",
        f"configured: {cfg['model']} (dtype {cfg['dtype']})",
        f"coverage: {report['coverage']:.0%} of entries embedded with the configured model",
    ]
    lines += [f"  in index: {m}: {n} entries" for m, n in sorted(report["models_in_index"].items())]
    if report["needs_reindex"]:
        lines.append(
            "recall is lexical until re-embedding finishes (it advances on normal use); "
            "finish now: python -m nachos_hermes.models reindex --profile-home <home>"
        )
    elif report["stale_models"]:
        lines.append(f"old vectors from {report['stale_models']} will be dropped on the next sync")
    return "\n".join(lines)


def reindex(home: Path, store_kind: str) -> dict[str, Any]:
    """Re-embed every entry with the configured model now; writes the profile's index."""
    cfg = profile_config(home)
    embedder = make_embedder_from_config(cfg)
    if embedder is None:
        raise ValueError("embeddings are off or unavailable in this profile's config")
    path = home / "nachos" / store_file(Path("."), store_kind).name
    if not path.is_file():
        raise ValueError(f"no store at {path}")
    scope = Scope(PROFILE, get_store(store_kind, path), MemoryIndex(index_path_for(path)))
    service = MemoryService(OrderedDict([(PROFILE, scope)]), embedder=embedder)
    try:
        return service.reindex()
    finally:
        service.close()


def _absolute_home(value: Path) -> Path:
    supplied = value.expanduser()
    if not supplied.is_absolute():
        raise ValueError("--profile-home must be an absolute path")
    return supplied.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["list", "status", "reindex"])
    parser.add_argument("--profile-home", type=Path, help="absolute Hermes profile home")
    parser.add_argument("--store", choices=["sqlite", "flatfile"], default="sqlite")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        home = _absolute_home(args.profile_home) if args.profile_home else None
        if args.command == "list":
            report: dict[str, Any] = list_report(home)
            text = _human_list(report)
        else:
            if home is None:
                raise ValueError("--profile-home is required")
            if args.command == "status":
                report = status_report(home, args.store)
                text = _human_status(report)
            else:
                report = reindex(home, args.store)
                text = json.dumps(report, indent=2)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2) if args.json else text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
