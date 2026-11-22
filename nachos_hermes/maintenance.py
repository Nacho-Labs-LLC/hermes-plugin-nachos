"""Offline, report-only memory maintenance; no Hermes runtime imports required."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

from nachos_core.capture import pending_jobs
from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.quality import flag_reason
from nachos_core.store import MDStore, SqliteStore

_CHECKS = ("summaries", "duplicates", "unused", "superseded", "suspicious_supersedes", "inbox")


def _summaries(store):
    flagged = []
    for key, title, summary, category in store.list():
        reason = flag_reason(title, summary, store.get(key) or "")
        if reason:
            flagged.append({"key": key, "category": category, "reason": reason})
    return {"count": len(flagged), "items": flagged}


def _check(name, service, home, days, index_available):
    if name == "summaries":
        return _summaries(service.profile.store)
    if name == "duplicates":
        pairs = service.duplicate_pairs(sync=False)
        return {"count": len(pairs), "items": pairs, "matching": "lexical"}
    index = service.profile.index
    if name == "unused":
        rows = service.unused(days)
        tracked = index.usage()
        unknown = sorted(k for k, *_ in service.profile.store.list() if k not in tracked)
        return {
            "count": len(rows),
            "items": rows,
            "untracked_keys": unknown,
            "days": days,
            "telemetry_available": index_available,
        }
    if name == "superseded":
        rows = service.superseded(days)
        return {"count": len(rows), "items": rows, "days": days}
    if name == "suspicious_supersedes":
        rows = service.suspicious_supersedes()
        return {"count": len(rows), "items": rows}
    inbox = home / "nachos" / "inbox"
    return {
        "pending_candidates": index.count_inbox() if index_available else None,
        "pending_jobs": len(pending_jobs(inbox)),
        "processing_jobs": len(list((inbox / "processing").glob("*.json.gz"))),
    }


def _validate_profile_paths(home: Path, path: Path, store_kind: str) -> None:
    index_path = index_path_for(path)
    targets = [
        path,
        index_path,
        home / "nachos" / "inbox" / "pending",
        home / "nachos" / "inbox" / "processing",
    ]
    sqlite_files = [index_path]
    if store_kind == "sqlite":
        sqlite_files.append(path)
    for db_file in sqlite_files:
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{db_file}{suffix}")
            if os.path.lexists(sidecar):
                targets.append(sidecar)
    for target in targets:
        if home not in target.resolve().parents:
            raise ValueError(f"path escapes selected profile: {target}")


def _report(args):
    if not math.isfinite(args.days) or args.days < 0:
        raise ValueError("--days must be finite and nonnegative")
    supplied = args.profile_home.expanduser()
    if not supplied.is_absolute():
        raise ValueError("--profile-home must be an absolute path")
    home = supplied.resolve()
    path = home / "nachos" / ("memory.md" if args.store == "flatfile" else "memory.db")
    _validate_profile_paths(home, path, args.store)
    if not path.is_file():
        raise ValueError(f"no store at {path}")
    store = (
        SqliteStore(path, read_only=True)
        if args.store == "sqlite"
        else MDStore(path, read_only=True)
    )
    index_path = index_path_for(path)
    try:
        index = (
            MemoryIndex(index_path, read_only=True)
            if index_path.exists()
            else MemoryIndex(":memory:")
        )
    except Exception:
        getattr(store, "close", lambda: None)()
        raise
    service = MemoryService(OrderedDict([(PROFILE, Scope(PROFILE, store, index))]))
    try:
        selected = _CHECKS if args.check == "all" else (args.check,)
        return {
            "schema_version": 1,
            "report_only": True,
            "profile_home": str(home),
            "store_path": str(path),
            "index_available": index_path.exists(),
            "checks": {
                name: _check(name, service, home, args.days, index_path.exists())
                for name in selected
            },
        }
    finally:
        service.close()


def _human_report(report):
    lines = [f"REPORT ONLY: {report['store_path']}"]
    if not report["index_available"]:
        lines.append("index unavailable: usage history and candidate inbox are unknown")
    for name, result in report["checks"].items():
        if name == "inbox":
            lines.append(f"inbox: {json.dumps(result, sort_keys=True)}")
            continue
        lines.append(f"{name}: {result['count']}")
        lines.extend("  " + json.dumps(item, sort_keys=True) for item in result["items"])
        if name == "unused":
            lines.append(f"  untracked keys: {json.dumps(result['untracked_keys'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-home",
        required=True,
        type=Path,
        help="absolute Hermes profile home; never inferred from the environment",
    )
    parser.add_argument(
        "--store",
        choices=["sqlite", "flatfile"],
        default="sqlite",
        help="store driver (default: sqlite)",
    )
    parser.add_argument("--json", action="store_true", help="emit one schema-versioned JSON report")
    parser.add_argument(
        "--days", type=float, default=30, help="minimum first-seen age for unused (default: 30)"
    )
    parser.add_argument("check", choices=["all", *_CHECKS])
    args = parser.parse_args(argv)
    try:
        report = _report(args)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"schema_version": 1, "report_only": True, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2) if args.json else _human_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
