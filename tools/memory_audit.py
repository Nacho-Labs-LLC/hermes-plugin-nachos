#!/usr/bin/env python3
"""Memory hygiene audit: duplicates, never-used entries, category purge.

Opens one Nachos store plus its sidecar index (``memory.index.db`` next to
the store) and reports on it. No LLM call is ever made. Read-only except
``purge-category --run`` and ``revert --run``.

Usage:
    memory_audit.py duplicates                   # likely duplicate pairs
    memory_audit.py unused --days 30             # never recalled, first seen > 30d ago
    memory_audit.py purge-category NAME          # dry-run: list entries in NAME
    memory_audit.py purge-category NAME --run    # actually delete them
    memory_audit.py history [--key K]            # recent memory changes
    memory_audit.py revert ID                    # dry-run: show what would be undone
    memory_audit.py revert ID --run              # undo that change
    memory_audit.py --json duplicates            # machine-readable output
    memory_audit.py --store flatfile --path ~/.hermes/nachos/memory.md unused

Embeddings are off by default (duplicates fall back to lexical overlap). Pass
--embeddings auto|nachos|openai|sentence-transformers, with --embedding-model
and --embedding-base-url as needed, to use semantic similarity.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import default_store_path, open_service

from nachos_core.embeddings import make_embedder
from nachos_core.memory import PROFILE


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Nachos memory hygiene audit")
    ap.add_argument("--store", default="sqlite", choices=["sqlite", "flatfile"])
    ap.add_argument(
        "--path", default=None, help="store file (default: ~/.hermes/nachos/memory.db or memory.md)"
    )
    ap.add_argument(
        "--embeddings",
        default="off",
        choices=["auto", "nachos", "openai", "sentence-transformers", "off"],
        help="embedding backend for similarity (default: off, lexical only)",
    )
    ap.add_argument("--embedding-model", default="")
    ap.add_argument("--embedding-base-url", default="")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("duplicates", help="list likely duplicate entry pairs")

    un = sub.add_parser("unused", help="list entries never recalled or returned by a query")
    un.add_argument(
        "--days",
        type=float,
        default=30,
        help="only entries first seen more than this many days ago (default 30)",
    )

    pc = sub.add_parser("purge-category", help="remove every entry in a category")
    pc.add_argument("name", help="category to purge, e.g. scratch")
    pc.add_argument("--run", action="store_true", help="actually delete (default: dry-run)")

    hi = sub.add_parser("history", help="list recent memory changes (newest first)")
    hi.add_argument("--key", default=None, help="only changes to this key")
    hi.add_argument("--limit", type=int, default=20)

    rv = sub.add_parser("revert", help="undo a logged memory change")
    rv.add_argument("id", type=int, help="change id from `history`")
    rv.add_argument("--run", action="store_true", help="actually revert (default: dry-run)")
    return ap


def _emit(args, payload: dict, lines: list[str]) -> None:
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("\n".join(lines))


def _cmd_duplicates(args, service) -> int:
    pairs = service.duplicate_pairs()
    lines = [f"Likely duplicates: {len(pairs)}"]
    lines += [
        f"  {p['similarity']:.2f} {p['match']:<8} {p['a']['key']}  ~  {p['b']['key']}"
        for p in pairs
    ]
    _emit(args, {"duplicates": pairs}, lines)
    return 0


def _cmd_unused(args, service) -> int:
    rows = service.unused(args.days)
    lines = [f"Never recalled, first seen > {args.days:g} days ago: {len(rows)}"]
    lines += [f"  {r['age_days']:>6}d  {r['category']}/{r['key']}" for r in rows]
    _emit(args, {"unused": rows}, lines)
    return 0


def _cmd_history(args, service) -> int:
    rows = service.history(key=args.key, limit=args.limit)
    lines = [f"Recent changes: {len(rows)}"]
    lines += [
        f"  {r['id']:>5} {r['action']:<6} {r['key']}"
        + (f"  (was: {r['preview']})" if r["preview"] else "")
        for r in rows
    ]
    _emit(args, {"changes": rows}, lines)
    return 0


def _cmd_revert(args, service) -> int:
    change = service.profile.index.get_change(args.id)
    if change is None:
        print(f"ERROR: no change {args.id}", file=sys.stderr)
        return 2
    if args.run:
        res = service.revert(args.id, PROFILE)
        lines = [f"Reverted {res['undid']} of {res['key']}: entry {res['result']}"]
        _emit(args, {"dry_run": False, **res}, lines)
        return 0
    lines = [f"Would revert {change['action']} of {change['key']} (dry-run; pass --run to apply)"]
    payload = {
        "dry_run": True,
        "id": change["id"],
        "key": change["key"],
        "action": change["action"],
    }
    _emit(args, payload, lines)
    return 0


def _cmd_remove_category(args, service) -> int:
    matches = [
        {"key": key, "title": title, "category": category}
        for key, title, _summary, category in service.profile.store.list()
        if category == args.name
    ]
    if args.run:
        for row in matches:
            service.remove(row["key"], PROFILE)
    verb = "Removed" if args.run else "Would remove"
    lines = [
        f"{verb} {len(matches)} entries in category {args.name!r}"
        + ("" if args.run else " (dry-run; pass --run to delete)")
    ]
    lines += [f"  {r['key']}  {r['title']}" for r in matches]
    payload = {
        "category": args.name,
        "dry_run": not args.run,
        "removed": len(matches) if args.run else 0,
        "matched": matches,
    }
    _emit(args, payload, lines)
    return 0


_COMMANDS = {
    "duplicates": _cmd_duplicates,
    "unused": _cmd_unused,
    "history": _cmd_history,
    "revert": _cmd_revert,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    path = Path(args.path).expanduser() if args.path else default_store_path(args.store)
    if not path.exists():
        print(f"ERROR: no store at {path}", file=sys.stderr)
        return 2

    embedder = make_embedder(
        args.embeddings,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
    )
    service = open_service(args.store, path, embedder=embedder)
    try:
        return _COMMANDS.get(args.command, _cmd_remove_category)(args, service)
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
