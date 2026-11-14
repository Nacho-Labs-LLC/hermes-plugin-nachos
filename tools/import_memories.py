#!/usr/bin/env python3
"""One-time import of memories from Hermes' own memory providers into Nachos.

Nachos replaces only Hermes' built-in general memory, MEMORY.md (set
``memory.memory_enabled: false``); the user profile, USER.md, stays in use by
Hermes. Nachos never copies writes automatically. Run this once to bring existing
memories across. Dry-run by default; nothing is written without ``--run``.
The source is only ever read.

Usage:
    import_memories.py builtin                  # preview MEMORY.md
    import_memories.py builtin --run            # import it
    import_memories.py builtin --include-user-profile --run   # MEMORY.md + USER.md
    import_memories.py holographic --run --category facts
    import_memories.py builtin --hermes-home ~/.hermes-work --json

Every item goes through ``MemoryService.put()``: quality gate, secret
redaction (Hermes' redactor when importable), duplicate flagging and the
change log (session id ``import:<source>``). An existing entry with the same
key but different content is skipped and reported unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from _common import default_store_path, open_service

# Hermes' tools/memory_tool.py: ENTRY_DELIMITER, files under <home>/memories/.
ENTRY_DELIMITER = "\n§\n"
TITLE_MAX = 60

_EPILOG = """\
Sources:
  builtin      <hermes-home>/memories/MEMORY.md (category "memory"); entries are
               separated by "\\n§\\n". USER.md stays in use by Hermes, so it is
               skipped unless --include-user-profile (category "user").
  holographic  facts table of <hermes-home>/memory_store.db (override with
               --source-path); each fact keeps its own category.
  byterover    markdown files under <hermes-home>/byterover/.brv/context-tree
               (best effort: that layout is owned by the external brv CLI).

Not supported (data lives on a server, no simple read-only export):
  hindsight, honcho, mem0, openviking, retaindb, supermemory
  Export from the provider's own dashboard/CLI/API into a text or markdown
  file (one entry per file or per "§" block), save it as MEMORY.md in a scratch
  directory, and run: import_memories.py builtin --hermes-home <scratch-dir>
  (it reads <scratch-dir>/memories/MEMORY.md).
"""


@dataclass
class Item:
    title: str
    body: str
    category: str
    origin: str


# -- sources ---------------------------------------------------------------


def derive_title(text: str) -> str:
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    first = first.lstrip("#").strip()
    return first[:TITLE_MAX].rstrip() or "untitled"


def _read_builtin(home: Path, include_user_profile: bool = False) -> list[Item]:
    items: list[Item] = []
    files = [("MEMORY.md", "memory")]
    if include_user_profile:
        files.append(("USER.md", "user"))
    for fname, category in files:
        path = home / "memories" / fname
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        entries = (e.strip() for e in raw.split(ENTRY_DELIMITER))
        items.extend(Item(derive_title(e), e, category, fname) for e in entries if e)
    return items


def _holographic_db(home: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    db = home / "memory_store.db"
    # honour plugins.hermes-memory-store.db_path like the provider does
    with contextlib.suppress(Exception):
        import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs; optional dep

        cfg = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8-sig")) or {}
        custom = ((cfg.get("plugins") or {}).get("hermes-memory-store") or {}).get("db_path")
        if isinstance(custom, str) and custom:
            custom = custom.replace("${HERMES_HOME}", str(home)).replace("$HERMES_HOME", str(home))
            db = Path(custom).expanduser()
    return db


def _read_holographic(home: Path, override: str | None) -> list[Item]:
    db = _holographic_db(home, override)
    if not db.is_file():
        raise FileNotFoundError(f"no holographic database at {db}")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT content, category FROM facts ORDER BY fact_id").fetchall()
    finally:
        conn.close()
    return [
        Item(derive_title(c), c.strip(), (cat or "general"), f"fact:{i}")
        for i, (c, cat) in enumerate(rows, 1)
        if c and c.strip()
    ]


def _read_byterover(home: Path, override: str | None) -> list[Item]:
    root = Path(override).expanduser() if override else home / "byterover" / ".brv" / "context-tree"
    if not root.is_dir():
        raise FileNotFoundError(f"no byterover context tree at {root}")
    items: list[Item] = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        title = ""
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                for ln in text[3:end].splitlines():
                    if ln.lower().startswith("title:"):
                        title = ln.split(":", 1)[1].strip().strip("\"'")
                text = text[end + 4 :]
        text = text.strip()
        if text:
            items.append(
                Item(
                    (title or derive_title(text))[:TITLE_MAX],
                    text,
                    "byterover",
                    str(path.relative_to(root)),
                )
            )
    return items


def read_source(
    source: str, home: Path, source_path: str | None, include_user_profile: bool = False
) -> list[Item]:
    if source == "builtin":
        return _read_builtin(home, include_user_profile)
    if source == "holographic":
        return _read_holographic(home, source_path)
    return _read_byterover(home, source_path)


# -- import ----------------------------------------------------------------


def hermes_redactor() -> Callable[[str], str]:
    try:
        from nachos_hermes.config import hermes_redactor as _hermes_redactor

        return _hermes_redactor() or (lambda text: text)
    except Exception:
        return lambda text: text


def _collision_bucket(existing: str | None, body: str, overwrite: bool) -> str | None:
    """Report bucket for an item whose key already exists, or None to go ahead."""
    if existing is None:
        return None
    if existing.strip() == body:
        return "unchanged"
    return None if overwrite else "collisions"


def run_import(
    service, items: list[Item], *, run: bool, overwrite: bool, category: str | None
) -> dict[str, list]:
    """Import ``items``; with ``run`` False nothing is written."""
    report: dict[str, list] = {
        "imported": [],
        "skipped_low_signal": [],
        "unchanged": [],
        "collisions": [],
        "duplicates": [],
        "errors": [],
    }
    store = service.profile.store
    planned: dict[str, str] = {}  # dry-run: key -> body already "written"
    for item in items:
        cat = category or item.category
        # prepare() is the same gate put() applies, so a dry-run cannot drift from --run.
        draft = service.prepare(title=item.title, body=item.body, category=cat)
        row = {"key": draft.key, "title": draft.title, "category": cat, "origin": item.origin}
        existing = planned.get(draft.key) if not run else None
        if existing is None:
            existing = store.get(draft.key)
        bucket = _collision_bucket(existing, draft.body, overwrite)
        if bucket:
            report[bucket].append(row)
            continue
        if run:
            result = service.put(title=item.title, body=item.body, category=cat)
            error, similar = result.get("error"), result.get("possible_duplicates")
        else:
            error = draft.error
            similar = (
                None
                if error
                else service.find_similar(
                    draft.title, draft.summary, exclude=(draft.scope, draft.key)
                )
            )
            if not error:
                planned[draft.key] = draft.body
        if error:
            report["skipped_low_signal"].append({**row, "reason": error})
            continue
        if similar:
            report["duplicates"].append({**row, "similar": similar})
        report["imported"].append(row)
    return report


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Import memories from a Hermes memory provider into Nachos (dry-run by default)",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("source", choices=["builtin", "holographic", "byterover"])
    ap.add_argument(
        "--hermes-home",
        default=None,
        help="Hermes home (default: from hermes_constants, else ~/.hermes)",
    )
    ap.add_argument(
        "--source-path",
        default=None,
        help="override the source file/dir (holographic db, byterover context tree)",
    )
    ap.add_argument("--store", default="sqlite", choices=["sqlite", "flatfile"])
    ap.add_argument(
        "--path",
        default=None,
        help="Nachos store file (default: <hermes-home>/nachos/memory.db or memory.md)",
    )
    ap.add_argument("--category", default=None, help="category for every imported entry")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing entry with the same key but different content",
    )
    ap.add_argument(
        "--include-user-profile",
        action="store_true",
        help="builtin only: also import USER.md as category 'user' "
        "(default: MEMORY.md only, since Hermes keeps using USER.md)",
    )
    ap.add_argument("--run", action="store_true", help="actually import (default: dry-run)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def _hermes_home(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _store_path(args: argparse.Namespace, home: Path) -> Path:
    if args.path:
        return Path(args.path).expanduser()
    if args.hermes_home:
        return home / "nachos" / ("memory.md" if args.store == "flatfile" else "memory.db")
    return default_store_path(args.store)


def _print_report(args: argparse.Namespace, found: int, report: dict[str, list]) -> None:
    verb = "Imported" if args.run else "Would import"
    print(
        f"{args.source}: {found} entries found"
        + ("" if args.run else " (dry-run; pass --run to import)")
    )
    print(f"  {verb}: {len(report['imported'])}")
    for r in report["imported"]:
        print(f"    {r['category']}/{r['key']}")
    labels = (
        ("skipped_low_signal", "Skipped (low-signal)"),
        ("unchanged", "Already present, unchanged"),
        ("collisions", "Skipped, key collides with a different entry (use --overwrite)"),
        ("duplicates", "Flagged as possible duplicates of existing entries"),
    )
    for name, label in labels:
        if report[name]:
            print(f"  {label}: {len(report[name])}")
            for r in report[name]:
                extra = f"  ({r['reason']})" if r.get("reason") else ""
                print(f"    {r['key']}{extra}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    home = _hermes_home(args.hermes_home)
    try:
        items = read_source(args.source, home, args.source_path, args.include_user_profile)
    except (FileNotFoundError, sqlite3.Error) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    path = _store_path(args, home)
    tmp = None
    if not args.run and not path.exists():
        # Dry-run against a store that does not exist yet: use a scratch one.
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / path.name
    elif args.run:
        path.parent.mkdir(parents=True, exist_ok=True)

    service = open_service(args.store, path, redact=hermes_redactor())
    service.session_id = f"import:{args.source}"
    try:
        report = run_import(
            service, items, run=args.run, overwrite=args.overwrite, category=args.category
        )
    finally:
        service.close()
        if tmp:
            tmp.cleanup()

    if args.json:
        print(
            json.dumps(
                {"source": args.source, "dry_run": not args.run, "found": len(items), **report},
                indent=2,
                default=str,
            )
        )
    else:
        _print_report(args, len(items), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
