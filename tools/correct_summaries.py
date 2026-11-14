#!/usr/bin/env python3
"""Companion cron — periodic memory-summary quality correction.

NOT wired to any live cron by default (point a Hermes cron at this if
you want it to run on a cadence — see below). This is the reflection
hygiene layer kept OUTSIDE the plugin hot path: nachos_memory_put
only runs a cheap heuristic quality gate (nachos_core.quality) at write
time; this script runs that SAME heuristic over the whole store and only
spends an LLM call on entries the heuristic actually flags — low-signal
summaries, summaries that just restate the title, or summaries that share
no vocabulary with their own body (stale after a body edit). Entries that
already look fine are skipped for free. The plugin itself never makes an
LLM call; this script does, on a cadence, when nobody is in the hot loop.

To schedule (example, M-F mornings), point a Hermes cron at:
    tools/correct_summaries.py --run
with a CHEAP model pinned:  auxiliary.nachos_summary.model in config.yaml.

Usage:
    correct_summaries.py                 # dry-run (default) — show diffs
    correct_summaries.py --limit 10      # dry-run, first 10 flagged entries
    correct_summaries.py --run           # actually write corrected summaries
    correct_summaries.py --store flatfile --path ~/.hermes/nachos/memory.md

Cost trap: if auxiliary.nachos_summary.model is unset, the aux client may
fall back to the MAIN model (possibly Opus/GPT-4o). This script prints the
resolved model at startup and WARNS if it looks expensive.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nachos_core.quality import (
    is_redundant_summary,
    low_signal_reason,
    shares_vocabulary,
)
from nachos_core.store import get_store

_API_DELAY_SECONDS = 1.5
_EXPENSIVE_HINTS = ("opus", "gpt-4o", "gemini-ultra")

_SYSTEM = (
    "You rewrite a ONE-LINE, keyword-dense summary of a memory entry so it "
    "accurately indexes the entry's body for retrieval. Name the concrete "
    "nouns (tools, repos, people, error strings). No fluff. Output only the "
    "single line, no quotes, no prefix."
)


def flag_reason(title: str, summary: str, body: str) -> str | None:
    """Why this entry needs LLM re-summarization, or None if it's fine.

    Shared logic with the write-time gate (nachos_core.quality) plus two
    audit-only checks that are too strict to enforce at write time
    (redundant-with-title, zero-vocabulary-overlap-with-body) but are
    still worth fixing on a periodic sweep.
    """
    reason = low_signal_reason(summary)
    if reason:
        return f"low-signal ({reason})"
    if is_redundant_summary(summary, title):
        return "redundant (just restates the title)"
    if not shares_vocabulary(summary, body):
        return "no shared vocabulary with body (likely stale)"
    return None


def _default_path(store_kind: str) -> Path:
    from hermes_constants import get_hermes_home

    base = Path(get_hermes_home()) / "nachos"
    return base / ("memory.md" if store_kind == "flatfile" else "memory.db")


def _resolve_client():
    from agent.auxiliary_client import get_text_auxiliary_client

    return get_text_auxiliary_client(task="nachos_summary")


def _regen_summary(client, model: str, title: str, body: str) -> str:
    user = f"Title: {title}\n\nBody:\n{body}\n\nOne-line summary:"
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        max_tokens=120,
    )
    try:
        resp = client.chat.completions.create(temperature=0.1, **kwargs)
    except TypeError:
        resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip().splitlines()[0].strip()


def _flag_entries(store, entries) -> list[tuple]:
    """Entries whose summary fails the quality bar, as (key, title, summary, category, body, reason)."""
    flagged = []
    for key, title, summary, category in entries:
        body = store.get(key) or ""
        if not body.strip():
            continue
        reason = flag_reason(title, summary, body)
        if reason:
            flagged.append((key, title, summary, category, body, reason))
    return flagged


def _report_no_client(flagged) -> int:
    print("ERROR: no auxiliary client configured (auxiliary.nachos_summary).")
    print("Flagged entries (no LLM available to fix them):")
    for key, _title, summary, _category, _body, reason in flagged:
        print(f"  [{key}] {reason}")
        print(f"    current: {summary!r}")
    return 2


def _correct(store, flagged, client, model, *, run: bool) -> int:
    """Regenerate each flagged summary; write it back when ``run``. Returns how many changed."""
    changed = 0
    for key, title, summary, category, body, reason in flagged:
        new_summary = _regen_summary(client, model, title, body)
        time.sleep(_API_DELAY_SECONDS)
        if new_summary and new_summary != (summary or "").strip():
            changed += 1
            print(f"[{key}] ({reason})")
            print(f"  old: {summary}")
            print(f"  new: {new_summary}")
            if run:
                store.put(key, title=title, summary=new_summary, category=category, body=body)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Nachos memory summary corrector")
    ap.add_argument("--store", default="sqlite", choices=["sqlite", "flatfile"])
    ap.add_argument("--path", default=None, help="store path (default: ~/.hermes/nachos/...)")
    ap.add_argument("--limit", type=int, default=0, help="max FLAGGED entries (0 = all)")
    ap.add_argument("--run", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()

    path = Path(args.path) if args.path else _default_path(args.store)
    store = get_store(args.store, path)
    entries = store.list()
    flagged = _flag_entries(store, entries)
    if args.limit:
        flagged = flagged[: args.limit]

    print(f"Store: {args.store} @ {path}")
    print(
        f"Entries: {len(entries)}   Flagged: {len(flagged)}   "
        f"Mode: {'RUN (writes)' if args.run else 'DRY-RUN'}"
    )
    if not flagged:
        print("Nothing flagged — every summary already passes the quality bar.")
        return 0

    client, model = _resolve_client()
    if not client or not model:
        return _report_no_client(flagged)
    print(f"Resolved extraction model: {model}")
    if any(h in model.lower() for h in _EXPENSIVE_HINTS):
        print(
            f"  ⚠ WARNING: {model!r} looks EXPENSIVE. Set "
            "auxiliary.nachos_summary.model to a cheap model in config.yaml."
        )
    print("-" * 60)
    changed = _correct(store, flagged, client, model, run=args.run)
    print("-" * 60)
    print(
        f"{'Corrected' if args.run else 'Would correct'}: {changed}/{len(flagged)} "
        f"flagged ({len(entries)} total entries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
