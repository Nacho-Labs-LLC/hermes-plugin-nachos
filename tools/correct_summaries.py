#!/usr/bin/env python3
"""Parked companion cron — periodic memory-summary self-correction.

NOT wired to any live cron. This is the reflection-pattern hygiene layer
the spec parks OUTSIDE the plugin hot path: it re-reads each memory entry's
body and, using the auxiliary client, regenerates the one-line manifest
summary IF it has drifted from the body. The plugin itself never makes an
LLM call — this script does, on a cadence, when nobody is in the hot loop.

To schedule (example, M-F mornings), point a Hermes cron at:
    tools/correct_summaries.py --run
with a CHEAP model pinned:  auxiliary.nachos_summary.model in config.yaml.

Usage:
    correct_summaries.py                 # dry-run (default) — show diffs
    correct_summaries.py --limit 10      # dry-run, first 10 entries
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

from nachos_core.store import get_store  # noqa: E402

_API_DELAY_SECONDS = 1.5
_EXPENSIVE_HINTS = ("opus", "gpt-4o", "gemini-ultra")

_SYSTEM = (
    "You rewrite a ONE-LINE, keyword-dense summary of a memory entry so it "
    "accurately indexes the entry's body for retrieval. Name the concrete "
    "nouns (tools, repos, people, error strings). No fluff. Output only the "
    "single line, no quotes, no prefix."
)


def _default_path(store_kind: str) -> Path:
    from hermes_constants import get_hermes_home  # type: ignore
    base = Path(get_hermes_home()) / "nachos"
    return base / ("memory.md" if store_kind == "flatfile" else "memory.db")


def _resolve_client():
    from agent.auxiliary_client import get_text_auxiliary_client
    return get_text_auxiliary_client(task="nachos_summary")


def _regen_summary(client, model: str, title: str, body: str) -> str:
    user = f"Title: {title}\n\nBody:\n{body}\n\nOne-line summary:"
    kwargs = dict(model=model,
                  messages=[{"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": user}],
                  max_tokens=120)
    try:
        resp = client.chat.completions.create(temperature=0.1, **kwargs)
    except TypeError:
        resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip().splitlines()[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Nachos memory summary corrector")
    ap.add_argument("--store", default="sqlite", choices=["sqlite", "flatfile"])
    ap.add_argument("--path", default=None, help="store path (default: ~/.hermes/nachos/...)")
    ap.add_argument("--limit", type=int, default=0, help="max entries (0 = all)")
    ap.add_argument("--run", action="store_true", help="write changes (default: dry-run)")
    args = ap.parse_args()

    path = Path(args.path) if args.path else _default_path(args.store)
    store = get_store(args.store, path)
    entries = store.list()
    if args.limit:
        entries = entries[: args.limit]

    client, model = _resolve_client()
    if not client or not model:
        print("ERROR: no auxiliary client configured (auxiliary.nachos_summary).")
        return 2
    print(f"Resolved extraction model: {model}")
    if any(h in model.lower() for h in _EXPENSIVE_HINTS):
        print(f"  ⚠ WARNING: {model!r} looks EXPENSIVE. Set "
              "auxiliary.nachos_summary.model to a cheap model in config.yaml.")
    print(f"Store: {args.store} @ {path}")
    print(f"Entries: {len(entries)}   Mode: {'RUN (writes)' if args.run else 'DRY-RUN'}")
    print("-" * 60)

    changed = 0
    for (key, title, summary, category) in entries:
        body = store.get(key) or ""
        if not body.strip():
            continue
        new_summary = _regen_summary(client, model, title, body)
        time.sleep(_API_DELAY_SECONDS)
        if new_summary and new_summary != (summary or "").strip():
            changed += 1
            print(f"[{key}]")
            print(f"  old: {summary}")
            print(f"  new: {new_summary}")
            if args.run:
                store.put(key, title=title, summary=new_summary,
                          category=category, body=body)
    print("-" * 60)
    print(f"{'Corrected' if args.run else 'Would correct'}: {changed}/{len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
