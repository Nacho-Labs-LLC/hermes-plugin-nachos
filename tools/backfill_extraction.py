#!/usr/bin/env python3
"""Nachos backfill extraction — replay past sessions through the fact extractor.

Reads from two sources:
  - Hermes state.db (all CLI sessions with enough messages)
  - Claude Code ~/.claude/projects/**/*.jsonl

Extracts durable facts into ~/.hermes/nachos/facts.jsonl using the same
extraction + dedup pipeline as the live plugin.

Usage:
    python backfill_extraction.py [--dry-run] [--hermes-only] [--claude-only]
    python backfill_extraction.py --dry-run          # estimate cost, no API calls
    python backfill_extraction.py --min-messages 5   # only sessions with >= N msgs
    python backfill_extraction.py --limit 10         # process at most N sessions
    python backfill_extraction.py --source hermes    # hermes only
    python backfill_extraction.py --source claude    # claude code only

Run from any directory — paths are resolved from HERMES_HOME.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve nachos_core from repo
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path.home() / "DEV" / "hermes-agent"
for p in [str(REPO_ROOT), str(HERMES_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nachos_core.extractor import ExtractionConfig, extract_facts
from nachos_core.types import MemoryFact
from adapters.hermes_extractor import JsonlFactStore, make_hermes_llm_call

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HERMES_HOME = Path.home() / ".hermes"
STATE_DB = HERMES_HOME / "state.db"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
FACTS_FILE = HERMES_HOME / "nachos" / "facts.jsonl"

# Max chars to send per session transcript to the extraction LLM.
# Keeps cost/latency bounded. Older/larger sessions are truncated from
# the beginning (keep recent context, most durable signal is in the tail).
MAX_CHARS_PER_SESSION = 40_000

# Delay between API calls (seconds) — avoids hammering Bedrock rate limits
API_DELAY_SECONDS = 1.5

# Code block filter — strips fenced code from assistant messages before extraction.
# These dominate Claude Code sessions but carry almost no durable-fact signal.
_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_XML_TOOL_RE = re.compile(r"<(?:function_calls|tool_use|parameter|local-command|command-name|command-message|command-args)[^>]*>[\s\S]*?</[^>]+>", re.IGNORECASE)
_LONG_LINE_RE = re.compile(r"^.{300,}$", re.MULTILINE)  # likely JSON/code dumps


def _clean_content(text: str, strip_code: bool = True) -> str:
    """Strip code blocks and tool-call XML from content."""
    if not text:
        return ""
    if strip_code:
        text = _CODE_FENCE_RE.sub("[code block omitted]", text)
    text = _XML_TOOL_RE.sub("", text)
    # Remove very long lines (likely minified code or JSON)
    text = _LONG_LINE_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Source: Hermes state.db
# ---------------------------------------------------------------------------

def iter_hermes_sessions(min_messages: int = 4) -> Generator[Tuple[str, str, List[Dict]], None, None]:
    """Yield (session_id, title, messages) for Hermes CLI sessions."""
    if not STATE_DB.exists():
        print(f"  [skip] Hermes state.db not found at {STATE_DB}")
        return

    con = sqlite3.connect(str(STATE_DB))
    try:
        # Get qualifying sessions
        cur = con.execute("""
            SELECT s.id, s.title, count(m.id) as msg_count
            FROM sessions s
            JOIN messages m ON m.session_id = s.id
                AND m.role IN ('user','assistant')
            WHERE s.source = 'cli'
            GROUP BY s.id
            HAVING msg_count >= ?
            ORDER BY s.started_at ASC
        """, (min_messages,))
        sessions = cur.fetchall()

        for sid, title, msg_count in sessions:
            cur2 = con.execute("""
                SELECT role, content FROM messages
                WHERE session_id = ? AND role IN ('user','assistant')
                ORDER BY timestamp ASC
            """, (sid,))
            messages = [
                {"role": role, "content": _clean_content(content or "", strip_code=False)}
                for role, content in cur2.fetchall()
                if (content or "").strip()
            ]
            if len(messages) >= min_messages:
                yield sid, title or "(untitled)", messages
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Source: Claude Code ~/.claude/projects/**/*.jsonl
# ---------------------------------------------------------------------------

def _extract_text_from_claude_content(content: Any) -> str:
    """Get text from Claude Code message content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "text":
                parts.append(blk.get("text", ""))
            elif blk.get("type") == "tool_result":
                # Include tool result text but truncated
                inner = blk.get("content", "")
                if isinstance(inner, list):
                    inner = " ".join(
                        b.get("text", "") for b in inner
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if inner and len(str(inner)) > 20:
                    parts.append(f"[tool result]: {str(inner)[:300]}")
        return "\n".join(parts)
    return ""


def iter_claude_sessions(min_messages: int = 4) -> Generator[Tuple[str, str, List[Dict]], None, None]:
    """Yield (session_id, title, messages) for Claude Code sessions."""
    if not CLAUDE_PROJECTS.exists():
        print(f"  [skip] Claude projects dir not found at {CLAUDE_PROJECTS}")
        return

    for project_dir in sorted(CLAUDE_PROJECTS.iterdir()):
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name.replace("-Users-naterichardson-DEV-", "")

        for jsonl_file in sorted(project_dir.glob("*.jsonl"),
                                  key=lambda f: f.stat().st_mtime):
            session_id = f"claude/{project_name}/{jsonl_file.stem}"
            try:
                lines = jsonl_file.read_text(errors="replace").strip().split("\n")
            except Exception as e:
                print(f"  [skip] Could not read {jsonl_file}: {e}")
                continue

            messages = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = obj.get("type")
                if entry_type not in ("user", "assistant"):
                    continue

                raw_content = obj.get("message", {}).get("content", "")
                text = _extract_text_from_claude_content(raw_content)

                # Skip system/tool-injection noise from Claude Code
                if any(skip in text[:120] for skip in [
                    "<local-command-caveat>", "<command-name>", "<function_calls>",
                    "<parameter", "CAVEAT:", "DO NOT respond to these messages",
                ]):
                    continue

                cleaned = _clean_content(text, strip_code=True)
                if len(cleaned.strip()) < 20:
                    continue

                messages.append({"role": entry_type, "content": cleaned})

            if len(messages) >= min_messages:
                # Use project + first user message as a synthetic title
                first_user = next(
                    (m["content"][:60] for m in messages if m["role"] == "user"),
                    "(code session)"
                )
                title = f"{project_name}: {first_user}"
                yield session_id, title, messages


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(sessions: List[Tuple[str, str, List[Dict]]]) -> Dict[str, Any]:
    """Estimate token usage and cost before running."""
    total_chars = 0
    for _, _, msgs in sessions:
        # Build the transcript the same way the extractor would
        transcript = "\n\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in msgs
        )
        total_chars += min(len(transcript), MAX_CHARS_PER_SESSION)

    # Input tokens (transcript) + output tokens (fact JSON, ~500 avg per session)
    input_tokens = total_chars // 4
    output_tokens = len(sessions) * 500

    # Sonnet 4 pricing: $3/1M input, $15/1M output (approximate).
    # If auxiliary.extraction.model is not set, falls back to main model.
    # Run: hermes config set auxiliary.extraction.model us.anthropic.claude-sonnet-4-6
    input_cost = input_tokens * 3 / 1_000_000
    output_cost = output_tokens * 15 / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "sessions": len(sessions),
        "total_chars": total_chars,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": total_cost,
        "estimated_time_minutes": len(sessions) * API_DELAY_SECONDS / 60 * 2,
    }


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------

def run_backfill(
    sessions: List[Tuple[str, str, List[Dict]]],
    fact_store: JsonlFactStore,
    llm_call,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, int]:
    """Run extraction over all sessions, upsert into fact store."""
    totals = {"sessions": 0, "kept": 0, "inserted": 0, "updated": 0,
              "failed": 0, "empty": 0}

    for i, (session_id, title, messages) in enumerate(sessions):
        prefix = f"  [{i+1}/{len(sessions)}]"
        title_short = title[:55]

        if dry_run:
            # Just report what we would do
            chars = sum(len(m["content"]) for m in messages)
            print(f"{prefix} {title_short}")
            print(f"         session={session_id} msgs={len(messages)} chars={chars:,}")
            totals["sessions"] += 1
            continue

        cfg = ExtractionConfig(
            max_conversation_chars=MAX_CHARS_PER_SESSION,
            min_confidence=0.65,
            max_response_tokens=4096,    # 2048 truncates dense sessions
            default_source_session=session_id,
        )

        try:
            result = extract_facts(messages, llm_call, cfg)
        except Exception as e:
            print(f"{prefix} [ERROR] {title_short[:40]} — {e}")
            totals["failed"] += 1
            time.sleep(API_DELAY_SECONDS)
            continue

        if not result.parse_success:
            print(f"{prefix} [PARSE FAIL] {title_short[:40]} — {result.error or 'unknown'}")
            totals["failed"] += 1
            time.sleep(API_DELAY_SECONDS)
            continue

        if result.kept == 0:
            if verbose:
                print(f"{prefix} [EMPTY] {title_short[:40]} (raw={result.raw_count})")
            totals["empty"] += 1
            time.sleep(API_DELAY_SECONDS)
            continue

        inserted, updated = fact_store.upsert(result.facts)
        totals["sessions"] += 1
        totals["kept"] += result.kept
        totals["inserted"] += inserted
        totals["updated"] += updated

        print(f"{prefix} {title_short}")
        print(f"         kept={result.kept} raw={result.raw_count} "
              f"inserted={inserted} updated={updated}")

        time.sleep(API_DELAY_SECONDS)

    return totals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nachos backfill extraction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost and list sessions, no API calls")
    parser.add_argument("--source", choices=["hermes", "claude", "both"],
                        default="both", help="Which sources to backfill")
    parser.add_argument("--min-messages", type=int, default=4,
                        help="Min user+assistant messages to include a session")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N sessions (for testing)")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true",
                        help="Only show errors + summary")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    print("🧀 Nachos Backfill Extraction")
    print("=" * 60)
    print()

    # Gather sessions
    sessions: List[Tuple[str, str, List[Dict]]] = []

    if args.source in ("hermes", "both"):
        print("Scanning Hermes state.db...")
        hermes_sessions = list(iter_hermes_sessions(args.min_messages))
        print(f"  Found {len(hermes_sessions)} Hermes sessions with >= {args.min_messages} messages")
        sessions.extend(hermes_sessions)

    if args.source in ("claude", "both"):
        print("Scanning Claude Code projects...")
        claude_sessions = list(iter_claude_sessions(args.min_messages))
        print(f"  Found {len(claude_sessions)} Claude Code sessions with >= {args.min_messages} messages")
        sessions.extend(claude_sessions)

    print()

    if not sessions:
        print("No sessions to process. Exiting.")
        return

    # Apply limit
    if args.limit:
        sessions = sessions[:args.limit]
        print(f"Limited to {args.limit} sessions (--limit flag)")

    # Cost estimate
    est = estimate_cost(sessions)
    print("Cost estimate:")
    print(f"  Sessions:           {est['sessions']}")
    print(f"  Transcript chars:   {est['total_chars']:,}")
    print(f"  Input tokens:       {est['input_tokens']:,}")
    print(f"  Output tokens:      {est['output_tokens']:,}")
    print(f"  Estimated cost:     ${est['estimated_cost_usd']:.2f}")
    print(f"  Estimated time:     ~{est['estimated_time_minutes']:.0f} min")
    print()

    if args.dry_run:
        print("[DRY RUN] Sessions that would be processed:")
        print()
        for i, (sid, title, msgs) in enumerate(sessions):
            chars = sum(len(m["content"]) for m in msgs)
            print(f"  [{i+1:>3}] {title[:60]}")
            print(f"        id={sid} msgs={len(msgs)} chars={chars:,}")
        print()
        print(f"[DRY RUN] Total: {len(sessions)} sessions, "
              f"estimated cost ${est['estimated_cost_usd']:.2f}")
        return

    # Confirm before running
    print(f"About to extract facts from {len(sessions)} sessions.")
    print(f"Facts will be saved to: {FACTS_FILE}")
    print()
    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Aborted.")
        return

    print()
    print("Initializing LLM call...")
    try:
        llm_call = make_hermes_llm_call()
        # Peek at what model resolved — import here to avoid circular deps
        from agent.auxiliary_client import get_text_auxiliary_client
        _, resolved_model = get_text_auxiliary_client(task="extraction")
        print(f"  Model: {resolved_model}")
        print(f"  Auxiliary client resolved.")
    except Exception as e:
        print(f"  ERROR: Could not initialize LLM call: {e}")
        print("  Make sure Hermes is configured with a working provider.")
        return

    fact_store = JsonlFactStore(FACTS_FILE)
    existing_count = len(fact_store.list_all())
    print(f"  Existing facts in store: {existing_count}")
    print()

    print("Running extraction...")
    print("-" * 60)
    started = time.time()

    totals = run_backfill(
        sessions=sessions,
        fact_store=fact_store,
        llm_call=llm_call,
        dry_run=False,
        verbose=verbose,
    )

    elapsed = time.time() - started
    final_count = len(fact_store.list_all())

    print()
    print("=" * 60)
    print("Backfill complete.")
    print(f"  Sessions processed:  {totals['sessions']}")
    print(f"  Sessions failed:     {totals['failed']}")
    print(f"  Sessions empty:      {totals['empty']}")
    print(f"  Facts kept:          {totals['kept']}")
    print(f"  Facts inserted:      {totals['inserted']}")
    print(f"  Facts updated:       {totals['updated']}")
    print(f"  Total facts in store: {existing_count} → {final_count}")
    print(f"  Wall time:           {elapsed:.0f}s")
    print()
    print(f"Facts saved to: {FACTS_FILE}")
    print("Run nachos_status.sh to see the updated fact store.")


if __name__ == "__main__":
    main()
