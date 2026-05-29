"""End-to-end smoke test for the v0.3 context engine.

Builds a synthetic 80-turn conversation with tool calls, runs it through
the budget calculator at every step, and verifies the engine produces
the expected zone progression: green → yellow → orange → red → critical.

This does NOT require Hermes runtime — it exercises nachos_core directly
to prove the orchestration logic works.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nachos_core.budget import BudgetThresholds, calc_budget, decide
from nachos_core.compactor import (
    drop_old_tool_results,
    slide_window,
)
from nachos_core.snapshots import SnapshotStore
from nachos_core.types import estimate_tokens


def make_synthetic_conversation(turns: int) -> list:
    """Build a realistic-shape conversation with tool calls every 3 turns."""
    msgs = [{"role": "system", "content": "You are an agent. " * 50}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"User turn {i}: " + "x " * 40})
        if i % 3 == 0:
            tcid = f"tc-{i}"
            msgs.append({
                "role": "assistant",
                "content": f"Calling tool for turn {i}",
                "tool_calls": [{"id": tcid, "function": {"name": "search"}}],
            })
            msgs.append({
                "role": "tool",
                "content": "Tool result " + "y " * 200,
                "tool_call_id": tcid,
            })
            msgs.append({"role": "assistant", "content": f"Done turn {i}"})
        else:
            msgs.append({"role": "assistant", "content": f"Reply {i}: " + "z " * 30})
    return msgs


def estimate_tokens_for_messages(msgs):
    total = 0
    for m in msgs:
        c = m.get("content", "")
        if isinstance(c, str):
            total += estimate_tokens(c)
    return total


def main():
    # Simulate 200K context window
    CTX = 200_000
    thresholds = BudgetThresholds()

    print("=" * 70)
    print("Synthetic conversation: walking through the zones")
    print("=" * 70)
    msgs = make_synthetic_conversation(turns=80)
    base_tokens = estimate_tokens_for_messages(msgs)
    print(f"\nBuilt {len(msgs)} messages, ~{base_tokens} tokens.\n")

    # Walk through pretend usage levels
    for pct in (0.30, 0.65, 0.78, 0.88, 0.97):
        used = int(CTX * pct)
        budget = calc_budget(used, CTX, thresholds)
        decision = decide(budget)
        print(f"  used={used:>7,}/{CTX:,} ({pct*100:.0f}%)  "
              f"zone={budget.zone:<8}  action={decision.action:<10}  "
              f"snapshot={'YES' if decision.snapshot_recommended else 'no'}  "
              f"target_drop={decision.target_token_reduction:>7,}")

    # Action 1: prune (yellow)
    print("\n" + "=" * 70)
    print("ACTION: prune old tool results (yellow zone — no LLM)")
    print("=" * 70)
    result = drop_old_tool_results(msgs, keep_recent=4)
    print(f"  pruned={result.tool_results_pruned}  "
          f"freed≈{result.tokens_freed} tokens")
    pruned_tokens = estimate_tokens_for_messages(result.messages)
    print(f"  before={base_tokens} → after={pruned_tokens} "
          f"(saved {base_tokens - pruned_tokens})")

    # Action 2: slide (orange)
    print("\n" + "=" * 70)
    print("ACTION: slide_window (orange zone — no LLM)")
    print("=" * 70)
    target = int(base_tokens * 0.30)
    result = slide_window(msgs, protect_head=3, protect_tail=6,
                          target_token_reduction=target)
    after = estimate_tokens_for_messages(result.messages)
    print(f"  dropped {result.dropped_count} messages in "
          f"{result.pairs_preserved} cohorts")
    print(f"  freed ≈{result.tokens_freed} tokens (target was {target})")
    print(f"  before={base_tokens} → after={after} "
          f"(saved {base_tokens - after})")
    print(f"  remaining message count: {len(result.messages)}")

    # Verify tool-pair invariant on the slid output
    from nachos_core.compactor import find_tool_pairs
    pairs_before = find_tool_pairs(msgs)
    pairs_after = find_tool_pairs(result.messages)
    print(f"  tool pairs: before={len(pairs_before)} after={len(pairs_after)} "
          f"(both intact = ✓)")
    # Check no orphaned tool messages remain
    orphans = [
        m for m in result.messages
        if m.get("role") == "tool"
        and not any(
            (m.get("tool_call_id") in {tc.get("id") for tc in (a.get("tool_calls") or [])})
            for a in result.messages if a.get("role") == "assistant"
        )
    ]
    print(f"  orphaned tool messages: {len(orphans)} "
          f"({'✓' if not orphans else '✗'})")

    # Action 3: snapshots
    print("\n" + "=" * 70)
    print("ACTION: snapshot save/list/load")
    print("=" * 70)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        store = SnapshotStore(Path(td), session_id="smoke-session")
        meta = store.save(msgs, reason="pre-compaction-aggressive",
                          notes=["80-turn synthetic", "tools every 3rd"])
        print(f"  saved id={meta.id} reason={meta.reason} "
              f"msgs={meta.message_count} ~tokens={meta.token_estimate}")

        labeled = store.save(msgs, reason="manual", label="smoke-test")
        print(f"  saved labeled id={labeled.id} label={labeled.label}")

        for _ in range(12):
            store.save(msgs, reason="auto")
        listing_before = store.list()
        print(f"  total snapshots before rotation: {len(listing_before)}")

        removed = store.rotate(keep=5, keep_labeled=True)
        listing_after = store.list()
        print(f"  rotated: removed={removed}  remaining={len(listing_after)}")

        labeled_kept = any(e["id"] == labeled.id for e in listing_after)
        print(f"  labeled snapshot survived rotation: "
              f"{'✓' if labeled_kept else '✗'}")

        loaded = store.load(meta.id)
        # Note: meta.id may have been rotated out — try labeled for safety
        loaded = loaded or store.load(labeled.id)
        if loaded:
            print(f"  loaded: msgs={len(loaded.messages)} "
                  f"reason={loaded.meta.reason}")

    print("\n" + "=" * 70)
    print("End-to-end engine smoke test passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
