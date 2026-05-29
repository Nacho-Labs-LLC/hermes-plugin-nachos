"""Tests for budget, compactor, and snapshots — no Hermes dependency."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nachos_core.budget import (
    BudgetThresholds,
    calc_budget,
    decide,
)
from nachos_core.compactor import (
    drop_old_tool_results,
    find_tool_pairs,
    is_safe_cut,
    slide_window,
)
from nachos_core.snapshots import SnapshotStore


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def test_budget_green_zone():
    b = calc_budget(10_000, 200_000)
    assert b.zone == "green"
    assert b.action == "none"
    assert b.target_token_reduction == 0


def test_budget_yellow_zone():
    b = calc_budget(125_000, 200_000)  # 62.5%
    assert b.zone == "yellow"
    assert b.action == "prune"
    assert b.target_token_reduction > 0


def test_budget_orange_zone():
    b = calc_budget(155_000, 200_000)  # 77.5%
    assert b.zone == "orange"
    assert b.action == "light"


def test_budget_red_zone():
    b = calc_budget(175_000, 200_000)  # 87.5%
    assert b.zone == "red"
    assert b.action == "aggressive"


def test_budget_critical_zone():
    b = calc_budget(195_000, 200_000)  # 97.5%
    assert b.zone == "critical"
    assert b.action == "emergency"


def test_budget_zero_context_safe():
    b = calc_budget(1_000, 0)
    assert b.zone == "green"
    assert b.utilization_ratio == 0.0


def test_decide_recommends_snapshot_for_aggressive():
    b = calc_budget(180_000, 200_000)
    d = decide(b)
    assert d.snapshot_recommended


def test_decide_no_snapshot_for_light():
    b = calc_budget(155_000, 200_000)
    d = decide(b)
    assert not d.snapshot_recommended


def test_thresholds_override():
    custom = BudgetThresholds(
        proactive_prune=0.5, light_compaction=0.7,
        aggressive_compaction=0.8, emergency=0.9,
    )
    b = calc_budget(100_000, 200_000, custom)  # 50%
    assert b.zone == "yellow"


# ---------------------------------------------------------------------------
# Compactor — tool-pair detection
# ---------------------------------------------------------------------------

def _msg(role, content="", **extra):
    m = {"role": role, "content": content}
    m.update(extra)
    return m


def test_find_tool_pairs_simple():
    messages = [
        _msg("user", "do thing"),
        _msg("assistant", "calling tool",
             tool_calls=[{"id": "t1", "function": {"name": "x"}}]),
        _msg("tool", "result", tool_call_id="t1"),
        _msg("assistant", "done"),
    ]
    pairs = find_tool_pairs(messages)
    assert pairs == {1: 2}


def test_find_tool_pairs_multiple():
    messages = [
        _msg("assistant", "", tool_calls=[{"id": "a"}]),
        _msg("tool", "", tool_call_id="a"),
        _msg("user", "more"),
        _msg("assistant", "", tool_calls=[{"id": "b"}]),
        _msg("tool", "", tool_call_id="b"),
    ]
    pairs = find_tool_pairs(messages)
    assert pairs == {0: 1, 3: 4}


def test_is_safe_cut_through_pair_unsafe():
    messages = [
        _msg("user"),
        _msg("assistant", tool_calls=[{"id": "t1"}]),
        _msg("tool", tool_call_id="t1"),
        _msg("user"),
    ]
    # Cut at index 2 separates assistant (1) from its tool result (2)
    assert not is_safe_cut(messages, cut_at=2)


def test_is_safe_cut_outside_pair_safe():
    messages = [
        _msg("user"),
        _msg("assistant", tool_calls=[{"id": "t1"}]),
        _msg("tool", tool_call_id="t1"),
        _msg("user"),
    ]
    # Cut at index 3 keeps the pair on one side
    assert is_safe_cut(messages, cut_at=3)


# ---------------------------------------------------------------------------
# Compactor — drop_old_tool_results
# ---------------------------------------------------------------------------

def test_drop_old_tool_results_keeps_recent():
    messages = [
        _msg("user", "1"),
        _msg("tool", "old result 1" * 50, tool_call_id="t1"),
        _msg("tool", "old result 2" * 50, tool_call_id="t2"),
        _msg("tool", "old result 3" * 50, tool_call_id="t3"),
        _msg("tool", "recent result", tool_call_id="t4"),
    ]
    result = drop_old_tool_results(messages, keep_recent=1)
    assert result.tool_results_pruned == 3
    # Recent stayed verbatim
    assert result.messages[4]["content"] == "recent result"
    # Old got placeholdered
    assert "[tool result pruned by Nachos]" in result.messages[1]["content"]


def test_drop_old_tool_results_zero_when_under_keep():
    messages = [
        _msg("tool", "result1", tool_call_id="t1"),
        _msg("tool", "result2", tool_call_id="t2"),
    ]
    result = drop_old_tool_results(messages, keep_recent=6)
    assert result.tool_results_pruned == 0


def test_drop_old_tool_results_does_not_mutate_input():
    messages = [_msg("tool", "x" * 200, tool_call_id="t1"),
                _msg("tool", "y" * 200, tool_call_id="t2")]
    snapshot_before = messages[0]["content"]
    drop_old_tool_results(messages, keep_recent=1)
    assert messages[0]["content"] == snapshot_before


# ---------------------------------------------------------------------------
# Compactor — slide_window
# ---------------------------------------------------------------------------

def test_slide_window_drops_middle():
    messages = [
        _msg("system", "sys"),
        _msg("user", "first"),
        _msg("assistant", "reply 1"),
        _msg("user", "middle 1 " * 100),
        _msg("assistant", "middle 2 " * 100),
        _msg("user", "middle 3 " * 100),
        _msg("assistant", "tail 1"),
        _msg("user", "tail 2"),
    ]
    result = slide_window(messages, protect_head=1, protect_tail=2,
                          target_token_reduction=100)
    # System + protected head + protected tail stay
    assert result.messages[0]["role"] == "system"
    assert result.messages[-1]["content"] == "tail 2"
    # At least one middle message dropped
    assert result.dropped_count > 0


def test_slide_window_preserves_tool_pairs():
    messages = [
        _msg("system", "sys"),
        _msg("user", "first"),                                 # protected head
        _msg("assistant", "a", tool_calls=[{"id": "t1"}]),     # cohort #1
        _msg("tool", "r" * 500, tool_call_id="t1"),            # cohort #1
        _msg("assistant", "b", tool_calls=[{"id": "t2"}]),     # cohort #2
        _msg("tool", "r" * 500, tool_call_id="t2"),            # cohort #2
        _msg("user", "tail 1"),                                # protected tail
        _msg("user", "tail 2"),                                # protected tail
    ]
    result = slide_window(messages, protect_head=1, protect_tail=2,
                          target_token_reduction=10)
    # If anything dropped, the tool pair must be intact in the output:
    # for every tool message remaining, its assistant must remain too.
    pairs = find_tool_pairs(result.messages)
    for a_idx, t_idx in pairs.items():
        # Both assistant and tool present, contiguous in remaining list
        assert result.messages[a_idx].get("tool_calls")
        assert result.messages[t_idx].get("role") == "tool"
    # No orphaned tool messages
    for i, m in enumerate(result.messages):
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            # Find its assistant somewhere before in the remaining list
            assert any(
                tcid in {(tc.get("id") if isinstance(tc, dict)
                         else getattr(tc, "id", None))
                        for tc in (prev.get("tool_calls") or [])}
                for prev in result.messages[:i]
                if prev.get("role") == "assistant"
            ), f"orphaned tool result at remaining idx {i}"


def test_slide_window_nothing_to_drop_when_only_head_and_tail():
    messages = [
        _msg("user", "1"),
        _msg("assistant", "2"),
        _msg("user", "3"),
        _msg("assistant", "4"),
    ]
    result = slide_window(messages, protect_head=2, protect_tail=2,
                          target_token_reduction=100)
    assert result.dropped_count == 0
    assert len(result.messages) == 4


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def test_snapshot_save_and_load(tmp_path):
    store = SnapshotStore(tmp_path, session_id="abc-123")
    msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
    meta = store.save(msgs, reason="manual", label="kickoff")
    assert meta.id
    assert meta.message_count == 2

    loaded = store.load(meta.id)
    assert loaded is not None
    assert loaded.meta.label == "kickoff"
    assert loaded.messages == msgs


def test_snapshot_list_newest_first(tmp_path):
    store = SnapshotStore(tmp_path, session_id="s1")
    import time
    a = store.save([_msg("user", "a")], reason="manual")
    time.sleep(0.01)
    b = store.save([_msg("user", "b")], reason="manual")
    listing = store.list()
    assert listing[0]["id"] == b.id
    assert listing[1]["id"] == a.id


def test_snapshot_rotation_keeps_recent(tmp_path):
    store = SnapshotStore(tmp_path, session_id="s1")
    ids = []
    for i in range(5):
        meta = store.save([_msg("user", f"m{i}")], reason="manual")
        ids.append(meta.id)
    removed = store.rotate(keep=2, keep_labeled=True)
    assert removed == 3
    remaining = {entry["id"] for entry in store.list()}
    # The two newest ids should remain
    assert ids[-1] in remaining
    assert ids[-2] in remaining


def test_snapshot_rotation_protects_labeled(tmp_path):
    store = SnapshotStore(tmp_path, session_id="s1")
    labeled = store.save([_msg("user", "important")], reason="manual",
                         label="keep-me")
    for i in range(5):
        store.save([_msg("user", f"trash-{i}")], reason="auto")
    store.rotate(keep=2, keep_labeled=True)
    listing = store.list()
    remaining_ids = {entry["id"] for entry in listing}
    assert labeled.id in remaining_ids


def test_snapshot_load_missing_returns_none(tmp_path):
    store = SnapshotStore(tmp_path, session_id="s1")
    assert store.load("nonexistent") is None


def test_snapshot_session_id_sanitized(tmp_path):
    store = SnapshotStore(tmp_path, session_id="weird/../session?!")
    meta = store.save([_msg("user", "x")], reason="manual")
    assert "/" not in store.session_dir.name
    assert ".." not in store.session_dir.name
    assert store.load(meta.id) is not None
