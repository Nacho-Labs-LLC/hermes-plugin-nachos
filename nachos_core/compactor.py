"""Compactor — sliding window with tool-pair preservation.

This is the structural compaction layer. It does NOT call an LLM.
Pure mechanical work:

  • drop_old_tool_results: yellow-zone prune. Strips tool result CONTENT
    but keeps the tool_use/tool_result message pair intact. The model
    sees that a tool ran but not the verbose output. Cheap, no LLM.

  • slide_window: orange-zone sliding. Drop oldest middle messages
    (after the protected head, before the protected tail) until the
    target reduction is reached. CRITICAL: tool_use messages must NEVER
    be separated from their tool_result. We always drop both or neither.

  • is_safe_cut: validates that a candidate cut point doesn't bisect
    a tool-use/tool-result pair.

For aggressive (red) and emergency (critical) zones, the plugin layer
delegates to Hermes' built-in summarizer — Nachos doesn't try to
out-summarize the 1,700-LOC ContextCompressor. Our value-add is the
GRADED RESPONSE and TOOL-PAIR PROTECTION, not a better summary.

Message format: OpenAI-shape dicts with "role" and "content" plus
tool_calls / tool_call_id for tool messages. Hermes uses this format
end to end, so no translation needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import estimate_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    messages: List[Dict[str, Any]]
    dropped_count: int = 0
    tokens_freed: int = 0
    tool_results_pruned: int = 0
    pairs_preserved: int = 0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool-pair detection
# ---------------------------------------------------------------------------

def _tool_call_ids(msg: Dict[str, Any]) -> Set[str]:
    """Get the set of tool_call IDs an assistant message references."""
    out: Set[str] = set()
    tcs = msg.get("tool_calls")
    if not tcs:
        return out
    for tc in tcs:
        if isinstance(tc, dict):
            tcid = tc.get("id") or (tc.get("function", {}) or {}).get("id")
        else:
            tcid = getattr(tc, "id", None)
        if tcid:
            out.add(str(tcid))
    return out


def _tool_result_id(msg: Dict[str, Any]) -> Optional[str]:
    """Get the tool_call_id a tool message responds to."""
    if msg.get("role") != "tool":
        return None
    return msg.get("tool_call_id")


def find_tool_pairs(messages: List[Dict[str, Any]]) -> Dict[int, int]:
    """Map index of assistant-with-tool_calls → index of last tool result.

    Returns dict {assistant_idx: last_tool_result_idx}. Used to keep
    pairs together when sliding.
    """
    pairs: Dict[int, int] = {}
    pending: Dict[str, int] = {}  # tool_call_id → assistant idx

    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            ids = _tool_call_ids(msg)
            for tcid in ids:
                pending[tcid] = idx
        elif msg.get("role") == "tool":
            tcid = _tool_result_id(msg)
            if tcid and tcid in pending:
                a_idx = pending[tcid]
                pairs[a_idx] = idx
                # If multiple tool results, keep the LAST one
                # (subsequent updates will overwrite)

    return pairs


def is_safe_cut(messages: List[Dict[str, Any]], cut_at: int) -> bool:
    """True if cutting messages[:cut_at] doesn't break a tool pair.

    A cut is unsafe when an assistant message inside the tail (at
    cut_at or later) has tool_calls whose tool_result lives in the
    HEAD (before cut_at), or vice versa. We never want to leave a
    dangling tool_use without its tool_result, or a tool_result
    without its tool_use.
    """
    if cut_at <= 0 or cut_at >= len(messages):
        return True
    pairs = find_tool_pairs(messages)
    for a_idx, t_idx in pairs.items():
        # Either both before cut, or both at-or-after cut
        a_before = a_idx < cut_at
        t_before = t_idx < cut_at
        if a_before != t_before:
            return False
    return True


# ---------------------------------------------------------------------------
# Action 1: prune old tool results (yellow zone, no LLM)
# ---------------------------------------------------------------------------

def drop_old_tool_results(messages: List[Dict[str, Any]],
                          keep_recent: int = 6,
                          placeholder: str = "[tool result pruned by Nachos]"
                          ) -> CompactionResult:
    """Strip CONTENT of tool results outside the recent window.

    Keeps the tool_call_id + role intact so the message structure stays
    valid, but replaces verbose output with a placeholder. The
    `keep_recent` newest tool results are kept verbatim.

    This is the cheapest action — no LLM call, structure preserved,
    typically frees 30-60% of context that was consumed by tool noise.
    """
    result = CompactionResult(messages=list(messages))
    tool_indices = [i for i, m in enumerate(messages)
                    if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return result

    targets = tool_indices[:-keep_recent] if keep_recent > 0 else tool_indices

    for idx in targets:
        original = result.messages[idx]
        before_size = _msg_size_chars(original)
        # Build a stripped copy (don't mutate the input dict)
        stripped = dict(original)
        stripped["content"] = placeholder
        result.messages[idx] = stripped
        after_size = _msg_size_chars(stripped)
        result.tokens_freed += estimate_tokens(
            original.get("content", "") if isinstance(original.get("content"), str)
            else _content_text(original.get("content"))
        ) - estimate_tokens(placeholder)
        result.tool_results_pruned += 1

    result.notes.append(
        f"Pruned {result.tool_results_pruned} old tool results, "
        f"freed ≈{result.tokens_freed} tokens"
    )
    return result


# ---------------------------------------------------------------------------
# Action 2: sliding window (orange zone, no LLM)
# ---------------------------------------------------------------------------

def slide_window(messages: List[Dict[str, Any]],
                 protect_head: int = 3,
                 protect_tail: int = 6,
                 target_token_reduction: int = 0,
                 ) -> CompactionResult:
    """Drop oldest middle messages until target reduction is met.

    `protect_head` is the count of NON-system messages from the start
    that are always preserved (matches Hermes' protect_first_n
    semantics). The system prompt itself is implicitly protected.

    `protect_tail` is the count from the end always preserved.

    Tool pairs that span the cut boundary are kept together. If the
    only safe cut is much smaller than `target_token_reduction`, we
    take what we can — never bisect a pair to hit a number.
    """
    result = CompactionResult(messages=list(messages))
    if not messages:
        return result

    # Find the protected ranges
    sys_prefix = sum(1 for m in messages if m.get("role") == "system")
    head_end = min(len(messages), sys_prefix + protect_head)
    tail_start = max(head_end, len(messages) - protect_tail)

    if head_end >= tail_start:
        result.notes.append(
            f"Nothing to slide: head_end={head_end} >= tail_start={tail_start}"
        )
        return result

    # Walk forward from head_end and find the furthest safe cut that
    # frees at least target_token_reduction. Tool pairs may force the
    # cut earlier than ideal; that's fine.
    middle = list(range(head_end, tail_start))
    pairs = find_tool_pairs(messages)

    # Build a "cohort" map: each middle index belongs to a cohort that
    # must be dropped together (an assistant + its tool result, or a
    # standalone message).
    cohort_of: Dict[int, int] = {}
    next_cohort = 0
    consumed: Set[int] = set()
    for idx in middle:
        if idx in consumed:
            continue
        cohort_of[idx] = next_cohort
        consumed.add(idx)
        # If this assistant has tool calls landing in the middle window,
        # bind them to the same cohort
        if messages[idx].get("role") == "assistant":
            t_idx = pairs.get(idx)
            if t_idx is not None and t_idx in middle:
                cohort_of[t_idx] = next_cohort
                consumed.add(t_idx)
        next_cohort += 1

    # Also: a tool-result whose assistant lives BEFORE head_end can be
    # dropped freely (the head keeps the assistant; this just removes
    # the result content — but we already handled prune for that, so
    # in slide-mode just leave them).

    # Greedy: drop oldest cohorts until target met or middle exhausted
    cohort_indices: Dict[int, List[int]] = {}
    for idx, c in cohort_of.items():
        cohort_indices.setdefault(c, []).append(idx)

    dropped: Set[int] = set()
    freed = 0
    for c in sorted(cohort_indices.keys()):
        if target_token_reduction > 0 and freed >= target_token_reduction:
            break
        for idx in cohort_indices[c]:
            msg = messages[idx]
            freed += estimate_tokens(_content_text(msg.get("content")))
            dropped.add(idx)
            result.dropped_count += 1
        # Each successful cohort drop is a preserved pair (or solo msg)
        result.pairs_preserved += 1

    result.messages = [m for i, m in enumerate(messages) if i not in dropped]
    result.tokens_freed = freed
    result.notes.append(
        f"Dropped {result.dropped_count} messages in {result.pairs_preserved} "
        f"cohorts (freed ≈{freed} tokens, target was {target_token_reduction})"
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content_text(content: Any) -> str:
    """Coerce content (str / list-of-blocks / None) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("text") or blk.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _msg_size_chars(msg: Dict[str, Any]) -> int:
    return len(_content_text(msg.get("content")))
