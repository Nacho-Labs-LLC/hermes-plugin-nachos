"""Context budget — zone-based pressure model.

Hermes reports tokens used vs context length. Nachos turns that into a
zone (green/yellow/orange/red/critical) and recommends an action. The
context engine plugin uses this to decide:

  green     — no compaction needed
  yellow    — proactive prune (cheap; drop old tool results only)
  orange    — light compaction (sliding window, no LLM call)
  red       — aggressive compaction (sliding + summarization)
  critical  — emergency compaction (drop more, summarize aggressively)

The point: instead of a single binary "should_compress?" bool, we get
a graded response that lets cheaper actions fire earlier and more often.
Most turns can be handled with prune-only — saving an LLM call.

This module is pure logic. The compactor.py wires it to actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Zone = Literal["green", "yellow", "orange", "red", "critical"]
Action = Literal["none", "prune", "light", "aggressive", "emergency"]


@dataclass
class BudgetThresholds:
    """Utilization ratios at which each zone activates.

    Defaults are tuned for typical 200K-token context windows. Smaller
    windows (32K) can keep these or override them via config; the
    relative pressure model still works.
    """
    proactive_prune: float = 0.60      # yellow
    light_compaction: float = 0.75     # orange (matches Hermes default 0.75)
    aggressive_compaction: float = 0.85  # red
    emergency: float = 0.95            # critical


@dataclass
class Budget:
    current_usage: int                 # tokens
    context_length: int                # tokens
    utilization_ratio: float           # current / context_length
    zone: Zone
    action: Action
    target_token_reduction: int        # how many tokens to free


@dataclass
class CompactionDecision:
    """Output of `decide()` — what should happen this turn."""
    zone: Zone
    action: Action
    reason: str
    target_token_reduction: int
    snapshot_recommended: bool         # for red/critical zones


def calc_budget(current_usage: int, context_length: int,
                thresholds: BudgetThresholds = BudgetThresholds()
                ) -> Budget:
    """Calculate utilization + zone + recommended action."""
    if context_length <= 0:
        return Budget(0, 0, 0.0, "green", "none", 0)

    ratio = current_usage / context_length

    if ratio >= thresholds.emergency:
        zone, action, target = "critical", "emergency", int(current_usage * 0.6)
    elif ratio >= thresholds.aggressive_compaction:
        zone, action, target = "red", "aggressive", int(current_usage * 0.4)
    elif ratio >= thresholds.light_compaction:
        zone, action, target = "orange", "light", int(current_usage * 0.3)
    elif ratio >= thresholds.proactive_prune:
        zone, action, target = "yellow", "prune", int(current_usage * 0.15)
    else:
        zone, action, target = "green", "none", 0

    return Budget(
        current_usage=current_usage,
        context_length=context_length,
        utilization_ratio=ratio,
        zone=zone,
        action=action,
        target_token_reduction=target,
    )


def decide(budget: Budget) -> CompactionDecision:
    """Wrap a Budget in an actionable decision with reasoning + advice."""
    reasons = {
        "none":       "Below proactive-prune threshold; no action needed.",
        "prune":      "Yellow zone — drop stale tool results without LLM call.",
        "light":      "Orange zone — sliding window compaction recommended.",
        "aggressive": "Red zone — sliding + summarization, snapshot first.",
        "emergency":  "Critical zone — emergency compaction, must snapshot.",
    }
    snapshot_recommended = budget.action in ("aggressive", "emergency")
    return CompactionDecision(
        zone=budget.zone,
        action=budget.action,
        reason=reasons[budget.action],
        target_token_reduction=budget.target_token_reduction,
        snapshot_recommended=snapshot_recommended,
    )
