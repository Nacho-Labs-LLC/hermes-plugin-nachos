# nachos_core — harness-agnostic context + memory-manifest layer.
#
# Pure Python, stdlib only. Does NOT import from hermes_* / agent.*.
# All host wiring lives in the plugin (plugins/memory/nachos, etc.).
from .budget import (
    Action,
    Budget,
    BudgetThresholds,
    CompactionDecision,
    Zone,
    calc_budget,
    decide,
)
from .compactor import (
    CompactionResult,
    drop_old_tool_results,
    find_tool_pairs,
    is_safe_cut,
    slide_window,
)
from .prefetch import LexicalScorer, Scorer, get_scorer
from .snapshots import (
    Snapshot,
    SnapshotMeta,
    SnapshotStore,
)
from .store import Entry, MDStore, MemoryStore, SqliteStore, get_store
from .toc import build_toc, render_toc
from .types import (
    ContextEntry,
    DurableFact,
    MemoryEntry,
    MemoryFact,
    PromptContribution,
    PromptContributionReport,
    PromptReport,
    PromptSectionReport,
)

__all__ = [
    # types
    "ContextEntry",
    "DurableFact",
    "PromptContribution",
    "PromptContributionReport",
    "PromptSectionReport",
    "PromptReport",
    "MemoryFact",
    "MemoryEntry",
    # budget
    "Action",
    "Budget",
    "BudgetThresholds",
    "CompactionDecision",
    "Zone",
    "calc_budget",
    "decide",
    # compactor
    "CompactionResult",
    "drop_old_tool_results",
    "find_tool_pairs",
    "is_safe_cut",
    "slide_window",
    # snapshots
    "Snapshot",
    "SnapshotMeta",
    "SnapshotStore",
    # store seam
    "MemoryStore",
    "Entry",
    "SqliteStore",
    "MDStore",
    "get_store",
    # scorer seam
    "Scorer",
    "LexicalScorer",
    "get_scorer",
    # manifest
    "build_toc",
    "render_toc",
]
