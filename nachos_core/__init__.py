# nachos_core — harness-agnostic context layer.
#
# This package contains the Nachos system: layered prompt assembly,
# memory manifest, fact extraction, and PromptReport observability.
# It does NOT import from hermes_*. All Hermes wiring lives in adapters/.
from .types import PromptSectionReport, PromptReport, MemoryFact, MemoryEntry
from .assembler import PromptAssembler, AssembleParams
from .manifest import build_manifest, render_manifest
from .extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionConfig,
    ExtractionResult,
    LLMCall,
    build_user_message,
    extract_facts,
)
from .dedup import (
    DedupResult,
    deduplicate_facts,
    is_exact_match,
    merge_fact,
)
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
from .snapshots import (
    Snapshot,
    SnapshotMeta,
    SnapshotStore,
)

__all__ = [
    # types
    "PromptSectionReport",
    "PromptReport",
    "MemoryFact",
    "MemoryEntry",
    # assembler
    "PromptAssembler",
    "AssembleParams",
    # manifest
    "build_manifest",
    "render_manifest",
    # extractor
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractionConfig",
    "ExtractionResult",
    "LLMCall",
    "build_user_message",
    "extract_facts",
    # dedup
    "DedupResult",
    "deduplicate_facts",
    "is_exact_match",
    "merge_fact",
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
]
