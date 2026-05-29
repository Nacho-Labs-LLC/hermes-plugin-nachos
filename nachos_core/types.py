"""Core type definitions for nachos_core.

These dataclasses are the cross-layer contract. Everything in nachos_core
imports from here; adapters do too. Keep this file dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Memory model — typed triples + entries
# ---------------------------------------------------------------------------

@dataclass
class MemoryFact:
    """A durable, structured fact extracted from conversation.

    Triples are easier to dedup, easier to render in prompts, and easier
    to score for confidence than free-form notes.
    """
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    kind: str = "general"          # 'preference' | 'environment' | 'project' | 'general'
    source_session: Optional[str] = None
    extracted_at: str = field(default_factory=lambda: _utcnow())

    def render(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass
class MemoryEntry:
    """Free-form memory note (the existing Hermes built-in shape)."""
    content: str
    kind: str = "note"             # 'note' | 'preference' | 'fact' | 'user'
    tags: List[str] = field(default_factory=list)
    target: str = "memory"         # 'memory' | 'user'
    created_at: str = field(default_factory=lambda: _utcnow())


# ---------------------------------------------------------------------------
# PromptReport — per-turn audit trail
# ---------------------------------------------------------------------------

@dataclass
class PromptSectionReport:
    name: str                      # 'base' | 'manifest' | 'memory' | 'skills' | ...
    size_chars: int
    size_tokens: Optional[int] = None
    hash: str = ""
    source: Optional[str] = None   # where this section came from


@dataclass
class PromptReport:
    """Per-turn audit of every section that landed in the system prompt.

    The killer feature. Surfaces what went where, what got dropped at
    compaction, what got extracted to memory. Persisted by the plugins
    so /nachos report can show history.
    """
    total_chars: int
    sections: List[PromptSectionReport]
    total_tokens: Optional[int] = None
    generated_at: str = field(default_factory=lambda: _utcnow())
    session_id: Optional[str] = None
    turn: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chars": self.total_chars,
            "total_tokens": self.total_tokens,
            "sections": [
                {
                    "name": s.name,
                    "size_chars": s.size_chars,
                    "size_tokens": s.size_tokens,
                    "hash": s.hash,
                    "source": s.source,
                }
                for s in self.sections
            ],
            "generated_at": self.generated_at,
            "session_id": self.session_id,
            "turn": self.turn,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def estimate_tokens(text: str, ratio: float = 4.0) -> int:
    """Conservative chars/token estimate. Heuristic; never trust precisely.

    Source comment: '4 chars/token is the GPT family rule of thumb. Real
    tokenizer count varies ±15%. We add no safety margin here — callers
    that care about budget enforcement should add their own.'
    """
    if not text:
        return 0
    return int(len(text) / ratio)
