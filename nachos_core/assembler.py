"""Prompt assembler — layered, ordered, observable.

This is the heart of Nachos. Instead of a single jumbled system prompt,
sections are assembled in declared order with each section reporting its
size, hash, and source. Output is the rendered prompt PLUS a PromptReport
the plugin persists for audit.

Deliberate design choices:

  • Sections are added explicitly in priority order. Caller controls.
  • Every section is hashable + token-countable. Auditable per turn.
  • Empty sections are dropped (don't pad the prompt with ': ').
  • The assembler does NOT enforce a budget. It reports. The compactor
    enforces. Separation of concerns.
  • Section ordering rationale (top → bottom):
      1. base prompt          (the agent's identity + role)
      2. memory manifest      (compact pointers — recall on demand)
      3. user profile         (durable user facts, structured)
      4. memory facts         (high-confidence triples)
      5. memory entries       (free-form notes)
      6. session state        (current task, optional)
      7. skills               (procedures, lower priority since indexed)
      8. instructions         (memory recall, delegation guidance)

Hashing: sha256 of section content. Used by /nachos report to spot
which sections actually changed turn-to-turn (most don't).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .types import (
    MemoryEntry,
    MemoryFact,
    PromptReport,
    PromptSectionReport,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------

@dataclass
class AssembleParams:
    """Everything the assembler can lay onto the plate.

    All fields optional — empty/None sections are silently dropped.
    """
    base_prompt: str = ""
    user_profile: Optional[str] = None
    memory_manifest: Optional[str] = None    # rendered manifest text
    memory_facts: List[MemoryFact] = field(default_factory=list)
    memory_entries: List[MemoryEntry] = field(default_factory=list)
    session_state: Optional[str] = None
    skills: Optional[str] = None
    include_memory_instructions: bool = False
    include_delegation_instructions: bool = False
    # Limits
    max_memory_entries: int = 50
    max_memory_facts: int = 50


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class PromptAssembler:
    """Layered system-prompt builder. Returns prompt + PromptReport."""

    def __init__(self, hash_algorithm: str = "sha256",
                 include_token_estimates: bool = True):
        self.hash_algorithm = hash_algorithm
        self.include_token_estimates = include_token_estimates

    def assemble(self, params: AssembleParams) -> Tuple[str, PromptReport]:
        """Build the prompt and matching report.

        Returns (rendered_prompt, PromptReport). Caller decides what to
        do with the report (persist, log, expose via /nachos report).
        """
        sections: List[Tuple[str, str, Optional[str]]] = []  # (name, content, source)

        if params.base_prompt:
            sections.append(("base", params.base_prompt.strip(), "agent.system_prompt"))

        if params.memory_manifest:
            sections.append(("memory_manifest", params.memory_manifest.strip(),
                             "nachos.manifest"))

        if params.user_profile:
            sections.append(("user_profile",
                             self._format_user_profile(params.user_profile),
                             "memory.user"))

        if params.memory_facts:
            content = self._format_facts(params.memory_facts[:params.max_memory_facts])
            if content:
                sections.append(("memory_facts", content, "nachos.extractor"))

        if params.memory_entries:
            content = self._format_entries(
                params.memory_entries[:params.max_memory_entries]
            )
            if content:
                sections.append(("memory", content, "memory.entries"))

        if params.session_state:
            sections.append(("session_state", params.session_state.strip(),
                             "session.state"))

        if params.skills:
            sections.append(("skills", params.skills.strip(), "skill-loader"))

        if params.include_memory_instructions:
            sections.append(("memory_instructions",
                             _MEMORY_INSTRUCTIONS, "nachos.assembler"))

        if params.include_delegation_instructions:
            sections.append(("delegation_instructions",
                             _DELEGATION_INSTRUCTIONS, "nachos.assembler"))

        prompt = "\n\n".join(content for _, content, _ in sections).strip()
        report = self._build_report(sections)
        return prompt, report

    # -- Formatters --------------------------------------------------------

    def _format_user_profile(self, profile: str) -> str:
        return f"# User Profile\n{profile.strip()}"

    def _format_facts(self, facts: List[MemoryFact]) -> str:
        if not facts:
            return ""
        lines = ["# Memory Facts"]
        for f in facts:
            tag = f" ({f.kind})" if f.kind and f.kind != "general" else ""
            lines.append(f"- {f.render()}{tag}")
        return "\n".join(lines)

    def _format_entries(self, entries: List[MemoryEntry]) -> str:
        if not entries:
            return ""
        lines = ["# Memory"]
        for e in entries:
            tags = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"- ({e.kind}) {e.content}{tags}")
        return "\n".join(lines)

    # -- Reporting ---------------------------------------------------------

    def _build_report(self, sections) -> PromptReport:
        section_reports = []
        for name, content, source in sections:
            size_chars = len(content)
            size_tokens = (estimate_tokens(content)
                           if self.include_token_estimates else None)
            section_reports.append(PromptSectionReport(
                name=name,
                size_chars=size_chars,
                size_tokens=size_tokens,
                hash=self._hash(content),
                source=source,
            ))
        total_chars = sum(s.size_chars for s in section_reports)
        total_tokens = (sum((s.size_tokens or 0) for s in section_reports)
                        if self.include_token_estimates else None)
        return PromptReport(
            total_chars=total_chars,
            total_tokens=total_tokens,
            sections=section_reports,
        )

    def _hash(self, content: str) -> str:
        h = hashlib.new(self.hash_algorithm)
        h.update(content.encode("utf-8"))
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Static instruction blocks (Nachos-flavored, harness-agnostic)
# ---------------------------------------------------------------------------

_MEMORY_INSTRUCTIONS = """## Memory Recall (Manifest-First)

The memory manifest section above lists what's known WITHOUT loading
full content. To pull full details, call `memory_recall` with a
specific pointer or query.

When to recall:
- User asks about past conversations or decisions
- User references something you should remember
- Checking for previous decisions before making new ones

When NOT to recall:
- Current conversation context (already visible)
- General knowledge questions (use training data)
- Real-time information (use web search)

Manifest pointers are stable: prefer them over free-text queries when
the manifest already lists what you need."""

_DELEGATION_INSTRUCTIONS = """## Task Delegation

For focused subtasks (research, file reading, scoped Q&A) prefer
delegated subagents — they run through the same context layer and
their results are observed at the parent layer.

For autonomous coding requiring full filesystem access, prefer the
harness's native code-agent tool (e.g. agent_exec) only when the task
needs unrestricted write access."""
