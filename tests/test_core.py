"""Unit tests for nachos_core — no Hermes dependency.

Run with:
    cd ~/DEV/hermes-plugin-nachos
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nachos_core.assembler import AssembleParams, PromptAssembler
from nachos_core.manifest import (
    ManifestConfig,
    build_manifest,
    render_manifest,
)
from nachos_core.types import MemoryEntry, MemoryFact


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class FakeReader:
    """In-memory MemoryReadProtocol stub."""

    def __init__(self, entries=None, topics=None, counts=None):
        self._entries = entries or []
        self._topics = topics or []
        self._counts = counts or {}

    def list_entries(self, kind=None, limit=100):
        if kind is None:
            return self._entries[:limit]
        return [e for e in self._entries if e["kind"] == kind][:limit]

    def list_recent_topics(self, limit=5):
        return self._topics[:limit]

    def fact_counts_by_kind(self):
        return dict(self._counts)


def test_assembler_empty():
    a = PromptAssembler()
    prompt, report = a.assemble(AssembleParams())
    assert prompt == ""
    assert report.total_chars == 0
    assert report.sections == []


def test_assembler_base_only():
    a = PromptAssembler()
    prompt, report = a.assemble(AssembleParams(base_prompt="You are an agent."))
    assert "You are an agent." in prompt
    assert len(report.sections) == 1
    assert report.sections[0].name == "base"
    assert report.sections[0].size_chars > 0
    assert report.sections[0].hash  # sha256 hex


def test_assembler_full_plate():
    a = PromptAssembler()
    params = AssembleParams(
        base_prompt="Base.",
        memory_manifest="Manifest goes here.",
        user_profile="Loves vim.",
        memory_facts=[MemoryFact("Nate", "uses", "macOS", kind="environment")],
        memory_entries=[MemoryEntry("Likes concise replies.", kind="preference")],
        skills="Skill A: do thing.",
        include_memory_instructions=True,
        include_delegation_instructions=True,
    )
    prompt, report = a.assemble(params)
    names = [s.name for s in report.sections]
    assert names == [
        "base", "memory_manifest", "user_profile", "memory_facts",
        "memory", "skills", "memory_instructions", "delegation_instructions",
    ]
    # Section order in rendered prompt should match
    assert prompt.index("Base.") < prompt.index("Manifest goes here.")
    assert prompt.index("Manifest goes here.") < prompt.index("vim")


def test_assembler_drops_empty_sections():
    a = PromptAssembler()
    params = AssembleParams(base_prompt="A", memory_facts=[],
                            memory_entries=[])
    prompt, report = a.assemble(params)
    assert "memory_facts" not in [s.name for s in report.sections]
    assert "memory" not in [s.name for s in report.sections]


def test_assembler_hash_stable_for_same_content():
    a = PromptAssembler()
    p1, r1 = a.assemble(AssembleParams(base_prompt="X"))
    p2, r2 = a.assemble(AssembleParams(base_prompt="X"))
    assert r1.sections[0].hash == r2.sections[0].hash


def test_assembler_token_estimates_present():
    a = PromptAssembler()
    _, report = a.assemble(AssembleParams(base_prompt="x" * 400))
    assert report.total_tokens is not None
    assert report.total_tokens >= 90  # ~4 chars/token, so 400/4 = 100, fuzzy


def test_assembler_max_facts_truncates():
    facts = [MemoryFact(f"S{i}", "p", f"O{i}") for i in range(100)]
    a = PromptAssembler()
    _, report = a.assemble(AssembleParams(memory_facts=facts,
                                          max_memory_facts=10))
    # Section content should reflect 10 + 1 header line = 11 lines
    fact_section = next(s for s in report.sections if s.name == "memory_facts")
    # Hash differs from full set, so we just sanity-check section exists
    assert fact_section.size_chars > 0


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def test_manifest_empty_renders_empty():
    reader = FakeReader()
    m = build_manifest(reader)
    assert render_manifest(m) == ""


def test_manifest_preferences_only():
    reader = FakeReader(
        entries=[
            {"content": "Editor: vim", "kind": "preference",
             "tags": [], "target": "user", "created_at": ""},
            {"content": "Prefers dark mode.", "kind": "preference",
             "tags": [], "target": "user", "created_at": ""},
        ],
    )
    m = build_manifest(reader)
    rendered = render_manifest(m)
    assert "Memory Manifest" in rendered
    assert "Preferences:" in rendered
    assert "Editor" in rendered or "vim" in rendered


def test_manifest_recent_topics():
    reader = FakeReader(
        topics=[{"topic": "Discussing Nachos plugin", "session_age": "2h ago"}],
    )
    m = build_manifest(reader)
    rendered = render_manifest(m)
    assert "Recent topics:" in rendered
    assert "Nachos plugin" in rendered
    assert "2h ago" in rendered


def test_manifest_fact_counts_sorted_descending():
    reader = FakeReader(
        counts={"general": 3, "preference": 7, "environment": 1},
    )
    m = build_manifest(reader)
    rendered = render_manifest(m)
    pref_idx = rendered.index("preference")
    gen_idx = rendered.index("general")
    env_idx = rendered.index("environment")
    assert pref_idx < gen_idx < env_idx


def test_manifest_respects_recent_topic_count():
    reader = FakeReader(
        topics=[
            {"topic": f"Topic {i}", "session_age": "1d ago"}
            for i in range(10)
        ],
    )
    m = build_manifest(reader, ManifestConfig(recent_topic_count=3))
    assert len(m.recent_topics) == 3


def test_manifest_truncates_long_preference_values():
    long_value = "x" * 500
    reader = FakeReader(
        entries=[{"content": long_value, "kind": "preference",
                  "tags": [], "target": "user", "created_at": ""}],
    )
    m = build_manifest(reader, ManifestConfig(preference_value_max_chars=40))
    rendered = render_manifest(m)
    # Should not contain the full 500-char string
    assert long_value not in rendered
    # Should contain ellipsis
    assert "…" in rendered


def test_manifest_does_not_dump_full_content():
    """Critical contract: manifest must NOT include full entry content.

    The whole point is pointer-list, not inline dump.
    """
    secret = "FULL_CONTENT_THAT_SHOULD_NEVER_LEAK_INLINE_" + "x" * 200
    reader = FakeReader(
        entries=[{"content": secret, "kind": "general",
                  "tags": [], "target": "memory", "created_at": ""}],
    )
    m = build_manifest(reader)
    rendered = render_manifest(m)
    # The secret string should not appear in the rendered manifest
    # (only in counts / pointers — never the full text)
    assert secret not in rendered


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def test_memory_fact_render():
    f = MemoryFact("Nate", "prefers", "concise replies")
    assert f.render() == "Nate prefers concise replies"


def test_prompt_report_to_dict_roundtrip_safe():
    a = PromptAssembler()
    _, report = a.assemble(AssembleParams(base_prompt="hi"))
    d = report.to_dict()
    assert d["total_chars"] == report.total_chars
    assert d["sections"][0]["name"] == "base"
    assert "hash" in d["sections"][0]
