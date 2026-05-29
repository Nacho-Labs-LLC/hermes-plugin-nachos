"""Tests for nachos_core.extractor and nachos_core.dedup.

No Hermes / no real LLM. The LLMCall is stubbed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from nachos_core.extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractionConfig,
    build_user_message,
    extract_facts,
)
from nachos_core.dedup import (
    deduplicate_facts,
    is_exact_match,
    merge_fact,
)
from nachos_core.types import MemoryFact


# ---------------------------------------------------------------------------
# build_user_message
# ---------------------------------------------------------------------------

def test_build_user_message_empty():
    msg = build_user_message([])
    assert "empty" in msg.lower()


def test_build_user_message_filters_non_user_assistant():
    transcript = build_user_message([
        {"role": "system", "content": "ignore me"},
        {"role": "tool", "content": "ignore me too"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ])
    assert "system" not in transcript.lower() or "User: Hello" in transcript
    assert "User: Hello" in transcript
    assert "Assistant: Hi" in transcript


def test_build_user_message_truncates_from_beginning():
    msgs = [{"role": "user", "content": f"Long old message {i} " * 200}
            for i in range(50)]
    msgs.append({"role": "user", "content": "RECENT"})
    out = build_user_message(msgs, max_chars=2000)
    assert "RECENT" in out
    assert "earlier turns truncated" in out


def test_build_user_message_handles_content_blocks():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Hello block"}]}
    ]
    out = build_user_message(msgs)
    assert "Hello block" in out


# ---------------------------------------------------------------------------
# extract_facts — full pipeline with stub LLM
# ---------------------------------------------------------------------------

class StubLLM:
    """Deterministic LLM stub for testing."""

    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.last_system = None
        self.last_user = None
        self.last_max_tokens = None

    def __call__(self, system_prompt, user_message, max_tokens):
        self.calls += 1
        self.last_system = system_prompt
        self.last_user = user_message
        self.last_max_tokens = max_tokens
        return self.response


def test_extract_facts_happy_path():
    stub = StubLLM('''[
      {"subject": "user", "predicate": "prefers", "object": "TypeScript",
       "kind": "preference", "confidence": 0.95},
      {"subject": "project deposco", "predicate": "uses",
       "object": "Angular 21.2", "kind": "project", "confidence": 0.9}
    ]''')
    msgs = [{"role": "user", "content": "I prefer TS"}]
    result = extract_facts(msgs, stub)
    assert result.parse_success
    assert result.kept == 2
    assert result.raw_count == 2
    assert result.facts[0].subject == "user"
    assert result.facts[0].kind == "preference"
    assert result.facts[1].kind == "project"
    # System prompt should be the canonical one
    assert stub.last_system == EXTRACTION_SYSTEM_PROMPT


def test_extract_facts_strips_markdown_fences():
    stub = StubLLM('```json\n[{"subject":"x","predicate":"y","object":"z",'
                   '"confidence":0.9}]\n```')
    result = extract_facts([{"role": "user", "content": "hi"}], stub)
    assert result.parse_success
    assert result.kept == 1


def test_extract_facts_empty_array():
    stub = StubLLM("[]")
    result = extract_facts([{"role": "user", "content": "hi"}], stub)
    assert result.parse_success
    assert result.kept == 0
    assert result.raw_count == 0


def test_extract_facts_below_min_confidence_dropped():
    stub = StubLLM('[{"subject":"a","predicate":"b","object":"c",'
                   '"confidence":0.3}]')
    cfg = ExtractionConfig(min_confidence=0.6)
    result = extract_facts([{"role": "user", "content": "hi"}], stub, cfg)
    assert result.parse_success
    assert result.raw_count == 1
    assert result.kept == 0


def test_extract_facts_invalid_kind_falls_back():
    stub = StubLLM('[{"subject":"a","predicate":"b","object":"c",'
                   '"kind":"weird-thing","confidence":0.9}]')
    result = extract_facts([{"role": "user", "content": "hi"}], stub)
    assert result.kept == 1
    assert result.facts[0].kind == "general"


def test_extract_facts_drops_secret_shapes():
    stub = StubLLM('''[
      {"subject":"user","predicate":"has","object":"sk-abc123def456ghi789jklmnopqrs",
       "kind":"general","confidence":0.95},
      {"subject":"user","predicate":"prefers","object":"vim",
       "kind":"preference","confidence":0.9}
    ]''')
    result = extract_facts([{"role": "user", "content": "hi"}], stub)
    # Secret-shaped object dropped, vim kept
    assert result.kept == 1
    assert result.facts[0].object == "vim"


def test_extract_facts_handles_llm_exception():
    def angry_llm(*_args, **_kwargs):
        raise RuntimeError("provider on fire")

    result = extract_facts([{"role": "user", "content": "hi"}], angry_llm)
    assert not result.parse_success
    assert "provider on fire" in (result.error or "")
    assert result.facts == []


def test_extract_facts_handles_invalid_json():
    stub = StubLLM("not json at all { [")
    result = extract_facts([{"role": "user", "content": "hi"}], stub)
    assert not result.parse_success
    assert "JSON" in (result.error or "")


def test_extract_facts_missing_required_field_dropped():
    stub = StubLLM('[{"subject":"a","object":"c","confidence":0.9}]')
    # Missing predicate
    result = extract_facts([{"role": "user", "content": "hi"}], stub)
    assert result.parse_success
    assert result.raw_count == 1
    assert result.kept == 0


def test_extract_facts_accepts_legacy_type_field():
    stub = StubLLM('[{"subject":"a","predicate":"b","object":"c",'
                   '"type":"preference","confidence":0.9}]')
    result = extract_facts([{"role": "user", "content": "hi"}], stub)
    assert result.kept == 1
    assert result.facts[0].kind == "preference"


def test_extract_facts_clamps_confidence():
    stub = StubLLM('[{"subject":"a","predicate":"b","object":"c",'
                   '"confidence":2.5}]')
    result = extract_facts([{"role": "user", "content": "hi"}], stub)
    assert result.kept == 1
    assert result.facts[0].confidence == 1.0


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

def test_dedup_exact_match_normalization():
    a = MemoryFact("User", "Prefers", "TypeScript")
    b = MemoryFact("user", "prefers ", " typescript")
    assert is_exact_match(a, b)


def test_dedup_different_objects_not_match():
    a = MemoryFact("user", "prefers", "TypeScript")
    b = MemoryFact("user", "prefers", "JavaScript")
    assert not is_exact_match(a, b)


def test_dedup_inserts_new_facts():
    incoming = [MemoryFact("u", "p", "x")]
    existing = []
    r = deduplicate_facts(incoming, existing)
    assert len(r.to_insert) == 1
    assert len(r.to_update) == 0


def test_dedup_merges_duplicate_facts():
    incoming = [MemoryFact("u", "p", "x", confidence=0.7)]
    existing = [MemoryFact("u", "p", "x", confidence=0.6)]
    r = deduplicate_facts(incoming, existing)
    assert len(r.to_insert) == 0
    assert len(r.to_update) == 1
    old, merged = r.to_update[0]
    # +0.05 nudge, capped
    assert merged.confidence > 0.65
    assert merged.confidence <= 1.0


def test_merge_fact_caps_confidence_at_one():
    a = MemoryFact("u", "p", "x", confidence=0.99)
    b = MemoryFact("u", "p", "x", confidence=0.99)
    m = merge_fact(a, b)
    assert m.confidence == 1.0


def test_merge_fact_prefers_specific_kind():
    existing = MemoryFact("u", "p", "x", kind="general")
    incoming = MemoryFact("u", "p", "x", kind="preference")
    m = merge_fact(existing, incoming)
    assert m.kind == "preference"


def test_merge_fact_keeps_existing_kind_when_incoming_is_general():
    existing = MemoryFact("u", "p", "x", kind="preference")
    incoming = MemoryFact("u", "p", "x", kind="general")
    m = merge_fact(existing, incoming)
    assert m.kind == "preference"


def test_dedup_handles_mixed_batch():
    existing = [
        MemoryFact("u", "p", "x", confidence=0.7),
        MemoryFact("u", "p", "y", confidence=0.7),
    ]
    incoming = [
        MemoryFact("u", "p", "x", confidence=0.8),    # match → update
        MemoryFact("u", "p", "z", confidence=0.9),    # new → insert
        MemoryFact("u", "q", "x", confidence=0.85),   # new (different pred)
    ]
    r = deduplicate_facts(incoming, existing)
    assert len(r.to_insert) == 2
    assert len(r.to_update) == 1
    inserted_objs = {f.object for f in r.to_insert}
    assert inserted_objs == {"z", "x"}  # 'z' object new, 'x' with q-pred new
