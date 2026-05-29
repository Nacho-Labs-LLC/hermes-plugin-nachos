"""End-to-end: extractor → fact store → manifest.

Uses a stub LLM and a temp directory so it runs anywhere. Validates
the full v0.2 pipeline without touching the real Hermes runtime or
real memory files.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.hermes_extractor import JsonlFactStore
from adapters.hermes_memory import HermesMemoryReader
from nachos_core.extractor import ExtractionConfig, extract_facts
from nachos_core.manifest import build_manifest, render_manifest


def stub_llm_response(_system, _user, _max_tokens):
    return """[
      {"subject": "user", "predicate": "prefers", "object": "TypeScript",
       "kind": "preference", "confidence": 0.95},
      {"subject": "user", "predicate": "uses", "object": "macOS",
       "kind": "environment", "confidence": 0.9},
      {"subject": "project deposco", "predicate": "uses",
       "object": "Angular 21.2", "kind": "project", "confidence": 0.92}
    ]"""


def test_full_pipeline_extracts_dedups_and_surfaces(tmp_path):
    # 1. Run extraction over a fake conversation
    messages = [
        {"role": "user", "content": "I prefer TypeScript and use macOS."},
        {"role": "assistant", "content": "Noted."},
    ]
    result = extract_facts(messages, stub_llm_response, ExtractionConfig())
    assert result.parse_success
    assert result.kept == 3

    # 2. Persist via the JsonlFactStore
    store = JsonlFactStore(tmp_path / "facts.jsonl")
    inserted, updated = store.upsert(result.facts)
    assert inserted == 3
    assert updated == 0

    # 3. Re-running with same facts dedups (no inserts, all updates)
    inserted2, updated2 = store.upsert(result.facts)
    assert inserted2 == 0
    assert updated2 == 3

    # 4. Reader sees the extracted facts via fact_counts_by_kind
    fake_home = tmp_path / "fake_hermes"
    fake_home.mkdir()
    (fake_home / "memories").mkdir()
    reader = HermesMemoryReader(hermes_home=fake_home)
    reader.set_fact_store(store)
    counts = reader.fact_counts_by_kind()

    assert "extracted:preference" in counts
    assert counts["extracted:preference"] == 1
    assert counts["extracted:environment"] == 1
    assert counts["extracted:project"] == 1

    # 5. Manifest renders the extracted-fact counts
    manifest = build_manifest(reader)
    rendered = render_manifest(manifest)
    assert "extracted:preference" in rendered
    assert "extracted:project" in rendered

    # 6. Confidence bumped on second observation
    facts_after = store.list_all()
    pref_fact = next(f for f in facts_after if f.kind == "preference")
    assert pref_fact.confidence > 0.95


def test_pipeline_handles_zero_facts(tmp_path):
    """Empty extraction should leave the store empty and not error."""
    def empty_llm(_s, _u, _t):
        return "[]"

    store = JsonlFactStore(tmp_path / "facts.jsonl")
    result = extract_facts(
        [{"role": "user", "content": "hi"}],
        empty_llm,
    )
    assert result.parse_success
    assert result.kept == 0
    inserted, updated = store.upsert(result.facts)
    assert inserted == 0 and updated == 0
    assert store.list_all() == []


def test_pipeline_handles_llm_failure_gracefully(tmp_path):
    def angry_llm(*_args):
        raise RuntimeError("upstream timeout")

    store = JsonlFactStore(tmp_path / "facts.jsonl")
    result = extract_facts([{"role": "user", "content": "hi"}], angry_llm)
    assert not result.parse_success
    # Don't try to upsert empty facts: just verify nothing landed
    store.upsert(result.facts)
    assert store.list_all() == []


def test_fact_store_survives_corruption(tmp_path):
    store = JsonlFactStore(tmp_path / "facts.jsonl")
    # Pre-corrupt the file
    store.path.write_text(
        '{"valid":"but missing fields"}\n'
        'not even json\n'
        '{"subject":"a","predicate":"b","object":"c","confidence":0.9}\n',
        encoding="utf-8",
    )
    facts = store.list_all()
    # Only the well-formed line survives
    assert len(facts) == 1
    assert facts[0].subject == "a"
