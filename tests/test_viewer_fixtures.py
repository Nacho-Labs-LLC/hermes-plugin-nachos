"""Viewer fixtures regenerate byte-for-byte and validate (or are rejected) through Python."""

import importlib.util
import json
from pathlib import Path

import pytest

from nachos_core.reconciliation import loads

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "viewer"
_SPEC = importlib.util.spec_from_file_location(
    "build_viewer_fixtures", FIXTURES / "build_viewer_fixtures.py"
)
builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(builder)

KINDS = {
    "packet-mixed.json": "packet",
    "packet-empty.json": "packet",
    "packet-large.json": "packet",
    "session-mixed-initial.json": "session",
    "session-mixed-reviewed.json": "session",
    "plan-mixed.json": "plan",
}


@pytest.fixture(scope="module")
def generated():
    return builder.build_all()


def test_checked_in_fixtures_match_regeneration(generated):
    assert builder.main(["--check"]) == 0
    on_disk = {
        str(p.relative_to(FIXTURES))
        for p in FIXTURES.rglob("*")
        if p.is_file() and p.suffix in (".json", ".bin")
    }
    assert set(generated) == on_disk


def test_regeneration_is_deterministic(generated):
    assert builder.build_all() == generated


@pytest.mark.parametrize(("name", "kind"), sorted(KINDS.items()))
def test_valid_fixtures_load(name, kind):
    doc = loads((FIXTURES / name).read_bytes(), expected_profile=builder.PROFILE)
    assert doc["schema"] == f"nachos.reconciliation.{kind}.v1"


def test_mixed_packet_covers_the_spec():
    packet = loads((FIXTURES / "packet-mixed.json").read_bytes())
    sources = packet["inventory"]["sources"]
    assert {s["kind"] for s in sources} == {
        "memory_record",
        "memory_file",
        "skill_file",
        "reference_file",
    }
    assert {f["type"] for f in packet["findings"]} == {
        "stale",
        "duplicate",
        "conflict",
        "unsupported",
        "clarification",
    }
    evidence = [e for f in packet["findings"] for e in f["evidence"]]
    assert {e["kind"] for e in evidence} >= {"schedule", "execution", "delivery", "usage"}
    assert {e["status"] for e in evidence} == {"verified", "unverified", "missing", "conflicting"}
    meta = [s["metadata"] for s in sources]
    assert {m["enabled"] for m in meta} == {True, False, None}
    assert {m["usage"]["status"] for m in meta} == {"available", "unavailable", "unknown"}
    assert len({tuple(k for k, v in s["coverage"].items() if v is True) for s in sources}) >= 3
    assert any("\r\n" in s["before"] for s in sources if isinstance(s["before"], str))
    assert any(len(p["changes"]) > 1 for p in packet["proposals"])
    assert any(p["depends_on"] for p in packet["proposals"])
    assert any(c["operation"] == "disable" for p in packet["proposals"] for c in p["changes"])
    assert any(
        not c["preserved"]
        for p in packet["proposals"]
        for a in p["assessments"]
        for c in a["preservation_checklist"]
    )
    text = json.dumps(packet, ensure_ascii=False)
    assert "onerror" in text
    assert "\U0001f600" in text
    assert "é" in text


def test_mixed_session_and_plan_lifecycle():
    session = loads((FIXTURES / "session-mixed-reviewed.json").read_bytes())
    plan = loads((FIXTURES / "plan-mixed.json").read_bytes())
    assert plan["session_digest"] == session["digest"]
    assert [a["proposal"]["id"] for a in plan["approved"]] == [
        "p-dep-base",
        "p-memory",
        "p-multi",
        "p-dep-child",
    ]
    assert {e["status"] for e in plan["ledger"]} >= {"blocked", "rejected", "deferred"}
    ids = {r["proposal_id"]: r for r in session["reviews"]}
    assert ids["p-memory"]["current_revision"] == 2  # edit invalidated then reaccepted
    assert "p-false-preservation" in ids  # accepting it is blocked by the contract
    assert ids["p-false-preservation"]["decision"] == "unreviewed"


def test_empty_and_large_shapes():
    empty = loads((FIXTURES / "packet-empty.json").read_bytes())
    assert empty["proposals"] == []
    assert empty["inventory"]["sources"] == []
    large = loads((FIXTURES / "packet-large.json").read_bytes())
    assert len(large["proposals"]) >= 300
    lines = max(
        len(s["before"].splitlines())
        for s in large["inventory"]["sources"]
        if isinstance(s["before"], str)
    )
    assert lines > 2000


MANIFEST = json.loads((FIXTURES / "errors" / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", MANIFEST["errors"], ids=lambda e: e["file"])
def test_error_corpus_is_rejected(entry):
    raw = (FIXTURES / "errors" / entry["file"]).read_bytes()
    assert entry["expected"] == "reject"
    if entry.get("valid_standalone"):
        doc = loads(raw)  # a valid artifact, wrong for the context named in the manifest
        if "expected_profile" in entry:
            with pytest.raises(ValueError, match="profile"):
                loads(raw, expected_profile=entry["expected_profile"])
        else:
            kind = entry["expected_kind"]
            assert doc["schema"] != f"nachos.reconciliation.{kind}.v1"
    else:
        with pytest.raises(ValueError, match=r".+"):
            loads(raw)


def test_validation_reports_are_unsealed_and_deterministic():
    statuses = {}
    for name in ("valid", "stale", "unavailable"):
        report = json.loads((FIXTURES / f"validation-{name}.json").read_text(encoding="utf-8"))
        assert report["schema"] == "nachos.reconciliation.validation.v1"
        assert "digest" not in report
        assert report["checked_at"] == builder.CHECKED_AT
        statuses[name] = report["status"]
    plan = loads((FIXTURES / "plan-mixed.json").read_bytes())
    assert statuses == {"valid": "valid_at_check", "stale": "stale", "unavailable": "unavailable"}
    report = json.loads((FIXTURES / "validation-valid.json").read_text(encoding="utf-8"))
    assert report["plan_digest"] == plan["digest"]
