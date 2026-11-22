"""Explicit point-in-time drift validation: synthetic data and a fake trusted reader."""

import copy
from datetime import datetime
from types import SimpleNamespace

import pytest

from nachos_core.reconciliation import (
    SCHEMAS,
    export_plan,
    file_sha256,
    new_session,
    review_proposal,
    seal_document,
)
from nachos_core.reconciliation import validate as module
from nachos_core.reconciliation.boundary import AcquisitionError, SourceBoundary
from nachos_core.reconciliation.validate import dumps_validation, validate_plan_sources

SPEC = SimpleNamespace(
    store="sqlite", skills=("skills/a/SKILL.md", "skills/b/SKILL.md"), references=()
)
NOW = "2026-11-16T13:00:00+00:00"


def make_plan(root):
    coverage = dict.fromkeys(
        ("discovered", "hashed", "content_searched", "section_reviewed", "body_reviewed"), False
    )
    coverage.update(discovered=True, hashed=True, limitations=["synthetic"])
    profile = {"id": "test", "root": str(root)}
    text = "old decision\n"
    source = {
        "id": "s1",
        "kind": "skill_file",
        "role": "primary",
        "metadata": {
            "byte_size": len(text),
            "char_size": len(text),
            "enabled": None,
            "provenance": ["synthetic"],
            "usage": {"status": "unknown", "references": []},
        },
        "identity": "skill:a",
        "locator": "skills/a/SKILL.md",
        "sha256": file_sha256(text.encode()),
        "before": text,
        "coverage": coverage,
    }
    inventory = seal_document(
        {
            "schema": SCHEMAS["inventory"],
            "profile": profile,
            "sources": [source],
            "coverage": coverage,
        }
    )
    evidence = {
        "kind": "manual",
        "status": "verified",
        "source_ids": ["s1"],
        "claim": "scope retained",
        "observation": "manual inspection",
        "observed_at": "2026-11-16T12:00:00Z",
    }
    finding = {
        "id": "f1",
        "type": "clarification",
        "source_ids": ["s1"],
        "excerpts": [{"source_id": "s1", "text": "old decision"}],
        "rationale": "Keep scoped permissions",
        "evidence": [evidence],
        "limitations": [],
    }
    proposal = {
        "id": "p1",
        "revision": 1,
        "finding_ids": ["f1"],
        "depends_on": [],
        "assessments": [
            {
                "finding_id": "f1",
                "revision": 1,
                "evidence": [evidence],
                "preservation_checklist": [
                    {"item": "decisions", "preserved": True, "explanation": "kept"}
                ],
            }
        ],
        "changes": [
            {
                "source_id": "s1",
                "source_sha256": source["sha256"],
                "operation": "edit",
                "before": text,
                "after": text + "clarification\n",
            }
        ],
    }
    packet = seal_document(
        {
            "schema": SCHEMAS["packet"],
            "profile": profile,
            "inventory": inventory,
            "findings": [finding],
            "proposals": [proposal],
        }
    )
    return export_plan(review_proposal(new_session(packet), "p1", "accept"))


class FakeReader:
    def __init__(self, sources=None, error=None):
        self.sources = sources
        self.error = error
        self.calls = []

    def __call__(self, boundary, spec):
        self.calls.append((boundary, spec))
        if self.error:
            raise AcquisitionError(self.error)
        return self.sources


def current(plan, **override):
    src = plan["approved"][0]["sources"][0]
    values = {
        "identity": src["identity"],
        "kind": src["kind"],
        "locator": src["locator"],
        "sha256": src["sha256"],
    }
    values.update(override)
    return {src["identity"]: SimpleNamespace(**values)}


def run(plan, root, reader, **kwargs):
    boundary = SourceBoundary(root, ())
    return validate_plan_sources(
        plan,
        boundary=boundary,
        profile=dict(plan["profile"]),
        spec=SPEC,
        reader=reader,
        checked_at=kwargs.pop("checked_at", NOW),
        **kwargs,
    )


def test_unchanged_is_valid_at_check_and_bound_to_plan(tmp_path):
    plan = make_plan(tmp_path)
    original = copy.deepcopy(plan)
    reader = FakeReader(current(plan))
    result = run(plan, tmp_path, reader)
    assert result["status"] == "valid_at_check"
    assert result["schema"] == "nachos.reconciliation.validation.v1"
    assert result["plan_digest"] == plan["digest"]
    assert result["sources"][0]["status"] == "unchanged"
    assert result["sources"][0]["proposal_ids"] == ["p1"]
    assert any("point-in-time" in item for item in result["limitations"])
    assert "perpetual" not in str(result["status"])
    assert plan == original
    assert plan["source_validation"] == "not_performed"
    assert "source_validation" not in result
    assert len(reader.calls) == 1
    assert reader.calls[0][1] is SPEC


@pytest.mark.parametrize(
    "override", [{"sha256": "0" * 64}, {"kind": "reference_file"}, {"locator": "skills/b/SKILL.md"}]
)
def test_changed_source_is_stale_with_detail(tmp_path, override):
    plan = make_plan(tmp_path)
    result = run(plan, tmp_path, FakeReader(current(plan, **override)))
    row = result["sources"][0]
    assert result["status"] == "stale"
    assert row["status"] == "changed"
    assert row["expected_sha256"] == plan["approved"][0]["sources"][0]["sha256"]
    assert row["actual_sha256"] is not None


def test_missing_record_is_stale(tmp_path):
    plan = make_plan(tmp_path)
    result = run(plan, tmp_path, FakeReader({}))
    assert result["status"] == "stale"
    assert result["sources"][0]["status"] == "missing"
    assert result["sources"][0]["actual_sha256"] is None


@pytest.mark.parametrize("code", ["source_unavailable", "invalid_sqlite"])
def test_reader_failure_is_unavailable_not_stale(tmp_path, code):
    plan = make_plan(tmp_path)
    result = run(plan, tmp_path, FakeReader(error=code))
    assert result["status"] == "unavailable"
    assert result["sources"][0]["status"] == "unavailable"
    assert result["sources"][0]["reason"] == code


def test_unstable_error_text_is_not_echoed(tmp_path):
    plan = make_plan(tmp_path)
    result = run(plan, tmp_path, FakeReader(error="/secret/path SECRET body"))
    assert result["sources"][0]["reason"] == "acquisition_failed"
    assert "SECRET" not in dumps_validation(result).decode()


def test_profile_and_root_must_match_explicit_authority(tmp_path):
    plan = make_plan(tmp_path)
    reader = FakeReader(current(plan))
    boundary = SourceBoundary(tmp_path, ())
    wrong = {"id": "other", "root": str(tmp_path)}
    with pytest.raises(ValueError, match="profile"):
        validate_plan_sources(plan, boundary=boundary, profile=wrong, spec=SPEC, reader=reader)
    other_root = tmp_path / "elsewhere"
    with pytest.raises(ValueError, match="root"):
        validate_plan_sources(
            plan,
            boundary=SourceBoundary(other_root, ()),
            profile=dict(plan["profile"]),
            spec=SPEC,
            reader=reader,
        )
    assert reader.calls == []


def test_resealed_traversal_locator_is_rejected_and_never_read(tmp_path):
    plan = make_plan(tmp_path)
    reader = FakeReader(current(plan))
    tampered = copy.deepcopy(plan)
    tampered["approved"][0]["sources"][0]["locator"] = "../../etc/passwd"
    with pytest.raises(ValueError, match=r".+"):
        run(seal_document(tampered), tmp_path, reader)
    with pytest.raises(ValueError, match="digest"):
        run(tampered, tmp_path, reader)
    assert reader.calls == []


def test_reader_only_receives_caller_boundary_and_spec(tmp_path):
    plan = make_plan(tmp_path)
    reader = FakeReader(current(plan))
    boundary = SourceBoundary(tmp_path, ("skills/a/SKILL.md",))
    validate_plan_sources(
        plan,
        boundary=boundary,
        profile=dict(plan["profile"]),
        spec=SPEC,
        reader=reader,
    )
    assert reader.calls == [(boundary, SPEC)]


def test_result_is_bound_to_each_plan_digest(tmp_path):
    first = make_plan(tmp_path)
    second = copy.deepcopy(first)
    second["approved"][0]["proposal"]["changes"][0]["after"] += "more"
    with pytest.raises(ValueError, match=r".+"):
        run(second, tmp_path, FakeReader(current(first)))
    assert run(first, tmp_path, FakeReader(current(first)))["plan_digest"] == first["digest"]
    empty = seal_document({**first, "approved": [], "ledger": []})
    result = run(empty, tmp_path, FakeReader({}))
    assert result["plan_digest"] == empty["digest"] != first["digest"]
    assert result["sources"] == []
    assert result["status"] == "valid_at_check"
    failed = run(empty, tmp_path, FakeReader(error="source_unavailable"))
    assert failed["status"] == "unavailable"


def test_non_plan_documents_are_rejected(tmp_path):
    plan = make_plan(tmp_path)
    reader = FakeReader(current(plan))
    inventory = plan["approved"][0]
    packet_like = seal_document(
        {
            "schema": SCHEMAS["inventory"],
            "profile": plan["profile"],
            "sources": inventory["sources"],
            "coverage": inventory["inventory_coverage"],
        }
    )
    with pytest.raises(ValueError, match="plan"):
        run(packet_like, tmp_path, reader)
    garbage = {"schema": "x", "profile": plan["profile"]}
    with pytest.raises(ValueError, match=r".+"):
        run(garbage, tmp_path, reader)
    assert reader.calls == []


def test_checked_at_defaults_to_aware_utc_and_must_be_aware(tmp_path):
    plan = make_plan(tmp_path)
    boundary = SourceBoundary(tmp_path, ())
    result = validate_plan_sources(
        plan,
        boundary=boundary,
        profile=dict(plan["profile"]),
        spec=SPEC,
        reader=FakeReader(current(plan)),
    )
    assert datetime.fromisoformat(result["checked_at"]).utcoffset().total_seconds() == 0
    with pytest.raises(ValueError, match="timezone"):
        run(plan, tmp_path, FakeReader(current(plan)), checked_at="2026-11-16T13:00:00")


def test_validation_module_has_no_apply_surface():
    assert not [name for name in dir(module) if "apply" in name.lower() or "write" in name.lower()]
    assert dumps_validation({"a": 1}) == b'{"a":1}'


def test_source_outside_callers_spec_is_unavailable_not_stale(tmp_path):
    plan = make_plan(tmp_path)
    reader = FakeReader(current(plan))
    narrow = SimpleNamespace(store="sqlite", skills=(), references=())
    result = validate_plan_sources(
        plan,
        boundary=SourceBoundary(tmp_path, ()),
        profile=dict(plan["profile"]),
        spec=narrow,
        reader=reader,
        checked_at=NOW,
    )
    assert result["status"] == "unavailable"
    assert result["sources"][0]["reason"] == "not_in_spec"
