"""Per-source validation with the real reader on disposable synthetic profiles."""

import os
from types import SimpleNamespace

import pytest

from nachos_core.reconciliation import export_plan, new_session, review_proposal, seal_document
from nachos_core.reconciliation.acquire import SourceSpec, read_each_source
from nachos_core.reconciliation.boundary import SourceBoundary
from nachos_core.reconciliation.inventory import build_inventory
from nachos_core.reconciliation.packets import build_packet
from nachos_core.reconciliation.validate import validate_plan_sources

MEMORY = "nachos/memory.md"
SKILL_A = "skills/a/SKILL.md"
SKILL_B = "skills/b/SKILL.md"
SPEC = SourceSpec("flatfile", (SKILL_A, SKILL_B))
NOW = "2026-11-16T13:00:00+00:00"
FLAT = "# ops\n\n## Alpha\nalpha body\n\n## Beta\nbeta body\n"


def _write(root, label, text):
    path = root / label
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _boundary(root):
    return SourceBoundary(root, (MEMORY, SKILL_A, SKILL_B))


def _obs(sid):
    return {
        "kind": "manual",
        "status": "verified",
        "source_ids": [sid],
        "claim": "reviewed",
        "observation": "synthetic",
        "observed_at": NOW,
    }


def _excerpt(source):
    before = source["before"]
    return before["body"] if isinstance(before, dict) else "old"


def _plan(root):
    """Approve one proposal editing skill a, skill b, record alpha and record beta."""
    _write(root, MEMORY, FLAT)
    _write(root, SKILL_A, "# A\nold a\n")
    _write(root, SKILL_B, "# B\nold b\n")
    profile = {"id": "p", "root": str(root)}
    inventory = build_inventory(_boundary(root), profile, SPEC)
    sources = inventory["sources"]
    ids = [s["id"] for s in sources]
    finding = {
        "id": "f1",
        "type": "stale",
        "source_ids": ids,
        "excerpts": [{"source_id": s["id"], "text": _excerpt(s)} for s in sources],
        "rationale": "outdated",
        "evidence": [_obs(ids[0])],
        "limitations": [],
    }
    changes = [
        {
            "source_id": s["id"],
            "source_sha256": s["sha256"],
            "operation": "edit",
            "before": s["before"],
            "after": dict(s["before"], body="new " + s["before"]["body"])
            if isinstance(s["before"], dict)
            else s["before"] + "new\n",
        }
        for s in sources
    ]
    proposal = {
        "id": "p1",
        "revision": 1,
        "finding_ids": ["f1"],
        "depends_on": [],
        "changes": changes,
        "assessments": [
            {
                "finding_id": "f1",
                "revision": 1,
                "evidence": [_obs(ids[0])],
                "preservation_checklist": [{"item": "i", "preserved": True, "explanation": "e"}],
            }
        ],
    }
    packet = build_packet(inventory, findings=[finding], proposals=[proposal])
    return export_plan(review_proposal(new_session(packet), "p1", "accept"))


def _check(root, plan, spec=SPEC, **kwargs):
    return validate_plan_sources(
        plan,
        boundary=_boundary(root),
        profile=dict(plan["profile"]),
        spec=spec,
        checked_at=NOW,
        **kwargs,
    )


def _rows(result):
    return {row["identity"]: row for row in result["sources"]}


def test_real_reader_all_unchanged_is_valid(tmp_path):
    plan = _plan(tmp_path)
    result = _check(tmp_path, plan)
    assert result["status"] == "valid_at_check"
    assert {r["status"] for r in result["sources"]} == {"unchanged"}
    assert len(result["sources"]) == 4


def test_deleted_skill_changed_record_and_unchanged_are_reported_individually(tmp_path):
    plan = _plan(tmp_path)
    (tmp_path / SKILL_A).unlink()
    _write(tmp_path, MEMORY, FLAT.replace("alpha body", "alpha drifted"))
    result = _check(tmp_path, plan)
    rows = _rows(result)
    assert result["status"] == "stale"
    assert rows["skill:" + SKILL_A]["status"] == "missing"
    assert rows["skill:" + SKILL_A]["reason"] == "source_not_found"
    assert rows["memory:alpha"]["status"] == "changed"
    assert rows["memory:beta"]["status"] == "unchanged"
    assert rows["skill:" + SKILL_B]["status"] == "unchanged"


def test_one_unreadable_label_does_not_mask_drift_elsewhere(tmp_path):
    plan = _plan(tmp_path)
    _write(tmp_path, MEMORY, FLAT.replace("alpha body", "alpha drifted"))
    (tmp_path / SKILL_B).unlink()
    (tmp_path / "elsewhere.md").write_text("x")
    os.symlink(tmp_path / "elsewhere.md", tmp_path / SKILL_B)
    result = _check(tmp_path, plan)
    rows = _rows(result)
    assert result["status"] == "unavailable"
    assert rows["skill:" + SKILL_B]["status"] == "unavailable"
    assert rows["skill:" + SKILL_B]["reason"] == "source_unavailable"
    assert rows["memory:alpha"]["status"] == "changed"
    assert rows["memory:beta"]["status"] == "unchanged"
    assert rows["skill:" + SKILL_A]["status"] == "unchanged"
    assert "elsewhere" not in str(result)


def test_deleted_flat_store_marks_its_records_missing(tmp_path):
    plan = _plan(tmp_path)
    (tmp_path / MEMORY).unlink()
    rows = _rows(_check(tmp_path, plan))
    assert rows["memory:alpha"]["status"] == rows["memory:beta"]["status"] == "missing"
    assert rows["skill:" + SKILL_A]["status"] == "unchanged"


def test_malformed_store_is_unavailable_only_for_its_records(tmp_path):
    plan = _plan(tmp_path)
    _write(tmp_path, MEMORY, "## a b\nx\n## a-b\ny\n")
    result = _check(tmp_path, plan)
    rows = _rows(result)
    assert result["status"] == "unavailable"
    assert rows["memory:alpha"]["reason"] == "duplicate_key"
    assert rows["skill:" + SKILL_A]["status"] == "unchanged"


def test_source_outside_spec_is_unavailable_individually(tmp_path):
    plan = _plan(tmp_path)
    narrow = SourceSpec("flatfile", (SKILL_A,))
    rows = _rows(_check(tmp_path, plan, narrow))
    assert rows["skill:" + SKILL_B]["reason"] == "not_in_spec"
    assert rows["skill:" + SKILL_A]["status"] == "unchanged"
    assert rows["memory:beta"]["status"] == "unchanged"


def test_empty_plan_with_working_reader_is_valid(tmp_path):
    plan = _plan(tmp_path)
    plan = {**plan, "approved": [], "ledger": []}
    result = _check(tmp_path, seal_document(plan))
    assert (result["status"], result["sources"]) == ("valid_at_check", [])


def test_injected_tuple_reader_supports_per_label_failures(tmp_path):
    plan = _plan(tmp_path)
    cur, _ = read_each_source(_boundary(tmp_path), SPEC)

    def reader(_boundary, _spec, **_kwargs):
        return (
            {k: v for k, v in cur.items() if k != "skill:" + SKILL_A},
            {SKILL_A: "wal_present"},
        )

    rows = _rows(_check(tmp_path, plan, reader=reader))
    assert rows["skill:" + SKILL_A]["status"] == "unavailable"
    assert rows["skill:" + SKILL_A]["reason"] == "wal_present"
    assert rows["memory:beta"]["status"] == "unchanged"


def test_injected_failure_text_is_not_echoed(tmp_path):
    plan = _plan(tmp_path)

    def reader(_boundary, _spec, **_kwargs):
        return {}, {SKILL_A: "/secret/path SECRET"}

    result = _check(tmp_path, plan, reader=reader)
    assert _rows(result)["skill:" + SKILL_A]["reason"] == "acquisition_failed"
    assert "SECRET" not in str(result)


def test_read_each_source_reports_failures_by_label(tmp_path):
    _write(tmp_path, MEMORY, FLAT)
    _write(tmp_path, SKILL_A, "a")
    found, failures = read_each_source(_boundary(tmp_path), SPEC)
    assert failures == {SKILL_B: "source_not_found"}
    assert sorted(found) == ["memory:alpha", "memory:beta", "skill:" + SKILL_A]
    spec = SimpleNamespace(store="bogus", skills=(), references=())
    with pytest.raises(ValueError, match="invalid_store"):
        read_each_source(_boundary(tmp_path), spec)  # type: ignore[arg-type]
