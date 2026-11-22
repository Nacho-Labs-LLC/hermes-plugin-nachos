"""Offline reconciliation trust boundary and lifecycle regression contracts."""

import copy
import json

import pytest

from nachos_core.reconciliation import (
    SCHEMAS,
    canonical_json,
    dumps,
    export_plan,
    file_sha256,
    loads,
    memory_sha256,
    new_session,
    proposal_digest,
    review_proposal,
    seal_document,
)


@pytest.mark.parametrize("target", ["locator", "root"])
def test_nul_labels_are_rejected(target):
    doc = packet()["inventory"]
    if target == "locator":
        doc["sources"][0]["locator"] += "\x00bad"
    else:
        doc["profile"]["root"] += "\x00bad"
    with pytest.raises(ValueError, match="NUL"):
        seal_document(doc)


def test_canonical_hashes_preserve_text_and_reject_non_json():
    assert canonical_json({"z": "é\r\n", "a": 1}) == '{"a":1,"z":"é\\r\\n"}'.encode()
    assert file_sha256(b"a") != file_sha256(b"a\n")
    record = dict(key="k", title="t", summary="s", category="c", body="b")
    assert len(memory_sha256(record)) == 64
    for key in record:
        assert memory_sha256(record) != memory_sha256({**record, key: "changed"})
    for value in (1.0, float("nan"), 2**53, {1: "x"}, ("a",), "\ud800"):
        with pytest.raises(ValueError, match=r".+"):
            canonical_json(value)


def assessment(finding_id, revision, preserved=True):
    return {
        "finding_id": finding_id,
        "revision": revision,
        "evidence": [
            {
                "kind": "manual",
                "status": "verified",
                "source_ids": ["s1"],
                "claim": "decisions retained",
                "observation": "manual comparison of this exact revision",
                "observed_at": "2026-11-16T12:00:00Z",
            }
        ],
        "preservation_checklist": [
            {"item": "decisions", "preserved": preserved, "explanation": "kept verbatim"}
        ],
    }


def fresh(item, revision, preserved=True):
    """Return a copy bound to a new revision with a fresh assessment."""
    result = copy.deepcopy(item)
    result["revision"] = revision
    result["assessments"] = [assessment(fid, revision, preserved) for fid in item["finding_ids"]]
    return result


def packet():
    coverage = {
        "discovered": True,
        "hashed": True,
        "content_searched": False,
        "section_reviewed": False,
        "body_reviewed": False,
        "limitations": ["offline"],
    }
    profile = {"id": "test", "root": "/inert/profile-label"}
    source = {
        "id": "s1",
        "kind": "skill_file",
        "role": "primary",
        "metadata": {
            "byte_size": len(b"old decision\n"),
            "char_size": len("old decision\n"),
            "enabled": None,
            "provenance": ["synthetic"],
            "usage": {"status": "unknown", "references": []},
        },
        "identity": "skill:a",
        "locator": "skills/a/SKILL.md",
        "sha256": file_sha256(b"old decision\n"),
        "before": "old decision\n",
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
    finding = {
        "id": "f1",
        "type": "clarification",
        "source_ids": ["s1"],
        "excerpts": [{"source_id": "s1", "text": "old decision"}],
        "rationale": "Keep scoped permissions",
        "evidence": [
            {
                "kind": "manual",
                "status": "verified",
                "source_ids": ["s1"],
                "claim": "scope retained",
                "observation": "manual inspection",
                "observed_at": "2026-11-16T12:00:00Z",
            }
        ],
        "limitations": ["No live validation"],
    }
    proposal = {
        "id": "p1",
        "revision": 1,
        "finding_ids": ["f1"],
        "depends_on": [],
        "assessments": [assessment("f1", 1)],
        "changes": [
            {
                "source_id": "s1",
                "source_sha256": source["sha256"],
                "operation": "edit",
                "before": source["before"],
                "after": "old decision\nclarification\n",
            }
        ],
    }
    return seal_document(
        {
            "schema": SCHEMAS["packet"],
            "profile": profile,
            "inventory": inventory,
            "findings": [finding],
            "proposals": [proposal],
        }
    )


def test_inventory_supports_typed_metadata_and_review_stages():
    doc = packet()["inventory"]
    src = doc["sources"][0]
    src.update(
        kind="reference_file",
        role="reference",
        metadata={
            "byte_size": len(src["before"].encode()),
            "char_size": len(src["before"]),
            "enabled": None,
            "provenance": ["synthetic fixture"],
            "usage": {"status": "unavailable", "references": []},
        },
    )
    for flags in (doc["coverage"], src["coverage"]):
        flags.update(content_searched=True, section_reviewed=True)
    checked = seal_document(doc)
    assert checked["sources"][0]["metadata"]["enabled"] is None
    assert checked["coverage"]["body_reviewed"] is False


def test_findings_have_typed_claim_bound_observations():
    doc = packet()
    finding = doc["findings"][0]
    finding["type"] = "stale"
    finding["evidence"][0].update(
        kind="schedule", status="unverified", source_ids=["s1"], claim="job executes"
    )
    checked = seal_document(doc)
    assert checked["findings"][0]["evidence"][0]["status"] == "unverified"
    finding["evidence"][0]["source_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="evidence source"):
        seal_document(doc)


def test_packet_roundtrip_is_profile_bound_and_nonmutating():
    doc = packet()
    original = copy.deepcopy(doc)
    assert loads(dumps(doc), expected_profile=doc["profile"]) == doc
    assert doc == original
    with pytest.raises(ValueError, match="profile"):
        loads(dumps(doc), expected_profile={"id": "other", "root": "/other"})


def _reseal_inner_then_outer(doc):
    doc["inventory"] = seal_document(doc["inventory"])
    seal_document(doc)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(schema="nachos.reconciliation.packet.v2"),
        lambda d: d.update(execute=True),
        lambda d: d["proposals"][0].update(apply=True),
        lambda d: d["inventory"]["sources"][0].update(sha256="a" * 64),
        lambda d: d["proposals"][0]["changes"][0].update(before="wrong"),
        lambda d: d["findings"][0]["excerpts"][0].update(text="not observed"),
        lambda d: d["proposals"][0]["assessments"][0]["preservation_checklist"][0].update(
            preserved="yes"
        ),
        lambda d: d["findings"].append(copy.deepcopy(d["findings"][0])),
        lambda d: d["inventory"].update(profile={"id": "other", "root": "/other"}),
        lambda d: d["inventory"]["sources"][0].update(locator="../escape"),
        lambda d: d["findings"][0].update(preservation_checklist=[]),
        lambda d: d["findings"][0].update(type="nonsense"),
        lambda d: d["proposals"][0]["assessments"][0].update(revision=2),
        lambda d: d["proposals"][0]["assessments"].append(
            copy.deepcopy(d["proposals"][0]["assessments"][0])
        ),
        lambda d: d["findings"][0]["evidence"][0].update(status="bogus"),
        lambda d: d["findings"][0]["evidence"][0].update(observed_at="2026-11-16T12:00:00"),
        lambda d: d["inventory"]["sources"][0]["metadata"].update(byte_size=1),
        lambda d: d["inventory"]["sources"][0]["metadata"].update(enabled="unknown"),
        lambda d: d["inventory"]["sources"][0]["metadata"]["usage"].update(status="unused"),
        lambda d: d["inventory"]["sources"][0]["metadata"]["usage"].update(references=["x"]),
    ],
)
def test_contract_rejects_invalid_or_tampered_content(mutation):
    doc = packet()
    mutation(doc)
    # Reseal the inner inventory first so the targeted rule, not a stale inner digest, rejects it.
    with pytest.raises(ValueError, match=r".+") as caught:
        _reseal_inner_then_outer(doc)
    assert "artifact digest mismatch" not in str(caught.value)


@pytest.mark.parametrize(
    "text",
    ['{"schema":1,"schema":2}', '{"x":NaN}', '{"x":1.0}', "[" * 40 + "0" + "]" * 40, '{"x":true}'],
)
def test_strict_json_import(text):
    with pytest.raises(ValueError, match=r".+"):
        loads(text)


def test_rehashing_cannot_hide_unknown_fields():
    doc = packet()
    doc["findings"][0]["evidence"][0]["command"] = "do not execute"
    with pytest.raises(ValueError, match=r".+"):
        seal_document(doc)
    with pytest.raises(ValueError, match=r".+"):
        loads(json.dumps(doc))


@pytest.mark.parametrize("mode", ["false", "missing"])
def test_false_or_missing_preservation_blocks_acceptance(mode):
    doc = packet()
    if mode == "false":
        doc["proposals"][0]["assessments"][0]["preservation_checklist"][0]["preserved"] = False
    else:
        doc["proposals"][0]["assessments"] = []
    session = new_session(seal_document(doc))
    with pytest.raises(ValueError, match="preservation"):
        review_proposal(session, "p1", "accept")
    assert export_plan(session)["approved"] == []


def test_false_preservation_blocks_resealed_session_and_imported_plan():
    session = review_proposal(new_session(packet()), "p1", "accept")
    bad = copy.deepcopy(session)
    check = bad["reviews"][0]["revisions"][0]["assessments"][0]["preservation_checklist"][0]
    check["preserved"] = False
    bad["packet"]["proposals"][0] = copy.deepcopy(bad["reviews"][0]["revisions"][0])
    bad["packet"] = seal_document(bad["packet"])
    with pytest.raises(ValueError, match="preservation"):
        seal_document(bad)
    plan = export_plan(session)
    plan["approved"][0]["proposal"]["assessments"][0]["preservation_checklist"][0]["preserved"] = (
        False
    )
    with pytest.raises(ValueError, match="preservation"):
        seal_document(plan)
    plan = export_plan(session)
    plan["approved"][0]["proposal"]["assessments"] = []
    with pytest.raises(ValueError, match="preservation"):
        seal_document(plan)


def test_edit_invalidates_preservation_until_fresh_revision_assessment():
    session = review_proposal(new_session(packet()), "p1", "accept")
    edited = copy.deepcopy(session["reviews"][0]["revisions"][-1])
    edited["revision"] = 2
    edited["changes"][0]["after"] = "unsafe replacement"
    session = review_proposal(session, "p1", "edit", edited_proposal=edited)
    # The copied revision-1 assessment is discarded, not carried forward.
    assert session["reviews"][0]["revisions"][-1]["assessments"] == []
    with pytest.raises(ValueError, match="preservation"):
        review_proposal(session, "p1", "accept")
    # A stale assessment bound to another revision cannot be smuggled in.
    stale = copy.deepcopy(edited)
    stale["revision"] = 3
    with pytest.raises(ValueError, match="exact revision"):
        review_proposal(session, "p1", "edit", edited_proposal=stale)
    fixed = fresh(edited, 3)
    session = review_proposal(session, "p1", "edit", edited_proposal=fixed)
    session = review_proposal(session, "p1", "accept")
    assert session["reviews"][0]["decision"] == "accept"


def test_review_edit_restore_and_revision_request_invalidate_acceptance():
    original = new_session(packet())
    accepted = review_proposal(original, "p1", "accept", note="checked")
    assert original["reviews"][0]["decision"] == "unreviewed"
    record = accepted["reviews"][0]
    assert record["accepted_digest"] == proposal_digest(accepted["packet"], record["revisions"][-1])
    edited = copy.deepcopy(record["revisions"][-1])
    edited["revision"] = 2
    edited["changes"][0]["after"] = "changed"
    session = review_proposal(accepted, "p1", "edit", edited_proposal=edited)
    assert session["reviews"][0]["accepted_digest"] is None
    restored = fresh(record["revisions"][0], 3)
    session = review_proposal(session, "p1", "edit", edited_proposal=restored)
    assert session["reviews"][0]["decision"] == "unreviewed"
    session = review_proposal(session, "p1", "accept")
    assert session["reviews"][0]["accepted_digest"] != record["accepted_digest"]
    session = review_proposal(session, "p1", "request_revision", note="recheck scope")
    assert session["reviews"][0]["current_revision"] == 4
    assert session["reviews"][0]["decision"] == "request_revision"
    assert session["reviews"][0]["accepted_digest"] is None
    assert loads(dumps(session)) == session
    assert len(session["reviews"][0]["history"]) == 5


@pytest.mark.parametrize(
    "field", ["rationale", "preservation_checklist", "evidence", "limitations", "metadata"]
)
def test_review_digest_binds_all_reviewed_finding_content(field):
    doc = packet()
    before = proposal_digest(doc, doc["proposals"][0])
    changed = copy.deepcopy(doc)
    if field == "rationale":
        changed["findings"][0][field] += " changed"
    elif field == "preservation_checklist":
        checks = changed["proposals"][0]["assessments"][0][field]
        checks.append({"item": "extra", "preserved": True, "explanation": "extra"})
    elif field == "metadata":
        changed["inventory"]["sources"][0]["metadata"]["provenance"] = ["different"]
        changed["inventory"] = seal_document(changed["inventory"])
    else:
        changed["findings"][0][field] = changed["findings"][0][field] * 2
        if field == "limitations":
            changed["findings"][0][field][-1] = "different limit"
    changed = seal_document(changed)
    assert proposal_digest(changed, changed["proposals"][0]) != before


def test_resume_rejects_resealed_approval_inconsistency():
    session = review_proposal(new_session(packet()), "p1", "accept")
    for field, value in (
        ("accepted_digest", "a" * 64),
        ("decision", "defer"),
        ("current_revision", 2),
    ):
        changed = copy.deepcopy(session)
        changed["reviews"][0][field] = value
        with pytest.raises(ValueError, match=r".+"):
            seal_document(changed)
    changed = copy.deepcopy(session)
    changed["reviews"][0]["revisions"][0]["changes"][0]["after"] = "tampered"
    with pytest.raises(ValueError, match=r".+"):
        seal_document(changed)
    changed = copy.deepcopy(session)
    changed["packet"]["findings"][0]["rationale"] += " tampered"
    changed["packet"] = seal_document(changed["packet"])
    with pytest.raises(ValueError, match=r".+"):
        seal_document(changed)


@pytest.mark.parametrize("action", ["reject", "defer", "request_revision"])
def test_decisions_revoke_approval(action):
    session = review_proposal(new_session(packet()), "p1", "accept")
    session = review_proposal(session, "p1", action)
    assert session["reviews"][0]["decision"] == action
    assert session["reviews"][0]["accepted_digest"] is None


def test_export_only_current_exact_accepted_revision_and_no_execution():
    session = new_session(packet())
    assert not export_plan(session)["approved"]
    session = review_proposal(session, "p1", "accept")
    plan = export_plan(loads(dumps(session)))
    assert plan["execution"] == "not_performed"
    assert plan["source_validation"] == "not_performed"
    assert plan["approved"][0]["proposal"] == session["reviews"][0]["revisions"][-1]
    assert plan["ledger"] == []
    assert loads(dumps(plan)) == plan
    session = review_proposal(session, "p1", "request_revision")
    assert export_plan(session)["approved"] == []
    assert export_plan(session)["ledger"][0]["status"] == "unreviewed"


@pytest.mark.parametrize(("decision", "status"), [("reject", "rejected"), ("defer", "deferred")])
def test_export_decision_ledger(decision, status):
    plan = export_plan(review_proposal(new_session(packet()), "p1", decision))
    assert plan["approved"] == []
    assert plan["ledger"][0]["status"] == status


@pytest.mark.parametrize("dependencies", [["missing"], ["p1"]])
def test_export_blocks_missing_or_cyclic_dependencies(dependencies):
    doc = packet()
    doc["proposals"][0]["depends_on"] = dependencies
    session = review_proposal(new_session(seal_document(doc)), "p1", "accept")
    plan = export_plan(session)
    assert plan["approved"] == []
    assert plan["ledger"][0]["status"] == "blocked"


def test_each_changed_source_requires_its_own_exact_excerpt():
    doc = packet()
    second = copy.deepcopy(doc["inventory"]["sources"][0])
    second.update(id="s2", identity="skill:b", locator="skills/b/SKILL.md")
    doc["inventory"]["sources"].append(second)
    doc["inventory"] = seal_document(doc["inventory"])
    doc["findings"][0]["source_ids"].append("s2")
    doc["proposals"][0]["changes"].append({**doc["proposals"][0]["changes"][0], "source_id": "s2"})
    with pytest.raises(ValueError, match="excerpt"):
        seal_document(doc)


def test_conflicting_multi_source_proposals_are_not_trimmed():
    doc = packet()
    second = copy.deepcopy(doc["inventory"]["sources"][0])
    second.update(id="s2", identity="skill:b", locator="skills/b/SKILL.md")
    doc["inventory"]["sources"].append(second)
    doc["inventory"] = seal_document(doc["inventory"])
    doc["findings"][0]["source_ids"].append("s2")
    other = copy.deepcopy(doc["proposals"][0])
    doc["findings"][0]["excerpts"].append({"source_id": "s2", "text": "old decision"})
    other["id"] = "p2"
    other["changes"][0]["after"] = "conflict"
    doc["proposals"][0]["changes"].append({**copy.deepcopy(other["changes"][0]), "source_id": "s2"})
    doc["proposals"].append(other)
    session = new_session(seal_document(doc))
    for pid in ("p1", "p2"):
        session = review_proposal(session, pid, "accept")
    plan = export_plan(session)
    assert plan["approved"] == []
    assert [item["status"] for item in plan["ledger"]] == ["blocked", "blocked"]
    assert len(session["reviews"][0]["revisions"][0]["changes"]) == 2


def test_disable_is_proposal_only_and_plan_rejects_resealed_tampering():
    doc = packet()
    doc["proposals"][0]["changes"][0].update(operation="disable", after=None)
    plan = export_plan(review_proposal(new_session(seal_document(doc)), "p1", "accept"))
    assert plan["approved"][0]["proposal"]["changes"][0]["operation"] == "disable"
    for key in ("execution", "source_validation"):
        bad = copy.deepcopy(plan)
        bad[key] = "performed"
        with pytest.raises(ValueError, match=r".+"):
            seal_document(bad)
    bad = copy.deepcopy(plan)
    bad["approved"][0]["findings"][0]["rationale"] = "tampered"
    with pytest.raises(ValueError, match=r".+"):
        seal_document(bad)


def test_wrong_nested_types_fail_closed_as_value_errors():
    # Exercise malformed behavioral types independently of stale outer checksums.
    doc = packet()
    for location in (
        ("findings", 0, "excerpts", 0, "source_id"),
        ("proposals", 0, "changes", 0, "source_id"),
        ("inventory", "sources"),
        ("inventory", "schema"),
    ):
        for value in (None, True, 0, [], {}):
            bad = copy.deepcopy(doc)
            target = bad
            for key in location[:-1]:
                target = target[key]
            target[location[-1]] = value
            with pytest.raises(ValueError, match=r".+"):
                seal_document(bad)
    session = review_proposal(new_session(doc), "p1", "accept")
    session["reviews"][0]["history"][0]["action"] = {}
    with pytest.raises(ValueError, match=r".+"):
        seal_document(session)
    plan = export_plan(review_proposal(new_session(doc), "p1", "defer"))
    plan["ledger"][0]["decision"] = {}
    with pytest.raises(ValueError, match=r".+"):
        seal_document(plan)


def test_memory_record_roundtrip_and_key_identity_precondition():
    doc = packet()
    record = {
        "key": "k",
        "title": "title",
        "summary": "summary",
        "category": "fact",
        "body": "old decision\n",
    }
    src = doc["inventory"]["sources"][0]
    src.update(
        kind="memory_record",
        before=record,
        sha256=memory_sha256(record),
        identity="memory:k",
        locator="index.sqlite3",
    )
    src["metadata"].update(
        byte_size=len(canonical_json(record)), char_size=len(canonical_json(record).decode())
    )
    doc["inventory"] = seal_document(doc["inventory"])
    change = doc["proposals"][0]["changes"][0]
    change.update(
        source_sha256=src["sha256"],
        before=copy.deepcopy(record),
        after={**record, "body": "old decision\nclarification"},
    )
    doc = seal_document(doc)
    plan = export_plan(review_proposal(new_session(doc), "p1", "accept"))
    assert loads(dumps(plan))["approved"][0]["sources"][0]["before"] == record
    doc["proposals"][0]["changes"][0]["after"]["key"] = "different"
    with pytest.raises(ValueError, match="key identity"):
        seal_document(doc)


def test_source_snapshot_metadata_and_coverage_are_review_bound():
    doc = packet()
    accepted = review_proposal(new_session(doc), "p1", "accept")
    digest = accepted["reviews"][0]["accepted_digest"]
    changed = copy.deepcopy(doc)
    changed["inventory"]["sources"][0]["coverage"]["body_reviewed"] = True
    changed["inventory"]["sources"][0]["identity"] = "different-label"
    changed["inventory"] = seal_document(changed["inventory"])
    changed = seal_document(changed)
    assert proposal_digest(changed, changed["proposals"][0]) != digest
    accepted["packet"] = changed
    with pytest.raises(ValueError, match="history digest"):
        seal_document(accepted)


def test_dependencies_require_current_acceptance_and_export_in_order():
    doc = packet()
    second = copy.deepcopy(doc["proposals"][0])
    second.update(id="p2", depends_on=["p1"])
    doc["proposals"].insert(0, second)
    session = new_session(seal_document(doc))
    session = review_proposal(session, "p2", "accept")
    plan = export_plan(session)
    assert plan["approved"] == []
    assert {item["proposal_id"]: item["status"] for item in plan["ledger"]} == {
        "p1": "unreviewed",
        "p2": "blocked",
    }
    session = review_proposal(session, "p1", "accept")
    assert [entry["proposal"]["id"] for entry in export_plan(session)["approved"]] == ["p1", "p2"]


def test_review_and_export_do_not_open_files_or_network(monkeypatch):
    def denied(*_args, **_kwargs):
        pytest.fail("offline contract attempted IO")

    doc = packet()
    monkeypatch.setattr("builtins.open", denied)
    monkeypatch.setattr("socket.socket", denied)
    session = review_proposal(new_session(doc), "p1", "accept")
    assert loads(dumps(export_plan(session)))["execution"] == "not_performed"


def test_json_is_utf8_only_and_size_depth_count_limited():
    with pytest.raises(ValueError, match=r".+"):
        loads(dumps(packet()).decode().encode("utf-16"))
    for value in ([0] * 10_001, "x" * (8 * 1024 * 1024 + 1)):
        with pytest.raises(ValueError, match=r".+"):
            canonical_json(value)
    with pytest.raises(ValueError, match=r".+"):
        loads(" " * (8 * 1024 * 1024 + 1))
