"""Deterministic findings generator: every proposal kind, validity and determinism."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from nachos_core.index import MemoryIndex
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.reconciliation.acquire import SourceSpec
from nachos_core.reconciliation.boundary import SourceBoundary
from nachos_core.reconciliation.findings import generate_findings
from nachos_core.reconciliation.inventory import build_inventory
from nachos_core.reconciliation.packets import build_packet
from nachos_core.reconciliation.review import new_session
from nachos_core.reconciliation.signals import collect_signals
from nachos_core.store import SqliteStore

AT = "2026-11-22T10:00:00+00:00"
DB = "nachos/memory.db"

ENTRIES = [
    # duplicate pair: same fact, the older one has an extra line
    (
        "pg-backup",
        "Postgres backup schedule",
        "Nightly pg_dump at 02:00 to s3",
        "ops",
        "Nightly pg_dump at 02:00 UTC to the s3 backups bucket.\nRetention is 30 days.",
        100,
    ),
    (
        "pg-backup-copy",
        "Postgres backup schedule",
        "Nightly pg_dump at 02:00 to s3",
        "ops",
        "Nightly pg_dump at 02:00 UTC to the s3 backups bucket.",
        200,
    ),
    # outdated: newer states an update with a changed value
    (
        "api-port-old",
        "API server port",
        "The API server listens on port 8080",
        "infra",
        "The API server listens on port 8080 in every environment.",
        100,
    ),
    (
        "api-port-new",
        "API server port",
        "The API server now listens on port 9090",
        "infra",
        "The API server now listens on port 9090 in every environment, no longer 8080.",
        200,
    ),
    # conflict: differing value, no cue
    (
        "cache-ttl-a",
        "Session cache ttl",
        "Session cache ttl is 300 seconds",
        "infra",
        "The session cache entries expire after a ttl of 300 seconds.",
        100,
    ),
    (
        "cache-ttl-b",
        "Session cache ttl",
        "Session cache ttl is 900 seconds",
        "infra",
        "The session cache entries expire after a ttl of 900 seconds.",
        200,
    ),
    # low quality: summary just restates the title; body has real content
    (
        "deploy-notes",
        "Deploy notes",
        "Deploy notes",
        "ops",
        "Blue green deploys are promoted by flipping the load balancer weights after smoke tests pass.",
        100,
    ),
    # unrelated, healthy
    (
        "editor",
        "Editor preference",
        "User prefers vim keybindings",
        "user",
        "The user prefers vim keybindings in every editor and terminal.",
        100,
    ),
]


def _service(tmp_path, *, link=None):
    (tmp_path / "nachos").mkdir()
    store = SqliteStore(tmp_path / DB)
    for key, title, summary, category, body, _ts in ENTRIES:
        store.put(key, title=title, summary=summary, category=category, body=body)
    for key, *_rest, ts in ENTRIES:
        store._conn.execute("UPDATE entries SET updated_at = ? WHERE key = ?", (float(ts), key))
    store._conn.commit()
    if link:
        store.set_superseded_by(*link)
    return MemoryService(OrderedDict([(PROFILE, Scope(PROFILE, store, MemoryIndex(":memory:")))]))


def _generate(tmp_path, **kwargs):
    service = _service(tmp_path, **kwargs)
    try:
        signals = collect_signals(service)
    finally:
        service.close()
    boundary = SourceBoundary(tmp_path, (DB,))
    inventory = build_inventory(boundary, {"id": "p", "root": str(tmp_path)}, SourceSpec("sqlite"))
    return inventory, generate_findings(inventory, signals, observed_at=AT)


def _by_kind(generated):
    return {p["kind"]: p for p in generated.proposals}


def test_every_kind_is_proposed_and_the_packet_validates(tmp_path):
    inventory, generated = _generate(tmp_path)
    kinds = _by_kind(generated)
    assert {"merge", "supersede", "choose-one", "rewrite"} <= kinds.keys()
    packet = build_packet(inventory, findings=generated.findings, proposals=generated.proposals)
    assert new_session(packet)["reviews"]
    for proposal in generated.proposals:
        assert proposal["assessments"] == []  # nothing is approvable without human verification
        assert proposal["confidence"] in ("low", "medium", "high")
        assert proposal["revision"] == 1


def test_merge_keeps_newer_folds_missing_lines_and_retires_older(tmp_path):
    _inv, generated = _generate(tmp_path)
    merge = _by_kind(generated)["merge"]
    edit, disable = merge["changes"]
    assert edit["operation"] == "edit"
    assert edit["after"]["key"] == "pg-backup-copy"
    assert "Retention is 30 days." in edit["after"]["body"]
    assert disable["operation"] == "disable"
    assert disable["before"]["key"] == "pg-backup"
    assert merge["confidence"] == "high"


def test_outdated_record_is_superseded_by_the_newer_one(tmp_path):
    _inv, generated = _generate(tmp_path)
    prop = next(
        p for p in generated.proposals if p["changes"][0]["before"]["key"] == "api-port-old"
    )
    assert prop["kind"] == "supersede"
    assert prop["changes"][0]["after"] == {"superseded_by": "api-port-new"}


def test_conflicting_facts_without_cue_are_choose_one_low_confidence(tmp_path):
    _inv, generated = _generate(tmp_path)
    prop = _by_kind(generated)["choose-one"]
    assert prop["confidence"] == "low"
    assert prop["changes"][0]["before"]["key"] == "cache-ttl-a"
    finding = next(f for f in generated.findings if f["id"] == prop["finding_ids"][0])
    assert finding["type"] == "conflict"
    assert "alternative" in finding["rationale"]


def test_low_quality_summary_gets_a_rewrite_from_the_body(tmp_path):
    _inv, generated = _generate(tmp_path)
    prop = next(p for p in generated.proposals if p["kind"] == "rewrite")
    after = prop["changes"][0]["after"]
    assert after["key"] == "deploy-notes"
    assert after["summary"].startswith("Blue green deploys")
    assert after["body"] == prop["changes"][0]["before"]["body"]


def test_suspicious_supersede_link_becomes_unlink(tmp_path):
    entries = [
        (
            "db-eu",
            "Database host for EU",
            "EU database host is eu-db-1",
            "infra",
            "The primary database host for EU region is eu-db-1.",
            100,
        ),
        (
            "db-us",
            "Database host for US",
            "US database host is us-db-1",
            "infra",
            "The primary database host for US region is us-db-1.",
            200,
        ),
    ]
    (tmp_path / "nachos").mkdir()
    store = SqliteStore(tmp_path / DB)
    for key, title, summary, category, body, _ts in entries:
        store.put(key, title=title, summary=summary, category=category, body=body)
    store.set_superseded_by("db-eu", "db-us")
    service = MemoryService(
        OrderedDict([(PROFILE, Scope(PROFILE, store, MemoryIndex(":memory:")))])
    )
    signals = collect_signals(service)
    service.close()
    inventory = build_inventory(
        SourceBoundary(tmp_path, (DB,)), {"id": "p", "root": str(tmp_path)}, SourceSpec("sqlite")
    )
    generated = generate_findings(inventory, signals, observed_at=AT)
    unlink = _by_kind(generated)["unlink"]
    assert unlink["changes"][0]["operation"] == "unlink"
    assert unlink["changes"][0]["after"] == {"superseded_by": "db-us"}
    build_packet(inventory, findings=generated.findings, proposals=generated.proposals)


def test_generation_is_deterministic_and_proposals_never_overlap(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _inv, first = _generate(tmp_path / "a")
    _inv, second = _generate(tmp_path / "b")
    assert first.proposals == second.proposals
    assert first.findings == second.findings
    touched = [c["source_id"] for p in first.proposals for c in p["changes"]]
    assert len(touched) == len(set(touched))


def test_drafter_can_replace_merged_text_but_not_the_key(tmp_path):
    service = _service(tmp_path)
    signals = collect_signals(service)
    service.close()
    inventory = build_inventory(
        SourceBoundary(tmp_path, (DB,)), {"id": "p", "root": str(tmp_path)}, SourceSpec("sqlite")
    )
    generated = generate_findings(
        inventory,
        signals,
        observed_at=AT,
        drafter=lambda _survivor, _other: {"body": "drafted", "key": "evil"},
    )
    edit = _by_kind(generated)["merge"]["changes"][0]
    assert edit["after"]["body"] == "drafted"
    assert edit["after"]["key"] == "pg-backup-copy"


def test_observed_at_requires_a_timezone(tmp_path):
    service = _service(tmp_path)
    signals = collect_signals(service)
    service.close()
    inventory = build_inventory(
        SourceBoundary(tmp_path, (DB,)), {"id": "p", "root": str(tmp_path)}, SourceSpec("sqlite")
    )
    with pytest.raises(ValueError, match="timezone"):
        generate_findings(inventory, signals, observed_at="2026-01-01")
