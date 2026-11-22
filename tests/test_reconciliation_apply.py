"""End to end: audit -> packet -> session -> approval -> export -> apply -> revert."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.reconciliation import dumps, export_plan, loads
from nachos_core.reconciliation.acquire import SourceSpec
from nachos_core.reconciliation.apply import apply_plan
from nachos_core.reconciliation.approval import approve_proposals
from nachos_core.reconciliation.boundary import SourceBoundary
from nachos_core.reconciliation.inventory import build_inventory
from nachos_core.reconciliation.packets import build_packet
from nachos_core.reconciliation.review import new_session
from nachos_core.store import SqliteStore
from nachos_hermes.reconcile_cli import main as cli
from tests.test_reconciliation_findings import ENTRIES

DB = "nachos/memory.db"
AT = "2026-11-22T10:00:00+00:00"


def _seed(home: Path) -> None:
    (home / "nachos").mkdir(parents=True)
    store = SqliteStore(home / DB)
    for key, title, summary, category, body, ts in ENTRIES:
        store.put(key, title=title, summary=summary, category=category, body=body)
        store._conn.execute("UPDATE entries SET updated_at = ? WHERE key = ?", (float(ts), key))
    store._conn.commit()
    store.close()


def _service(home: Path, redact=None) -> MemoryService:
    path = home / DB
    scope = Scope(PROFILE, SqliteStore(path), MemoryIndex(index_path_for(path)))
    return MemoryService(OrderedDict([(PROFILE, scope)]), redact=redact)


@pytest.fixture
def world(tmp_path):
    home = tmp_path / "home"
    _seed(home)
    out = tmp_path / "out"
    out.mkdir()
    base = ["--profile-home", str(home)]
    assert cli(["inventory", *base, "--out", str(out / "inv.json")]) == 0
    assert (
        cli(
            [
                "packet",
                "--inventory",
                str(out / "inv.json"),
                "--audit",
                *base,
                "--observed-at",
                AT,
                "--out",
                str(out / "packet.json"),
            ]
        )
        == 0
    )
    return home, out, base


def _accepted_plan(out: Path, proposal_ids: list[str] | None = None) -> dict:
    session = new_session(loads((out / "packet.json").read_bytes()))
    ids = proposal_ids or [r["proposal_id"] for r in session["reviews"]]
    approved = approve_proposals(session, ids, observed_at=AT)
    return export_plan(approved)


def _run(plan, home, *, run=False, allow_remove=False, redact=None):
    boundary = SourceBoundary(home, (DB,))
    profile = {"id": home.name, "root": str(home)}
    service = _service(home, redact)
    try:
        return apply_plan(
            plan,
            service,
            boundary,
            profile,
            SourceSpec("sqlite"),
            run=run,
            allow_remove=allow_remove,
        )
    finally:
        service.close()


def _keys(home):
    store = SqliteStore(home / DB)
    try:
        return {k for k, *_ in store.list()}
    finally:
        store.close()


def test_dry_run_is_the_default_and_changes_nothing(world):
    home, out, _base = world
    before = (home / DB).read_bytes()
    report = _run(_accepted_plan(out), home, allow_remove=True)
    assert report["dry_run"] is True
    assert report["status"] == "planned"
    assert {a["status"] for a in report["actions"]} == {"planned"}
    assert (home / DB).read_bytes() == before


def test_removal_needs_explicit_opt_in(world):
    home, out, _base = world
    report = _run(_accepted_plan(out), home, run=True)
    assert report["status"] == "refused"
    assert any("allow_remove" in r["reason"] for r in report["refusals"])
    assert "pg-backup" in _keys(home)


def test_run_applies_everything_through_the_service_and_revert_undoes_it(world):
    home, out, _base = world
    report = _run(_accepted_plan(out), home, run=True, allow_remove=True)
    assert report["status"] == "applied"
    keys = _keys(home)
    assert "pg-backup" not in keys
    store = SqliteStore(home / DB)
    assert "Retention is 30 days." in store.get("pg-backup-copy")
    assert store.get_meta("api-port-old").superseded_by == "api-port-new"
    store.close()
    ids = [a["change_id"] for a in report["actions"]]
    assert all(isinstance(i, int) for i in ids)
    service = _service(home)
    for change_id in sorted(ids, reverse=True):
        assert "error" not in service.revert(change_id)
    service.close()
    store = SqliteStore(home / DB)
    assert store.get_meta("api-port-old").superseded_by is None
    assert "pg-backup" in {k for k, *_ in store.list()}
    store.close()


def test_drifted_source_is_refused_before_any_write(world):
    home, out, _base = world
    plan = _accepted_plan(out)
    service = _service(home)
    service.put(title="Editor preference", body="The user now prefers emacs keybindings.")
    service.close()
    store = SqliteStore(home / DB)
    store.put(
        "api-port-new",
        title="API server port",
        summary="The API server port changed again",
        category="infra",
        body="The API server now listens on port 7070 everywhere.",
    )
    store.close()
    before = (home / DB).read_bytes()
    report = _run(plan, home, run=True, allow_remove=True)
    assert report["status"] == "refused"
    assert report["source_status"] == "stale"
    assert (home / DB).read_bytes() == before


def test_redaction_applies_to_edits_and_is_reported(world):
    home, out, _base = world
    plan = _accepted_plan(out)
    report = _run(
        plan, home, run=True, allow_remove=True, redact=lambda t: t.replace("30 days", "[x]")
    )
    assert report["status"] == "applied"
    edit = next(
        a for a in report["actions"] if a["operation"] == "edit" and a["key"] == "pg-backup-copy"
    )
    assert edit["redacted"] is True
    store = SqliteStore(home / DB)
    assert "[x]" in store.get("pg-backup-copy")
    store.close()


def test_a_tampered_plan_is_rejected_before_anything_runs(world):
    home, out, _base = world
    plan = _accepted_plan(out)
    for approval in plan["approved"]:
        for change in approval["proposal"]["changes"]:
            if change["operation"] == "edit" and change["before"]["key"] == "deploy-notes":
                change["after"]["summary"] = "tbd"
    report = None
    with pytest.raises(ValueError, match=r"digest|mismatch|malformed"):
        report = _run(plan, home, run=True, allow_remove=True)
    assert report is None


def test_unaccepted_plan_applies_nothing(world):
    home, out, _base = world
    session = new_session(loads((out / "packet.json").read_bytes()))
    report = _run(export_plan(session), home, run=True, allow_remove=True)
    assert report["status"] in ("applied", "planned")
    assert report["actions"] == []


def test_cli_apply_dry_run_then_run(world, capsys):
    home, out, base = world
    (out / "plan.json").write_bytes(dumps(_accepted_plan(out)))
    args = ["apply", "--plan", str(out / "plan.json"), *base, "--allow-unredacted"]
    assert cli([*args, "--out", str(out / "refused.json")]) == 3  # removal not opted into
    assert cli([*args, "--allow-remove", "--out", str(out / "dry.json")]) == 0
    assert json.loads((out / "dry.json").read_text())["status"] == "planned"
    assert "pg-backup" in _keys(home)
    assert cli([*args, "--allow-remove", "--run", "--out", str(out / "run.json")]) == 0
    assert json.loads((out / "run.json").read_text())["status"] == "applied"
    assert "pg-backup" not in _keys(home)
    assert "applied" in capsys.readouterr().out


def test_cli_apply_needs_a_redactor_unless_allowed(world, monkeypatch):
    _home, out, base = world
    monkeypatch.setattr("nachos_hermes.reconcile_cli.hermes_redactor", lambda: None)
    (out / "plan.json").write_bytes(dumps(_accepted_plan(out)))
    code = cli(["apply", "--plan", str(out / "plan.json"), *base, "--out", str(out / "r.json")])
    assert code == 2
    assert not (out / "r.json").exists()


def test_build_packet_helper_roundtrip_is_valid(world):
    _home, out, _base = world
    packet = loads((out / "packet.json").read_bytes())
    assert build_packet(
        packet["inventory"], findings=packet["findings"], proposals=packet["proposals"]
    )
    assert build_inventory  # imported for parity with other suites
