"""The agent-facing flow: review -> (user) approve -> apply, with its guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nachos_core.store import SqliteStore
from nachos_hermes.memory_provider import NachosMemoryProvider
from tests.test_reconciliation_findings import ENTRIES


def _provider(home: Path, *, primary: bool = True) -> NachosMemoryProvider:
    (home / "nachos").mkdir(parents=True)
    store = SqliteStore(home / "nachos" / "memory.db")
    for key, title, summary, category, body, ts in ENTRIES:
        store.put(key, title=title, summary=summary, category=category, body=body)
        store._conn.execute("UPDATE entries SET updated_at = ? WHERE key = ?", (float(ts), key))
    store._conn.commit()
    store.close()
    provider = NachosMemoryProvider()
    provider.save_config({"embeddings": "off", "redact_secrets": "off"}, str(home))
    provider.initialize(
        "s1", hermes_home=str(home), agent_context="primary" if primary else "subagent"
    )
    return provider


def _call(provider, **args):
    return json.loads(provider.handle_tool_call("nachos_memory_reconcile", args))


@pytest.fixture
def provider(tmp_path):
    return _provider(tmp_path / "home")


def _review(provider):
    result = _call(provider, action="review")
    assert result["review_id"]
    return result


def test_review_writes_private_files_and_changes_no_memory(provider):
    before = {k for k, *_ in provider._service.profile.store.list()}
    result = _review(provider)
    folder = Path(result["folder"])
    assert {p.name for p in folder.iterdir()} == {"packet.json", "session.json", "viewer.html"}
    assert (folder / "session.json").stat().st_mode & 0o777 == 0o600
    assert result["proposals"]
    assert {p["kind"] for p in result["proposals"]} >= {"merge", "supersede"}
    assert {k for k, *_ in provider._service.profile.store.list()} == before


def test_approve_requires_user_confirmation_and_explicit_ids(provider):
    review_id = _review(provider)["review_id"]
    assert "error" in _call(provider, action="approve", review_id=review_id, proposal_ids=["x"])
    assert "error" in _call(
        provider, action="approve", review_id=review_id, confirm_user_approved=True
    )
    assert "error" in _call(
        provider,
        action="approve",
        review_id=review_id,
        confirm_user_approved=True,
        proposal_ids=["not-a-proposal"],
    )


def test_full_flow_apply_is_dry_run_first_then_writes_and_is_undoable(provider):
    result = _review(provider)
    review_id = result["review_id"]
    ids = [p["id"] for p in result["proposals"]]
    approved = _call(
        provider,
        action="approve",
        review_id=review_id,
        proposal_ids=ids,
        confirm_user_approved=True,
    )
    assert set(approved["approved"]) == set(ids)

    dry = _call(provider, action="apply", review_id=review_id, allow_remove=True)
    assert dry["dry_run"] is True
    assert dry["status"] == "planned"
    assert "pg-backup" in {k for k, *_ in provider._service.profile.store.list()}

    refused = _call(provider, action="apply", review_id=review_id, run=True)
    assert refused["status"] == "refused"

    done = _call(provider, action="apply", review_id=review_id, run=True, allow_remove=True)
    assert done["status"] == "applied"
    assert "pg-backup" not in {k for k, *_ in provider._service.profile.store.list()}
    assert all(isinstance(a["change_id"], int) for a in done["actions"])
    for change_id in sorted((a["change_id"] for a in done["actions"]), reverse=True):
        assert "error" not in provider._service.revert(change_id)
    assert "pg-backup" in {k for k, *_ in provider._service.profile.store.list()}


def test_status_lists_reviews_and_decisions(provider):
    assert _call(provider, action="status") == {"reviews": []}
    review_id = _review(provider)["review_id"]
    listing = _call(provider, action="status")
    assert listing["reviews"] == [{"review_id": review_id, "has_plan": False}]
    detail = _call(provider, action="status", review_id=review_id)
    assert {p["decision"] for p in detail["proposals"]} == {"unreviewed"}


def test_non_primary_cannot_write_but_can_read_status(tmp_path):
    provider = _provider(tmp_path / "home", primary=False)
    assert _call(provider, action="review") == {"skipped": "non-primary agent context"}
    assert _call(provider, action="approve", review_id="a" * 12) == {
        "skipped": "non-primary agent context"
    }
    assert _call(provider, action="status") == {"reviews": []}


@pytest.mark.parametrize("review_id", ["../../etc", "abc", "g" * 12, "A" * 12, 5])
def test_review_ids_are_validated_not_joined_into_paths(provider, review_id):
    result = _call(provider, action="status", review_id=review_id)
    assert "error" in result


def test_unknown_action_is_an_error(provider):
    assert "error" in _call(provider, action="wipe")


def test_clean_store_reports_nothing_to_review(tmp_path):
    home = tmp_path / "home"
    (home / "nachos").mkdir(parents=True)
    store = SqliteStore(home / "nachos" / "memory.db")
    store.put(
        "editor",
        title="Editor preference",
        summary="User prefers vim keybindings",
        category="user",
        body="The user prefers vim keybindings in every editor and terminal.",
    )
    store.close()
    provider = NachosMemoryProvider()
    provider.save_config({"embeddings": "off"}, str(home))
    provider.initialize("s", hermes_home=str(home))
    result = _call(provider, action="review")
    assert result["review"] is None
    assert not (home / "nachos" / "reconcile").exists()
