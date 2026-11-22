"""Apply an exported, accepted plan through ``MemoryService``; dry-run unless told otherwise.

Nothing here trusts the plan on its own: it re-validates the plan against the CURRENT
sources immediately before acting and refuses on any drift, re-checks each record right
before each write, and performs every write through the service so redaction, the quality
gate and the change log all apply (``revert`` undoes any applied change). The whole plan is
preflighted first: one refusal and nothing is written. Memory records only; skill and
reference file edits stay with the primary session. Removal needs an explicit opt-in.
"""

from __future__ import annotations

from typing import Any

from ..memory import PROFILE, MemoryService
from .acquire import SourceSpec, read_current_sources
from .boundary import AcquisitionError, SourceBoundary
from .models import SCHEMAS, validate_document
from .validate import validate_plan_sources

APPLY_SCHEMA = "nachos.reconciliation.apply.v1"
_LINK_OPS = ("supersede", "unlink")


def _refusal(proposal_id: str, reason: str) -> dict[str, str]:
    return {"proposal_id": proposal_id, "reason": reason}


def _preflight(
    service: MemoryService, plan: dict[str, Any], allow_remove: bool
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Ordered action list and refusals, using only dry-run service calls."""
    actions: list[dict[str, Any]] = []
    refusals: list[dict[str, str]] = []
    for approval in plan["approved"]:
        item = approval["proposal"]
        for change in item["changes"]:
            reason = _check_change(service, change, allow_remove)
            if reason:
                refusals.append(_refusal(item["id"], reason))
            actions.append(
                {
                    "proposal_id": item["id"],
                    "kind": item.get("kind", "other"),
                    "operation": change["operation"],
                    "key": change["before"]["key"] if isinstance(change["before"], dict) else "",
                    "change": change,
                    "status": "planned",
                }
            )
    return actions, refusals


def _check_change(service: MemoryService, change: dict[str, Any], allow_remove: bool) -> str | None:
    before = change["before"]
    if not isinstance(before, dict):
        return "only memory records can be applied here; use the primary session for files"
    op = change["operation"]
    if op == "disable" and not allow_remove:
        return "removal is not allowed (re-run with allow_remove)"
    if op == "edit":
        after = change["after"]
        draft = service.prepare(
            title=after["title"],
            body=after["body"],
            summary=after["summary"],
            category=after["category"],
            key=before["key"],
        )
        if draft.error:
            return f"quality gate: {draft.error}"
    return None


def _live_record_matches(
    boundary: SourceBoundary, spec: SourceSpec, change: dict[str, Any]
) -> bool:
    try:
        current = read_current_sources(boundary, spec)
    except AcquisitionError:
        return False
    found = current.get(f"memory:{change['before']['key']}")
    return found is not None and found.sha256 == change["source_sha256"]


def _perform(service: MemoryService, change: dict[str, Any]) -> dict[str, Any]:
    op, before, after = change["operation"], change["before"], change["after"]
    key = before["key"]
    if op == "edit":
        return service.put(
            title=after["title"],
            body=after["body"],
            summary=after["summary"],
            category=after["category"],
            key=key,
        )
    if op == "disable":
        removed = service.remove(key, PROFILE)
        return {"removed": removed} if removed else {"error": f"{key!r} not found"}
    if op == "supersede":
        return service.link_supersede(key, after["superseded_by"])
    return service.unlink_supersede(key, expect=after["superseded_by"])


def _change_id(service: MemoryService, key: str) -> int | None:
    rows = service.history(key=key, limit=1)
    return int(rows[0]["id"]) if rows else None


def apply_plan(
    plan: dict[str, Any],
    service: MemoryService,
    boundary: SourceBoundary,
    profile: dict[str, str],
    spec: SourceSpec,
    *,
    run: bool = False,
    allow_remove: bool = False,
) -> dict[str, Any]:
    """Validate, preflight, then (only with ``run``) apply. Returns a detached report."""
    checked = validate_document(plan, expected_profile=profile)
    if checked["schema"] != SCHEMAS["plan"]:
        raise ValueError("apply requires an exported plan")
    report: dict[str, Any] = {
        "schema": APPLY_SCHEMA,
        "plan_digest": checked["digest"],
        "dry_run": not run,
        "status": "refused",
        "refusals": [],
        "actions": [],
    }
    validation = validate_plan_sources(checked, boundary=boundary, profile=profile, spec=spec)
    report["source_status"] = validation["status"]
    if validation["status"] != "valid_at_check":
        report["refusals"] = [_refusal("*", f"sources are {validation['status']}; re-inventory")]
        return report
    actions, refusals = _preflight(service, checked, allow_remove)
    report["refusals"] = refusals
    if refusals:
        report["actions"] = [_public(a) for a in actions]
        return report
    if not run:
        report["status"] = "planned"
        report["actions"] = [_public(a) for a in actions]
        return report
    applied = _execute(service, boundary, spec, actions)
    report["actions"] = [_public(a) for a in actions]
    report["status"] = "applied" if applied else "partial_failure"
    return report


def _execute(
    service: MemoryService,
    boundary: SourceBoundary,
    spec: SourceSpec,
    actions: list[dict[str, Any]],
) -> bool:
    for position, action in enumerate(actions):
        change = action["change"]
        if not _live_record_matches(boundary, spec, change):
            return _halt(actions, position, "source changed since validation")
        result = _perform(service, change)
        if "error" in result:
            return _halt(actions, position, str(result["error"]))
        action["status"] = "applied"
        action["change_id"] = _change_id(service, action["key"])
        if change["operation"] == "edit":
            action["verified"], action["redacted"] = _readback(service, change)
    return True


def _halt(actions: list[dict[str, Any]], position: int, reason: str) -> bool:
    actions[position]["status"] = "failed"
    actions[position]["note"] = reason
    for later in actions[position + 1 :]:
        later["status"] = "not_attempted"
    return False


def _readback(service: MemoryService, change: dict[str, Any]) -> tuple[bool, bool]:
    """(stored body matches what the service wrote, service changed the approved text)."""
    after = change["after"]
    draft = service.prepare(
        title=after["title"],
        body=after["body"],
        summary=after["summary"],
        category=after["category"],
        key=change["before"]["key"],
    )
    stored = service.scopes[PROFILE].store.get(draft.key)
    redacted = (draft.title, draft.summary, draft.body) != (
        after["title"],
        after["summary"],
        after["body"],
    )
    return stored is not None and draft.body in stored, redacted


def _public(action: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in action.items() if k != "change"}
