"""Pure immutable-style review transitions and fail-closed history replay."""

from copy import deepcopy
from typing import Any, Optional

from .hashing import content_digest
from .models import (
    SCHEMAS,
    approval_safe,
    fields,
    indexed,
    proposal,
    seal_document,
    sequence,
    text,
    validate_document,
)

ACTIONS = ("accept", "edit", "request_revision", "reject", "defer")


def review_context(packet: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """Exact human-reviewed content, including evidence, coverage and source snapshots."""
    findings = indexed(packet["findings"])
    selected = [findings[fid] for fid in item["finding_ids"]]
    source_ids = {sid for finding in selected for sid in finding["source_ids"]}
    return {
        "profile": packet["profile"],
        "packet_digest": packet["digest"],
        "proposal": item,
        "findings": selected,
        "sources": [src for src in packet["inventory"]["sources"] if src["id"] in source_ids],
        "inventory_coverage": packet["inventory"]["coverage"],
    }


def proposal_digest(packet: dict[str, Any], item: dict[str, Any]) -> str:
    """Bind a revision and everything presented for its review, not merely its diff."""
    validate_document(packet)
    proposal(item, packet)
    return content_digest("proposal", review_context(packet, item))


def new_session(packet: dict[str, Any]) -> dict[str, Any]:
    checked = validate_document(packet)
    if checked["schema"] != SCHEMAS["packet"]:
        raise ValueError("session requires a packet")
    reviews = [
        {
            "proposal_id": item["id"],
            "revisions": [deepcopy(item)],
            "history": [],
            "current_revision": 1,
            "decision": "unreviewed",
            "accepted_digest": None,
        }
        for item in checked["proposals"]
    ]
    return seal_document(
        {
            "schema": SCHEMAS["session"],
            "profile": checked["profile"],
            "packet": checked,
            "reviews": reviews,
        }
    )


def _event_revision(event: Any, revisions: list[Any], current: int) -> int:
    fields(event, "revision action digest note")
    text(event["note"], empty=True)
    action = event["action"]
    if action not in ACTIONS:
        raise ValueError("unknown review action")
    if action in ("edit", "request_revision"):
        current += 1
    if (
        type(event["revision"]) is not int
        or event["revision"] != current
        or current > len(revisions)
    ):
        raise ValueError("history revision mismatch")
    if action == "request_revision" and revisions[current - 1] != {
        **revisions[current - 2],
        "revision": current,
        "assessments": [],
    }:
        raise ValueError("revision request cannot edit content")
    return current


def _replay(record: dict[str, Any], packet: dict[str, Any], original: dict[str, Any]) -> None:
    fields(record, "proposal_id revisions history current_revision decision accepted_digest")
    sequence(record["revisions"], nonempty=True)
    sequence(record["history"])
    revisions = record["revisions"]
    if revisions[0] != original:
        raise ValueError("initial revision differs from packet")
    for number, item in enumerate(revisions, 1):
        proposal(item, packet)
        if item["id"] != original["id"] or item["revision"] != number:
            raise ValueError("noncontiguous or mismatched revision identity")
    current = 1
    decision = "unreviewed"
    accepted = None
    for event in record["history"]:
        current = _event_revision(event, revisions, current)
        action = event["action"]
        item = revisions[current - 1]
        if action == "accept" and not approval_safe(item):
            raise ValueError("preservation assessment blocks acceptance")
        actual = content_digest("proposal", review_context(packet, item))
        if event["digest"] != actual:
            raise ValueError("review history digest mismatch")
        decision = "unreviewed" if action == "edit" else action
        accepted = actual if action == "accept" else None
    if (
        type(record["current_revision"]) is not int
        or record["current_revision"] != current
        or len(revisions) != current
        or record["decision"] != decision
        or record["accepted_digest"] != accepted
    ):
        raise ValueError("inconsistent approval/revision state")


def validate_session(value: dict[str, Any]) -> None:
    """Internal semantic validator, invoked by validate_document after digest validation."""
    fields(value, "schema profile digest packet reviews")
    packet = value["packet"]
    validate_document(packet, expected_profile=value["profile"])
    if packet["schema"] != SCHEMAS["packet"]:
        raise ValueError("session requires packet schema")
    originals = indexed(packet["proposals"])
    sequence(value["reviews"])
    seen = set()
    for record in value["reviews"]:
        fields(record, "proposal_id revisions history current_revision decision accepted_digest")
        pid = record["proposal_id"]
        text(pid)
        if pid not in originals or pid in seen:
            raise ValueError("unknown or duplicate review ID")
        seen.add(pid)
        _replay(record, packet, originals[pid])
    if seen != originals.keys():
        raise ValueError("missing proposal review state")


def _append_revision(
    result: dict[str, Any],
    record: dict[str, Any],
    action: str,
    edited_proposal: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and append the next revision for an edit or revision request."""
    current = record["revisions"][-1]
    item = (
        deepcopy(edited_proposal)
        if action == "edit"
        else {**deepcopy(current), "revision": current["revision"] + 1, "assessments": []}
    )
    if item is None:
        raise ValueError("edit requires full edited_proposal")
    if type(item) is dict and type(item.get("assessments")) is list:
        # A copied prior assessment is not fresh confirmation of edited text.
        item["assessments"] = [
            a
            for a in item["assessments"]
            if type(a) is not dict or a.get("revision") != current["revision"]
        ]
    proposal(item, result["packet"])
    if item["id"] != record["proposal_id"] or item["revision"] != current["revision"] + 1:
        raise ValueError("edit must preserve ID and increment revision exactly")
    record["revisions"].append(item)
    return item


def review_proposal(
    session: dict[str, Any],
    proposal_id: str,
    action: str,
    *,
    edited_proposal: Optional[dict[str, Any]] = None,
    note: str = "",
) -> dict[str, Any]:
    """Return a new sealed session; edits require a full next-numbered revision.

    request_revision clones content into a new revision. Both it and edit revoke
    acceptance even when prose is unchanged/restored. No inputs are modified.
    """
    result = validate_document(session)
    if result["schema"] != SCHEMAS["session"] or action not in ACTIONS:
        raise ValueError("invalid session/action")
    text(note, empty=True)
    matches = [record for record in result["reviews"] if record["proposal_id"] == proposal_id]
    if not matches:
        raise ValueError("unknown proposal ID")
    record = matches[0]
    current = record["revisions"][-1]
    if action in ("edit", "request_revision"):
        current = _append_revision(result, record, action, edited_proposal)
    if action != "edit" and edited_proposal is not None:
        raise ValueError("edited_proposal is only permitted for edit")
    digest = content_digest("proposal", review_context(result["packet"], current))
    if action == "accept" and not approval_safe(current):
        raise ValueError("preservation assessment blocks acceptance")
    record["current_revision"] = current["revision"]
    record["decision"] = "unreviewed" if action == "edit" else action
    record["accepted_digest"] = digest if action == "accept" else None
    record["history"].append(
        {"revision": current["revision"], "action": action, "digest": digest, "note": note}
    )
    return seal_document(result)
