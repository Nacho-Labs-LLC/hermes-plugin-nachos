"""Accepted-current-revision export only. No apply or live source validation."""

from collections import defaultdict, deque
from typing import Any, cast

from .hashing import canonical_json, content_digest
from .models import (
    SCHEMAS,
    approval_safe,
    coverage,
    digest_text,
    fields,
    finding,
    indexed,
    proposal,
    seal_document,
    sequence,
    source,
    text,
    validate_document,
)
from .review import review_context


def _mark_conflicts(owners: dict[str, list[tuple[str, bytes]]], blocked: dict[str, str]) -> None:
    for group in owners.values():
        if len({raw for _, raw in group}) > 1:
            for pid, _ in group:
                blocked[pid] = "conflicting changes to the same source"


def _dependency_order(
    pending: dict[str, int], followers: dict[str, list[str]], blocked: dict[str, str]
) -> list[str]:
    queue = deque(
        sorted(pid for pid, count in pending.items() if count == 0 and pid not in blocked)
    )
    order = []
    while queue:
        pid = queue.popleft()
        order.append(pid)
        for follower in followers[pid]:
            pending[follower] -= 1
            if pending[follower] == 0 and follower not in blocked:
                queue.append(follower)
    return order


def _eligible(items: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Conservative source conflict detection and dependency ordering, all-or-nothing."""
    blocked: dict[str, str] = {}
    owners: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    followers: dict[str, list[str]] = defaultdict(list)
    pending: dict[str, int] = {}
    for pid, item in items.items():
        for change in item["changes"]:
            owners[change["source_id"]].append((pid, canonical_json(change)))
        pending[pid] = len(item["depends_on"])
        for dep in item["depends_on"]:
            followers[dep].append(pid)
            if dep not in items:
                blocked[pid] = "missing or unaccepted dependency"
    _mark_conflicts(owners, blocked)
    order = _dependency_order(pending, followers, blocked)
    selected = set(order)
    for pid in items:
        if pid not in selected and pid not in blocked:
            blocked[pid] = "cyclic or blocked dependency"
    return order, blocked


def export_plan(session: dict[str, Any]) -> dict[str, Any]:
    """Export accepted exact current revisions; all other states go to a separate ledger.

    Hashes attest integrity, NOT authorship. Offline export neither validates live
    sources nor authorizes execution. A multi-source proposal is never trimmed.
    """
    checked = validate_document(session)
    if checked["schema"] != SCHEMAS["session"]:
        raise ValueError("export requires a review session")
    records = {record["proposal_id"]: record for record in checked["reviews"]}
    candidates = {
        pid: record["revisions"][-1]
        for pid, record in records.items()
        if record["decision"] == "accept"
    }
    order, blocked = _eligible(candidates)
    approved = []
    for pid in order:
        context = review_context(checked["packet"], candidates[pid])
        context.pop("profile")
        context.pop("packet_digest")
        approved.append({**context, "review_digest": records[pid]["accepted_digest"]})
    ledger = []
    for pid in sorted(records):
        if pid in order:
            continue
        record = records[pid]
        decision = record["decision"]
        status = {"reject": "rejected", "defer": "deferred"}.get(decision, "unreviewed")
        ledger.append(
            {
                "proposal_id": pid,
                "revision": record["current_revision"],
                "decision": decision,
                "status": "blocked" if pid in blocked else status,
                "reason": blocked.get(pid, "current revision is not explicitly accepted"),
            }
        )
    return seal_document(
        {
            "schema": SCHEMAS["plan"],
            "profile": checked["profile"],
            "packet_digest": checked["packet"]["digest"],
            "session_digest": checked["digest"],
            "approved": approved,
            "ledger": ledger,
            "execution": "not_performed",
            "source_validation": "not_performed",
        }
    )


def _approved(entry: Any, plan: dict[str, Any]) -> dict[str, Any]:
    fields(entry, "proposal findings sources inventory_coverage review_digest")
    sources = indexed(entry["sources"])
    for src in sources.values():
        source(src)
    findings = indexed(entry["findings"])
    for item in findings.values():
        finding(item, sources)
    coverage(entry["inventory_coverage"])
    item = entry["proposal"]
    proposal(item, {"findings": entry["findings"], "inventory": {"sources": entry["sources"]}})
    if not approval_safe(item):
        raise ValueError("preservation assessment blocks approval")
    if set(item["finding_ids"]) != findings.keys():
        raise ValueError("extraneous plan findings")
    wanted = {sid for item_finding in findings.values() for sid in item_finding["source_ids"]}
    if wanted != sources.keys():
        raise ValueError("extraneous plan sources")
    context = {key: value for key, value in entry.items() if key != "review_digest"}
    context.update(profile=plan["profile"], packet_digest=plan["packet_digest"])
    if entry["review_digest"] != content_digest("proposal", context):
        raise ValueError("approved review content digest mismatch")
    return cast("dict[str, Any]", item)


def validate_plan(value: dict[str, Any]) -> None:
    """Internal validator; plan integrity is not evidence of authentic approval."""
    fields(
        value,
        "schema profile digest packet_digest session_digest approved ledger execution source_validation",
    )
    digest_text(value["packet_digest"])
    digest_text(value["session_digest"])
    if value["execution"] != "not_performed" or value["source_validation"] != "not_performed":
        raise ValueError("offline plan must not claim execution or source validation")
    sequence(value["approved"])
    items: dict[str, Any] = {}
    for entry in value["approved"]:
        item = _approved(entry, value)
        if item["id"] in items:
            raise ValueError("duplicate approved ID")
        items[item["id"]] = item
    order, blocked = _eligible(items)
    if list(items) != order:
        raise ValueError("approved plan entries are not in dependency order")
    if blocked:
        raise ValueError("plan includes conflicting or unsatisfied dependencies")
    sequence(value["ledger"])
    seen = set(items)
    for entry in value["ledger"]:
        fields(entry, "proposal_id revision decision status reason")
        text(entry["proposal_id"])
        text(entry["reason"])
        if entry["proposal_id"] in seen:
            raise ValueError("duplicate ledger/approved ID")
        seen.add(entry["proposal_id"])
        if type(entry["revision"]) is not int or entry["revision"] < 1:
            raise ValueError("invalid ledger revision")
        allowed = {
            "reject": "rejected",
            "defer": "deferred",
            "accept": "blocked",
            "unreviewed": "unreviewed",
            "request_revision": "unreviewed",
        }
        if entry["decision"] not in allowed or allowed[entry["decision"]] != entry["status"]:
            raise ValueError("invalid decision ledger state")
