"""``nachos_memory_reconcile``: the agent-facing reconciliation flow.

review  -> audit the store, write a review packet, session and offline viewer into a private
           review folder, and return short proposal summaries (nothing is changed)
status  -> list review folders, or show one with each proposal's decision
approve -> record the USER's named approvals (never the agent's own) and export a plan
apply   -> re-validate against current sources and apply through the live MemoryService;
           dry-run unless ``run`` is true; removal needs ``allow_remove``

Everything that writes is primary-agent only. Approval needs ``confirm_user_approved`` and
explicit proposal ids; the careful path is the viewer, where a reviewer verifies evidence.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nachos_core.memory import MemoryService
from nachos_core.reconciliation import dumps, export_plan, loads, new_session
from nachos_core.reconciliation.acquire import SourceSpec
from nachos_core.reconciliation.apply import apply_plan
from nachos_core.reconciliation.approval import approve_proposals
from nachos_core.reconciliation.boundary import SourceBoundary
from nachos_core.reconciliation.findings import generate_findings
from nachos_core.reconciliation.hashing import canonical_json
from nachos_core.reconciliation.inventory import build_inventory
from nachos_core.reconciliation.packets import build_packet
from nachos_core.reconciliation.signals import collect_signals
from nachos_core.reconciliation.viewer import viewer_html

_ID = re.compile(r"^[0-9a-f]{12}$")
_MAX_LISTED = 25
_NOTE_CHARS = 140

RECONCILE_SCHEMA = {
    "name": "nachos_memory_reconcile",
    "description": (
        "Review and clean up memory: find duplicates, outdated or conflicting entries, "
        "suspicious supersede links and low-quality entries, and propose fixes. "
        "action='review' audits the store and writes a review packet plus an offline viewer "
        "(changes nothing) and returns proposal summaries; show them to the user. "
        "action='status' lists reviews or one review's decisions. action='approve' records "
        "approvals the USER named (set confirm_user_approved only after they said so; never "
        "approve on your own) and exports a plan. action='apply' applies the plan through the "
        "normal memory path (redaction, quality gate, change log, undo with revert); it is a "
        "dry run unless run=true, and removing an entry needs allow_remove=true. Proposals "
        "start unreviewed and the user can instead review them in the viewer file."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["review", "status", "approve", "apply"]},
            "review_id": {"type": "string", "description": "Id returned by action='review'."},
            "proposal_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact proposal ids the user approved (action='approve').",
            },
            "confirm_user_approved": {
                "type": "boolean",
                "description": "True only when the user explicitly approved these proposals.",
            },
            "plan_path": {
                "type": "string",
                "description": "Plan exported from the viewer (default: this review's plan).",
            },
            "run": {"type": "boolean", "description": "Actually write (default false: dry run)."},
            "allow_remove": {
                "type": "boolean",
                "description": "Permit removing entries (merges, deletes). Undoable via revert.",
            },
        },
        "required": ["action"],
    },
}


def _root(home: Path) -> Path:
    return home / "nachos" / "reconcile"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _label(store_kind: str) -> str:
    return "nachos/memory.md" if store_kind == "flatfile" else "nachos/memory.db"


def _folder(home: Path, review_id: Any) -> Path:
    if not isinstance(review_id, str) or not _ID.match(review_id):
        raise ValueError("review_id must be an id returned by action='review'")
    folder = _root(home) / review_id
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError(f"no review {review_id}")
    return folder


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(data)
    path.chmod(0o600)


def _summary(proposal: dict[str, Any]) -> dict[str, Any]:
    keys = [c["before"]["key"] for c in proposal["changes"] if isinstance(c["before"], dict)]
    return {
        "id": proposal["id"],
        "kind": proposal.get("kind", "other"),
        "confidence": proposal.get("confidence", "unspecified"),
        "operations": [c["operation"] for c in proposal["changes"]],
        "keys": keys,
    }


def _rationales(packet: dict[str, Any]) -> dict[str, str]:
    by_finding = {f["id"]: f["rationale"] for f in packet["findings"]}
    return {
        p["id"]: by_finding[p["finding_ids"][0]][: _NOTE_CHARS * 2] for p in packet["proposals"]
    }


def _review(service: MemoryService, home: Path, store_kind: str) -> dict[str, Any]:
    boundary = SourceBoundary(home, (_label(store_kind),))
    profile = {"id": home.name, "root": str(home)}
    inventory = build_inventory(boundary, profile, SourceSpec(store_kind))
    generated = generate_findings(inventory, collect_signals(service), observed_at=_now())
    if not generated.proposals:
        return {
            "review": None,
            "message": "Nothing to propose: no duplicates, outdated, "
            "conflicting or low-quality entries found.",
            "flagged": len(generated.findings),
        }
    packet = build_packet(inventory, findings=generated.findings, proposals=generated.proposals)
    review_id = secrets.token_hex(6)
    folder = _root(home) / review_id
    _write(folder / "packet.json", dumps(packet))
    _write(folder / "session.json", dumps(new_session(packet)))
    _write(folder / "viewer.html", viewer_html())
    why = _rationales(packet)
    return {
        "review_id": review_id,
        "folder": str(folder),
        "proposals": [{**_summary(p), "why": why[p["id"]]} for p in packet["proposals"]][
            :_MAX_LISTED
        ],
        "more": max(0, len(packet["proposals"]) - _MAX_LISTED),
        "flag_only_findings": len(packet["findings"]) - len(packet["proposals"]),
        "next": (
            "Show these to the user. They can approve by naming proposal ids (then call "
            "action='approve' with confirm_user_approved=true) or open viewer.html in a "
            "browser, import session.json, review, and export a plan for action='apply'."
        ),
    }


def _status(home: Path, review_id: Any) -> dict[str, Any]:
    if review_id is None:
        root = _root(home)
        folders = sorted(root.iterdir()) if root.is_dir() else []
        rows = [
            {"review_id": f.name, "has_plan": (f / "plan.json").exists()}
            for f in folders
            if _ID.match(f.name) and f.is_dir()
        ]
        return {"reviews": rows}
    folder = _folder(home, review_id)
    session = loads((folder / "session.json").read_bytes())
    decisions = {r["proposal_id"]: r["decision"] for r in session["reviews"]}
    return {
        "review_id": review_id,
        "has_plan": (folder / "plan.json").exists(),
        "proposals": [
            {**_summary(r["revisions"][-1]), "decision": decisions[r["proposal_id"]]}
            for r in session["reviews"]
        ],
    }


def _approve(home: Path, args: dict[str, Any]) -> dict[str, Any]:
    ids = args.get("proposal_ids")
    if args.get("confirm_user_approved") is not True:
        return {"error": "Set confirm_user_approved=true only after the user approved by name."}
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        return {"error": "Provide the exact proposal_ids the user approved."}
    folder = _folder(home, args.get("review_id"))
    session = loads((folder / "session.json").read_bytes())
    approved = approve_proposals(session, ids, observed_at=_now())
    plan = export_plan(approved)
    _write(folder / "session.json", dumps(approved))
    _write(folder / "plan.json", dumps(plan))
    return {
        "review_id": args["review_id"],
        "approved": [a["proposal"]["id"] for a in plan["approved"]],
        "not_exported": [
            {"proposal_id": e["proposal_id"], "status": e["status"], "reason": e["reason"]}
            for e in plan["ledger"]
            if e["decision"] == "accept"
        ],
        "next": "Call action='apply' (dry run first) with the same review_id.",
    }


def _plan_file(home: Path, args: dict[str, Any]) -> Path:
    if args.get("plan_path"):
        path = Path(str(args["plan_path"])).expanduser()
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("plan_path must be an absolute path to a regular file")
        return path
    return _folder(home, args.get("review_id")) / "plan.json"


def _apply(
    service: MemoryService, home: Path, store_kind: str, args: dict[str, Any]
) -> dict[str, Any]:
    plan = loads(_plan_file(home, args).read_bytes())
    boundary = SourceBoundary(home, (_label(store_kind),))
    profile = {"id": home.name, "root": str(home)}
    report = apply_plan(
        plan,
        service,
        boundary,
        profile,
        SourceSpec(store_kind),
        run=args.get("run") is True,
        allow_remove=args.get("allow_remove") is True,
    )
    if args.get("review_id") and _ID.match(str(args["review_id"])):
        folder = _folder(home, args["review_id"])
        _write(folder / "apply-report.json", canonical_json(report))
    return report


def handle_reconcile(
    service: MemoryService, home: Path, store_kind: str, primary: bool, args: dict[str, Any]
) -> str:
    """Dispatch one tool call; always returns a JSON string, never raises for bad input."""
    action = args.get("action")
    if action not in ("review", "status", "approve", "apply"):
        return json.dumps({"error": "action must be review, status, approve or apply"})
    if action != "status" and not primary:
        return json.dumps({"skipped": "non-primary agent context"})
    resolved = home.resolve()
    try:
        if action == "review":
            result = _review(service, resolved, store_kind)
        elif action == "status":
            result = _status(resolved, args.get("review_id"))
        elif action == "approve":
            result = _approve(resolved, args)
        else:
            result = _apply(service, resolved, store_kind, args)
    except (ValueError, OSError) as exc:
        result = {"error": str(exc)[:200]}
    return json.dumps(result)
