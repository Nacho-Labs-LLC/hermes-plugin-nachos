"""Record an explicit human approval made outside the viewer (for example in chat).

The viewer is the careful path: a reviewer verifies each finding's evidence and
preservation checklist. This helper is for a user who has read the proposal summaries
and approves them by name. It writes the approval as what it is: ``manual`` evidence,
``verified`` only in the sense that the user said so, with a checklist item stating
that the approval came from the user's own instruction. It cannot be derived from
anything the agent wrote; callers must pass the proposal ids the user named.
"""

from __future__ import annotations

from typing import Any

from .models import timestamp
from .review import review_proposal

NOTE = "Approved by the user's explicit instruction outside the viewer."


def _assessments(
    packet: dict[str, Any], proposal: dict[str, Any], observed_at: str
) -> list[dict[str, Any]]:
    findings = {f["id"]: f for f in packet["findings"]}
    return [
        {
            "finding_id": fid,
            "revision": proposal["revision"] + 1,
            "evidence": [
                {
                    "kind": "manual",
                    "status": "verified",
                    "source_ids": findings[fid]["source_ids"],
                    "claim": "The user approved this proposal.",
                    "observation": NOTE,
                    "observed_at": observed_at,
                }
            ],
            "preservation_checklist": [
                {
                    "item": "User reviewed the proposal summary and approved it by name",
                    "preserved": True,
                    "explanation": NOTE,
                }
            ],
        }
        for fid in proposal["finding_ids"]
    ]


def approve_proposals(
    session: dict[str, Any], proposal_ids: list[str], *, observed_at: str
) -> dict[str, Any]:
    """Return a new sealed session in which the named proposals are accepted."""
    timestamp(observed_at)
    result = session
    by_id = {r["proposal_id"]: r for r in session["reviews"]}
    for pid in proposal_ids:
        if pid not in by_id:
            raise ValueError("unknown proposal id")
        current = by_id[pid]["revisions"][-1]
        edited = {
            **current,
            "revision": current["revision"] + 1,
            "assessments": _assessments(session["packet"], current, observed_at),
        }
        result = review_proposal(result, pid, "edit", edited_proposal=edited, note=NOTE)
        result = review_proposal(result, pid, "accept", note=NOTE)
    return result
