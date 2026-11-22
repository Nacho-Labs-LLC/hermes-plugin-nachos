"""Synthetic offline interoperability vectors; run from the checkout root.

Only this fixture directory is written. No source/profile discovery or live data.
"""

import copy
import json
from pathlib import Path

from nachos_core.reconciliation import (
    SCHEMAS,
    canonical_json,
    content_digest,
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

HERE = Path(__file__).resolve().parent
PROFILE = {"id": "synthetic-review", "root": "/inert/synthetic-profile"}
COVERAGE = {
    "discovered": True,
    "hashed": True,
    "content_searched": False,
    "section_reviewed": False,
    "body_reviewed": False,
    "limitations": ["Synthetic snapshots only; no live inspection or execution telemetry."],
}
DECISION = "Decision: retain the offline export-only boundary."
PERMISSION = "Permission: staging cleanup only after explicit per-run approval; never production."
HISTORY = (
    "History: 2025-03-04 staging cleanup failed; retained for audit, not a current instruction."
)
HOSTILE = '<img src="https://invalid.example/probe" onerror="window.fixtureExecuted=true"> </script> & café 😀'


def source(sid, kind, locator, before):
    snapshot = canonical_json(before).decode("utf-8") if kind == "memory_record" else before
    return {
        "id": sid,
        "kind": kind,
        "role": "primary",
        "metadata": {
            "byte_size": len(snapshot.encode("utf-8")),
            "char_size": len(snapshot),
            "enabled": None,
            "provenance": ["synthetic fixture"],
            "usage": {"status": "unknown", "references": []},
        },
        "identity": "synthetic:" + sid,
        "locator": locator,
        "before": before,
        "coverage": copy.deepcopy(COVERAGE),
        "sha256": memory_sha256(before)
        if kind == "memory_record"
        else file_sha256(before.encode("utf-8")),
    }


def assessment(fid, src, revision):
    return {
        "finding_id": fid,
        "revision": revision,
        "evidence": [
            {
                "kind": "manual",
                "status": "verified",
                "source_ids": [src["id"]],
                "claim": "Authority clauses are retained verbatim in this revision.",
                "observation": "Synthetic fixture author compared before and after text.",
                "observed_at": "2026-11-16T12:00:00Z",
            }
        ],
        "preservation_checklist": [
            {"item": name, "preserved": True, "explanation": explanation}
            for name, explanation in [
                ("Decision", "Offline export-only decision remains verbatim."),
                (
                    "Scoped permission",
                    "Staging-only, per-run approval and production prohibition remain verbatim.",
                ),
                (
                    "Historical evidence",
                    "Dated failure remains verbatim and explicitly historical.",
                ),
            ]
        ],
    }


def finding(fid, src, excerpt, rationale):
    return {
        "id": fid,
        "type": "clarification",
        "source_ids": [src["id"]],
        "excerpts": [{"source_id": src["id"], "text": excerpt}],
        "rationale": rationale,
        "evidence": [
            {
                "kind": "source_content",
                "status": "verified",
                "source_ids": [src["id"]],
                "claim": "The excerpt appears in the supplied snapshot.",
                "observation": "Synthetic fixture author inspected the complete supplied snapshot.",
                "observed_at": "2026-11-16T12:00:00Z",
            }
        ],
        "limitations": ["Not live evidence; preservation flags are claims requiring human review."],
    }


def proposal(pid, src, fid, after, operation="edit"):
    return {
        "id": pid,
        "revision": 1,
        "finding_ids": [fid],
        "depends_on": [],
        "assessments": [assessment(fid, src, 1)],
        "changes": [
            {
                "source_id": src["id"],
                "source_sha256": src["sha256"],
                "operation": operation,
                "before": copy.deepcopy(src["before"]),
                "after": after,
            }
        ],
    }


def write(name, doc):
    raw = dumps(doc)
    assert loads(raw, expected_profile=PROFILE) == doc
    (HERE / name).write_bytes(raw + b"\n")


def main():
    body = (
        "\n".join([DECISION, PERMISSION, HISTORY, "Preference: concise reports.", HOSTILE]) + "\n"
    )
    record = {
        "key": "mixed-authority",
        "title": "Mixed authority / café 😀",
        "summary": "Decision, scoped permission, history and preference",
        "category": "mixed",
        "body": body,
    }
    memory = source("s-memory", "memory_record", "memory/index.sqlite3", record)
    skill_text = (
        "---\nname: synthetic-cleanup\ndescription: Synthetic review fixture only.\n---\n# Guardrails\n"
        + "\n".join([DECISION, PERMISSION, HISTORY])
        + "\n\n## Reporting\nWrite a short report.\n"
    )
    skill = source("s-skill", "skill_file", "skills/synthetic-cleanup/SKILL.md", skill_text)
    disabled = source(
        "s-disable",
        "skill_file",
        "skills/synthetic-legacy/SKILL.md",
        skill_text.replace("synthetic-cleanup", "synthetic-legacy"),
    )
    sources = [memory, skill, disabled]
    findings = [
        finding(
            "f-memory-clarity",
            memory,
            "Preference: concise reports.",
            "Clarify preference without weakening decision, permission or historical evidence.",
        ),
        finding(
            "f-skill-section",
            skill,
            "Write a short report.",
            "Make the reporting section precise; preserve the entire guardrail section.",
        ),
        finding(
            "f-disable-review",
            disabled,
            DECISION,
            "Propose disabling a redundant synthetic skill; keep full source available for review.",
        ),
    ]
    after_record = {
        **record,
        "body": body.replace(
            "Preference: concise reports.",
            "Preference: concise reports; include verification gaps.",
        ),
    }
    proposals = [
        proposal("p-memory", memory, findings[0]["id"], after_record),
        proposal(
            "p-skill",
            skill,
            findings[1]["id"],
            skill_text.replace(
                "Write a short report.",
                "Write a short report with verification evidence and remaining gaps.",
            ),
        ),
        proposal("p-disable", disabled, findings[2]["id"], None, "disable"),
    ]
    inventory = seal_document(
        {
            "schema": SCHEMAS["inventory"],
            "profile": PROFILE,
            "sources": sources,
            "coverage": copy.deepcopy(COVERAGE),
        }
    )
    packet = seal_document(
        {
            "schema": SCHEMAS["packet"],
            "profile": PROFILE,
            "inventory": inventory,
            "findings": findings,
            "proposals": proposals,
        }
    )
    session = new_session(packet)
    write("packet.json", packet)
    write("session-initial.json", session)
    session = review_proposal(
        session, "p-memory", "accept", note="Checked all five fields and authority clauses."
    )
    accepted_digest = session["reviews"][0]["accepted_digest"]
    edited = copy.deepcopy(session["reviews"][0]["revisions"][-1])
    edited["revision"] = 2
    edited["changes"][0]["after"]["summary"] += "; preserve authority"
    edited["assessments"] = [assessment("f-memory-clarity", memory, 2)]
    session = review_proposal(
        session,
        "p-memory",
        "edit",
        edited_proposal=edited,
        note="Summary clarified; approval revoked.",
    )
    assert session["reviews"][0]["accepted_digest"] is None
    write("session-edit-invalidated.json", session)
    session = review_proposal(
        session, "p-memory", "accept", note="Explicitly reaccepted revision 2."
    )
    assert session["reviews"][0]["accepted_digest"] != accepted_digest
    session = review_proposal(
        session, "p-skill", "request_revision", note="Recheck preservation before acceptance."
    )
    write("session-revision-requested.json", session)
    session = review_proposal(
        session, "p-skill", "reject", note="Retain original skill for this review."
    )
    session = review_proposal(
        session, "p-disable", "defer", note="Proposal only; no skill state changed."
    )
    session = loads(dumps(session))
    write("session-reviewed.json", session)
    plan = export_plan(session)
    assert [x["proposal"]["id"] for x in plan["approved"]] == ["p-memory"]
    assert {x["proposal_id"]: x["status"] for x in plan["ledger"]} == {
        "p-disable": "deferred",
        "p-skill": "rejected",
    }
    write("plan.json", plan)
    disable_session = review_proposal(
        new_session(packet),
        "p-disable",
        "accept",
        note="Accept proposal only, not permission to execute.",
    )
    write("plan-disable-proposal-only.json", export_plan(disable_session))
    empty_inventory = seal_document(
        {
            "schema": SCHEMAS["inventory"],
            "profile": PROFILE,
            "sources": [],
            "coverage": copy.deepcopy(COVERAGE),
        }
    )
    write(
        "packet-empty.json",
        seal_document(
            {
                "schema": SCHEMAS["packet"],
                "profile": PROFILE,
                "inventory": empty_inventory,
                "findings": [],
                "proposals": [],
            }
        ),
    )
    values = [
        {"z": "é\r\n", "a": 1},
        {"😀": "astral", "\ue000": "BMP", "2": "two", "10": "ten"},
        {
            "control": "\b\f\n\r\t\u0000",
            "slash": "/",
            "html": HOSTILE,
            "combining": "e\u0301",
            "composed": "é",
        },
        [None, True, False, -9007199254740991, 9007199254740991],
    ]
    vectors = {
        "canonical": [
            {
                "value": v,
                "utf8_hex": canonical_json(v).hex(),
                "proposal_domain_digest": content_digest("proposal", v),
            }
            for v in values
        ],
        "proposal_digests": {p["id"]: proposal_digest(packet, p) for p in proposals},
        "invalid_json": [
            '{"a":1,"a":2}',
            '{"n":1.0}',
            '{"n":1e0}',
            '{"n":NaN}',
            '{"n":9007199254740992}',
            '{"s":"\\ud800"}',
        ],
    }
    (HERE / "hash-vectors.json").write_text(
        json.dumps(vectors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    names = sorted(p.name for p in HERE.glob("*.json"))
    print(
        json.dumps(
            {
                "validated_artifacts": len(names) - 1,
                "files": names,
                "sources": len(sources),
                "proposals": len(proposals),
                "approved": len(plan["approved"]),
                "ledger": len(plan["ledger"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
