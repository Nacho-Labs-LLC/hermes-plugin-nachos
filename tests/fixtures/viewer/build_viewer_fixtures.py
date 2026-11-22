"""Deterministic synthetic viewer fixtures, built only with the public Python contract API.

    python tests/fixtures/viewer/build_viewer_fixtures.py           # (re)write fixtures
    python tests/fixtures/viewer/build_viewer_fixtures.py --check   # fail if bytes differ

All data is invented. No profile, store or network is touched; ``https://invalid.example``
URLs inside hostile text are inert canaries (a browser test asserts nothing ever fetches them).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nachos_core.reconciliation import (  # noqa: E402
    SCHEMAS,
    content_digest,
    dumps,
    export_plan,
    file_sha256,
    loads,
    memory_sha256,
    new_session,
    review_proposal,
    seal_document,
)
from nachos_core.reconciliation.validate import (  # noqa: E402
    dumps_validation,
    validate_plan_sources,
)

HERE = Path(__file__).resolve().parent
PROFILE = {"id": "synthetic-viewer", "root": "/inert/synthetic-profile"}
OTHER_PROFILE = {"id": "synthetic-other", "root": "/inert/other-profile"}
WHEN = "2026-11-16T12:00:00Z"
CHECKED_AT = "2026-11-16T13:00:00+00:00"
DB = "nachos/memory.db"
MD = "nachos/memory.md"

# Inert canaries: any execution sets window.__pwned; any fetch hits invalid.example.
HOSTILE = (
    '<img src="https://invalid.example/probe-img" onerror="window.__pwned=1">',
    "</script><script>window.__pwned=2</script>",
    '<a href="javascript:window.__pwned=3">x</a> &amp; &lt;b&gt; <!-- c -->',
    '"><svg onload=window.__pwned=4> {{7*7}} ${7*7} `x`',
)
# Astral (emoji, math script, CJK ext B), combining marks, RTL override and zero-width space.
UNICODE = " ".join(
    [
        "\U0001f600",
        "\U0001d49c",
        "\U00020bb7",
        "e" + chr(0x301),
        "a" + chr(0x308) + chr(0x323),
        chr(0x202E) + "rtl" + chr(0x202C),
        chr(0x200B) + "z",
    ]
)


PLAIN = []  # non-empty while building the large fixture (keeps it small)


def hot(label, n=0):
    """Hostile + astral + combining text, tagged so tests can find where it renders."""
    if PLAIN:
        return label
    return f"{label} {HOSTILE[n % len(HOSTILE)]} {UNICODE}"


def coverage(stage=0, note="synthetic"):
    """Stage 0 discovered+hashed; 1 +content search; 2 +section review; 3 +body review."""
    return {
        "discovered": True,
        "hashed": True,
        "content_searched": stage >= 1,
        "section_reviewed": stage >= 2,
        "body_reviewed": stage >= 3,
        "limitations": [f"Stage {stage}: {note}", hot("coverage limitation", stage)],
    }


def make_source(sid, kind, locator, before, **kw):
    snap = json.dumps(before, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    snap = snap if kind == "memory_record" else before
    enabled, usage, stage = kw.get("enabled"), kw.get("usage", ("unknown", [])), kw.get("stage", 0)
    return {
        "id": sid,
        "kind": kind,
        "role": kw.get("role", "primary"),
        "identity": kw.get("identity", f"{kind}:{sid}"),
        "locator": locator,
        "sha256": memory_sha256(before)
        if kind == "memory_record"
        else file_sha256(before.encode("utf-8")),
        "before": before,
        "coverage": coverage(stage),
        "metadata": {
            "byte_size": len(snap.encode("utf-8")),
            "char_size": len(snap),
            "enabled": enabled,
            "provenance": kw.get("provenance", []),
            "usage": {"status": usage[0], "references": list(usage[1])},
        },
    }


def memory(key, n=0, plain=False, **overrides):
    if plain:
        return {
            "key": key,
            "title": f"Bulk {key}",
            "summary": f"summary {key}",
            "category": "bulk",
            "body": f"Decision: keep {key} offline.\nNote {n}.",
        }
    record = {
        "key": key,
        "title": hot(f"title {key}", n),
        "summary": hot(f"summary {key}", n + 1),
        "category": hot(f"category {key}", n + 2),
        "body": f"Decision: keep {key} offline.\nPreference: concise reports.\n"
        + hot("body", n + 3),
    }
    record.update(overrides)
    return record


def obs(kind, status, sids, claim, observation=None):
    return {
        "kind": kind,
        "status": status,
        "source_ids": list(sids),
        "claim": hot(f"{kind}/{status} claim", len(claim)) if claim == "hot" else claim,
        "observation": observation or hot(f"{kind} observation", len(kind)),
        "observed_at": WHEN,
    }


def make_finding(fid, ftype, excerpts, evidence, **kw):
    sids = list(dict.fromkeys(sid for sid, _ in excerpts))
    return {
        "id": fid,
        "type": ftype,
        "source_ids": sids,
        "excerpts": [{"source_id": sid, "text": text} for sid, text in excerpts],
        "rationale": kw.get("rationale", hot(f"{ftype} rationale", len(fid))),
        "evidence": evidence,
        "limitations": kw.get("limitations", [hot("finding limitation", 1)]),
    }


def checklist(*preserved):
    return [
        {"item": hot(f"preserve item {i}", i), "preserved": ok, "explanation": hot("because", i)}
        for i, ok in enumerate(preserved)
    ]


def make_proposal(pid, findings, changes, **kw):
    """changes: [(source, after, operation)]; assessments default to fully preserved."""
    fids = [f["id"] for f in findings]
    assessments = kw.get("assessments")
    if assessments is None:
        assessments = [
            {
                "finding_id": f["id"],
                "revision": 1,
                "evidence": [obs("manual", "verified", f["source_ids"], "scope retained")],
                "preservation_checklist": checklist(True, True),
            }
            for f in findings
        ]
    return {
        "id": pid,
        "revision": 1,
        "finding_ids": fids,
        "depends_on": kw.get("depends_on", []),
        "assessments": assessments,
        "changes": [
            {
                "source_id": src["id"],
                "source_sha256": src["sha256"],
                "operation": op,
                "before": copy.deepcopy(src["before"]),
                "after": after,
            }
            for src, after, op in changes
        ],
    }


def inventory_of(sources, profile=PROFILE):
    return seal_document(
        {
            "schema": SCHEMAS["inventory"],
            "profile": profile,
            "sources": sources,
            "coverage": coverage(0, "inventory"),
        }
    )


def packet_of(sources, findings, proposals, profile=PROFILE):
    return seal_document(
        {
            "schema": SCHEMAS["packet"],
            "profile": profile,
            "inventory": inventory_of(sources, profile),
            "findings": findings,
            "proposals": proposals,
        }
    )


SKILL = (
    "---\nname: synthetic-cleanup\ndescription: Synthetic viewer fixture only.\n---\n"
    "# Guardrails\nDecision: retain the offline export-only boundary.\n"
    "Permission: staging cleanup only after explicit per-run approval; never production.\n\n"
    "## Reporting\nWrite a short report.\n\n"
    "## Cleanup\nRemove only the named temporary files.\n\n"
    "## History\nHistory: 2025-03-04 staging cleanup failed; retained for audit.\n"
)
CRLF_REF = "Reference notes\r\nline two with trailing space \r\nline three\r\n\r\nlast line no EOL"


def _mixed_sources():
    s = {}
    s["mem1"] = make_source(
        "s-mem-1",
        "memory_record",
        DB,
        memory("mixed-authority", 0),
        identity="memory:mixed-authority",
        enabled=True,
        usage=("available", ["skills/synthetic-cleanup/SKILL.md", hot("reference", 1)]),
        provenance=["synthetic fixture", hot("provenance", 2)],
        stage=1,
    )
    s["mem2"] = make_source(
        "s-mem-2",
        "memory_record",
        DB,
        memory("duplicate-a", 1),
        identity="memory:duplicate-a",
        enabled=False,
        usage=("unavailable", []),
        stage=0,
    )
    s["mem3"] = make_source(
        "s-mem-3",
        "memory_record",
        DB,
        memory("duplicate-b", 2),
        identity="memory:duplicate-b",
        usage=("unknown", []),
        stage=2,
    )
    s["skill"] = make_source(
        "s-skill",
        "skill_file",
        "skills/synthetic-cleanup/SKILL.md",
        SKILL,
        identity="skill:skills/synthetic-cleanup/SKILL.md",
        enabled=True,
        usage=("available", ["cron:nightly-synthetic"]),
        stage=2,
    )
    s["legacy"] = make_source(
        "s-legacy",
        "skill_file",
        "skills/synthetic-legacy/SKILL.md",
        SKILL.replace("synthetic-cleanup", "synthetic-legacy"),
        identity="skill:skills/synthetic-legacy/SKILL.md",
        enabled=False,
        usage=("unavailable", []),
        stage=3,
    )
    s["ref"] = make_source(
        "s-ref-crlf",
        "reference_file",
        "references/crlf-notes.md",
        CRLF_REF,
        role="reference",
        identity="reference:references/crlf-notes.md",
        stage=3,
    )
    s["empty"] = make_source(
        "s-ref-empty",
        "reference_file",
        "references/empty.md",
        "",
        role="reference",
        identity="reference:references/empty.md",
    )
    s["memfile"] = make_source(
        "s-memfile",
        "memory_file",
        MD,
        "# ops\n\n## Beta Note\nbeta line\n" + hot("flatfile", 3) + "\n",
        identity="memory:beta-note",
        enabled=None,
    )
    s["hostile"] = make_source(
        "s-hostile-<b>1</b>",
        "skill_file",
        "skills/<img src=x onerror=window.__pwned=5>/SKILL.md",
        "# Hostile locator\n" + hot("text", 0) + "\n",
        identity=hot("identity", 2),
        provenance=[hot("provenance", 0)],
    )
    for n in (1, 2, 3, 4, 5):
        s[f"x{n}"] = make_source(
            f"s-extra-{n}",
            "memory_record",
            DB,
            memory(f"extra-{n}", n),
            identity=f"memory:extra-{n}",
        )
    return s


def _edit_memory(src, **fields):
    return {**src["before"], **fields}


def _mixed_findings(s):
    mem1, mem2, mem3 = (s[k]["before"] for k in ("mem1", "mem2", "mem3"))
    pref = "Preference: concise reports."
    f = {}
    f["clar"] = make_finding(
        "f-clarification",
        "clarification",
        [("s-mem-1", pref)],
        [obs("source_content", "verified", ["s-mem-1"], "excerpt is present")],
    )
    f["section"] = make_finding(
        "f-skill-section",
        "clarification",
        [("s-skill", "Write a short report.")],
        [obs("manual", "verified", ["s-skill"], "reporting is vague")],
    )
    f["stale"] = make_finding(
        "f-stale-legacy",
        "stale",
        [("s-legacy", "History: 2025-03-04 staging cleanup failed; retained for audit.")],
        [
            obs("schedule", "verified", ["s-legacy"], "A cron entry names this skill."),
            obs("execution", "missing", ["s-legacy"], "No execution record was supplied."),
            obs("delivery", "unverified", ["s-legacy"], "Delivery was not inspected."),
            obs("usage", "conflicting", ["s-legacy"], "Two usage sources disagree."),
        ],
        limitations=["A schedule is not proof of execution or delivery."],
    )
    f["dup"] = make_finding(
        "f-duplicate",
        "duplicate",
        [("s-mem-2", mem2["body"].splitlines()[0]), ("s-mem-3", mem3["body"].splitlines()[0])],
        [obs("source_content", "unverified", ["s-mem-2", "s-mem-3"], "texts are near-identical")],
    )
    f["conflict"] = make_finding(
        "f-conflict",
        "conflict",
        [
            ("s-mem-1", mem1["body"].splitlines()[0]),
            ("s-extra-5", "Decision: keep extra-5 offline."),
        ],
        [
            obs("manual", "conflicting", ["s-mem-1", "s-extra-5"], "hot"),
            obs("usage", "missing", ["s-mem-1"], "No usage data."),
        ],
    )
    f["unsupported"] = make_finding(
        "f-unsupported",
        "unsupported",
        [("s-ref-crlf", "line two with trailing space")],
        [obs("delivery", "missing", ["s-ref-crlf"], "No delivery evidence for the claim.")],
    )
    f["memfile"] = make_finding(
        "f-memfile",
        "clarification",
        [("s-memfile", "beta line")],
        [obs("manual", "verified", ["s-memfile"], "ok")],
    )
    f["hostile"] = make_finding(
        "f-hostile-<script>",
        "stale",
        [("s-hostile-<b>1</b>", "# Hostile locator")],
        [obs("source_content", "verified", ["s-hostile-<b>1</b>"], "hot")],
        rationale=hot("rationale", 1),
    )
    f["x1"] = make_finding(
        "f-x1",
        "clarification",
        [("s-extra-1", "Decision: keep extra-1 offline.")],
        [obs("manual", "verified", ["s-extra-1"], "ok")],
    )
    f["x2"] = make_finding(
        "f-x2",
        "clarification",
        [("s-extra-2", "Decision: keep extra-2 offline.")],
        [obs("manual", "verified", ["s-extra-2"], "ok")],
    )
    f["x3"] = make_finding(
        "f-x3",
        "clarification",
        [("s-extra-3", "Decision: keep extra-3 offline.")],
        [obs("manual", "verified", ["s-extra-3"], "ok")],
    )
    f["x4"] = make_finding(
        "f-x4",
        "clarification",
        [("s-extra-4", "Decision: keep extra-4 offline.")],
        [obs("manual", "verified", ["s-extra-4"], "ok")],
    )
    return f


def _mixed_proposals(s, f):
    mem1, mem2, mem3 = s["mem1"], s["mem2"], s["mem3"]
    p = []
    p.append(
        make_proposal(
            "p-memory",
            [f["clar"]],
            [
                (
                    mem1,
                    _edit_memory(
                        mem1,
                        title=mem1["before"]["title"] + " (edited)",
                        summary="Edited summary \U0001d49c",
                        category=mem1["before"]["category"] + "/edited",
                        body=mem1["before"]["body"].replace(
                            "Preference: concise reports.",
                            "Preference: concise reports; list verification gaps.",
                        ),
                    ),
                    "edit",
                )
            ],
        )
    )
    p.append(
        make_proposal(
            "p-skill-section",
            [f["section"]],
            [
                (
                    s["skill"],
                    SKILL.replace(
                        "Write a short report.",
                        "Write a short report with evidence and remaining gaps.",
                    ),
                    "edit",
                )
            ],
        )
    )
    p.append(make_proposal("p-disable", [f["stale"]], [(s["legacy"], None, "disable")]))
    p.append(
        make_proposal(
            "p-multi",
            [f["dup"]],
            [
                (mem2, _edit_memory(mem2, summary="Merged duplicate A"), "edit"),
                (mem3, _edit_memory(mem3, summary="Merged duplicate B"), "edit"),
            ],
        )
    )
    p.append(
        make_proposal(
            "p-dep-base",
            [f["unsupported"]],
            [(s["ref"], CRLF_REF.replace("line three", "line three edited"), "edit")],
        )
    )
    p.append(
        make_proposal(
            "p-dep-child",
            [f["memfile"]],
            [(s["memfile"], s["memfile"]["before"].replace("beta line", "beta line 2"), "edit")],
            depends_on=["p-dep-base"],
        )
    )
    p.append(
        make_proposal(
            "p-dep-missing",
            [f["x1"]],
            [(s["x1"], _edit_memory(s["x1"], summary="needs absent dependency"), "edit")],
            depends_on=["p-does-not-exist"],
        )
    )
    p.append(
        make_proposal(
            "p-cycle-a",
            [f["x2"]],
            [(s["x2"], _edit_memory(s["x2"], summary="cycle a"), "edit")],
            depends_on=["p-cycle-b"],
        )
    )
    p.append(
        make_proposal(
            "p-cycle-b",
            [f["x3"]],
            [(s["x3"], _edit_memory(s["x3"], summary="cycle b"), "edit")],
            depends_on=["p-cycle-a"],
        )
    )
    p.extend(
        make_proposal(
            f"p-conflict-{tag}",
            [f["conflict"]],
            [(s["x5"], _edit_memory(s["x5"], summary=f"conflicting summary {tag}"), "edit")],
        )
        for tag in ("a", "b")
    )
    p.append(
        make_proposal(
            "p-false-preservation",
            [f["x4"]],
            [(s["x4"], _edit_memory(s["x4"], body="Decision removed."), "edit")],
            assessments=[
                {
                    "finding_id": "f-x4",
                    "revision": 1,
                    "evidence": [obs("manual", "verified", ["s-extra-4"], "decision dropped")],
                    "preservation_checklist": checklist(True, False),
                }
            ],
        )
    )
    p.append(
        make_proposal(
            "p-unassessed",
            [f["hostile"]],
            [(s["hostile"], s["hostile"]["before"] + "appended\n", "edit")],
            assessments=[],
        )
    )
    return p


def build_mixed():
    s = _mixed_sources()
    f = _mixed_findings(s)
    sources = list(s.values())
    findings = list(f.values())
    return packet_of(sources, findings, _mixed_proposals(s, f))


def build_empty():
    return packet_of([], [], [])


def build_reviewed(packet):
    """A real lifecycle: accept/edit/reaccept, request-revision, reject, defer, blocked accepts."""
    session = new_session(packet)
    initial = copy.deepcopy(session)
    note = "synthetic review note"
    for pid in ("p-memory", "p-dep-base", "p-dep-child", "p-multi", "p-skill-section"):
        session = review_proposal(session, pid, "accept", note=note)
    edited = copy.deepcopy(session["reviews"][0]["revisions"][-1])
    edited["revision"] = 2
    edited["changes"][0]["after"]["summary"] += "; edited"
    for assessment in edited["assessments"]:
        assessment["revision"] = 2
    session = review_proposal(session, "p-memory", "edit", edited_proposal=edited, note=note)
    session = review_proposal(session, "p-memory", "accept", note="reaccepted revision 2")
    session = review_proposal(session, "p-skill-section", "request_revision", note=note)
    session = review_proposal(session, "p-disable", "reject", note=note)
    session = review_proposal(session, "p-dep-missing", "accept", note="accepted with absent dep")
    session = review_proposal(session, "p-cycle-a", "accept", note=note)
    session = review_proposal(session, "p-cycle-b", "accept", note=note)
    for tag in ("a", "b"):
        session = review_proposal(session, f"p-conflict-{tag}", "accept", note="same source")
    session = review_proposal(session, "p-unassessed", "defer", note=note)
    session = loads(dumps(session))
    return initial, session


def _large_text(lines, step_tag):
    return "".join(f"L{i:05d} {step_tag} é\U0001f600 pad\n" for i in range(lines))


def build_large():
    PLAIN.append(True)
    try:
        return _build_large()
    finally:
        PLAIN.clear()


def _build_large():
    sources, findings, proposals = [], [], []
    for n in range(300):
        key = f"bulk-{n:03d}"
        src = make_source(
            f"s-{key}",
            "memory_record",
            DB,
            memory(key, n, plain=True),
            identity=f"memory:{key}",
            stage=n % 4,
        )
        fnd = make_finding(
            f"f-{key}",
            ("stale", "duplicate", "conflict", "unsupported", "clarification")[n % 5],
            [(src["id"], f"Decision: keep {key} offline.")],
            [
                obs(
                    "manual",
                    ("verified", "unverified", "missing", "conflicting")[n % 4],
                    [src["id"]],
                    "bulk claim",
                )
            ],
        )
        after = _edit_memory(src, summary=f"bulk summary {n}")
        deps = [f"p-{n - 1:03d}"] if n % 10 == 5 else []
        proposals.append(
            make_proposal(f"p-{n:03d}", [fnd], [(src, after, "edit")], depends_on=deps)
        )
        sources.append(src)
        findings.append(fnd)
    for name, lines, every in (("big-sparse", 2600, 0), ("big-dense", 2200, 7)):
        text = _large_text(lines, name)
        src = make_source(
            f"s-{name}",
            "skill_file",
            f"skills/{name}/SKILL.md",
            text,
            identity=f"skill:skills/{name}/SKILL.md",
        )
        fnd = make_finding(
            f"f-{name}",
            "stale",
            [(src["id"], "L00010 ")],
            [obs("source_content", "verified", [src["id"]], "large source")],
        )
        rows = text.splitlines(keepends=True)
        if every:
            rows = [
                r.replace("pad", "changed") if i % every == 0 else r for i, r in enumerate(rows)
            ]
        else:
            rows[1500] = "L01500 replaced single line\n"
        proposals.append(make_proposal(f"p-{name}", [fnd], [(src, "".join(rows), "edit")]))
        sources.append(src)
        findings.append(fnd)
    return packet_of(sources, findings, proposals)


def _plan_of(session):
    return export_plan(session)


def _fake_reader(mutations):
    def reader(_boundary, _spec):
        return {k: SimpleNamespace(identity=k, **v) for k, v in mutations.items() if v is not None}

    return reader


def build_validation_reports(plan):
    """Unsealed validation.v1 results from the real validator with an injected fake reader."""
    spec = SimpleNamespace(
        store="sqlite",
        skills=("skills/synthetic-cleanup/SKILL.md", "skills/synthetic-legacy/SKILL.md"),
        references=("references/crlf-notes.md", MD),
    )
    boundary = SimpleNamespace(root=PROFILE["root"])
    rows = {}
    for entry in plan["approved"]:
        for src in entry["sources"]:
            rows[src["identity"]] = {
                "kind": src["kind"],
                "locator": src["locator"],
                "sha256": src["sha256"],
            }
    from nachos_core.reconciliation.boundary import AcquisitionError  # noqa: PLC0415

    def run(reader):
        return validate_plan_sources(
            plan,
            boundary=boundary,
            profile=dict(PROFILE),
            spec=spec,
            reader=reader,
            checked_at=CHECKED_AT,
        )

    stale = copy.deepcopy(rows)
    first, second, *_ = sorted(stale)
    stale[first]["sha256"] = "0" * 64
    stale[second] = None

    def failing(_boundary, _spec):
        raise AcquisitionError("source_unavailable")

    return {
        "validation-valid.json": run(_fake_reader(rows)),
        "validation-stale.json": run(_fake_reader(stale)),
        "validation-unavailable.json": run(failing),
    }


def _text(doc):
    return dumps(doc).decode("utf-8")


def build_errors(packet, session, plan):
    """(filename, bytes, manifest-entry); every entry carries an expected-rejection label."""
    good = _text(packet)
    out = []

    def add(name, raw, category, **extra):
        out.append((name, raw if isinstance(raw, bytes) else raw.encode("utf-8"), category, extra))

    tampered = good.replace("Decision: keep mixed-authority offline.", "Decision: keep OFFLINE.", 1)
    assert tampered != good
    add("tampered-digest.json", tampered, "digest_mismatch")
    dig = packet["digest"]
    add(
        "duplicate-key.json",
        good.replace(f'{{"digest":"{dig}",', f'{{"digest":"{dig}","digest":"{dig}",', 1),
        "duplicate_key",
    )
    add("float.json", good.replace('"revision":1', '"revision":1.0', 1), "float")
    add("exponent-number.json", good.replace('"revision":1', '"revision":1e0', 1), "float")
    add("nan.json", good.replace('"revision":1', '"revision":NaN', 1), "nan")
    add(
        "big-integer.json",
        good.replace('"revision":1', '"revision":9007199254740992', 1),
        "integer_range",
    )
    add("lone-surrogate.json", good.replace("Decision:", "\\ud800 Decision:", 1), "lone_surrogate")
    add("truncated.json", good[: len(good) // 2], "truncated")
    add("empty.json", b"", "empty")
    add("not-json.json", "this is not json", "not_json")
    add("bom-prefixed.json", b"\xef\xbb\xbf" + good.encode("utf-8"), "bom")
    raw = good.encode("utf-8")
    add("invalid-utf8.bin", raw.replace(b"Decision:", b"Decision:\xff\xfe", 1), "invalid_utf8")
    nest = 200
    add("huge-nesting.json", '{"schema":"x","a":' + "[" * nest + "]" * nest + "}", "depth_limit")
    add("unknown-schema.json", good.replace("packet.v1", "packet.v2", 1), "unsupported_schema")
    doc = json.loads(good)
    doc["unexpected"] = 1
    doc.pop("digest")
    doc["digest"] = content_digest("packet", doc)
    add("extra-key-resealed.json", json.dumps(doc, separators=(",", ":")), "extra_key")
    add(
        "wrong-profile-session.json",
        _text(new_session(packet_of([], [], [], OTHER_PROFILE))),
        "profile_mismatch",
        valid_standalone=True,
        expected_profile=PROFILE,
    )
    add(
        "plan-as-session.json",
        _text(plan),
        "wrong_artifact_kind",
        valid_standalone=True,
        expected_kind="session",
    )
    add(
        "session-as-packet.json",
        _text(session),
        "wrong_artifact_kind",
        valid_standalone=True,
        expected_kind="packet",
    )
    return out


def build_all():
    """Return {relative path: bytes} for every fixture."""
    packet = build_mixed()
    initial, reviewed = build_reviewed(packet)
    plan = _plan_of(reviewed)
    assert plan["approved"]
    assert plan["ledger"]
    files = {
        "packet-mixed.json": dumps(packet),
        "session-mixed-initial.json": dumps(initial),
        "session-mixed-reviewed.json": dumps(reviewed),
        "plan-mixed.json": dumps(plan),
        "packet-empty.json": dumps(build_empty()),
        "packet-large.json": dumps(build_large()),
    }
    for name, report in build_validation_reports(plan).items():
        files[name] = dumps_validation(report)
    manifest = {"profile": PROFILE, "errors": []}
    for name, raw, category, extra in build_errors(packet, reviewed, plan):
        files[f"errors/{name}"] = raw
        manifest["errors"].append(
            {"file": name, "category": category, "expected": "reject", **extra}
        )
    files["errors/manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    return {
        k: v if k.startswith("errors/") and k != "errors/manifest.json" else v + b"\n"
        for k, v in files.items()
    }


def main(argv=None):
    check = "--check" in (argv if argv is not None else sys.argv[1:])
    files = build_all()
    bad = []
    for rel, raw in sorted(files.items()):
        path = HERE / rel
        if check:
            if not path.is_file() or path.read_bytes() != raw:
                bad.append(rel)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    if check:
        extra = sorted(
            str(p.relative_to(HERE))
            for p in HERE.rglob("*.json")
            if str(p.relative_to(HERE)) not in files
        )
        if bad or extra:
            print(f"viewer fixtures out of date: changed={bad} unexpected={extra}", file=sys.stderr)
            return 1
        print(f"viewer fixtures up to date ({len(files)} files)")
        return 0
    print(f"wrote {len(files)} fixtures to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
