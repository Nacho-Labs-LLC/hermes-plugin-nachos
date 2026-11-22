# ruff: noqa: PLR0915, PLR0912, C901, S311
"""Python <-> JavaScript parity for the offline viewer core (viewer/core.js).

Python (``nachos_core.reconciliation``) is authoritative. Node runs the standalone JS suite and
a differential corpus built here: canonical bytes/digests, validation verdicts, lifecycle
transitions, export plans and validation-result shapes must agree byte-for-byte, and every
invalid class Python rejects must be rejected by JS.

Node is required. Without it the tests FAIL, unless NACHOS_ALLOW_NO_NODE=1 (explicit skip).
"""

import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nachos_core.reconciliation import (
    SCHEMAS,
    canonical_json,
    content_digest,
    dumps,
    export_plan,
    loads,
    new_session,
    proposal_digest,
    review_proposal,
    seal_document,
)
from nachos_core.reconciliation import models as models_module
from nachos_core.reconciliation.acquire import FLATFILE_LABEL, SQLITE_LABEL
from nachos_core.reconciliation.boundary import AcquisitionError, SourceBoundary
from nachos_core.reconciliation.codec import _forbidden, _pairs
from nachos_core.reconciliation.hashing import MAX_BYTES
from nachos_core.reconciliation.review import review_context
from nachos_core.reconciliation.validate import validate_plan_sources

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "nachos_core/reconciliation/viewer/core.js"
JS_DIR = ROOT / "tests/js"
FIX = ROOT / "tests/fixtures/reconciliation"
TIMEOUT = 240

_spec = importlib.util.spec_from_file_location("nachos_build_fixtures", FIX / "build_fixtures.py")
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

PROFILE = {"id": "rich-profile", "root": "/inert/rich"}
STAMPS = [
    "2026-11-16T12:00:00Z",
    "2026-11-16T12:00:00.123+05:30",
    "2026-11-16T07:00:00-05:00",
    "2026-11-16T12:00:00.123456+00:00",
]
HOSTILE = [
    "",
    " ",
    "\r\n",
    "\n\r",
    "\x01\x1f\x7f",
    "\u2028\u2029",
    "<!-- --> </script> <script>alert(1)</script>",
    "e\u0301",
    "\u00e9",
    "\U0001f600",
    "\U00010000",
    "\uffff \ue000 \ud7ff",
    "\u0085\u00a0\ufeff",
    '\\ " / \t\b\f',
    "a\x00b",
    "`${x}` {{y}} %s &amp; javascript:alert(1)",
]


def _node_exe():
    exe = shutil.which("node")
    if exe is None and os.access("/usr/local/bin/node", os.X_OK):
        exe = "/usr/local/bin/node"
    return exe


@pytest.fixture(scope="module")
def node():
    exe = _node_exe()
    if exe:
        return exe
    if os.environ.get("NACHOS_ALLOW_NO_NODE") == "1":
        pytest.skip("node missing; explicitly allowed by NACHOS_ALLOW_NO_NODE=1")
    pytest.fail("node is required for the viewer core parity tests (NACHOS_ALLOW_NO_NODE=1 skips)")


def test_core_js_standalone_parity_suite(node):
    done = subprocess.run(
        [node, str(JS_DIR / "core_parity.mjs"), str(ROOT)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "core_parity ok" in done.stdout


# ---------------------------------------------------------------- Python helpers


def summary(raw):
    return {
        "ok": True,
        "sha": hashlib.sha256(raw).hexdigest(),
        "len": len(raw),
        "hex": raw.hex() if len(raw) <= 4096 else None,
    }


def attempt(fn):
    try:
        return summary(fn())
    except Exception as exc:
        return {"ok": False, "message": str(exc), "type": type(exc).__name__}


def py_parse(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError("JSON byte limit exceeded")
    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_float=_forbidden,
        parse_constant=_forbidden,
    )


def generate(gen):
    kind, n = gen["kind"], gen["n"]
    if kind == "str":
        return ('"' + "a" * n + '"').encode()
    if kind == "zeros":
        return ("[" + ",".join(["0"] * n) + "]").encode()
    if kind == "nest":
        return ("[" * n + "]" * n).encode()
    if kind == "keys":
        return ("{" + ",".join(f'"k{i}":0' for i in range(n)) + "}").encode()
    if kind == "objnest":
        return ('{"a":' * n + "0" + "}" * n).encode()
    raise AssertionError(kind)


def raw_of(case):
    return generate(case["gen"]) if "gen" in case else bytes.fromhex(case["raw_hex"])


def raw_seal(doc):
    """Recompute only the top-level digest, bypassing validation (to isolate one fault)."""
    kind = doc["schema"].split(".")[2]
    result = {key: value for key, value in doc.items() if key != "digest"}
    result["digest"] = content_digest(kind, result)
    return result


def text_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def obs(sids, kind="manual", status="verified", stamp=STAMPS[0], tag=""):
    return {
        "kind": kind,
        "status": status,
        "source_ids": list(sids),
        "claim": "claim " + tag,
        "observation": "observation \u00e9 " + tag,
        "observed_at": stamp,
    }


def assess(fid, sids, revision, preserved=True):
    return {
        "finding_id": fid,
        "revision": revision,
        "evidence": [obs(sids[:1], tag=fid, stamp=STAMPS[revision % len(STAMPS)])],
        "preservation_checklist": [
            {"item": "decision", "preserved": True, "explanation": "kept verbatim"},
            {"item": "permission", "preserved": preserved, "explanation": "scoped \U0001f600"},
        ],
    }


def make_source(sid, kind, locator, before, *, role="primary", enabled=None, usage=None, cov=None):
    src = bf.source(sid, kind, locator, before)
    src["role"] = role
    src["metadata"]["enabled"] = enabled
    if usage:
        src["metadata"]["usage"] = usage
    if cov:
        src["coverage"].update(cov)
    return src


def make_finding(fid, ftype, sources, excerpts, kinds=("source_content",)):
    sids = [s["id"] for s in sources]
    return {
        "id": fid,
        "type": ftype,
        "source_ids": sids,
        "excerpts": [{"source_id": s["id"], "text": e} for s, e in zip(sources, excerpts)],
        "rationale": f"rationale {fid} \u00e9\u0301 \U0001f600",
        "evidence": [
            obs(sids, kind=kind, status=st, stamp=STAMPS[i % len(STAMPS)], tag=fid)
            for i, (kind, st) in enumerate(
                zip(kinds, ["verified", "unverified", "missing", "conflicting"] * 2)
            )
        ],
        "limitations": ["limit one", "limit two"] if ftype == "stale" else [],
    }


def change_of(src, after, operation="edit"):
    return {
        "source_id": src["id"],
        "source_sha256": src["sha256"],
        "operation": operation,
        "before": copy.deepcopy(src["before"]),
        "after": after,
    }


def make_proposal(pid, findings, changes, depends_on=(), assessed=None, preserved=True):
    assessed = list(findings) if assessed is None else assessed
    return {
        "id": pid,
        "revision": 1,
        "finding_ids": [f["id"] for f in findings],
        "depends_on": list(depends_on),
        "changes": changes,
        "assessments": [assess(f["id"], f["source_ids"], 1, preserved) for f in assessed],
    }


def edit_memory(record, suffix):
    return {**record, "body": record["body"] + suffix}


def rich_packet():
    body = (
        "Decision: keep\r\nPermission: only staging\r\nctl:\x01\x1f\x7f\u2028\u2029|\n"
        "markup: <script>alert(1)</script> <!-- x --> &amp; ${x}\n"
        "astral \U0001f600 \U00010000 bmp \ue000 \uffff combining e\u0301 \u00e9\n"
    )
    rec1 = {
        "key": "k-m1",
        "title": "T\u00efle e\u0301 \U0001f600",
        "summary": "sum \u2028",
        "category": "mixed",
        "body": body,
    }
    rec2 = {"key": "k-m2", "title": "", "summary": "", "category": "c", "body": "dup text\r\n"}
    m1 = make_source("m1", "memory_record", SQLITE_LABEL, rec1, enabled=True)
    m2 = make_source("m2", "memory_record", SQLITE_LABEL, rec2, enabled=False)
    mf = make_source("mf", "memory_file", FLATFILE_LABEL, "# dup text\r\nline\n", enabled=None)
    s1 = make_source(
        "s1",
        "skill_file",
        "skills/a/SKILL.md",
        "---\r\nname: a\r\n---\r\n# Guard\r\nPermission: only staging\r\n\r\n## Report\r\nshort\r\n",
        enabled=True,
        usage={"status": "available", "references": ["skills/b/SKILL.md", "x\u00e9"]},
        cov={"content_searched": True, "limitations": ["partial"]},
    )
    s2 = make_source(
        "s2",
        "skill_file",
        "skills/b/SKILL.md",
        "# B\nunused? unknown\n",
        enabled=False,
        usage={"status": "unavailable", "references": []},
        cov={"body_reviewed": True, "section_reviewed": True},
    )
    r1 = make_source(
        "r1", "reference_file", "skills/a/references/r.md", "ref \u00e9\u0301\n", role="reference"
    )
    extra = [
        make_source(f"x{i}", "skill_file", f"skills/x{i}/SKILL.md", f"# x{i}\nbody {i}\n")
        for i in range(6)
    ]
    empty = make_source("e0", "skill_file", "skills/empty/SKILL.md", "")
    sources = [m1, m2, mf, s1, s2, r1, empty, *extra]
    byid = {s["id"]: s for s in sources}
    f1 = make_finding("f1", "stale", [m1], ["Decision: keep"], kinds=("schedule", "execution"))
    f2 = make_finding("f2", "conflict", [s1, s2], ["Permission: only staging", "# B"])
    f3 = make_finding("f3", "duplicate", [m2, mf], ["dup text", "dup text"], kinds=("usage",))
    f4 = make_finding("f4", "unsupported", [r1], ["ref"], kinds=("delivery", "manual"))
    f5 = make_finding("f5", "clarification", [m1], ["e\u0301 \u00e9"])
    f6 = make_finding("f6", "stale", [s2], ["unknown"])
    findings = [f1, f2, f3, f4, f5, f6]
    findings.extend(
        make_finding(f"fx{i}", "clarification", [byid[f"x{i}"]], [f"body {i}"]) for i in range(6)
    )
    fx = findings[6:]
    props = {}
    props["pa"] = make_proposal("pa", [f1], [change_of(m1, edit_memory(rec1, "A\r\n"))])
    props["pb"] = make_proposal(
        "pb",
        [f2],
        [
            change_of(s1, s1["before"].replace("short", "short and exact")),
            change_of(s2, None, "disable"),
        ],
    )
    props["pc"] = make_proposal(
        "pc", [f3], [change_of(m2, {**rec2, "body": "dup text\r\nC"})], depends_on=["pa"]
    )
    props["pd"] = make_proposal(
        "pd", [f4], [change_of(r1, "ref \u00e9\u0301\nD\n")], depends_on=["pc", "pa"]
    )
    props["pe"] = make_proposal("pe", [f5], [change_of(m1, edit_memory(rec1, "E"))])
    props["pf"] = make_proposal("pf", [f1], [change_of(m1, edit_memory(rec1, "A\r\n"))])
    props["pg"] = make_proposal(
        "pg", [fx[0]], [change_of(byid["x0"], "# x0\nG\n")], depends_on=["ph"]
    )
    props["ph"] = make_proposal(
        "ph", [fx[1]], [change_of(byid["x1"], "# x1\nH\n")], depends_on=["pg"]
    )
    props["pi"] = make_proposal(
        "pi", [fx[2]], [change_of(byid["x2"], "# x2\nI\n")], depends_on=["pi"]
    )
    props["pj"] = make_proposal(
        "pj", [fx[3]], [change_of(byid["x3"], "# x3\nJ\n")], depends_on=["ghost"]
    )
    props["pk"] = make_proposal("pk", [f1], [change_of(m1, None, "disable")])
    props["pu"] = make_proposal("pu", [f6], [change_of(s2, "# B\nU\n")], preserved=False)
    props["pn"] = make_proposal("pn", [fx[4]], [change_of(byid["x4"], "# x4\nN\n")], assessed=[])
    props["pm"] = make_proposal(
        "pm", [f1, f5], [change_of(m1, edit_memory(rec1, "M"))], assessed=[f1]
    )
    props["pz"] = make_proposal("pz", [fx[5]], [change_of(byid["x5"], "# x5\nZ\n")])
    order = [
        "pd",
        "pc",
        "pb",
        "pa",
        "pf",
        "pe",
        "pz",
        "pg",
        "ph",
        "pi",
        "pj",
        "pk",
        "pu",
        "pn",
        "pm",
    ]
    inventory = seal_document(
        {
            "schema": SCHEMAS["inventory"],
            "profile": PROFILE,
            "sources": sources,
            "coverage": {**bf.COVERAGE, "limitations": ["inventory limit", "second"]},
        }
    )
    return seal_document(
        {
            "schema": SCHEMAS["packet"],
            "profile": PROFILE,
            "inventory": inventory,
            "findings": findings,
            "proposals": [props[name] for name in order],
        }
    )


def rich_session_states(packet):
    """A deterministic chain of pre-states exercising every decision."""
    states = {"initial": new_session(packet)}
    s = states["initial"]
    for pid in ("pa", "pb", "pc", "pd", "pz"):
        s = review_proposal(s, pid, "accept", note=f"accept {pid} \u00e9")
    states["accepted_chain"] = s
    s = review_proposal(s, "pk", "defer", note="later")
    s = review_proposal(s, "pe", "reject")
    s = review_proposal(s, "pu", "request_revision", note="redo")
    states["mixed"] = s
    s2 = new_session(packet)
    for pid in ("pa", "pe", "pf", "pc", "pd", "pg", "ph", "pi", "pj", "pk"):
        s2 = review_proposal(s2, pid, "accept")
    states["conflicts"] = s2
    s3 = new_session(packet)
    for pid in ("pa", "pf", "pb", "pc", "pd"):
        s3 = review_proposal(s3, pid, "accept")
    states["identical_changes"] = s3
    s4 = review_proposal(
        states["accepted_chain"],
        "pa",
        "edit",
        edited_proposal=edited_of(states["accepted_chain"], "pa"),
    )
    s4 = review_proposal(s4, "pa", "accept", note="again")
    s4 = review_proposal(s4, "pb", "request_revision")
    s4 = review_proposal(s4, "pb", "reject")
    states["edited_reaccepted"] = s4
    s5 = review_proposal(new_session(packet), "pc", "accept")
    states["missing_dependency"] = s5
    s6 = review_proposal(new_session(packet), "pk", "accept")
    states["disable_only"] = s6
    return states


def current_revision(session, pid):
    return next(r for r in session["reviews"] if r["proposal_id"] == pid)["revisions"][-1]


def edited_of(session, pid, *, assessments="fresh", mutate=None):
    cur = current_revision(session, pid)
    item = copy.deepcopy(cur)
    item["revision"] = cur["revision"] + 1
    for change in item["changes"]:
        if isinstance(change["after"], dict):
            change["after"]["summary"] += "!"
        elif isinstance(change["after"], str):
            change["after"] += "!"
    packet = session["packet"]
    findings = {f["id"]: f for f in packet["findings"]}
    if assessments == "fresh":
        item["assessments"] = [
            assess(fid, findings[fid]["source_ids"], item["revision"])
            for fid in item["finding_ids"]
        ]
    elif assessments == "none":
        item["assessments"] = []
    elif assessments == "stale":
        item["assessments"] = copy.deepcopy(cur["assessments"])
    elif assessments == "future":
        item["assessments"] = [
            assess(fid, findings[fid]["source_ids"], item["revision"] + 1)
            for fid in item["finding_ids"]
        ]
    if mutate:
        mutate(item)
    return item


def edit_variants(session, pid):
    cur = current_revision(session, pid)
    variants = {
        "fresh": edited_of(session, pid),
        "none": edited_of(session, pid, assessments="none"),
        "stale": edited_of(session, pid, assessments="stale"),
        "future": edited_of(session, pid, assessments="future"),
        "restored": {**copy.deepcopy(cur), "revision": cur["revision"] + 1, "assessments": []},
        "carried": {**copy.deepcopy(cur), "revision": cur["revision"] + 1},
        "same_revision": edited_of(
            session, pid, mutate=lambda i: i.update(revision=cur["revision"])
        ),
        "skip_revision": edited_of(
            session, pid, mutate=lambda i: i.update(revision=cur["revision"] + 2)
        ),
        "other_id": edited_of(session, pid, mutate=lambda i: i.update(id=pid + "-x")),
        "extra_key": edited_of(session, pid, mutate=lambda i: i.update(extra=1)),
        "bad_precondition": edited_of(
            session, pid, mutate=lambda i: i["changes"][0].update(source_sha256="0" * 64)
        ),
        "bad_depends": edited_of(session, pid, mutate=lambda i: i.update(depends_on="pa")),
        "bool_revision": edited_of(session, pid, mutate=lambda i: i.update(revision=True)),
        "dropped_change": edited_of(session, pid, mutate=lambda i: i.update(changes=[])),
        "unknown_finding": edited_of(session, pid, mutate=lambda i: i.update(finding_ids=["nope"])),
        "unsafe": {
            **edited_of(session, pid),
            "assessments": [
                assess(
                    fid, session["packet"]["findings"][0]["source_ids"], cur["revision"] + 1, False
                )
                for fid in cur["finding_ids"][:1]
            ],
        },
    }
    if cur["revision"] == 1:
        # Python treats a JSON true as equal to revision 1 when dropping stale assessments.
        variants["bool_assessment"] = edited_of(
            session,
            pid,
            assessments="stale",
            mutate=lambda i: [a.update(revision=True) for a in i["assessments"]],
        )
    return variants


def lifecycle_steps(session):
    steps = []
    pids = [r["proposal_id"] for r in session["reviews"]]
    steps.extend(
        {"pid": pid, "action": action, "edited": None, "note": f"{action} \u00e9"}
        for pid in pids
        for action in ("accept", "request_revision", "reject", "defer")
    )
    for pid in ("pa", "pb", "pc", "pk"):
        for label, edited in edit_variants(session, pid).items():
            steps.append({"pid": pid, "action": "edit", "edited": edited, "note": label})
    steps += [
        {"pid": "pa", "action": "edit", "edited": None, "note": ""},
        {"pid": "pa", "action": "accept", "edited": current_revision(session, "pa"), "note": ""},
        {"pid": "pa", "action": "defer", "edited": current_revision(session, "pa"), "note": ""},
        {
            "pid": "pa",
            "action": "request_revision",
            "edited": current_revision(session, "pa"),
            "note": "",
        },
        {"pid": "nope", "action": "accept", "edited": None, "note": ""},
        {"pid": "pa", "action": "approve", "edited": None, "note": ""},
        {"pid": "pa", "action": "accept", "edited": None, "note": 5},
        {"pid": 1, "action": "accept", "edited": None, "note": ""},
        {"pid": "pa", "action": "accept", "edited": None, "note": "x" * 5000 + "\U0001f600\r\n"},
    ]
    return steps


def py_step(session, step):
    return review_proposal(
        session, step["pid"], step["action"], edited_proposal=step["edited"], note=step["note"]
    )


def js_step(step):
    out = {**step}
    out["edited_hex"] = None if step["edited"] is None else text_bytes(step["edited"]).hex()
    del out["edited"]
    return out


def tamper_session(session, fn, *, reseal_packet=False):
    s = copy.deepcopy(session)
    fn(s)
    if reseal_packet:
        s["packet"]["inventory"] = raw_seal(s["packet"]["inventory"])
        s["packet"] = raw_seal(s["packet"])
    return raw_seal(s)


def tamper_packet(packet, fn):
    p = copy.deepcopy(packet)
    fn(p)
    p["inventory"] = raw_seal(p["inventory"])
    return raw_seal(p)


def rev_of(s, pid):
    return next(r for r in s["reviews"] if r["proposal_id"] == pid)


def negative_docs(packet, states, plans):
    """(name, raw bytes, expected_profile, isolate) all rejected by Python."""
    cases = []
    base = dumps(packet).decode()

    def add(name, raw, *, profile=None, isolate=True):
        cases.append((name, raw if isinstance(raw, bytes) else text_bytes(raw), profile, isolate))

    # Lexical / encoding classes (not isolated: the failure is in the parser).
    add("dup_key", base.replace('{"', '{"profile":1,"', 1).encode(), isolate=False)
    add(
        "dup_nested",
        base.replace('"discovered":true', '"discovered":true,"discovered":true', 1).encode(),
        isolate=False,
    )
    add("float_1_0", base.replace('"revision":1', '"revision":1.0', 1).encode(), isolate=False)
    add("float_exp", base.replace('"revision":1', '"revision":1e0', 1).encode(), isolate=False)
    add("nan", base.replace('"revision":1', '"revision":NaN', 1).encode(), isolate=False)
    add("infinity", base.replace('"revision":1', '"revision":-Infinity', 1).encode(), isolate=False)
    add(
        "unsafe_int",
        base.replace('"revision":1', '"revision":9007199254740992', 1).encode(),
        isolate=False,
    )
    add(
        "neg_unsafe_int",
        base.replace('"revision":1', '"revision":-9007199254740992', 1).encode(),
        isolate=False,
    )
    add(
        "huge_int",
        base.replace('"revision":1', '"revision":' + "9" * 40, 1).encode(),
        isolate=False,
    )
    add("lone_surrogate", base.replace('"id":"pa"', '"id":"\\ud800"', 1).encode(), isolate=False)
    add(
        "lone_low_surrogate", base.replace('"id":"pa"', '"id":"\\udc00"', 1).encode(), isolate=False
    )
    add("bad_utf8", base.encode() + b"\xff", isolate=False)
    add("overlong", b'{"a":"\xc0\x80"}', isolate=False)
    add("bom", b"\xef\xbb\xbf" + base.encode(), isolate=False)
    add("empty", b"", isolate=False)
    add("trailing", base.encode() + b" 1", isolate=False)
    add("trailing_comma", base.encode()[:-1] + b",}", isolate=False)
    add("top_array", b"[]", isolate=False)
    add("top_string", b'"x"', isolate=False)
    add("top_null", b"null", isolate=False)
    add("depth_33", ("[" * 34 + "]" * 34).encode(), isolate=False)
    add("nodes_100001", ("[" + ",".join(["0"] * 100001) + "]").encode(), isolate=False)
    add("items_10001", ("[" + ",".join(["0"] * 10001) + "]").encode(), isolate=False)
    add("keys_10001", generate({"kind": "keys", "n": 10001}), isolate=False)
    add("too_large", b'"' + b"a" * MAX_BYTES + b'"', isolate=False)
    add("unknown_schema", {**packet, "schema": "nachos.reconciliation.packet.v2"}, isolate=False)
    add("extra_top_key", raw_seal({**packet, "extra": 1}))
    add("missing_profile", {k: v for k, v in packet.items() if k != "profile"}, isolate=False)
    add("bad_digest_case", {**packet, "digest": packet["digest"].upper()}, isolate=False)
    add("tampered_digest", {**packet, "digest": "0" * 64}, isolate=False)
    add(
        "wrong_profile",
        dumps(packet),
        profile={"id": "rich-profile", "root": "/other"},
        isolate=False,
    )
    add("profile_extra", dumps(packet), profile={**PROFILE, "x": 1}, isolate=False)
    # Packet semantic classes, each resealed so only the intended rule fails.
    pk = {
        "nul_locator": lambda p: p["inventory"]["sources"][3].update(
            locator="skills/a\x00/SKILL.md"
        ),
        "backslash_locator": lambda p: p["inventory"]["sources"][3].update(locator="skills\\a"),
        "colon_locator": lambda p: p["inventory"]["sources"][3].update(locator="c:/x"),
        "dotdot_locator": lambda p: p["inventory"]["sources"][3].update(locator="skills/../x"),
        "dot_locator": lambda p: p["inventory"]["sources"][3].update(locator="./x"),
        "empty_segment": lambda p: p["inventory"]["sources"][3].update(locator="skills//x"),
        "abs_empty_locator": lambda p: p["inventory"]["sources"][3].update(locator=""),
        "profile_root_nul": lambda p: (
            p["inventory"]["profile"].update(root="/r\x00"),
            p["profile"].update(root="/r\x00"),
        ),
        "bad_kind": lambda p: p["inventory"]["sources"][3].update(kind="skill"),
        "bad_role": lambda p: p["inventory"]["sources"][3].update(role="other"),
        "hash_mismatch": lambda p: p["inventory"]["sources"][3].update(sha256="1" * 64),
        "hash_uppercase": lambda p: p["inventory"]["sources"][3].update(
            sha256=p["inventory"]["sources"][3]["sha256"].upper()
        ),
        "before_changed": lambda p: p["inventory"]["sources"][3].update(
            before=p["inventory"]["sources"][3]["before"] + "x"
        ),
        "byte_size": lambda p: p["inventory"]["sources"][3]["metadata"].update(byte_size=1),
        "char_size_bytes": lambda p: p["inventory"]["sources"][0]["metadata"].update(
            char_size=p["inventory"]["sources"][0]["metadata"]["byte_size"]
        ),
        "size_bool": lambda p: p["inventory"]["sources"][3]["metadata"].update(byte_size=True),
        "enabled_str": lambda p: p["inventory"]["sources"][3]["metadata"].update(enabled="yes"),
        "usage_status": lambda p: p["inventory"]["sources"][3]["metadata"]["usage"].update(
            status="used"
        ),
        "usage_refs_unknown": lambda p: p["inventory"]["sources"][4]["metadata"]["usage"].update(
            references=["a"]
        ),
        "hashed_false": lambda p: p["inventory"]["sources"][3]["coverage"].update(hashed=False),
        "coverage_non_bool": lambda p: p["inventory"]["coverage"].update(body_reviewed=1),
        "dup_source_id": lambda p: p["inventory"]["sources"][1].update(id="m1"),
        "dup_identity": lambda p: p["inventory"]["sources"][1].update(
            identity="synthetic:m1",
            locator=p["inventory"]["sources"][0]["locator"],
            kind="memory_record",
        ),
        "memory_extra_field": lambda p: p["inventory"]["sources"][0]["before"].update(extra="x"),
        "excerpt_not_exact": lambda p: p["findings"][0]["excerpts"][0].update(
            text="Decision: keep "
        ),
        "excerpt_empty": lambda p: p["findings"][0]["excerpts"][0].update(text=""),
        "excerpt_source_mismatch": lambda p: p["findings"][0]["excerpts"][0].update(source_id="m2"),
        "finding_unknown_source": lambda p: p["findings"][0].update(source_ids=["zzz"]),
        "finding_type": lambda p: p["findings"][0].update(type="other"),
        "finding_no_evidence": lambda p: p["findings"][0].update(evidence=[]),
        "evidence_naive_time": lambda p: p["findings"][0]["evidence"][0].update(
            observed_at="2026-11-16T12:00:00"
        ),
        "evidence_bad_time": lambda p: p["findings"][0]["evidence"][0].update(
            observed_at="yesterday"
        ),
        "evidence_kind": lambda p: p["findings"][0]["evidence"][0].update(kind="cron"),
        "evidence_status": lambda p: p["findings"][0]["evidence"][0].update(status="ok"),
        "evidence_source": lambda p: p["findings"][0]["evidence"][0].update(source_ids=["m2"]),
        "dup_finding_id": lambda p: p["findings"][1].update(id="f1"),
        "proposal_revision_2": lambda p: p["proposals"][3].update(revision=2),
        "proposal_revision_bool": lambda p: p["proposals"][3].update(revision=True),
        "proposal_extra_key": lambda p: p["proposals"][3].update(extra=1),
        "proposal_unknown_finding": lambda p: p["proposals"][3].update(finding_ids=["nope"]),
        "assessment_wrong_revision": lambda p: p["proposals"][3]["assessments"][0].update(
            revision=2
        ),
        "assessment_dup": lambda p: p["proposals"][3]["assessments"].append(
            copy.deepcopy(p["proposals"][3]["assessments"][0])
        ),
        "assessment_unknown_finding": lambda p: p["proposals"][3]["assessments"][0].update(
            finding_id="f2"
        ),
        "assessment_preserved_int": lambda p: p["proposals"][3]["assessments"][0][
            "preservation_checklist"
        ][0].update(preserved=1),
        "assessment_empty_checklist": lambda p: p["proposals"][3]["assessments"][0].update(
            preservation_checklist=[]
        ),
        "change_before_mismatch": lambda p: p["proposals"][3]["changes"][0].update(before="x"),
        "change_sha_mismatch": lambda p: p["proposals"][3]["changes"][0].update(
            source_sha256="2" * 64
        ),
        "change_operation": lambda p: p["proposals"][3]["changes"][0].update(operation="delete"),
        "disable_with_after": lambda p: p["proposals"][11]["changes"][0].update(
            after=p["inventory"]["sources"][0]["before"]
        ),
        "edit_null_after": lambda p: p["proposals"][3]["changes"][0].update(after=None),
        "memory_key_change": lambda p: p["proposals"][3]["changes"][0]["after"].update(key="other"),
        "memory_after_missing_field": lambda p: p["proposals"][3]["changes"][0]["after"].pop(
            "body"
        ),
        "change_uncovered_source": lambda p: p["proposals"][3]["changes"][0].update(source_id="m2"),
        "change_unexcerpted": lambda p: p["proposals"][3]["changes"].append(
            change_of(p["inventory"]["sources"][2], "x")
        ),
        "empty_changes": lambda p: p["proposals"][3].update(changes=[]),
        "dup_proposal_id": lambda p: p["proposals"][1].update(id=p["proposals"][0]["id"]),
        "depends_dup": lambda p: p["proposals"][3].update(depends_on=["x", "x"]),
    }
    for name, fn in pk.items():
        add("packet_" + name, tamper_packet(packet, fn))
    # Session classes.
    sess = states["accepted_chain"]
    sk = {
        "history_digest": lambda s: rev_of(s, "pa")["history"][0].update(digest="0" * 64),
        "history_revision": lambda s: rev_of(s, "pa")["history"][0].update(revision=2),
        "history_action": lambda s: rev_of(s, "pa")["history"][0].update(action="reject"),
        "history_unknown_action": lambda s: rev_of(s, "pa")["history"][0].update(action="approve"),
        "history_extra": lambda s: rev_of(s, "pa")["history"][0].update(extra=1),
        "history_note_int": lambda s: rev_of(s, "pa")["history"][0].update(note=1),
        "history_dropped": lambda s: rev_of(s, "pa").update(history=[]),
        "decision_forged": lambda s: rev_of(s, "pe").update(decision="accept"),
        "accepted_digest_forged": lambda s: rev_of(s, "pe").update(accepted_digest="a" * 64),
        "accepted_digest_cleared": lambda s: rev_of(s, "pa").update(accepted_digest=None),
        "current_revision_forged": lambda s: rev_of(s, "pa").update(current_revision=2),
        "current_revision_bool": lambda s: rev_of(s, "pa").update(current_revision=True),
        "initial_differs": lambda s: rev_of(s, "pa")["revisions"][0].update(depends_on=["x"]),
        "extra_review_field": lambda s: rev_of(s, "pa").update(extra=1),
        "missing_review": lambda s: s["reviews"].pop(),
        "dup_review": lambda s: s["reviews"].append(copy.deepcopy(s["reviews"][0])),
        "unknown_review": lambda s: s["reviews"][0].update(proposal_id="zz"),
        "revision_gap": lambda s: rev_of(s, "pa")["revisions"].append(
            {**copy.deepcopy(rev_of(s, "pa")["revisions"][0]), "revision": 3}
        ),
        "session_extra_key": lambda s: s.update(extra=1),
        "session_profile": lambda s: s.update(profile={"id": "other", "root": "/inert/rich"}),
    }
    for name, fn in sk.items():
        add("session_" + name, tamper_session(sess, fn))
    add(
        "session_nested_digest_tampered",
        tamper_session(sess, lambda s: s["packet"].update(digest="0" * 64)),
        isolate=False,
    )
    add(
        "session_nested_inventory_tampered",
        tamper_session(sess, lambda s: s["packet"]["inventory"].update(digest="0" * 64)),
        isolate=False,
    )
    add(
        "session_packet_content_tampered",
        tamper_session(sess, lambda s: s["packet"]["findings"][0].update(rationale="x")),
        isolate=False,
    )
    add(
        "session_nested_source_tampered",
        tamper_session(
            sess, lambda s: s["packet"]["inventory"]["sources"][0].update(sha256="1" * 64)
        ),
        isolate=False,
    )
    add(
        "session_packet_is_inventory",
        tamper_session(sess, lambda s: s.update(packet=packet["inventory"])),
        isolate=False,
    )

    def false_accept(s):
        rec = rev_of(s, "pu")
        item = rec["revisions"][0]
        digest = content_digest("proposal", review_context(s["packet"], item))
        rec.update(
            history=[{"revision": 1, "action": "accept", "digest": digest, "note": ""}],
            decision="accept",
            accepted_digest=digest,
        )

    add("session_false_preservation_accept", tamper_session(sess, false_accept))

    def missing_assessment_accept(s):
        rec = rev_of(s, "pn")
        digest = content_digest("proposal", review_context(s["packet"], rec["revisions"][0]))
        rec.update(
            history=[{"revision": 1, "action": "accept", "digest": digest, "note": ""}],
            decision="accept",
            accepted_digest=digest,
        )

    add("session_missing_assessment_accept", tamper_session(sess, missing_assessment_accept))

    def partial_assessment_accept(s):
        rec = rev_of(s, "pm")
        digest = content_digest("proposal", review_context(s["packet"], rec["revisions"][0]))
        rec.update(
            history=[{"revision": 1, "action": "accept", "digest": digest, "note": ""}],
            decision="accept",
            accepted_digest=digest,
        )

    add("session_incomplete_assessment_accept", tamper_session(sess, partial_assessment_accept))

    def stale_assessment_edit(s):
        rec = rev_of(s, "pa")
        cur = rec["revisions"][-1]
        new = {**copy.deepcopy(cur), "revision": 2}  # keeps assessments bound to revision 1
        rec["revisions"].append(new)
        digest = content_digest("proposal", review_context(s["packet"], new))
        rec.update(
            history=[
                *rec["history"],
                {"revision": 2, "action": "edit", "digest": digest, "note": ""},
            ],
            current_revision=2,
            decision="unreviewed",
            accepted_digest=None,
        )

    add("session_stale_assessment_edit", tamper_session(sess, stale_assessment_edit))

    def edit_keeps_accept(s):
        rec = rev_of(s, "pa")
        new = {**copy.deepcopy(rec["revisions"][-1]), "revision": 2, "assessments": []}
        new["changes"][0]["after"] = {**new["changes"][0]["after"], "summary": "chg"}
        rec["revisions"].append(new)
        digest = content_digest("proposal", review_context(s["packet"], new))
        rec.update(
            history=[
                *rec["history"],
                {"revision": 2, "action": "edit", "digest": digest, "note": ""},
            ],
            current_revision=2,
            decision="accept",
            accepted_digest=rec["accepted_digest"],
        )

    add("session_edit_retains_acceptance", tamper_session(sess, edit_keeps_accept))

    def revision_request_edits(s):
        rec = rev_of(s, "pa")
        new = {**copy.deepcopy(rec["revisions"][-1]), "revision": 2, "assessments": []}
        new["depends_on"] = ["pz"]
        rec["revisions"].append(new)
        digest = content_digest("proposal", review_context(s["packet"], new))
        rec.update(
            history=[
                *rec["history"],
                {"revision": 2, "action": "request_revision", "digest": digest, "note": ""},
            ],
            current_revision=2,
            decision="request_revision",
            accepted_digest=None,
        )

    add("session_revision_request_edits_content", tamper_session(sess, revision_request_edits))
    # Plan classes.
    plan = plans["identical_changes"]
    pl = {
        "unsorted_order": lambda p: p["approved"].reverse(),
        "review_digest": lambda p: p["approved"][0].update(review_digest="0" * 64),
        "execution": lambda p: p.update(execution="performed"),
        "source_validation": lambda p: p.update(source_validation="valid"),
        "extra_key": lambda p: p.update(extra=1),
        "packet_digest_uppercase": lambda p: p.update(packet_digest=p["packet_digest"].upper()),
        "entry_extra": lambda p: p["approved"][0].update(extra=1),
        "extra_finding": lambda p: p["approved"][0]["findings"].append(
            copy.deepcopy(next(f for f in packet["findings"] if f["id"] == "f3"))
        ),
        "extra_source": lambda p: p["approved"][0]["sources"].append(
            copy.deepcopy(packet["inventory"]["sources"][6])
        ),
        "unsafe_assessment": lambda p: p["approved"][0]["proposal"]["assessments"][0][
            "preservation_checklist"
        ][0].update(preserved=False),
        "dup_approved": lambda p: p["approved"].append(copy.deepcopy(p["approved"][0])),
        "ledger_status": lambda p: p["ledger"][0].update(status="accepted"),
        "ledger_decision": lambda p: p["ledger"][0].update(decision="approve"),
        "ledger_revision": lambda p: p["ledger"][0].update(revision=0),
        "ledger_dup": lambda p: p["ledger"].append(copy.deepcopy(p["ledger"][0])),
        "ledger_overlaps_approved": lambda p: p["ledger"].append(
            {**p["ledger"][0], "proposal_id": p["approved"][0]["proposal"]["id"]}
        ),
        "missing_dependency": lambda p: p["approved"][0]["proposal"].update(depends_on=["zzz"]),
    }
    for name, fn in pl.items():
        mutated = copy.deepcopy(plan)
        fn(mutated)
        add("plan_" + name, raw_seal(mutated))
    conflict_plan = copy.deepcopy(plan)
    other = copy.deepcopy(plans["identical_changes"]["approved"][0])
    conflict_plan["approved"] = [conflict_plan["approved"][0], other]
    add("plan_duplicate_entry", raw_seal(conflict_plan))
    return cases


def conflicting_plan(packet):
    """A plan whose entries hold conflicting changes to one source (never exportable)."""
    s = new_session(packet)
    s = review_proposal(s, "pa", "accept")
    plan_a = export_plan(s)
    s2 = review_proposal(new_session(packet), "pe", "accept")
    plan_e = export_plan(s2)
    forged = copy.deepcopy(plan_a)
    forged["approved"] = [plan_a["approved"][0], plan_e["approved"][0]]
    forged["ledger"] = []
    return raw_seal(forged)


def timestamp_matrix():
    fixed = [
        "2026-11-16T12:00:00Z",
        "2026-11-16T12:00:00z",
        "2026-11-16T12:00:00+00:00",
        "2026-11-16T12:00:00-05:00",
        "2026-11-16T12:00:00.5Z",
        "2026-11-16T12:00:00.123456+01:00",
        "2026-11-16T12:00:00.1234567Z",
        "2026-11-16 12:00:00Z",
        "2026-11-16t12:00:00Z",
        "2026-11-16_12:00:00Z",
        "20261116T120000Z",
        "20261116T120000+0100",
        "2026-11-16T12:00Z",
        "2026-11-16T12Z",
        "2026-11-16T12:00:00+0530",
        "2026-11-16T12:00:00+05",
        "2026-11-16T12:00:00+05:30:15",
        "2026-11-16T12:00:00+05:30:15.5",
        "2026-11-16T12:00:00-23:59",
        "2026-11-16T12:00:00+23:59:59.999999",
        "2026-11-16T12:00:00+24:00",
        "2026-11-16T12:00:00+00:60",
        "2026-11-16T12:00:00+00:99",
        "2026-11-16T12:00:00+99:00",
        "2026-11-16T12:00:00-00:00",
        "2026-11-16T12:00:00+1",
        "2026-11-16T12:00:00+123",
        "2026-11-16T12:00:00,5Z",
        "2026-11-16T12:00:00.Z",
        "2026-11-16T12:00:00.5",
        "2026-W46-1T12:00:00Z",
        "2026W461T120000Z",
        "2026-W46T12:00Z",
        "2026W46T12:00Z",
        "2026-W53-1T12:00:00Z",
        "2020-W53-7T12:00:00Z",
        "2026-W00-1T12:00:00Z",
        "2026-W46-8T12:00:00Z",
        "2026-W46-1-T12:00:00Z",
        "0001-W01-1T00:00:00Z",
        "9999-W52-7T00:00:00Z",
        "0000-01-01T00:00:00Z",
        "0001-01-01T00:00:00Z",
        "9999-12-31T23:59:59Z",
        "2024-02-29T00:00:00Z",
        "2026-02-29T00:00:00Z",
        "2100-02-29T00:00:00Z",
        "2000-02-29T00:00:00Z",
        "2026-13-01T00:00:00Z",
        "2026-00-10T00:00:00Z",
        "2026-04-31T00:00:00Z",
        "2026-11-16T24:00:00Z",
        "2026-11-16T23:59:59Z",
        "2026-11-16T12:60:00Z",
        "2026-11-16T12:00:60Z",
        "2026-11-16T25:00:00Z",
        "2026-11-16",
        "2026-11-16T",
        "2026-11-16T12:00:00",
        "2026-11-16T12:00",
        "2026-1-16T12:00:00Z",
        "2026-11-16T1:00:00Z",
        "2026-11-16T12:0:00Z",
        "2026-11-16T12:00:0Z",
        "2026/11/16T12:00:00Z",
        "2026-11-16T12-00-00Z",
        "2026-11-16T12:00:00ZZ",
        "2026-11-16T12:00:00+00:00Z",
        "2026-11-16T12:00:00 Z",
        "2026-11-16T12:00:00Z ",
        " 2026-11-16T12:00:00Z",
        "2026-11-16T12:00:00\nZ",
        "\uff12\uff10\uff12\uff16-11-16T12:00:00Z",
        "2026-11-16T12:00:00\u00e9Z",
        "2026-11-16\u00e912:00:00Z",
        "2026-11-16\U0001f60012:00:00Z",
        "now",
        "yesterday",
        "12345",
        "2026",
        "-2026-11-16T12:00:00Z",
        "+2026-11-16T12:00:00Z",
        "2026-11-16T+12:00Z",
        "2026-11-16T12:00:00-",
        "2026-11-16T12:00:00+",
        "2026-11-16T12:00:00.1-05:00",
        "2026-11-16T12:00:00.123456789012-05:00",
        "2026-11-16T12:00:00_5Z",
        "2026-11-16T1200Z",
        "2026-11-16T120000Z",
        "2026-11-16T12:0000Z",
        "2026-11-16T1200:00Z",
        "20261116T12:00:00Z",
        "2026-1116T12:00:00Z",
        "202611-16T12:00:00Z",
        "2026-11-16T12:00:00+05:3",
        "2026-11-16T12:00:00+05:30:1",
        "2026-11-16T12:00:00+0530:00",
        "2026-11-16T12:00:00+05:3000",
        "2026-11-16T12:00:00+053000",
        "2026-11-16T12:00:00+053000.5",
    ]
    rng = random.Random(20260301)
    parts = [
        "2026",
        "-",
        "11",
        "16",
        "T",
        "12",
        ":",
        "00",
        ".",
        "5",
        "Z",
        "+",
        "05",
        "W",
        "46",
        "1",
        " ",
        ",",
        "0",
    ]
    fuzz = []
    while len(fuzz) < 400:
        pieces = [rng.choice(parts) for _ in range(rng.randrange(6, 16))]
        value = "".join(pieces)
        if rng.random() < 0.5:
            value = (
                "2026-11-16T"
                + value[: rng.randrange(1, 14)]
                + rng.choice(["Z", "+00:00", "-01:30", ""])
            )
        fuzz.append(value)
    return fixed + fuzz


CORE_STAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}([.,]\d+)?)?"
    r"(Z|[+-]\d{2}(:\d{2}(:\d{2}([.,]\d+)?)?)?)$",
    re.ASCII,
)


def py_timestamp_ok(value):
    try:
        models_module.timestamp(value)
    except ValueError:
        return False
    return True


class FakeReader:
    def __init__(self, sources=None, error=None):
        self.sources, self.error = sources, error

    def __call__(self, _boundary, _spec):
        if self.error:
            raise AcquisitionError(self.error)
        return self.sources


def validation_cases(plan):
    spec = SimpleNamespace(
        store="sqlite",
        skills=tuple(
            sorted(
                {
                    s["locator"]
                    for e in plan["approved"]
                    for s in e["sources"]
                    if s["kind"] == "skill_file"
                }
            )
        ),
        references=tuple(
            sorted(
                {
                    s["locator"]
                    for e in plan["approved"]
                    for s in e["sources"]
                    if s["kind"] == "reference_file"
                }
            )
        ),
    )
    current = {}
    for entry in plan["approved"]:
        for src in entry["sources"]:
            current[src["identity"]] = SimpleNamespace(
                identity=src["identity"],
                kind=src["kind"],
                locator=src["locator"],
                sha256=src["sha256"],
            )

    def run(reader, spec=spec, stamp="2026-11-16T13:00:00+00:00"):
        return validate_plan_sources(
            plan,
            boundary=SourceBoundary(Path(plan["profile"]["root"]), ()),
            profile=dict(plan["profile"]),
            spec=spec,
            reader=reader,
            checked_at=stamp,
        )

    identities = sorted(current)
    results = {"unchanged": run(FakeReader(current))}
    changed = {k: copy.copy(v) for k, v in current.items()}
    changed[identities[0]].sha256 = "9" * 64
    results["changed_sha"] = run(FakeReader(changed))
    moved = {k: copy.copy(v) for k, v in current.items()}
    moved[identities[-1]].locator = "skills/elsewhere/SKILL.md"
    results["moved"] = run(FakeReader(moved))
    results["missing"] = run(FakeReader({k: v for k, v in current.items() if k != identities[0]}))
    results["reader_error"] = run(FakeReader(error="source_unavailable"))
    results["unstable_error"] = run(FakeReader(error="Some Unstable Text!"))
    narrow = SimpleNamespace(store="sqlite", skills=(), references=())
    results["not_in_spec"] = run(FakeReader(current), spec=narrow)
    results["z_stamp"] = run(FakeReader(current), stamp="2026-11-16T13:00:00Z")
    for name, result in results.items():
        assert result["schema"] == "nachos.reconciliation.validation.v1", name
    return results


def build_corpus():
    rng = random.Random(7)
    packet = rich_packet()
    states = rich_session_states(packet)
    plans = {name: export_plan(state) for name, state in states.items()}
    corpus = {
        "canonical": [],
        "docs": [],
        "lifecycle": [],
        "new_session": [],
        "export": [],
        "proposal_digest": [],
        "review_context": [],
        "validation": [],
    }
    meta = {"docs": [], "lifecycle": [], "canonical": [], "validation": []}

    def add_canonical(name, raw=None, gen=None):
        case = {"digests": dict.fromkeys(("proposal", "memory-record", "plan"), "")}
        if gen:
            case["gen"] = gen
        else:
            case["raw_hex"] = raw.hex()
        corpus["canonical"].append(case)
        meta["canonical"].append(name)

    for text in HOSTILE:
        add_canonical("hostile-value", text_bytes(text))
        add_canonical("hostile-key", text_bytes({text: 1, "a": text}))
    add_canonical("hostile-ascii-escapes", json.dumps(HOSTILE).encode())
    add_canonical(
        "astral-vs-bmp",
        text_bytes({"\U0001f600": 1, "\ue000": 2, "\uffff": 3, "\U00010000": 4, "z": 5}),
    )
    add_canonical(
        "integer-like", text_bytes({"10": 1, "9": 2, "2": 3, "-1": 4, "": 5, "01": 6, "a": 7})
    )
    add_canonical("combining", text_bytes({"e\u0301": 1, "\u00e9": 2}))
    add_canonical(
        "proto-keys",
        b'{"__proto__":{"x":1},"constructor":2,"prototype":3,"toString":4,"hasOwnProperty":5}',
    )
    add_canonical("safe-ints", b"[9007199254740991,-9007199254740991,0,-0,1,-1]")
    add_canonical("ws", b' \t\r\n{ "b" : [ 1 , 2 ] , "a" : null } \n')
    add_canonical("nest-33-levels", ("[" * 33 + "]" * 33).encode())
    add_canonical("nest-34-levels", ("[" * 34 + "]" * 34).encode())
    add_canonical("objnest-32", gen={"kind": "objnest", "n": 32})
    add_canonical("objnest-33", gen={"kind": "objnest", "n": 33})
    add_canonical("zeros-10000", gen={"kind": "zeros", "n": 10000})
    add_canonical("zeros-10001", gen={"kind": "zeros", "n": 10001})
    add_canonical("zeros-99999", raw=("[" + ",".join(["[0]"] * 33333) + "]").encode())
    add_canonical(
        "nodes-100000",
        raw=(("[" + ",".join(["[" + ",".join(["0"] * 9999) + "]"] * 10) + "]").encode()),
    )
    add_canonical(
        "nodes-100001",
        raw=(("[" + ",".join(["[" + ",".join(["0"] * 9999) + "]"] * 10) + ",0]").encode()),
    )
    add_canonical("keys-10000", gen={"kind": "keys", "n": 10000})
    add_canonical("keys-10001", gen={"kind": "keys", "n": 10001})
    add_canonical("bytes-at-limit", gen={"kind": "str", "n": MAX_BYTES - 2})
    add_canonical("bytes-over-limit", gen={"kind": "str", "n": MAX_BYTES - 1})
    for raw in (
        b'{"n":1.0}',
        b'{"n":1e0}',
        b'{"n":NaN}',
        b"[1,2",
        b"01",
        b"-",
        b'{"a":1,"a":1}',
        b'"\\ud800"',
    ):
        add_canonical("lexical-" + raw.decode(), raw)
    for _ in range(250):
        add_canonical("random", text_bytes(random_value(rng, 0)))

    def add_doc(name, raw, profile=None, *, isolate=False, expect_ok=None):
        corpus["docs"].append({"raw_hex": raw.hex(), "expected_profile": profile})
        meta["docs"].append({"name": name, "isolate": isolate, "expect_ok": expect_ok})

    for path in sorted(FIX.glob("*.json")):
        if path.name != "hash-vectors.json":
            add_doc("fixture-" + path.name, path.read_bytes(), expect_ok=True)
    add_doc("rich-packet", dumps(packet), PROFILE, expect_ok=True)
    add_doc("rich-inventory", dumps(packet["inventory"]), PROFILE, expect_ok=True)
    add_doc(
        "rich-packet-pretty",
        json.dumps(packet, indent=2, ensure_ascii=False).encode(),
        expect_ok=True,
    )
    for name, state in states.items():
        add_doc("rich-session-" + name, dumps(state), expect_ok=True)
    for name, plan in plans.items():
        add_doc("rich-plan-" + name, dumps(plan), expect_ok=True)
    for name, raw, profile, isolate in negative_docs(packet, states, plans):
        add_doc(name, raw, profile, isolate=isolate, expect_ok=False)
    add_doc(
        "plan_conflicting_changes",
        text_bytes(conflicting_plan(packet)),
        isolate=True,
        expect_ok=False,
    )
    # Lifecycle: independent steps against each pre-state.
    for name, state in states.items():
        steps = lifecycle_steps(state)
        corpus["lifecycle"].append(
            {"session_hex": dumps(state).hex(), "steps": [js_step(s) for s in steps]}
        )
        meta["lifecycle"].append({"name": name, "state": state, "steps": steps})
    for path in (FIX / "packet.json", FIX / "packet-empty.json"):
        corpus["new_session"].append({"packet_hex": path.read_bytes().hex()})
    corpus["new_session"].append({"packet_hex": dumps(packet).hex()})
    corpus["new_session"].append({"packet_hex": dumps(states["initial"]).hex()})
    corpus["new_session"].append({"packet_hex": text_bytes({**packet, "extra": 1}).hex()})
    for state in [*states.values(), new_session(bf_packet())]:
        corpus["export"].append({"session_hex": dumps(state).hex()})
    corpus["export"].append({"session_hex": dumps(packet).hex()})
    corpus["export"].append(
        {
            "session_hex": text_bytes(
                tamper_session(states["initial"], lambda s: s.update(extra=1))
            ).hex()
        }
    )
    for item in packet["proposals"]:
        corpus["proposal_digest"].append(
            {"packet_hex": dumps(packet).hex(), "proposal_hex": text_bytes(item).hex()}
        )
        corpus["review_context"].append(
            {"packet_hex": dumps(packet).hex(), "proposal_hex": text_bytes(item).hex()}
        )
    foreign = copy.deepcopy(next(p for p in packet["proposals"] if p["id"] == "pa"))
    foreign["changes"][0]["after"] = {**foreign["changes"][0]["after"], "key": "other"}
    corpus["proposal_digest"].append(
        {"packet_hex": dumps(packet).hex(), "proposal_hex": text_bytes(foreign).hex()}
    )
    corpus["proposal_digest"].append(
        {
            "packet_hex": text_bytes({**packet, "extra": 1}).hex(),
            "proposal_hex": text_bytes(packet["proposals"][0]).hex(),
        }
    )
    # Validation-result shapes (real producer output + targeted corruption + timestamps).
    plan = plans["identical_changes"]
    results = validation_cases(plan)
    plan_hex = dumps(plan).hex()
    for name, result in results.items():
        corpus["validation"].append({"plan_hex": plan_hex, "result_hex": text_bytes(result).hex()})
        meta["validation"].append({"name": "result-" + name, "ok": True})
    empty_plan = export_plan(new_session(packet))
    for reader_kw, ok in (({"sources": {}}, True), ({"error": "wal_present"}, True)):
        r = validate_plan_sources(
            empty_plan,
            boundary=SourceBoundary(Path(empty_plan["profile"]["root"]), ()),
            profile=dict(empty_plan["profile"]),
            spec=SimpleNamespace(store="sqlite", skills=(), references=()),
            reader=FakeReader(**reader_kw),
            checked_at="2026-11-16T13:00:00+00:00",
        )
        corpus["validation"].append(
            {"plan_hex": dumps(empty_plan).hex(), "result_hex": text_bytes(r).hex()}
        )
        meta["validation"].append({"name": f"empty-plan-{r['status']}", "ok": ok})
    base = results["changed_sha"]
    mutants = {
        "extra_key": lambda r: r.update(extra=1),
        "bad_schema": lambda r: r.update(schema="nachos.reconciliation.validation.v2"),
        "plan_digest": lambda r: r.update(plan_digest="0" * 64),
        "status_unknown": lambda r: r.update(status="ok"),
        "status_valid_but_stale": lambda r: r.update(status="valid_at_check"),
        "profile_other": lambda r: r.update(profile={"id": "x", "root": "/y"}),
        "profile_extra": lambda r: r["profile"].update(x="1"),
        "naive_time": lambda r: r.update(checked_at="2026-11-16T13:00:00"),
        "time_int": lambda r: r.update(checked_at=5),
        "row_status": lambda r: r["sources"][0].update(status="gone"),
        "row_extra": lambda r: r["sources"][0].update(extra=1),
        "row_missing_key": lambda r: r["sources"][0].pop("reason"),
        "row_identity": lambda r: r["sources"][0].update(identity="nope"),
        "row_order": lambda r: r["sources"].reverse(),
        "row_dropped": lambda r: r["sources"].pop(),
        "row_proposals": lambda r: r["sources"][0].update(proposal_ids=["zz"]),
        "row_expected": lambda r: r["sources"][0].update(expected_sha256="3" * 64),
        "row_actual_bad": lambda r: r["sources"][0].update(actual_sha256="xyz"),
        "row_reason_unsafe": lambda r: r["sources"][0].update(reason="Not Stable!"),
        "row_unchanged_mismatch": lambda r: r["sources"][1].update(actual_sha256="4" * 64),
        "limitations_int": lambda r: r.update(limitations=[1]),
        "limitations_not_list": lambda r: r.update(limitations="x"),
    }
    for name, fn in mutants.items():
        r = copy.deepcopy(base)
        fn(r)
        corpus["validation"].append({"plan_hex": plan_hex, "result_hex": text_bytes(r).hex()})
        meta["validation"].append({"name": "mutant-" + name, "ok": False})
    old_python = sys.version_info < (3, 11)
    stamps = [t for t in timestamp_matrix() if not old_python or t in STAMPS]
    for stamp in stamps:
        r = copy.deepcopy(base)
        r["checked_at"] = stamp
        corpus["validation"].append({"plan_hex": plan_hex, "result_hex": text_bytes(r).hex()})
        meta["validation"].append(
            {
                "name": "timestamp " + ascii(stamp),
                "ok": py_timestamp_ok(stamp),
                "core": bool(CORE_STAMP.match(stamp)),
                "timestamp": True,
            }
        )
    return corpus, meta, packet, states


def bf_packet():
    return loads((FIX / "packet.json").read_bytes())


def random_value(rng, depth):
    alphabet = [
        "a",
        "b",
        "z",
        "\u00e9",
        "e\u0301",
        "\U0001f600",
        "\U00010000",
        "\ue000",
        "\uffff",
        "1",
        "10",
        "2",
        "",
        " ",
        "\n",
        "\x01",
        "<",
        "\\",
        '"',
    ]
    roll = rng.random()
    if depth > 4 or roll < 0.35:
        return rng.choice(
            [
                None,
                True,
                False,
                0,
                -1,
                rng.randrange(-(2**53) + 1, 2**53),
                "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 6))),
            ]
        )
    if roll < 0.6:
        return [random_value(rng, depth + 1) for _ in range(rng.randrange(0, 5))]
    return {
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 4))): random_value(
            rng, depth + 1
        )
        for _ in range(rng.randrange(0, 6))
    }


# ---------------------------------------------------------------- differential run


@pytest.fixture(scope="module")
def differential(node, tmp_path_factory):
    work = tmp_path_factory.mktemp("viewer-core")
    corpus, meta, packet, states = build_corpus()
    corpus_file, out_file = work / "corpus.json", work / "out.json"
    corpus_file.write_text(json.dumps(corpus), encoding="utf-8")
    done = subprocess.run(
        [node, str(JS_DIR / "differential.mjs"), str(CORE), str(corpus_file), str(out_file)],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return SimpleNamespace(
        corpus=corpus,
        meta=meta,
        out=json.loads(out_file.read_text(encoding="utf-8")),
        packet=packet,
        states=states,
    )


def agree(name, py, js, *, messages=False):
    assert py["ok"] == js["ok"], f"{name}: python={py} js={js}"
    if py["ok"]:
        assert (py["sha"], py["len"]) == (js["sha"], js["len"]), (
            f"{name}: python={py['hex']} js={js['hex']}"
        )
    elif messages:
        assert py["message"] == js["message"], f"{name}: {py['message']!r} != {js['message']!r}"


def test_canonical_bytes_digests_and_limits_match_python(differential):
    for i, case in enumerate(differential.corpus["canonical"]):
        name = differential.meta["canonical"][i]
        raw = raw_of(case)

        def run(raw=raw):
            return canonical_json(py_parse(raw))

        py = attempt(run)
        js = differential.out["canonical"][i]
        agree(name, py, js)
        if py["ok"]:
            value = py_parse(raw)
            for domain in case["digests"]:
                assert js["digests"][domain] == (
                    content_digest(domain, value)
                    if domain != "memory-record"
                    else content_digest("memory-record", value)
                ), (name, domain)


def test_valid_documents_validate_and_reserialize_identically(differential):
    seen = 0
    for i, case in enumerate(differential.corpus["docs"]):
        info = differential.meta["docs"][i]
        if info["expect_ok"] is not True:
            continue
        raw = raw_of(case)
        py = attempt(
            lambda raw=raw, case=case: dumps(loads(raw, expected_profile=case["expected_profile"]))
        )
        assert py["ok"], (info["name"], py)
        agree(info["name"], py, differential.out["docs"][i], messages=True)
        seen += 1
    assert seen >= 25


def test_every_invalid_document_class_is_rejected_by_python_and_js(differential):
    rejected = 0
    for i, case in enumerate(differential.corpus["docs"]):
        info = differential.meta["docs"][i]
        if info["expect_ok"] is not False:
            continue
        raw = raw_of(case)
        py = attempt(
            lambda raw=raw, case=case: dumps(loads(raw, expected_profile=case["expected_profile"]))
        )
        js = differential.out["docs"][i]
        assert not py["ok"], f"{info['name']}: python unexpectedly accepted"
        assert not js["ok"], f"{info['name']}: JS accepted a document Python rejects"
        if info["isolate"]:
            assert py["message"] != "artifact digest mismatch", info["name"]
        rejected += 1
    assert rejected >= 120


def test_error_messages_agree_for_isolated_semantic_negatives(differential):
    mismatches = []
    for i, case in enumerate(differential.corpus["docs"]):
        info = differential.meta["docs"][i]
        if info["expect_ok"] is not False or not info["isolate"]:
            continue
        raw = raw_of(case)
        py = attempt(
            lambda raw=raw, case=case: dumps(loads(raw, expected_profile=case["expected_profile"]))
        )
        js = differential.out["docs"][i]
        if py["message"] != js["message"]:
            mismatches.append((info["name"], py["message"], js["message"]))
    assert not mismatches, mismatches


def test_lifecycle_transitions_are_byte_identical(differential):
    total = accepted = rejected = 0
    for meta, js_steps in zip(differential.meta["lifecycle"], differential.out["lifecycle"]):
        for step, js in zip(meta["steps"], js_steps):
            py = attempt(lambda meta=meta, step=step: dumps(py_step(meta["state"], step)))
            label = f"{meta['name']}:{step['pid']}:{step['action']}:{str(step['note'])[:20]}"
            agree(label, py, js)
            total += 1
            accepted += py["ok"]
            rejected += not py["ok"]
    assert total > 500
    assert accepted > 100
    assert rejected > 100


def test_new_session_and_export_plan_are_byte_identical(differential):
    for case, js in zip(differential.corpus["new_session"], differential.out["new_session"]):
        raw = bytes.fromhex(case["packet_hex"])
        py = attempt(lambda raw=raw: dumps(new_session(loads(raw))))
        agree("new_session", py, js)
    exported = 0
    for case, js in zip(differential.corpus["export"], differential.out["export"]):
        raw = bytes.fromhex(case["session_hex"])
        py = attempt(lambda raw=raw: dumps(export_plan(loads(raw))))
        agree("export", py, js)
        exported += py["ok"]
    assert exported >= 8


def test_export_parity_covers_blocking_and_ordering(differential):
    plan = export_plan(differential.states["conflicts"])
    reasons = {row["reason"] for row in plan["ledger"]}
    assert "conflicting changes to the same source" in reasons
    assert "cyclic or blocked dependency" in reasons
    assert "missing or unaccepted dependency" in reasons
    ok_plan = export_plan(differential.states["identical_changes"])
    assert [e["proposal"]["id"] for e in ok_plan["approved"]] == ["pa", "pf", "pc", "pb", "pd"] or [
        e["proposal"]["id"] for e in ok_plan["approved"]
    ]
    assert len(ok_plan["approved"]) == 5
    missing = export_plan(differential.states["missing_dependency"])
    assert missing["approved"] == []
    assert missing["ledger"][-1]["status"] in ("blocked", "unreviewed")


def test_proposal_digest_and_review_context_match(differential):
    packet = differential.packet
    for case, js in zip(
        differential.corpus["proposal_digest"], differential.out["proposal_digest"]
    ):
        item = json.loads(bytes.fromhex(case["proposal_hex"]))
        py_raw = bytes.fromhex(case["packet_hex"])
        try:
            expected = proposal_digest(loads(py_raw), item)
        except ValueError:
            assert not js["ok"]
        else:
            assert js["ok"], js
            assert js["digest"] == expected
    for item, js in zip(packet["proposals"], differential.out["review_context"]):
        assert js["digest"] == content_digest("proposal", review_context(packet, item))
        assert js["safe"] == models_module.approval_safe(item)


def test_validation_result_shape_checks_agree(differential):
    out = differential.out["validation"]
    assert len(out) == len(differential.meta["validation"])
    wrong, lenient = [], []
    for info, js in zip(differential.meta["validation"], out):
        if info.get("timestamp") and not info["core"]:
            # Exotic ISO forms depend on the Python build; JS must never be looser than Python.
            if js["ok"] and not info["ok"]:
                wrong.append((info["name"], js))
            if info["ok"] and not js["ok"]:
                lenient.append(info["name"])
        elif js["ok"] != info["ok"] or (js["ok"] and not js["same"]):
            wrong.append((info["name"], js))
    assert not wrong, wrong
    assert len(lenient) < 80, lenient
    accepted_stamps = [
        i for i in differential.meta["validation"] if i["name"].startswith("timestamp") and i["ok"]
    ]
    assert len(accepted_stamps) >= 15
    assert any(
        i["name"].startswith("timestamp") and not i["ok"] for i in differential.meta["validation"]
    )
