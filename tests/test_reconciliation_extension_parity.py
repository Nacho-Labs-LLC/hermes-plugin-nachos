"""Python and the viewer's JS core agree on the supersede/unlink operations and the optional
proposal ``kind``/``confidence`` fields (accept the valid forms, reject every malformed one)."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from nachos_core.reconciliation.hashing import content_digest
from nachos_core.reconciliation.models import validate_document
from nachos_core.reconciliation.packets import build_packet
from tests.test_reconciliation_findings import _generate

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "nachos_core/reconciliation/viewer/core.js"

_RUNNER = """
const C = require(process.argv[2]);
const cases = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
(async () => {
  const out = [];
  for (const c of cases) {
    try { await C.validateDocument(c.doc); out.push(true); } catch (e) { out.push(false); }
  }
  process.stdout.write(JSON.stringify(out));
})();
"""


def _node():
    exe = shutil.which("node") or (
        "/usr/local/bin/node" if os.access("/usr/local/bin/node", os.X_OK) else None
    )
    if exe is None:
        if os.environ.get("NACHOS_ALLOW_NO_NODE") == "1":
            pytest.skip("node missing; explicitly allowed")
        pytest.fail("node is required (NACHOS_ALLOW_NO_NODE=1 skips)")
    return exe


def _reseal(packet):
    payload = {k: v for k, v in packet.items() if k != "digest"}
    return {**payload, "digest": content_digest("packet", payload)}


def _variants(tmp_path):
    inventory, generated = _generate(tmp_path)
    packet = build_packet(inventory, findings=generated.findings, proposals=generated.proposals)
    link_index = next(i for i, p in enumerate(packet["proposals"]) if p["kind"] == "supersede")

    def mutated(name, edit, ok):
        doc = copy.deepcopy(packet)
        edit(doc["proposals"][link_index], doc)
        return {"name": name, "doc": _reseal(doc), "ok": ok}

    def change(doc):
        return doc["changes"][0]

    return [
        {"name": "valid", "doc": packet, "ok": True},
        mutated("no kind or confidence", lambda p, _d: (p.pop("kind"), p.pop("confidence")), True),
        mutated("bad kind", lambda p, _d: p.update(kind="wipe"), False),
        mutated("bad confidence", lambda p, _d: p.update(confidence="certain"), False),
        mutated(
            "unknown target", lambda p, _d: change(p)["after"].update(superseded_by="nope"), False
        ),
        mutated(
            "self target",
            lambda p, _d: change(p)["after"].update(superseded_by=change(p)["before"]["key"]),
            False,
        ),
        mutated("extra after key", lambda p, _d: change(p)["after"].update(extra="x"), False),
        mutated("null after", lambda p, _d: change(p).update(after=None), False),
        mutated("unknown operation", lambda p, _d: change(p).update(operation="rename"), False),
    ]


def _python_verdict(doc):
    try:
        validate_document(doc)
    except ValueError:
        return False
    return True


def test_python_and_js_agree_on_link_operations_and_optional_fields(tmp_path):
    cases = _variants(tmp_path)
    (tmp_path / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    (tmp_path / "run.js").write_text(_RUNNER, encoding="utf-8")
    done = subprocess.run(
        [_node(), str(tmp_path / "run.js"), str(CORE), str(tmp_path / "cases.json")],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    js = json.loads(done.stdout)
    py = [_python_verdict(c["doc"]) for c in cases]
    expected = [c["ok"] for c in cases]
    assert py == expected, [c["name"] for c, a, b in zip(cases, py, expected) if a != b]
    assert js == expected, [c["name"] for c, a, b in zip(cases, js, expected) if a != b]
