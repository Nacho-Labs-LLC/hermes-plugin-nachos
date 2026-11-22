"""Installed wheel -> real browser UI -> installed CLI, over a disposable source tree.

The packet comes from the installed ``nachos-reconcile`` CLI, the viewer is the INSTALLED
package's ``viewer_html()``, a proposal is accepted through visible UI actions, the exported plan
is validated by the installed CLI (valid), a real source file then drifts and validation reports
it stale. Directory entries and bytes of the source tree are proven unchanged through the UI
export and the valid ``validate`` run. Nothing here is skipped silently: a missing toolchain fails
under NACHOS_REQUIRE_BROWSER=1 (see conftest) and the wheel build itself raises ``LoopError``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path

import pytest
import wheel_loop
from browser_harness import gate_required
from viewer_ui import click, open_viewer, select, wait_decision

pytestmark = pytest.mark.usefixtures("browser_toolchain")

VIEWER_SNIPPET = """
import sys
from pathlib import Path
import nachos_core
from nachos_core.reconciliation.viewer import viewer_html
html = viewer_html()
data = html.read_bytes() if isinstance(html, Path) else html
data = data.encode("utf-8") if isinstance(data, str) else data
Path(sys.argv[1]).write_bytes(data)
print(nachos_core.__file__)
"""


def fs_state(root: Path) -> dict[str, tuple]:
    """Name, type, mode, mtime_ns, size and content hash of every entry (no symlink follow)."""
    out = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        kind = stat.S_IFMT(info.st_mode)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if stat.S_ISREG(kind) else None
        out[str(path.relative_to(root))] = (
            kind, stat.S_IMODE(info.st_mode), info.st_mtime_ns, info.st_size, digest
        )  # fmt: skip
    return out


@pytest.fixture(scope="module")
def installed():
    if not wheel_loop.has_wheel_toolchain():
        message = "no interpreter with pip+setuptools for the installed-wheel acceptance test"
        if gate_required():
            pytest.fail(f"BROWSER GATE FAILED (required): {message}", pytrace=False)
        pytest.skip(f"BROWSER GATE NOT RUN: {message}")
    scratch = wheel_loop.make_scratch("nachos-wheel-ui-")
    try:
        wheel = wheel_loop.build_wheel(scratch / "build")
        python = wheel_loop.make_venv(scratch / "venv")
        wheel_loop.install_wheel(python, wheel)
        yield scratch, python
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _make_packet(python, profile, work):
    flags = profile.flags()
    inventory = work / "inventory.json"
    wheel_loop.run_cli(python, ["inventory", *flags, "--out", str(inventory)], cwd=work)
    findings, proposals = wheel_loop.make_findings_and_proposals(
        json.loads(inventory.read_text(encoding="utf-8"))
    )
    (work / "findings.json").write_text(json.dumps(findings), encoding="utf-8")
    (work / "proposals.json").write_text(json.dumps(proposals), encoding="utf-8")
    packet = work / "packet.json"
    args = ["packet", "--inventory", str(inventory), "--out", str(packet)]
    args += ["--findings", str(work / "findings.json"), "--proposals", str(work / "proposals.json")]
    wheel_loop.run_cli(python, args, cwd=work)
    return packet


def _installed_viewer(python, work):
    target = work / "installed-viewer.html"
    done = wheel_loop._must(
        [python, "-I", "-c", VIEWER_SNIPPET, target],
        "installed viewer_html()",
        cwd=work,
        env=wheel_loop.isolated_env(work / ".iso"),
    )
    where = Path(done.stdout.strip()).resolve()
    assert python.parent.parent.resolve() in where.parents, where  # not the checkout
    assert Path(__file__).resolve().parents[2] not in where.parents
    return target


EXTERNAL_ATTR = re.compile(
    r"""\b(?:src|href|action|poster|srcset)\s*=\s*["']?\s*(?:https?:)?//""", re.I
)
SCRIPT_SRC = re.compile(r"<script\b[^>]*\bsrc\s*=", re.I)


def assert_no_external_references(html: str) -> None:
    """No http(s)/protocol-relative src/href attributes, no external script, no CSS url(http)."""
    assert EXTERNAL_ATTR.findall(html) == []
    assert SCRIPT_SRC.findall(html) == []
    assert re.findall(r"url\(\s*[\"']?\s*(?:https?:)?//", html, re.I) == []
    assert re.findall(r"@import\s+[\"']?\s*(?:https?:)?//", html, re.I) == []


def _validate(python, profile, plan, work, name, expect):
    out = work / name
    args = ["validate", "--plan", str(plan), *profile.flags(), "--out", str(out)]
    done = wheel_loop.run_cli(python, args, cwd=work, expect=expect)
    return done.stdout.strip(), json.loads(out.read_text(encoding="utf-8"))


def test_installed_ui_export_validates_then_detects_drift(viewer_page, installed):
    scratch, python = installed
    work = scratch / "work"
    work.mkdir()
    profile = wheel_loop.make_profile(scratch)
    packet = _make_packet(python, profile, work)
    viewer = _installed_viewer(python, work)
    assert_no_external_references(viewer.read_text(encoding="utf-8"))
    initial = fs_state(profile.home)

    open_viewer(viewer_page, viewer, packet)
    select(viewer_page, "p-skill")
    click(viewer_page, "accept")
    wait_decision(viewer_page, "p-skill", "accept")
    plan_path = viewer_page.capture_download(
        lambda: viewer_page.page.locator('[data-testid="export-plan"]').click()
    )
    viewer_page.assert_isolated()
    recorder = viewer_page.recorder
    assert recorder.page_errors == []
    assert [m for m in recorder.console if m[0] == "error"] == []
    outside = [
        u for u in recorder.requests if not u.lower().startswith(("file:", "data:", "blob:"))
    ]
    assert outside == [], outside
    assert not [u for u in recorder.requests if u.lower().startswith(("http:", "https:"))]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert [a["proposal"]["id"] for a in plan["approved"]] == ["p-skill"]
    assert fs_state(profile.home) == initial  # UI export wrote nothing into the source tree

    line, report = _validate(python, profile, plan_path, work, "validation-ok.json", 0)
    assert line.startswith("valid_at_check"), line
    assert report["plan_digest"] == plan["digest"]
    assert fs_state(profile.home) == initial  # validate is read-only (bytes, mtimes, entries)

    target = profile.home / wheel_loop.SKILL
    target.write_bytes(target.read_bytes() + b"# real drift\n")
    changed = {k for k, v in fs_state(profile.home).items() if initial.get(k) != v}
    assert changed == {wheel_loop.SKILL}
    stat_after = os.lstat(target)
    assert stat_after.st_size == initial[wheel_loop.SKILL][3] + len(b"# real drift\n")

    line, report = _validate(python, profile, plan_path, work, "validation-stale.json", 3)
    assert line.startswith("stale"), line
    statuses = {r["status"] for r in report["sources"]}
    assert "changed" in statuses
    assert report["plan_digest"] == plan["digest"]
    after_validate = {k for k, v in fs_state(profile.home).items() if initial.get(k) != v}
    assert after_validate == {wheel_loop.SKILL}  # stale validate mutated nothing further
