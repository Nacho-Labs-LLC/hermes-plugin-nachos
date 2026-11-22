"""The installed-wheel loop helpers work end to end (no browser needed)."""

import json
import shutil
import sys

import pytest
import wheel_loop
from browser_harness import gate_required

pytestmark = pytest.mark.no_browser_needed


@pytest.fixture(scope="module")
def built_wheel():
    if not wheel_loop.has_wheel_toolchain():
        message = "no interpreter with pip+setuptools for the wheel loop"
        if gate_required():
            pytest.fail(f"BROWSER GATE FAILED (required): {message}", pytrace=False)
        pytest.skip(f"BROWSER GATE NOT RUN: {message}")
    scratch = wheel_loop.make_scratch("nachos-wheel-test-")
    yield wheel_loop.build_wheel(scratch / "build")
    shutil.rmtree(scratch, ignore_errors=True)


@pytest.fixture
def scratch():
    path = wheel_loop.make_scratch("nachos-wheel-run-")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_default_loop_validates_then_detects_the_mutation(built_wheel, scratch):
    paths = wheel_loop.run_loop(scratch, wheel=built_wheel)
    assert paths.results["validate_ok"].startswith("valid_at_check ")
    assert paths.results["validate_stale"].startswith("stale ")
    assert paths.results["mutated"] == wheel_loop.SKILL
    plan = json.loads(paths.plan.read_text(encoding="utf-8"))
    assert len(plan["approved"]) == 4
    stale = json.loads(paths.validation_stale.read_text(encoding="utf-8"))
    assert {r["status"] for r in stale["sources"]} == {"changed", "unchanged"}
    assert stale["plan_digest"] == plan["digest"]


def test_viewer_step_hook_receives_paths(built_wheel, scratch):
    seen = {}

    def step(paths):
        seen["packet"] = paths.packet.is_file()
        seen["session"] = json.loads(paths.session.read_text(encoding="utf-8"))["schema"]
        wheel_loop.default_review(paths)

    wheel_loop.run_loop(scratch, step, wheel=built_wheel)
    assert seen == {"packet": True, "session": "nachos.reconciliation.session.v1"}


def test_command_hook_gets_environment(built_wheel, scratch):
    script = scratch / "hook.py"
    script.write_text(
        "import os, subprocess, sys\n"
        "subprocess.run([os.environ['LOOP_VENV_PYTHON'], '-I', '-c', sys.argv[1],"
        " os.environ['LOOP_SESSION'], os.environ['LOOP_REVIEWED_SESSION']], check=True)\n",
        encoding="utf-8",
    )
    command = f"{sys.executable} {script} '{wheel_loop.REVIEW_SNIPPET}'"
    paths = wheel_loop.run_loop(
        scratch / "run", wheel_loop.command_step(command), wheel=built_wheel
    )
    assert paths.reviewed.is_file()


def test_failed_review_step_is_an_error(built_wheel, scratch):
    with pytest.raises(wheel_loop.LoopError, match="did not write"):
        wheel_loop.run_loop(scratch / "run", lambda _paths: None, wheel=built_wheel)


def test_wheel_contents(built_wheel):
    wheel_loop.check_wheel(built_wheel, strict=False)
    import zipfile  # noqa: PLC0415

    names = zipfile.ZipFile(built_wheel).namelist()
    assert "nachos_hermes/skills/reconcile/SKILL.md" in names
    assert not any(n.startswith(("tests/", "docs/")) for n in names)
