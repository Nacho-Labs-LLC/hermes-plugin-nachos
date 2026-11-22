"""Installed-wheel end-to-end loop on a disposable synthetic profile.

Builds a wheel from a scratch COPY of the tree (offline: ``pip wheel --no-deps
--no-build-isolation --no-index``), installs it into a fresh ``--without-pip`` venv, then drives
the real installed ``nachos-reconcile``: inventory -> packet -> session -> [review step] ->
export -> validate -> mutate one source -> validate again (expects exit 3). Everything lives in
per-run scratch directories under $TMPDIR that are removed afterwards.

CLI:  python tests/browser/wheel_loop.py [--keep]   (see scripts/loop_wheel.sh)
      LOOP_VIEWER_STEP="cmd args"  runs between `session` and `export` and must write the
      reviewed session to $LOOP_REVIEWED_SESSION (env: LOOP_PACKET, LOOP_SESSION,
      LOOP_REVIEWED_SESSION, LOOP_WORKDIR, LOOP_PROFILE_HOME, LOOP_VENV_PYTHON).
Library: build_wheel(), make_venv(), install_wheel(), make_profile(), run_cli(), run_loop().
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOW = "2026-11-16T12:00:00+00:00"
SKILL = "skills/demo/SKILL.md"
LEGACY = "skills/legacy/SKILL.md"
REFERENCE = "references/notes.md"
COPY_ITEMS = ("pyproject.toml", "README.md", "LICENSE", "nachos_core", "nachos_hermes")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".pytest_cache", "build")
MAX_STEP_SECONDS = 600
REVIEW_SNIPPET = """
import sys
from nachos_core.reconciliation import dumps, loads, review_proposal
session = loads(open(sys.argv[1], "rb").read())
for item in session["reviews"]:
    session = review_proposal(session, item["proposal_id"], "accept", note="loop default review")
open(sys.argv[2], "xb").write(dumps(session))
"""
INSTALLED_SNIPPET = (
    "import nachos_core, nachos_hermes.reconcile_cli as c; print(nachos_core.__file__)"
)


class LoopError(RuntimeError):
    pass


def scratch_root() -> Path:
    return Path(os.environ.get("TMPDIR") or tempfile.gettempdir())


def make_scratch(prefix: str = "nachos-wheel-loop-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=scratch_root()))


def clean_env() -> dict[str, str]:
    """Parent environment without PYTHON* (a PYTHONPATH of the checkout would mask the wheel)."""
    return {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}


def isolated_env(base: Path) -> dict[str, str]:
    """Env for pip/build/install: HOME, caches and TMPDIR live inside ``base``; no real HOME."""
    for name in ("home", "cache", "tmp"):
        (base / name).mkdir(parents=True, exist_ok=True)
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "HOME": str(base / "home"),
        "XDG_CACHE_HOME": str(base / "cache"),
        "PIP_CACHE_DIR": str(base / "cache" / "pip"),
        "TMPDIR": str(base / "tmp"),
        "PYTHONNOUSERSITE": "1",
    }


def _run(cmd, **kwargs) -> subprocess.CompletedProcess:
    kwargs.setdefault("env", clean_env())
    return subprocess.run(  # noqa: PLW1510 -- callers inspect returncode
        [str(c) for c in cmd], capture_output=True, text=True, timeout=MAX_STEP_SECONDS, **kwargs
    )


def _must(cmd, what: str, **kwargs) -> subprocess.CompletedProcess:
    done = _run(cmd, **kwargs)
    if done.returncode != 0:
        raise LoopError(f"{what} failed ({done.returncode}):\n{done.stdout}\n{done.stderr}")
    return done


def build_python() -> str:
    """An interpreter with pip AND setuptools for the offline wheel build."""
    candidates = [os.environ.get("NACHOS_WHEEL_BUILD_PYTHON"), sys.executable]
    candidates += [shutil.which("python3"), shutil.which("python")]
    for exe in filter(None, candidates):
        probe = _run([exe, "-I", "-c", "import setuptools, pip"])
        if probe.returncode == 0:
            return exe
    raise LoopError(
        "no interpreter with pip+setuptools for `pip wheel --no-build-isolation --no-index`; "
        "set NACHOS_WHEEL_BUILD_PYTHON"
    )


def has_wheel_toolchain() -> bool:
    try:
        build_python()
    except LoopError:
        return False
    return True


def build_wheel(dest: Path, *, tree: Path = ROOT) -> Path:
    """Build a wheel from a scratch copy of ``tree``; the checkout is never written."""
    dest.mkdir(parents=True, exist_ok=True)
    copy = dest / "src"
    copy.mkdir()
    for name in COPY_ITEMS:
        source = tree / name
        if source.is_dir():
            shutil.copytree(source, copy / name, ignore=IGNORE)
        elif source.exists():
            shutil.copy2(source, copy / name)
    out = dest / "wheels"
    _must(
        [
            build_python(),
            "-I",
            *("-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index"),
            *("--disable-pip-version-check", "--no-cache-dir", "-w", out, copy),
        ],
        "wheel build",
        env=isolated_env(dest / "iso"),
    )
    (wheel,) = out.glob("*.whl")
    check_wheel(wheel, tree)
    return wheel


def check_wheel(wheel: Path, tree: Path = ROOT, *, strict: bool | None = None) -> None:
    """The wheel ships the CLI entry point, the skill and every non-Python viewer asset.

    Missing viewer assets are an error when ``strict`` (default: NACHOS_REQUIRE_BROWSER=1 or
    LOOP_STRICT_ASSETS=1), otherwise a loud warning on stderr.
    """
    if strict is None:
        strict = "1" in (
            os.environ.get("NACHOS_REQUIRE_BROWSER"),
            os.environ.get("LOOP_STRICT_ASSETS"),
        )
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        points = next(
            (archive.read(n).decode() for n in names if n.endswith("entry_points.txt")), ""
        )
    if "nachos-reconcile" not in points:
        raise LoopError("wheel lacks the nachos-reconcile entry point")
    if "nachos_hermes/skills/reconcile/SKILL.md" not in names:
        raise LoopError("wheel lacks nachos_hermes/skills/reconcile/SKILL.md")
    viewer = tree / "nachos_core" / "reconciliation" / "viewer"
    if viewer.is_dir():
        missing = [
            str(p.relative_to(tree))
            for p in sorted(viewer.rglob("*"))
            if p.is_file()
            and "__pycache__" not in p.parts
            and str(p.relative_to(tree)) not in names
        ]
        if missing and strict:
            raise LoopError(f"wheel is missing viewer files (package-data?): {missing}")
        if missing:
            print(
                f"WARNING: wheel is missing viewer files (package-data?): {missing}",
                file=sys.stderr,
            )


def make_venv(dest: Path) -> Path:
    _must(
        [sys.executable, "-I", "-m", "venv", "--without-pip", dest],
        "venv creation",
        env=isolated_env(dest.parent / f"{dest.name}-iso"),
    )
    return dest / "bin" / "python"


def install_wheel(venv_python: Path, wheel: Path) -> None:
    venv = venv_python.parent.parent
    env = isolated_env(venv.parent / f"{venv.name}-iso-install")
    _must(
        [
            build_python(),
            "-I",
            *("-m", "pip", "--python", venv_python, "install", "--no-deps", "--no-cache-dir"),
            *("--no-index", "--ignore-installed", "--disable-pip-version-check", wheel),
        ],
        "wheel install",
        env=env,
    )
    done = _must([venv_python, "-I", "-c", INSTALLED_SNIPPET], "installed import check", env=env)
    where = Path(done.stdout.strip()).resolve()
    if venv_python.parent.parent.resolve() not in where.parents:
        raise LoopError(f"nachos_core imported from outside the venv: {where}")


@dataclass
class Profile:
    home: Path
    store: str = "sqlite"
    skills: tuple[str, ...] = (SKILL, LEGACY)
    references: tuple[str, ...] = (REFERENCE,)

    def flags(self) -> list[str]:
        out = ["--profile-home", str(self.home), "--store", self.store]
        for label in self.skills:
            out += ["--skill", label]
        for label in self.references:
            out += ["--reference", label]
        return out


def make_profile(root: Path) -> Profile:
    """A synthetic profile: sqlite memory store, two skills and one reference file."""
    home = root / "profile"
    (home / "nachos").mkdir(parents=True)
    conn = sqlite3.connect(home / "nachos" / "memory.db")
    conn.execute(
        "CREATE TABLE entries (key TEXT PRIMARY KEY, title TEXT, summary TEXT,"
        " category TEXT, body TEXT)"
    )
    rows = [
        ("k1", "Alpha", "alpha summary", "general", "Decision: stay offline.\nPreference: brief."),
        ("k2", "Beta <b>", "beta summary", "ops", "beta body \U0001f600"),
    ]
    conn.executemany("INSERT INTO entries VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    for label, text in (
        (SKILL, "# Demo\nold step\nkeep this guardrail\n"),
        (LEGACY, "# Legacy\nredundant skill\n"),
        (REFERENCE, "reference line one\r\nreference line two\r\n"),
    ):
        path = home / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return Profile(home)


def run_cli(venv_python: Path, args, *, cwd: Path, expect: int | None = 0):
    """Run the INSTALLED console script with a scrubbed environment (no PYTHONPATH)."""
    exe = venv_python.parent / "nachos-reconcile"
    env = isolated_env(cwd / ".iso")
    env["PATH"] = f"{venv_python.parent}:/usr/bin:/bin"
    done = _run([exe, *args], cwd=cwd, env=env)
    if expect is not None and done.returncode != expect:
        raise LoopError(
            f"nachos-reconcile {args[0]} exited {done.returncode}, wanted {expect}:\n"
            f"{done.stdout}\n{done.stderr}"
        )
    return done


def snapshot_tree(home: Path) -> dict[str, str]:
    """Every directory entry and every file's bytes (sha256), relative to ``home``."""
    out = {}
    for path in sorted(home.rglob("*")):
        key = str(path.relative_to(home))
        out[key] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "<directory>"
        )
    return out


def _evidence(sid: str) -> dict:
    return {
        "kind": "manual",
        "status": "verified",
        "source_ids": [sid],
        "claim": "reviewed by the loop script",
        "observation": "synthetic loop fixture",
        "observed_at": NOW,
    }


def _entry(src: dict, fid: str, pid: str, excerpt: str, after, op: str = "edit"):
    finding = {
        "id": fid,
        "type": "stale",
        "source_ids": [src["id"]],
        "excerpts": [{"source_id": src["id"], "text": excerpt}],
        "rationale": "synthetic loop finding",
        "evidence": [_evidence(src["id"])],
        "limitations": ["synthetic"],
    }
    proposal = {
        "id": pid,
        "revision": 1,
        "finding_ids": [fid],
        "depends_on": [],
        "assessments": [
            {
                "finding_id": fid,
                "revision": 1,
                "evidence": [_evidence(src["id"])],
                "preservation_checklist": [
                    {"item": "intent", "preserved": True, "explanation": "unchanged purpose"}
                ],
            }
        ],
        "changes": [
            {
                "source_id": src["id"],
                "source_sha256": src["sha256"],
                "operation": op,
                "before": src["before"],
                "after": after,
            }
        ],
    }
    return finding, proposal


def make_findings_and_proposals(inventory: dict) -> tuple[list[dict], list[dict]]:
    """Explicit findings/proposals as plain JSON built from an inventory (no contract API)."""
    by_identity = {s["identity"]: s for s in inventory["sources"]}
    skill, legacy = by_identity[f"skill:{SKILL}"], by_identity[f"skill:{LEGACY}"]
    ref, mem = by_identity[f"reference:{REFERENCE}"], by_identity["memory:k1"]
    parts = [
        _entry(skill, "f-skill", "p-skill", "old step", skill["before"].replace("old", "new")),
        _entry(legacy, "f-legacy", "p-legacy", "redundant skill", None, "disable"),
        _entry(
            ref, "f-ref", "p-ref", "reference line one", ref["before"] + "reference line three\r\n"
        ),
        _entry(
            mem,
            "f-mem",
            "p-mem",
            "Preference: brief.",
            {**mem["before"], "body": mem["before"]["body"].replace("brief", "brief, with gaps")},
        ),
    ]
    return [f for f, _ in parts], [p for _, p in parts]


@dataclass
class LoopPaths:
    work: Path
    profile: Profile
    venv_python: Path
    wheel: Path | None = None
    inventory: Path | None = None
    packet: Path | None = None
    session: Path | None = None
    reviewed: Path | None = None
    plan: Path | None = None
    validation_ok: Path | None = None
    validation_stale: Path | None = None
    results: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, str]:
        return {
            k: str(v) for k, v in vars(self).items() if isinstance(v, Path) and k != "profile"
        } | {"profile_home": str(self.profile.home)}


def default_review(paths: LoopPaths) -> None:
    """Review step used without a viewer: accept every proposal via the INSTALLED API."""
    assert paths.session is not None
    _must(
        [paths.venv_python, "-I", "-c", REVIEW_SNIPPET, paths.session, paths.reviewed],
        "default review",
        env=isolated_env(paths.work / ".iso"),
    )


def command_step(command: str) -> Callable[[LoopPaths], None]:
    """Turn LOOP_VIEWER_STEP into a review step (argv split, never a shell)."""

    def step(paths: LoopPaths) -> None:
        env = {
            **isolated_env(paths.work / ".iso-step"),
            "LOOP_PACKET": str(paths.packet),
            "LOOP_SESSION": str(paths.session),
            "LOOP_REVIEWED_SESSION": str(paths.reviewed),
            "LOOP_WORKDIR": str(paths.work),
            "LOOP_PROFILE_HOME": str(paths.profile.home),
            "LOOP_VENV_PYTHON": str(paths.venv_python),
        }
        _must(shlex.split(command), "LOOP_VIEWER_STEP", cwd=paths.work, env=env)

    return step


def _validate(paths: LoopPaths, out: Path, expect: int):
    assert paths.plan is not None
    args = ["validate", "--plan", paths.plan, *paths.profile.flags(), "--out", out]
    return run_cli(paths.venv_python, [str(a) for a in args], cwd=paths.work, expect=expect)


def _drive(paths: LoopPaths, review: Callable[[LoopPaths], None]) -> None:
    py, work, flags = paths.venv_python, paths.work, paths.profile.flags()
    before = snapshot_tree(paths.profile.home)
    paths.inventory = work / "inventory.json"
    run_cli(py, ["inventory", *flags, "--out", str(paths.inventory)], cwd=work)
    inventory = json.loads(paths.inventory.read_text(encoding="utf-8"))
    findings, proposals = make_findings_and_proposals(inventory)
    (work / "findings.json").write_text(json.dumps(findings), encoding="utf-8")
    (work / "proposals.json").write_text(json.dumps(proposals), encoding="utf-8")
    paths.packet = work / "packet.json"
    paths.session = work / "session.json"
    paths.reviewed = work / "session-reviewed.json"
    paths.plan = work / "plan.json"
    run_cli(
        py,
        [
            *("packet", "--inventory", str(paths.inventory), "--out", str(paths.packet)),
            *("--findings", str(work / "findings.json")),
            *("--proposals", str(work / "proposals.json")),
        ],
        cwd=work,
    )
    run_cli(py, ["session", "--packet", str(paths.packet), "--out", str(paths.session)], cwd=work)
    review(paths)
    if not paths.reviewed.is_file():
        raise LoopError("review step did not write the reviewed session")
    run_cli(py, ["export", "--session", str(paths.reviewed), "--out", str(paths.plan)], cwd=work)
    paths.validation_ok = work / "validation-ok.json"
    ok = _validate(paths, paths.validation_ok, 0)
    if not ok.stdout.startswith("valid_at_check"):
        raise LoopError(f"unexpected validate output: {ok.stdout!r}")
    if snapshot_tree(paths.profile.home) != before:
        raise LoopError("sources or directory entries changed during the read-only loop")
    target = paths.profile.home / SKILL
    target.write_bytes(target.read_bytes() + b"# drift\n")
    paths.validation_stale = work / "validation-stale.json"
    stale = _validate(paths, paths.validation_stale, 3)
    after = snapshot_tree(paths.profile.home)
    changed = {k for k in after if before.get(k) != after[k]}
    if changed != {SKILL} or after.keys() != before.keys():
        raise LoopError(f"unexpected profile change after mutation: {sorted(changed)}")
    paths.results.update(
        validate_ok=ok.stdout.strip(), validate_stale=stale.stdout.strip(), mutated=SKILL
    )


def run_loop(
    scratch: Path,
    viewer_step: Callable[[LoopPaths], None] | None = None,
    *,
    wheel: Path | None = None,
) -> LoopPaths:
    """Run the whole loop under ``scratch``; ``viewer_step`` replaces the default review."""
    wheel = wheel or build_wheel(scratch / "build")
    venv_python = make_venv(scratch / "venv")
    install_wheel(venv_python, wheel)
    work = scratch / "work"
    work.mkdir()
    paths = LoopPaths(work, make_profile(scratch), venv_python, wheel)
    _drive(paths, viewer_step or default_review)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    keep = "--keep" in args or os.environ.get("LOOP_KEEP") == "1"
    command = os.environ.get("LOOP_VIEWER_STEP")
    scratch = make_scratch()
    try:
        paths = run_loop(scratch, command_step(command) if command else None)
        print(json.dumps({**paths.summary(), **paths.results}, indent=2, sort_keys=True))
        print("loop OK" + (" (scratch kept)" if keep else " (scratch removed; use --keep)"))
        return 0
    except LoopError as exc:
        print(f"loop FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if not keep:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
