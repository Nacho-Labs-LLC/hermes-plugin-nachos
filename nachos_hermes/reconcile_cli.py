"""``nachos-reconcile``: offline inventory -> packet -> session -> export -> validate.

Single explicit profile, no network/LLM, no apply. Inputs are explicit files; outputs go
only to explicit new files (never overwritten, never inside the selected profile).
``cron-evidence`` is an explicit opt-in read of one profile cron file (never run, never
triggered); its observations are attached to a finding with ``packet --attach-evidence``.
Exit codes: 0 ok/valid, 2 any error, 3 stale, 4 unavailable.
"""

from __future__ import annotations

import argparse
import errno
import os
import secrets
import stat
import sys
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.reconciliation import dumps, export_plan, loads, new_session
from nachos_core.reconciliation.acquire import SourceSpec
from nachos_core.reconciliation.apply import apply_plan
from nachos_core.reconciliation.boundary import AcquisitionError, SourceBoundary
from nachos_core.reconciliation.cron import read_cron_evidence
from nachos_core.reconciliation.findings import Generated, generate_findings
from nachos_core.reconciliation.hashing import canonical_json
from nachos_core.reconciliation.inventory import build_inventory
from nachos_core.reconciliation.models import timestamp
from nachos_core.reconciliation.packets import build_packet, heuristic_findings, load_findings_json
from nachos_core.reconciliation.signals import collect_signals
from nachos_core.reconciliation.validate import dumps_validation, validate_plan_sources
from nachos_core.store import get_store
from nachos_hermes.config import hermes_redactor
from nachos_hermes.maintenance import open_readonly_service, resolve_profile_home

_MAX_INPUT = 8 * 1024 * 1024
_EXIT = {"valid_at_check": 0, "stale": 3, "unavailable": 4}
_APPLY_EXIT = {"planned": 0, "applied": 0, "refused": 3, "partial_failure": 5}
_CRON_LABEL = "cron/jobs.json"


def _read_input(path: Path) -> bytes:
    """Read one explicit regular input file, bounded; no symlinks."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_INPUT:
            raise ValueError(f"input must be a regular file under {_MAX_INPUT} bytes")
        return handle.read(_MAX_INPUT + 1)


def _write_new(path: Path, data: bytes, forbidden: Path | None) -> None:
    """Create a brand-new 0600 file atomically; never overwrite, never inside the profile.

    The bytes go to an O_EXCL temp file in the same directory which is then hard-linked to
    the final name (link refuses an existing name, including a symlink) and removed, so a
    failed write never leaves a partial or blocking ``--out``.
    """
    target = path.expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    parent = target.parent.resolve()
    if forbidden is not None:
        root = forbidden.resolve()
        if parent == root or root in parent.parents:
            raise ValueError("--out must be outside the selected profile")
    temp = parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, parent / target.name, follow_symlinks=False)
    finally:
        temp.unlink(missing_ok=True)


def _source_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--profile-home", type=Path, required=True, help="absolute profile home")
    sub.add_argument("--profile-id", help="label recorded in artifacts (default: dir name)")
    sub.add_argument("--store", choices=("sqlite", "flatfile"), default="sqlite")
    sub.add_argument("--skill", action="append", default=[], help="explicit skill file label")
    sub.add_argument("--reference", action="append", default=[], help="explicit reference label")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nachos-reconcile", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="read one profile's sources into an inventory")
    _source_args(inv)
    inv.add_argument("--out", type=Path, required=True)
    cron = commands.add_parser(
        "cron-evidence",
        help="opt-in: read one profile cron file into schedule/execution/delivery observations",
        description=(
            "Reads <profile>/cron/jobs.json (never runs or triggers jobs) and writes a JSON "
            "array of observations about the given packet source ids. Attach it to a "
            "finding with: packet --attach-evidence FINDING_ID=FILE."
        ),
    )
    cron.add_argument("--profile-home", type=Path, required=True, help="absolute profile home")
    cron.add_argument("--source-id", action="append", required=True, help="packet source id")
    cron.add_argument("--jobs-label", default=_CRON_LABEL, help="cron file label in the profile")
    cron.add_argument("--observed-at", required=True, help="ISO 8601 time with timezone")
    cron.add_argument("--out", type=Path, required=True)
    pkt = commands.add_parser("packet", help="bounded review packet from an inventory")
    pkt.add_argument("--inventory", type=Path, required=True)
    pkt.add_argument("--findings", type=Path, help="explicit findings JSON array")
    pkt.add_argument("--proposals", type=Path, help="explicit proposals JSON array")
    pkt.add_argument("--heuristics", action="store_true", help="add quality-heuristic findings")
    pkt.add_argument(
        "--audit",
        action="store_true",
        help="generate findings and proposals (merge, supersede, unlink, rewrite, delete, "
        "choose-one) from the offline audit; needs --profile-home and --observed-at",
    )
    pkt.add_argument("--profile-home", type=Path, help="absolute profile home (with --audit)")
    pkt.add_argument("--store", choices=("sqlite", "flatfile"), default="sqlite")
    pkt.add_argument("--observed-at", help="ISO 8601 time with timezone (required by --heuristics)")
    pkt.add_argument(
        "--attach-evidence",
        action="append",
        default=[],
        metavar="FINDING_ID=FILE",
        help="append the observations in FILE to that finding's evidence (repeatable)",
    )
    pkt.add_argument("--out", type=Path, required=True)
    ses = commands.add_parser("session", help="open an unreviewed review session")
    ses.add_argument("--packet", type=Path, required=True)
    ses.add_argument("--out", type=Path, required=True)
    exp = commands.add_parser("export", help="export approved proposals as an offline plan")
    exp.add_argument("--session", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    val = commands.add_parser("validate", help="explicit point-in-time source drift check")
    val.add_argument("--plan", type=Path, required=True)
    _source_args(val)
    val.add_argument("--out", type=Path, required=True)
    app = commands.add_parser(
        "apply",
        help="apply an accepted plan through MemoryService (dry-run unless --run)",
        description=(
            "Re-validates the plan against current sources, refuses on any drift, "
            "preflights every change, then (with --run) applies through MemoryService so "
            "redaction, the quality gate and the change log apply. Undo with "
            "`nachos-maintain`-adjacent history/revert (tools/memory_audit.py revert ID, "
            "or /nachos-memory-revert). Memory records only."
        ),
    )
    app.add_argument("--plan", type=Path, required=True)
    _source_args(app)
    app.add_argument("--run", action="store_true", help="write (default is a dry run)")
    app.add_argument(
        "--allow-remove", action="store_true", help="permit 'disable' changes (removes the entry)"
    )
    app.add_argument(
        "--allow-unredacted",
        action="store_true",
        help="proceed when Hermes' secret redactor is unavailable (not recommended)",
    )
    app.add_argument("--out", type=Path, required=True, help="where to write the apply report")
    vwr = commands.add_parser("viewer", help="write the offline review viewer (one HTML file)")
    vwr.add_argument("--out", type=Path, required=True)
    vwr.add_argument("--profile-home", type=Path, help="refuse an --out inside this profile")
    skl = commands.add_parser("skill", help="write the companion workflow skill (SKILL.md)")
    skl.add_argument("--out", type=Path, required=True)
    skl.add_argument("--profile-home", type=Path, help="refuse an --out inside this profile")
    return parser


def _context(args: argparse.Namespace) -> tuple[SourceBoundary, dict[str, str], SourceSpec]:
    home = _home(args)
    profile = {"id": args.profile_id or home.name, "root": str(home)}
    spec = SourceSpec(args.store, tuple(args.skill), tuple(args.reference))
    boundary = SourceBoundary(home, allowed_labels=_labels(spec))
    return boundary, profile, spec


def _home(args: argparse.Namespace) -> Path:
    home: Path = args.profile_home.expanduser()
    if not home.is_absolute():
        raise ValueError("--profile-home must be an absolute path")
    return home


def _labels(spec: SourceSpec) -> tuple[str, ...]:
    memory = "nachos/memory.db" if spec.store == "sqlite" else "nachos/memory.md"
    return (memory, *spec.skills, *spec.references)


def _root(artifact: dict[str, Any]) -> Path:
    return Path(artifact["profile"]["root"])


def _cmd_inventory(args: argparse.Namespace) -> int:
    boundary, profile, spec = _context(args)
    inventory = build_inventory(boundary, profile, spec)
    _write_new(args.out, dumps(inventory), args.profile_home)
    return 0


def _cmd_cron_evidence(args: argparse.Namespace) -> int:
    home = _home(args)
    boundary = SourceBoundary(home, allowed_labels=(args.jobs_label,))
    items = read_cron_evidence(
        boundary, args.source_id, observed_at=args.observed_at, jobs_label=args.jobs_label
    )
    _write_new(args.out, canonical_json(items), home)
    return 0


def _attach(
    findings: list[dict[str, Any]], specs: list[str], read: Callable[[Path], bytes]
) -> None:
    """Append observation files to named findings' evidence (explicit mapping only)."""
    by_id = {f.get("id"): f for f in findings}
    for spec in specs:
        finding_id, sep, name = spec.partition("=")
        if not sep or not name or finding_id not in by_id:
            raise ValueError("--attach-evidence needs FINDING_ID=FILE naming a known finding")
        target = by_id[finding_id]
        target["evidence"] = [*target.get("evidence", []), *load_findings_json(read(Path(name)))]


def _audit(args: argparse.Namespace, inventory: dict[str, Any]) -> Generated:
    if args.profile_home is None or args.observed_at is None:
        raise ValueError("--audit requires --profile-home and --observed-at")
    home = resolve_profile_home(args.profile_home)
    if str(home) != inventory["profile"]["root"]:
        raise ValueError("--profile-home does not match the inventory's profile")
    service, _path, _index = open_readonly_service(home, args.store)
    try:
        signals = collect_signals(service)
    finally:
        service.close()
    return generate_findings(inventory, signals, observed_at=args.observed_at)


def _cmd_packet(args: argparse.Namespace) -> int:
    inventory = loads(_read_input(args.inventory))
    findings: list[dict[str, Any]] = (
        load_findings_json(_read_input(args.findings)) if args.findings else []
    )
    if args.observed_at is not None:
        timestamp(args.observed_at)
    if args.heuristics:
        if args.observed_at is None:
            raise ValueError("--heuristics requires --observed-at")
        findings += heuristic_findings(inventory, observed_at=args.observed_at)
    proposals: list[dict[str, Any]] = (
        load_findings_json(_read_input(args.proposals)) if args.proposals else []
    )
    if args.audit:
        found = _audit(args, inventory)
        findings += found.findings
        proposals += found.proposals
        print(
            f"audit: {len(found.findings)} findings, {len(found.proposals)} proposals, "
            f"{len(found.skipped)} skipped",
            file=sys.stderr,
        )
    _attach(findings, args.attach_evidence, _read_input)
    packet = build_packet(inventory, findings=findings, proposals=proposals)
    _write_new(args.out, dumps(packet), _root(packet))
    return 0


def _cmd_session(args: argparse.Namespace) -> int:
    session = new_session(loads(_read_input(args.packet)))
    _write_new(args.out, dumps(session), _root(session))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    plan = export_plan(loads(_read_input(args.session)))
    _write_new(args.out, dumps(plan), _root(plan))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    plan = loads(_read_input(args.plan))
    boundary, profile, spec = _context(args)
    result = validate_plan_sources(
        plan,
        boundary=boundary,
        profile=profile,
        spec=spec,
    )
    _write_new(args.out, dumps_validation(result), args.profile_home)
    print(f"{result['status']} plan_digest={result['plan_digest']}")
    return _EXIT[result["status"]]


def _cmd_apply(args: argparse.Namespace) -> int:
    boundary, profile, spec = _context(args)
    plan = loads(_read_input(args.plan))
    redact = hermes_redactor()
    if redact is None and not args.allow_unredacted:
        raise ValueError(
            "Hermes' secret redactor is unavailable; run inside the Hermes environment "
            "or pass --allow-unredacted"
        )
    home = _home(args).resolve()
    path = home / "nachos" / ("memory.md" if args.store == "flatfile" else "memory.db")
    if not path.is_file():
        raise ValueError(f"no store at {path}")
    scope = Scope(PROFILE, get_store(args.store, path), MemoryIndex(index_path_for(path)))
    service = MemoryService(OrderedDict([(PROFILE, scope)]), redact=redact)
    service.session_id = "nachos-reconcile"
    try:
        report = apply_plan(
            plan, service, boundary, profile, spec, run=args.run, allow_remove=args.allow_remove
        )
    finally:
        service.close()
    _write_new(args.out, canonical_json(report), args.profile_home)
    print(f"{report['status']} actions={len(report['actions'])} refusals={len(report['refusals'])}")
    return _APPLY_EXIT[report["status"]]


def _forbidden_home(args: argparse.Namespace) -> Path | None:
    return args.profile_home.expanduser() if args.profile_home else None


def _cmd_viewer(args: argparse.Namespace) -> int:
    from nachos_core.reconciliation.viewer import viewer_html

    _write_new(args.out, viewer_html(), _forbidden_home(args))
    return 0


def _cmd_skill(args: argparse.Namespace) -> int:
    source = Path(__file__).parent / "skills" / "reconcile" / "SKILL.md"
    _write_new(args.out, source.read_bytes(), _forbidden_home(args))
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "inventory": _cmd_inventory,
    "cron-evidence": _cmd_cron_evidence,
    "packet": _cmd_packet,
    "session": _cmd_session,
    "export": _cmd_export,
    "validate": _cmd_validate,
    "apply": _cmd_apply,
    "viewer": _cmd_viewer,
    "skill": _cmd_skill,
}


def _run(args: argparse.Namespace) -> int:
    return _COMMANDS[args.command](args)


def _reason(exc: Exception) -> str:
    """Stable, content-free reason: never echo quoted or multi-line exception text."""
    if isinstance(exc, OSError) and not isinstance(exc, AcquisitionError):
        return errno.errorcode.get(exc.errno or 0, "OSError")
    text = str(exc)
    safe = (
        isinstance(exc, (AcquisitionError, ValueError))
        and 0 < len(text) <= 120
        and text.isprintable()
        and not any(c in text for c in "'\"`")
    )
    return text if safe else type(exc).__name__


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (AcquisitionError, ValueError, OSError, KeyError, RecursionError) as exc:
        print(f"nachos-reconcile: {args.command} failed: {_reason(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
