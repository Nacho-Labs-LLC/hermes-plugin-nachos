"""Explicit, point-in-time source drift check for an exported plan. Never applies anything.

The caller supplies the trusted profile, the ``SourceBoundary`` and the source spec
(allowlist). Locators and identities inside an imported plan are only compared with
what the trusted reader returns; they are never used to read anything. The plan is
not mutated and ``source_validation`` is never set: the result is a separate,
unsealed record bound to one exact plan digest and one moment in time.

Per-source status: the default reader (``acquire.read_each_source``) reads each
explicit label independently, so one unreadable label marks only the sources recorded
under that label ``unavailable`` and real drift in the others is still reported. An
injected ``reader`` keeps the original contract (returns ``{identity: source}`` or
raises ``AcquisitionError``, which makes every source ``unavailable``) and may instead
return ``(sources_by_identity, failures_by_locator_label)`` for per-label results; a
failure reason ``source_not_found`` means the source itself is gone (``missing``).

Profile policy: a plan whose profile differs from the explicit profile, or an
explicit profile whose root differs from the boundary root, raises ValueError.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Optional

from .acquire import FLATFILE_LABEL, SQLITE_LABEL
from .boundary import AcquisitionError, SourceBoundary
from .hashing import canonical_json
from .models import SCHEMAS, timestamp, validate_document
from .models import profile as check_profile

VALIDATION_SCHEMA = "nachos.reconciliation.validation.v1"
LIMITATIONS = [
    "point-in-time check bound to this exact plan digest; it is not a lock or lease",
    "sources may change immediately after checked_at; re-validate before any later use",
    "hash comparison detects ordinary drift, not a malicious concurrent writer",
    "no changes were applied and the plan was not modified",
]

Reader = Callable[..., Any]


def _reviewed(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collect reviewed sources by identity; conflicting duplicates are malformed."""
    found: dict[str, dict[str, Any]] = {}
    for entry in plan["approved"]:
        for src in entry["sources"]:
            record = found.setdefault(
                src["identity"],
                {
                    "sha256": src["sha256"],
                    "kind": src["kind"],
                    "locator": src["locator"],
                    "proposals": set(),
                },
            )
            if (record["sha256"], record["kind"], record["locator"]) != (
                src["sha256"],
                src["kind"],
                src["locator"],
            ):
                raise ValueError("plan reviews one identity with inconsistent sources")
            record["proposals"].add(entry["proposal"]["id"])
    return found


def _reason(exc: AcquisitionError) -> str:
    code = str(exc)
    ok = code != "" and len(code) <= 64 and all(c.islower() or c == "_" for c in code)
    return code if ok else "acquisition_failed"


def _compare(expected: dict[str, Any], actual: Any) -> tuple[Optional[str], str, str]:
    if actual is None:
        return None, "missing", "source_not_found"
    if (actual.kind, actual.locator) != (expected["kind"], expected["locator"]):
        return actual.sha256, "changed", "kind_or_locator_changed"
    if actual.sha256 != expected["sha256"]:
        return actual.sha256, "changed", "sha256_changed"
    return actual.sha256, "unchanged", "sha256_match"


def _in_spec(spec: Any, locator: str) -> bool:
    """Whether the caller's allowlist covers a reviewed source's recorded locator."""
    memory = {"sqlite": SQLITE_LABEL, "flatfile": FLATFILE_LABEL}.get(getattr(spec, "store", ""))
    labels = {*getattr(spec, "skills", ()), *getattr(spec, "references", ())}
    if memory is not None:
        labels.add(memory)
    return locator in labels


def _row(
    identity: str, expected: dict[str, Any], actual: Any, unavailable: Optional[str]
) -> dict[str, Any]:
    if unavailable == "source_not_found":
        sha, status, reason = None, "missing", unavailable
    elif unavailable is not None:
        sha, status, reason = None, "unavailable", unavailable
    else:
        sha, status, reason = _compare(expected, actual)
    return {
        "identity": identity,
        "proposal_ids": sorted(expected["proposals"]),
        "expected_sha256": expected["sha256"],
        "actual_sha256": sha,
        "status": status,
        "reason": reason,
    }


def _overall(rows: list[dict[str, Any]], reader_failed: bool) -> str:
    if reader_failed:
        return "unavailable"
    statuses = {row["status"] for row in rows}
    if "unavailable" in statuses:
        return "unavailable"
    return "valid_at_check" if statuses <= {"unchanged"} else "stale"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_reader() -> Reader:
    from .acquire import read_each_source  # noqa: PLC0415 -- Optional sibling module.

    return read_each_source


def _bind(
    plan: dict[str, Any], explicit: dict[str, str], boundary: SourceBoundary
) -> dict[str, Any]:
    check_profile(explicit)
    checked = validate_document(plan, expected_profile=explicit)
    if checked["schema"] != SCHEMAS["plan"]:
        raise ValueError("validation requires an exported plan")
    if str(boundary.root) != explicit["root"]:
        raise ValueError("boundary root does not match explicit profile")
    return checked


def _read(
    reader: Optional[Reader], boundary: SourceBoundary, spec: Any
) -> tuple[dict[str, Any], dict[str, str], Optional[str]]:
    """Return (current sources, per-label failures, whole-read failure reason)."""
    try:
        read = reader if reader is not None else _default_reader()
        result = read(boundary, spec)
    except AcquisitionError as exc:
        return {}, {}, _reason(exc)
    current, failures = result if isinstance(result, tuple) else (result, {})
    return current, {label: _reason_text(code) for label, code in failures.items()}, None


def _reason_text(code: object) -> str:
    return _reason(AcquisitionError(code))


def _failure(
    spec: Any, expected: dict[str, Any], failures: dict[str, str], whole: Optional[str]
) -> Optional[str]:
    if whole is not None:
        return whole
    if not _in_spec(spec, expected["locator"]):
        return "not_in_spec"
    return failures.get(expected["locator"])


def validate_plan_sources(
    plan: dict[str, Any],
    *,
    boundary: SourceBoundary,
    profile: dict[str, str],
    spec: Any,
    reader: Optional[Reader] = None,
    checked_at: Optional[str] = None,
) -> dict[str, Any]:
    """Compare reviewed source hashes with current sources; return a detached result.

    Each reviewed source is ``unchanged``/``changed``/``missing``/``unavailable`` on its
    own. Overall: any ``unavailable`` gives ``unavailable``; else any ``changed`` or
    ``missing`` gives ``stale``; else ``valid_at_check`` (also for an empty plan whose
    reader works). Unavailable reasons are stable codes only.
    """
    checked = _bind(plan, profile, boundary)
    stamp = _now() if checked_at is None else checked_at
    timestamp(stamp)
    reviewed = _reviewed(checked)
    current, failures, whole = _read(reader, boundary, spec)
    rows = [
        _row(
            identity,
            reviewed[identity],
            current.get(identity),
            _failure(spec, reviewed[identity], failures, whole),
        )
        for identity in sorted(reviewed)
    ]
    return {
        "schema": VALIDATION_SCHEMA,
        "plan_digest": checked["digest"],
        "profile": dict(profile),
        "checked_at": stamp,
        "status": _overall(rows, whole is not None),
        "sources": rows,
        "limitations": list(LIMITATIONS),
    }


def dumps_validation(result: dict[str, Any]) -> bytes:
    """Canonical UTF-8 bytes of a validation result."""
    return canonical_json(result)
