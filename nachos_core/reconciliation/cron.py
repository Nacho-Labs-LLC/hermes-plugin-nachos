"""Optional, explicit, read-only cron evidence adapter (schema verified from Hermes source).

Layout verified by reading the installed Hermes source (``cron/jobs.py``,
``cron/delivery_outcome.py``; ``create_job``/``mark_job_run``/``save_jobs``), not any
live data: ``<profile>/cron/jobs.json`` is ``{"jobs": [...], "updated_at": str}`` (a bare
list is also accepted, as Hermes' own loader does). Per job, only these fields are read
(others, such as the prompt, are never read into output):

- definition: ``id`` (str, required, unique), ``schedule`` (object with ``kind`` and
  ``display``), ``schedule_display`` (str), ``enabled`` (bool), ``state`` (str),
  ``deliver`` (str target spec; never echoed);
- execution state (set by ``mark_job_run``): ``last_run_at`` (ISO8601), ``last_status``
  (``ok``/``error``/``delivery_failed``/``delivery_queued``/...), ``last_error``;
- delivery state: ``last_delivery_error`` (null unless the send failed) and
  ``last_delivery_unverified``. Hermes records NO per-job delivery timestamp or receipt in
  ``jobs.json`` (receipts live in its execution ledger, which is not read here), and a
  silent or suppressed run also records ``ok``. So delivery is never ``verified`` here: a
  recorded failure is ``conflicting``; a run without a recorded error is ``unverified``.

Schedule, execution and delivery stay separate: a configured schedule never implies a run,
and a run never implies a delivery. An absent field means "unknown" (a ``missing``
observation), never a default. Error text, targets and names are never copied.

Nothing is executed, listed, discovered or followed; only one allowlisted label (default
``cron/jobs.json``) is read through the caller's ``SourceBoundary`` (bounded, strict JSON;
duplicate keys, floats, NUL, excessive depth/size rejected). A missing file raises
``AcquisitionError``. ``observed_at`` is required and explicit (ISO8601 with timezone).
"""

import json
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from .boundary import SourceBoundary
from .codec import _forbidden, _pairs
from .hashing import canonical_json
from .models import observations, timestamp

MAX_CRON_BYTES = 1024 * 1024
_TEXT_LIMIT = 200
_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_TEXT_FIELDS = (
    "schedule_display",
    "state",
    "deliver",
    "last_run_at",
    "last_status",
    "last_error",
    "last_delivery_error",
)


def _parse(data: bytes) -> Any:
    try:
        raw = data.decode("utf-8")
        if "\x00" in raw or "\\u0000" in raw:
            raise ValueError("NUL in cron JSON")
        value = json.loads(
            raw, object_pairs_hook=_pairs, parse_float=_forbidden, parse_constant=_forbidden
        )
        canonical_json(value)
    except (UnicodeError, RecursionError, TypeError) as exc:
        raise ValueError("invalid cron JSON") from exc
    return value


def _check_job(job: Any) -> None:
    if type(job) is not dict or type(job.get("id")) is not str or not job["id"]:
        raise ValueError("cron job requires a string id")
    for key in _TEXT_FIELDS:
        if job.get(key) is not None and type(job[key]) is not str:
            raise ValueError("cron job fields must be strings or null")
    if job.get("enabled") is not None and type(job["enabled"]) is not bool:
        raise ValueError("cron job enabled must be boolean or null")
    if job.get("schedule") is not None and type(job["schedule"]) not in (dict, str):
        raise ValueError("cron job schedule must be an object, string or null")


def _jobs(value: Any, max_jobs: int) -> list[dict[str, Any]]:
    if type(value) is dict and "jobs" in value and set(value) <= {"jobs", "updated_at"}:
        value = value["jobs"]
    if type(value) is not list or len(value) > max_jobs:
        raise ValueError("cron jobs must be a bounded list")
    seen = set()
    for job in value:
        _check_job(job)
        if job["id"] in seen:
            raise ValueError("duplicate cron job id")
        seen.add(job["id"])
    return value


def _short(value: str) -> str:
    return value if len(value) <= _TEXT_LIMIT else value[:_TEXT_LIMIT] + "..."


def _token(value: Any) -> str:
    return value if type(value) is str and _TOKEN.match(value) else "unrecognised"


def _aware(value: str) -> bool:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _item(
    kind: str, status: str, ids: Sequence[str], claim: str, text: str, at: str
) -> dict[str, Any]:
    return {
        "kind": kind,
        "status": status,
        "source_ids": list(ids),
        "claim": claim,
        "observation": text,
        "observed_at": at,
    }


def _display(job: dict[str, Any]) -> str:
    shown = job.get("schedule_display")
    schedule = job.get("schedule")
    if not shown and type(schedule) is dict and type(schedule.get("display")) is str:
        shown = schedule["display"]
    if not shown and type(schedule) is str:
        shown = schedule
    return shown or ""


def _schedule(job: dict[str, Any], ids: Sequence[str], at: str) -> dict[str, Any]:
    claim = f"cron job {_short(job['id'])} has a configured schedule"
    shown = _display(job)
    if not shown:
        return _item("schedule", "missing", ids, claim, "no schedule is configured", at)
    enabled = {True: "true", False: "false", None: "unrecorded"}[job.get("enabled")]
    text = (
        f"configured schedule (not execution evidence): {_short(shown)}; "
        f"enabled={enabled}; state={_token(job.get('state'))}"
    )
    return _item("schedule", "verified", ids, claim, text, at)


def _recorded(job: dict[str, Any]) -> tuple[str, str]:
    """Return (status, detail) for last_run_at, independent of schedule and delivery."""
    stamp = job.get("last_run_at")
    if not stamp:
        return "missing", "no last_run_at is recorded"
    if not _aware(stamp):
        return "unverified", "last_run_at is not an ISO8601 timestamp with a timezone"
    return "verified", f"last_run_at={_short(stamp)}"


def _execution(job: dict[str, Any], ids: Sequence[str], at: str) -> dict[str, Any]:
    claim = f"cron job {_short(job['id'])} ran"
    status, detail = _recorded(job)
    if status == "verified":
        detail += f"; last_status={_token(job.get('last_status'))}"
        if job.get("last_error"):
            detail += "; a run error is recorded (text not copied)"
    return _item("execution", status, ids, claim, detail, at)


def _delivery_state(job: dict[str, Any]) -> tuple[str, str]:
    if job.get("last_delivery_error") or job.get("last_status") == "delivery_failed":
        return "conflicting", "a delivery failure is recorded (text not copied)"
    if job.get("last_status") == "delivery_queued":
        return "unverified", "delivery was queued; no terminal outcome is recorded"
    if job.get("last_delivery_unverified"):
        return "unverified", "a target acknowledged the send without a message id"
    if type(job.get("deliver")) is str and job["deliver"] != "local":
        return "unverified", (
            "a delivery target is configured and no delivery error is recorded; jobs.json "
            "holds no delivery receipt and silent or suppressed runs also record ok"
        )
    return "missing", "no remote delivery target is configured or recorded"


def _delivery(job: dict[str, Any], ids: Sequence[str], at: str) -> dict[str, Any]:
    claim = f"cron job {_short(job['id'])} delivered its output"
    if not job.get("last_run_at"):
        return _item("delivery", "missing", ids, claim, "no run is recorded, so no delivery", at)
    status, detail = _delivery_state(job)
    return _item("delivery", status, ids, claim, detail, at)


def read_cron_evidence(
    boundary: SourceBoundary,
    source_ids: Sequence[str],
    *,
    observed_at: str,
    jobs_label: str = "cron/jobs.json",
    max_jobs: int = 64,
) -> list[dict[str, Any]]:
    """Return schedule/execution/delivery observations per job, in file order.

    ``source_ids`` names the packet sources the evidence is about (required, as the
    observation contract demands). ``observed_at`` is required (ISO8601 with timezone):
    it states when the caller read the file; no clock is consulted.
    """
    timestamp(observed_at)
    if type(max_jobs) is not int or not 0 < max_jobs <= 1024:
        raise ValueError("max_jobs out of range")
    ids = list(source_ids)
    data = boundary.read_file(jobs_label, max_bytes=MAX_CRON_BYTES).data
    result = []
    for job in _jobs(_parse(data), max_jobs):
        result += [
            _schedule(job, ids, observed_at),
            _execution(job, ids, observed_at),
            _delivery(job, ids, observed_at),
        ]
    if result:
        observations(result, ids)
    return result
