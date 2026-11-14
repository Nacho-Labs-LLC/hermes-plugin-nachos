"""Session-end capture: queue a trimmed transcript, extract memory candidates later.

Two phases, so the session-end hook stays fast and never needs an LLM:

1. ``write_job`` stores a trimmed transcript (user/assistant text only) as a
   gzipped JSON file under ``<inbox>/pending/``.
2. ``process_pending`` runs later (background thread or on demand). It claims
   each job by atomic rename (``pending`` -> ``processing``), asks an injected
   ``llm(messages) -> str`` for durable-fact candidates, runs them through the
   quality gate and redaction, and stores them in the index's ``inbox`` table
   for the user to approve or reject. Nothing is written to memory here.

No Hermes imports: the adapter supplies the ``llm`` callable.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from .memory import PROFILE, MemoryService

logger = logging.getLogger(__name__)

MIN_USER_MESSAGES = 3
MIN_TEXT_CHARS = 1500
MAX_MESSAGE_CHARS = 2000
MAX_TOTAL_CHARS = 60000
MAX_CANDIDATES = 5
MAX_ATTEMPTS = 2  # malformed output: first try + one retry
MAX_FAILURES = 5  # consecutive llm-call exceptions before a job is dropped
STALE_PROCESSING_SECONDS = 3600


class CaptureUnavailable(Exception):
    """The LLM backend is not available; jobs stay pending without a retry penalty."""


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in (
                None,
                "text",
                "input_text",
                "output_text",
            ):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def trim_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """User/assistant text only, each message truncated, newest kept under the total cap."""
    kept: list[dict[str, str]] = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        text = text_of(m.get("content")).strip()
        if not text:
            continue
        kept.append({"role": m["role"], "content": text[:MAX_MESSAGE_CHARS]})
    out: list[dict[str, str]] = []
    total = 0
    for m in reversed(kept):
        if total + len(m["content"]) > MAX_TOTAL_CHARS:
            break
        total += len(m["content"])
        out.append(m)
    out.reverse()
    return out


def is_trivial(messages: list[dict[str, Any]]) -> bool:
    users = [m for m in messages if m.get("role") == "user"]
    chars = sum(len(m.get("content", "")) for m in messages)
    return len(users) < MIN_USER_MESSAGES or chars < MIN_TEXT_CHARS


def _safe_name(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", session_id or "")[:100] or "session"


def write_job(
    inbox_dir: Path, session_id: str, messages: list[dict[str, Any]], now: float | None = None
) -> Path | None:
    """Queue a pending job; returns its path, or None for a trivial session."""
    trimmed = trim_messages(messages)
    if is_trivial(trimmed):
        return None
    pending = Path(inbox_dir) / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    path = pending / f"{_safe_name(session_id)}.json.gz"
    payload = {
        "session_id": session_id,
        "timestamp": now if now is not None else time.time(),
        "attempts": 0,
        "messages": trimmed,
    }
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    return path


def pending_jobs(inbox_dir: Path) -> list[Path]:
    return sorted(Path(inbox_dir, "pending").glob("*.json.gz"))


def has_pending(inbox_dir: Path) -> bool:
    return bool(pending_jobs(inbox_dir))


def outstanding_count(inbox_dir: Path) -> int:
    """Jobs not yet turned into candidates (pending or being processed)."""
    return len(pending_jobs(inbox_dir)) + len(list(Path(inbox_dir, "processing").glob("*.json.gz")))


def _recover_stale(inbox_dir: Path, now: float) -> None:
    """Return jobs abandoned in ``processing`` (crashed process) to ``pending``."""
    for path in Path(inbox_dir, "processing").glob("*.json.gz"):
        with contextlib.suppress(OSError):
            if now - path.stat().st_mtime > STALE_PROCESSING_SECONDS:
                os.replace(path, Path(inbox_dir, "pending", path.name))


def _claim(path: Path, inbox_dir: Path) -> Path | None:
    processing = Path(inbox_dir) / "processing"
    processing.mkdir(parents=True, exist_ok=True)
    target = processing / path.name
    try:
        os.rename(path, target)  # atomic: only one process wins
    except OSError:
        return None
    return target


def _load(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) and isinstance(data.get("messages"), list) else None
    except (OSError, ValueError, EOFError):
        return None


def _release(claimed: Path, inbox_dir: Path, data: dict[str, Any] | None) -> None:
    """Put a claimed job back in pending (with updated attempts when ``data`` is given)."""
    pending = Path(inbox_dir) / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    if data is not None:
        with gzip.open(claimed, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)
    os.replace(claimed, pending / claimed.name)


# -- prompt and parsing ----------------------------------------------------

_SYSTEM = (
    "You curate a long-term memory for an AI assistant. From the conversation, "
    "extract at most 5 DURABLE facts worth remembering across future sessions: "
    "projects, environment facts, decisions and their rationale, recurring "
    "procedures, and history. Ignore transient task chatter, one-off debugging, "
    "and anything already in the existing memory list. Never include secrets. "
    "Do NOT propose facts about the user's preferences, communication style or "
    "personal details: those belong to the separate user profile, which the host "
    "maintains. Do NOT propose persona or identity content (who the assistant is, "
    "its tone). Propose only durable knowledge for the memory layer. "
    "Answer with ONLY a JSON array (no prose, no code fences) of objects with "
    'keys "title" (short), "summary" (one keyword-dense line), "category" '
    '(one word, e.g. project, decision, environment, procedure), "body" '
    '(the full fact, self-contained) and "reason" (why it is durable). '
    "An empty array [] is a good answer when nothing qualifies."
)


def build_prompt(
    messages: list[dict[str, str]],
    manifest: list[Any],
    user_facts: list[str] | None = None,
) -> list[dict[str, str]]:
    known = "\n".join(f"- {e[1]}: {e[2]}" for e in manifest[:200]) or "(none)"
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    about_user = ""
    if user_facts:
        lines = "\n".join(f"- {f}" for f in user_facts[:100])
        about_user = f"Already known about the user (do not propose these or similar):\n{lines}\n\n"
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Existing memory (do not re-propose):\n{known}\n\n{about_user}"
                f"Conversation:\n{convo}"
            ),
        },
    ]


def parse_candidates(raw: str) -> list[dict[str, Any]] | None:
    """Parse the LLM answer into a list of dicts, or None if malformed."""
    text = (raw or "").strip()
    fence = re.match(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```\s*$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
    if not isinstance(data, list):
        return None
    return [d for d in data if isinstance(d, dict)]


def _field(service: MemoryService, item: dict[str, Any], name: str) -> str:
    value = item.get(name)
    return service.redact(value.strip()) if isinstance(value, str) else ""


def _duplicates(service: MemoryService, key: str, title: str, summary: str) -> list[dict[str, Any]]:
    """Existing entries a candidate would update or resemble (the review inbox shows these)."""
    dups: list[dict[str, Any]] = []
    if service.profile.store.get(key) is not None:
        dups.append(
            {"key": key, "scope": PROFILE, "title": title, "similarity": 1.0, "match": "update"}
        )
    try:
        dups.extend(
            d
            for d in service.find_similar(title, summary, exclude=(PROFILE, key))
            if d["key"] != key or d["scope"] != PROFILE
        )
    except Exception as e:
        logger.debug("Nachos capture duplicate check failed: %s", e)
    return dups


def _clean_candidates(items: list[dict[str, Any]], service: MemoryService) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for item in items[:MAX_CANDIDATES]:
        # prepare() applies the same redaction and quality gates as a real write.
        draft = service.prepare(
            title=_field(service, item, "title"),
            body=_field(service, item, "body"),
            summary=_field(service, item, "summary"),
            category=_field(service, item, "category"),
        )
        if not draft.key or draft.key in seen:
            continue
        if draft.error:
            logger.debug("Nachos capture rejected %r: %s", draft.title, draft.error)
            continue
        seen.add(draft.key)
        out.append(
            {
                "title": draft.title,
                "summary": draft.summary,
                "category": draft.category,
                "body": draft.body,
                "reason": _field(service, item, "reason"),
                "duplicates": _duplicates(service, draft.key, draft.title, draft.summary),
            }
        )
    return out


def _on_llm_failure(
    path: Path, inbox_dir: Path, data: dict[str, Any], e: Exception
) -> tuple[str, int]:
    """Count a failed LLM call; drop the job after MAX_FAILURES, else keep it pending."""
    failures = int(data.get("failures") or 0) + 1
    data["failures"] = failures
    if failures >= MAX_FAILURES:
        logger.warning(
            "Nachos capture dropped job %s after %d failed LLM calls (%s)",
            path.name,
            failures,
            e,
        )
        path.unlink(missing_ok=True)
        return "done", 0
    logger.debug("Nachos capture LLM call failed (%d/%d): %s", failures, MAX_FAILURES, e)
    _release(path, inbox_dir, data)
    return "error", 0


def _on_unparsed(
    path: Path, inbox_dir: Path, data: dict[str, Any], attempts: int
) -> tuple[str, int]:
    """The model answered but not with parseable candidates: retry, then drop."""
    data["attempts"] = attempts + 1
    if data["attempts"] >= MAX_ATTEMPTS:
        logger.warning(
            "Nachos capture dropped job %s after %d failed attempts",
            path.name,
            data["attempts"],
        )
        path.unlink(missing_ok=True)
        return "done", 0
    _release(path, inbox_dir, data)
    return "retry", 0


def process_job(
    path: Path,
    inbox_dir: Path,
    service: MemoryService,
    llm: Callable[[list[dict[str, str]]], str],
    user_facts: list[str] | None = None,
) -> tuple[str, int]:
    """Process one claimed job file.

    Returns ("done", candidates_stored), ("retry", 0) when the job went back to
    pending, ("unavailable", 0) when the LLM backend is missing, or ("error", 0) when the
    call raised (job kept pending with a failure count).
    """
    data = _load(path)
    if data is None:
        logger.warning("Nachos capture dropped unreadable job %s", path.name)
        path.unlink(missing_ok=True)
        return "done", 0
    attempts = int(data.get("attempts") or 0)
    try:
        raw = llm(build_prompt(data["messages"], service.manifest_entries(), user_facts))
    except CaptureUnavailable as e:
        logger.debug("Nachos capture LLM unavailable (%s); job stays pending", e)
        _release(path, inbox_dir, None)
        return "unavailable", 0
    except Exception as e:
        return _on_llm_failure(path, inbox_dir, data, e)
    if data.get("failures"):
        data["failures"] = 0  # only consecutive failures count
    items = parse_candidates(raw)
    if items is None:
        return _on_unparsed(path, inbox_dir, data, attempts)
    candidates = _clean_candidates(items, service)
    for c in candidates:
        service.profile.index.add_inbox(session_id=str(data.get("session_id") or ""), **c)
    path.unlink(missing_ok=True)
    return "done", len(candidates)


def process_pending(
    inbox_dir: Path,
    service: MemoryService,
    llm: Callable[[list[dict[str, str]]], str],
    now: float | None = None,
    user_facts: list[str] | None = None,
) -> dict[str, int]:
    """Process every pending job once. Thread-free; safe against concurrent processes."""
    now = now if now is not None else time.time()
    stats = {"jobs": 0, "candidates": 0, "retry": 0}
    _recover_stale(inbox_dir, now)
    service.profile.index.trim_inbox()
    for job in pending_jobs(inbox_dir):
        claimed = _claim(job, inbox_dir)
        if claimed is None:
            continue
        try:
            status, n = process_job(claimed, inbox_dir, service, llm, user_facts)
        except Exception as e:  # never lose a job to an unexpected bug
            logger.warning("Nachos capture failed on %s: %s", job.name, e)
            with contextlib.suppress(OSError):
                _release(claimed, inbox_dir, None)
            stats["retry"] += 1
            continue
        if status == "done":
            stats["jobs"] += 1
            stats["candidates"] += n
        else:
            stats["retry"] += 1
            if status in ("unavailable", "error"):
                break
    return stats
