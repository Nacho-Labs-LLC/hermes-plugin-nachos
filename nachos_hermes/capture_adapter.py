"""Hermes-side adapter for session-end capture.

``nachos_core.capture`` is host-agnostic (job queue, prompt, candidate parsing).
This module supplies what only the Hermes host can: the auxiliary-LLM callable,
transcript redaction when queueing a job, and the single background worker.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from nachos_core import capture as _capture
from nachos_core.memory import MemoryService

logger = logging.getLogger(__name__)

LLM = Callable[[list[dict[str, str]]], str]
IDLE_STATS = {"jobs": 0, "candidates": 0, "retry": 0}


def capture_llm(messages: list[dict[str, str]]) -> str:
    """Aux-LLM callable for nachos_core.capture; model from auxiliary.nachos_capture.*."""
    try:
        from agent.auxiliary_client import (
            call_llm,
            extract_content_or_reasoning,
        )
    except Exception as e:
        raise _capture.CaptureUnavailable(f"auxiliary client unavailable: {e}") from e
    try:
        response = call_llm(
            task="nachos_capture",
            messages=messages,
            max_tokens=1500,
            timeout=120,
            temperature=0.2,
        )
    except RuntimeError as e:  # "No LLM provider configured"
        raise _capture.CaptureUnavailable(str(e)) from e
    return extract_content_or_reasoning(response) or ""


USER_PROFILE_DELIMITER = "\n§\n"  # Hermes tools/memory_tool.py ENTRY_DELIMITER


def read_user_profile(hermes_home: Path | None) -> list[str]:
    """Entries of Hermes' USER.md (``<home>/memories/USER.md``); empty when missing."""
    if hermes_home is None:
        return []
    try:
        raw = (Path(hermes_home) / "memories" / "USER.md").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    return [e.strip() for e in raw.split(USER_PROFILE_DELIMITER) if e.strip()]


def queue_session(
    inbox_dir: Path,
    session_id: str,
    messages: list[dict[str, Any]] | None,
    redact: Callable[[str], str],
) -> bool:
    """Write a redacted transcript job to the inbox. True when a job was queued."""
    cleaned = [
        dict(m, content=redact(_capture.text_of(m.get("content"))))
        for m in messages or []
        if isinstance(m, dict)
    ]
    return _capture.write_job(inbox_dir, session_id, cleaned) is not None


class CaptureWorker:
    """Runs capture jobs under a lock, on at most one daemon thread at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def process(
        self,
        inbox_dir: Path,
        service: MemoryService,
        llm: LLM,
        user_facts: list[str] | None = None,
    ) -> dict[str, int]:
        """Process every pending job now (thread-free)."""
        with self._lock:
            return _capture.process_pending(inbox_dir, service, llm, user_facts=user_facts)

    def start(self, inbox_dir: Path, target: Callable[[], Any]) -> None:
        """Start the daemon worker running ``target`` if jobs are pending and none is running."""
        try:
            if self._thread is not None and self._thread.is_alive():
                return
            if not _capture.has_pending(inbox_dir):
                return
            self._thread = threading.Thread(target=target, name="nachos-capture", daemon=True)
            self._thread.start()
        except Exception as e:
            logger.debug("Nachos capture thread not started: %s", e)


def outstanding_count(inbox_dir: Path) -> int:
    return _capture.outstanding_count(inbox_dir)
