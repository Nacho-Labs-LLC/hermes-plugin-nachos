"""Transcript checkpoints — gzipped copies of the conversation, one dir per session.

The memory provider writes one before each Hermes compression pass
(``NachosMemoryProvider.on_pre_compress``); ``nachos_core.history`` searches
and prunes them. They are raw transcripts, not redacted.

Storage layout:

    ~/.hermes/<profile>/nachos/snapshots/
      <session_id>/
        <timestamp>-<id>.json.gz       — gzipped checkpoint blob
        index.jsonl                    — append-only metadata index
"""

from __future__ import annotations

import builtins
import gzip
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint model
# ---------------------------------------------------------------------------


@dataclass
class SnapshotMeta:
    id: str
    session_id: str
    created_at: float  # unix epoch
    message_count: int
    token_estimate: int
    reason: str  # e.g. "pre-compress-checkpoint"
    label: str | None = None  # human-friendly tag
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Snapshot:
    meta: SnapshotMeta
    messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "messages": self.messages,
        }


# ---------------------------------------------------------------------------
# SnapshotStore — filesystem-backed, gzip+jsonl
# ---------------------------------------------------------------------------


class SnapshotStore:
    """Per-session snapshot store. One directory per session_id."""

    def __init__(self, base_dir: Path, session_id: str):
        if not session_id:
            raise ValueError("session_id is required")
        self.base_dir = Path(base_dir)
        self.session_id = session_id
        self.session_dir = self.base_dir / self._safe_session_id(session_id)
        self.index_file = self.session_dir / "index.jsonl"

    # -- Save -----------------------------------------------------------

    def save(
        self,
        messages: builtins.list[dict[str, Any]],
        reason: str = "manual",
        label: str | None = None,
        notes: builtins.list[str] | None = None,
    ) -> SnapshotMeta:
        """Persist a snapshot, return its metadata."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        snap_id = uuid.uuid4().hex[:12]
        ts = time.time()
        meta = SnapshotMeta(
            id=snap_id,
            session_id=self.session_id,
            created_at=ts,
            message_count=len(messages),
            token_estimate=_estimate_messages_tokens(messages),
            reason=reason,
            label=label,
            notes=list(notes or []),
        )
        snap = Snapshot(
            meta=meta,
            messages=messages,
        )

        ts_str = time.strftime("%Y%m%dT%H%M%S", time.gmtime(ts))
        blob_name = f"{ts_str}-{snap_id}.json.gz"
        blob_path = self.session_dir / blob_name

        try:
            payload = json.dumps(snap.to_dict(), default=str).encode("utf-8")
            with gzip.open(blob_path, "wb") as f:
                f.write(payload)
        except Exception as e:
            logger.warning("Snapshot save failed for %s: %s", blob_path, e)
            raise

        # Append to index
        try:
            with self.index_file.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "id": snap_id,
                            "blob": blob_name,
                            **meta.to_dict(),
                        }
                    )
                    + "\n"
                )
        except Exception as e:
            logger.warning("Snapshot index append failed: %s", e)

        logger.info(
            "Nachos snapshot saved: id=%s session=%s reason=%s msgs=%d ~tokens=%d",
            snap_id,
            self.session_id,
            reason,
            len(messages),
            meta.token_estimate,
        )
        return meta

    # -- List -----------------------------------------------------------

    def _read_index(self) -> builtins.list[dict[str, Any]]:
        """Read index entries without checking blob existence."""
        if not self.index_file.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with self.index_file.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug("Snapshot index read failed: %s", e)
            return []
        return rows

    def list(self) -> builtins.list[dict[str, Any]]:
        """Return index entries newest-first."""
        rows = self._read_index()

        # Filter to only entries whose blob still exists; sort newest first
        present = []
        for r in rows:
            blob = self.session_dir / r.get("blob", "")
            if blob.exists():
                present.append(r)
        present.sort(key=lambda r: float(r.get("created_at", 0)), reverse=True)
        return present

    # -- Helpers --------------------------------------------------------

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        """Sanitize for filesystem use."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)[:128]


def _estimate_block_tokens(blk: Any) -> int:
    """Estimate tokens for a single message block."""
    if isinstance(blk, dict):
        t = blk.get("text") or ""
        if isinstance(t, str):
            return len(t) // 4
    return 0


def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough estimate so the index doesn't lie about size."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 4
        elif isinstance(c, list):
            for blk in c:
                total += _estimate_block_tokens(blk)
    return total
