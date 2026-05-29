"""Snapshot lifecycle — capture conversation state before destructive ops.

Hermes already has filesystem checkpoints (tools/checkpoint_manager.py)
for git-based file snapshots. Nachos snapshots are different: they
capture the CONVERSATION state — message list + system prompt hash +
PromptReport history — at a point in time, so the user can `restore`
back to before a bad compaction or risky tool sequence.

Storage layout:

    ~/.hermes/<profile>/nachos/snapshots/
      <session_id>/
        <timestamp>-<id>.json.gz       — gzipped snapshot blob
        index.jsonl                    — append-only metadata index

Each snapshot is one gzipped JSON file. The index lists them with
metadata (timestamp, message_count, reason, label) so /nachos
snapshots list is fast without unzipping anything.

Rotation: keep the N most recent per session by default. Manual
snapshots (label != None) are exempt from rotation by default —
the user asked for them.

Restore: load the JSON, return the message list. Caller decides
whether to atomically swap it into the live agent.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Snapshot model
# ---------------------------------------------------------------------------

@dataclass
class SnapshotMeta:
    id: str
    session_id: str
    created_at: float                  # unix epoch
    message_count: int
    token_estimate: int
    reason: str                        # "manual", "pre-compaction-aggressive", etc.
    label: Optional[str] = None        # human-friendly tag
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Snapshot:
    meta: SnapshotMeta
    messages: List[Dict[str, Any]]
    system_prompt_hash: Optional[str] = None
    prompt_reports_tail: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "messages": self.messages,
            "system_prompt_hash": self.system_prompt_hash,
            "prompt_reports_tail": self.prompt_reports_tail,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Snapshot":
        meta_d = d["meta"]
        meta = SnapshotMeta(
            id=meta_d["id"],
            session_id=meta_d["session_id"],
            created_at=float(meta_d["created_at"]),
            message_count=int(meta_d["message_count"]),
            token_estimate=int(meta_d.get("token_estimate", 0)),
            reason=meta_d.get("reason", ""),
            label=meta_d.get("label"),
            notes=list(meta_d.get("notes", [])),
        )
        return Snapshot(
            meta=meta,
            messages=d.get("messages", []),
            system_prompt_hash=d.get("system_prompt_hash"),
            prompt_reports_tail=d.get("prompt_reports_tail", []),
        )


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

    def save(self, messages: List[Dict[str, Any]],
             reason: str = "manual",
             label: Optional[str] = None,
             system_prompt_hash: Optional[str] = None,
             prompt_reports_tail: Optional[List[Dict[str, Any]]] = None,
             notes: Optional[List[str]] = None) -> SnapshotMeta:
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
            system_prompt_hash=system_prompt_hash,
            prompt_reports_tail=list(prompt_reports_tail or []),
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
                f.write(json.dumps({
                    "id": snap_id,
                    "blob": blob_name,
                    **meta.to_dict(),
                }) + "\n")
        except Exception as e:
            logger.warning("Snapshot index append failed: %s", e)

        logger.info(
            "Nachos snapshot saved: id=%s session=%s reason=%s msgs=%d ~tokens=%d",
            snap_id, self.session_id, reason, len(messages), meta.token_estimate,
        )
        return meta

    # -- List -----------------------------------------------------------

    def list(self) -> List[Dict[str, Any]]:
        """Return index entries newest-first."""
        if not self.index_file.exists():
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with self.index_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug("Snapshot index read failed: %s", e)
            return []

        # Filter to only entries whose blob still exists; sort newest first
        present = []
        for r in rows:
            blob = self.session_dir / r.get("blob", "")
            if blob.exists():
                present.append(r)
        present.sort(key=lambda r: float(r.get("created_at", 0)), reverse=True)
        return present

    # -- Load -----------------------------------------------------------

    def load(self, snap_id: str) -> Optional[Snapshot]:
        """Load a snapshot by id."""
        for entry in self.list():
            if entry.get("id") == snap_id:
                blob = self.session_dir / entry["blob"]
                try:
                    with gzip.open(blob, "rb") as f:
                        d = json.loads(f.read().decode("utf-8"))
                    return Snapshot.from_dict(d)
                except Exception as e:
                    logger.warning("Snapshot load failed for %s: %s", blob, e)
                    return None
        return None

    # -- Rotate ---------------------------------------------------------

    def rotate(self, keep: int = 10, keep_labeled: bool = True) -> int:
        """Delete oldest snapshots beyond `keep`. Returns count removed.

        When `keep_labeled` is True, snapshots with a non-None label
        are exempt from rotation. The user explicitly tagged them.
        """
        entries = self.list()
        if len(entries) <= keep:
            return 0

        # Sort oldest-first for deletion
        entries_old_first = sorted(entries, key=lambda r: float(r["created_at"]))
        deletable = [
            e for e in entries_old_first
            if not (keep_labeled and e.get("label"))
        ]

        # Now delete from the OLDEST deletable until we're at `keep` total
        target_remove = max(0, len(entries) - keep)
        if target_remove == 0 or not deletable:
            return 0

        removed = 0
        kept_ids = set(e["id"] for e in entries) - set(
            d["id"] for d in deletable[:target_remove]
        )

        for entry in deletable[:target_remove]:
            blob = self.session_dir / entry["blob"]
            try:
                if blob.exists():
                    blob.unlink()
                removed += 1
            except Exception as e:
                logger.debug("Snapshot rotate delete failed: %s", e)

        # Rewrite the index without the removed entries (atomic-ish)
        if removed:
            try:
                tmp = self.index_file.with_suffix(self.index_file.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for entry in self.list():  # filters present blobs
                        if entry["id"] in kept_ids:
                            f.write(json.dumps(entry) + "\n")
                tmp.replace(self.index_file)
            except Exception as e:
                logger.debug("Snapshot index rewrite failed: %s", e)

        logger.info("Nachos snapshot rotation: removed=%d session=%s",
                    removed, self.session_id)
        return removed

    # -- Helpers --------------------------------------------------------

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        """Sanitize for filesystem use."""
        return "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in session_id)[:128]


def _estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough estimate so the index doesn't lie about size."""
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 4
        elif isinstance(c, list):
            for blk in c:
                if isinstance(blk, dict):
                    t = blk.get("text") or ""
                    if isinstance(t, str):
                        total += len(t) // 4
    return total
