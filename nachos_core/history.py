"""Search and retention over pre-compression transcript checkpoints.

The memory provider snapshots the full transcript before every compression
(see ``NachosMemoryProvider.on_pre_compress``). Those checkpoints are the
recovery path for anything a compressor summarized away. This module makes
them searchable without copying fragments into curated memory, so nothing
auto-generated ever lands in the always-on manifest.

Checkpoints overlap heavily (each one contains everything still in context
at that moment), so results are de-duplicated by message content.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[\w'-]{2,}", re.UNICODE)
_EXCERPT_CHARS = 400


def _terms(query: str) -> list[str]:
    return list(dict.fromkeys(w.lower() for w in _WORD_RE.findall(query or "")))


def message_text(message: dict[str, Any]) -> str:
    """Flatten a chat message (string or block content, plus tool calls) to text."""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
    for call in message.get("tool_calls") or []:
        fn = call.get("function") if isinstance(call, dict) else None
        if isinstance(fn, dict):
            parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
    return "\n".join(p for p in parts if p)


def _excerpt(text: str, terms: list[str]) -> str:
    lower = text.lower()
    positions = [lower.find(t) for t in terms if lower.find(t) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, anchor - _EXCERPT_CHARS // 4)
    end = min(len(text), start + _EXCERPT_CHARS)
    snippet = text[start:end].strip()
    return ("…" if start else "") + snippet + ("…" if end < len(text) else "")


def _iter_index(session_dir: Path) -> Iterator[dict[str, Any]]:
    index = session_dir / "index.jsonl"
    if not index.is_file():
        return
    with index.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _snapshots_newest_first(
    root: Path, session_ids: Iterable[str] | None
) -> list[tuple[Path, dict[str, Any]]]:
    if not root.is_dir():
        return []
    wanted = None if session_ids is None else set(session_ids)
    rows: list[tuple[Path, dict[str, Any]]] = []
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        for row in _iter_index(session_dir):
            if wanted is not None and row.get("session_id") not in wanted:
                continue
            rows.append((session_dir, row))
    rows.sort(key=lambda item: float(item[1].get("created_at", 0)), reverse=True)
    return rows


def search_history(
    root: Path,
    query: str,
    *,
    limit: int = 10,
    session_ids: Iterable[str] | None = None,
    max_snapshots: int = 200,
) -> list[dict[str, Any]]:
    """Return messages containing every query term, best match first.

    Ranking: number of term occurrences, then recency. Each distinct message
    text is returned once, attributed to the newest checkpoint holding it.
    ``session_ids`` restricts the search to those sessions (None = all; an
    empty iterable matches nothing).
    """
    terms = _terms(query)
    if not terms:
        return []
    seen: set = set()
    hits: list[tuple[int, float, dict[str, Any]]] = []
    for session_dir, row in _snapshots_newest_first(Path(root), session_ids)[:max_snapshots]:
        blob = session_dir / str(row.get("blob", ""))
        try:
            with gzip.open(blob, "rb") as f:
                messages = json.loads(f.read().decode("utf-8")).get("messages", [])
        except (OSError, ValueError) as e:
            logger.debug("Skipping unreadable checkpoint %s: %s", blob, e)
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            text = message_text(message)
            lower = text.lower()
            if not text or not all(t in lower for t in terms):
                continue
            digest = hashlib.sha1(  # dedupe key only, not a security use
                f"{message.get('role')}\0{text}".encode(), usedforsecurity=False
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            score = sum(lower.count(t) for t in terms)
            created = float(row.get("created_at", 0))
            hits.append(
                (
                    score,
                    created,
                    {
                        "session_id": row.get("session_id"),
                        "checkpoint_id": row.get("id"),
                        "checkpoint_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(created)),
                        "role": message.get("role"),
                        "excerpt": _excerpt(text, terms),
                    },
                )
            )
    hits.sort(key=lambda h: (-h[0], -h[1]))
    return [h[2] for h in hits[:limit]]


def checkpoint_stats(root: Path) -> dict[str, Any]:
    rows = _snapshots_newest_first(Path(root), None)
    present = [(d, r) for d, r in rows if (d / str(r.get("blob", ""))).is_file()]
    if not present:
        return {"count": 0, "sessions": 0, "oldest_days": None}
    oldest = min(float(r.get("created_at", 0)) for _d, r in present)
    return {
        "count": len(present),
        "sessions": len({d for d, _r in present}),
        "oldest_days": round((time.time() - oldest) / 86400, 1),
    }


def prune_checkpoints(root: Path, max_age_days: float) -> int:
    """Delete unlabeled checkpoints older than ``max_age_days``. Returns count.

    Age-based, not count-based: within one session the OLDEST checkpoints hold
    the content that later compressions summarized away, so keeping "the N
    newest" would delete exactly the history worth searching.
    """
    root = Path(root)
    if max_age_days <= 0 or not root.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for session_dir in [d for d in root.iterdir() if d.is_dir()]:
        rows = list(_iter_index(session_dir))
        keep = []
        for row in rows:
            if float(row.get("created_at", 0)) < cutoff and not row.get("label"):
                blob = session_dir / str(row.get("blob", ""))
                try:
                    blob.unlink()
                except FileNotFoundError:
                    pass
                except OSError as e:
                    logger.debug("Checkpoint delete failed for %s: %s", blob, e)
                    keep.append(row)
                    continue
                removed += 1
            else:
                keep.append(row)
        if len(keep) == len(rows):
            continue
        if keep:
            tmp = session_dir / "index.jsonl.tmp"
            tmp.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
            tmp.replace(session_dir / "index.jsonl")
        else:
            shutil.rmtree(session_dir, ignore_errors=True)
    return removed
