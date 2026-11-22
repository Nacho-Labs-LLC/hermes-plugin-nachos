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
import math
import re
import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[\w'-]{2,}", re.UNICODE)
_EXCERPT_CHARS = 400
_PART_CHARS = 600  # per context message; the matched message gets twice this
_PAIR_SCAN = 12  # how far to look for the other half of a tool call/result pair
_BM25_K1 = 1.2
_BM25_B = 0.75


_STOP_TEXT = (
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "do does did done have has had it its this that these those i we you he she they them our "
    "your my me us what which who whom whose when where why how can could should would will "
    "shall may might must not no yes so than then there here about into over also just any "
    "some all more most other such"
)
_STOPWORDS = frozenset(_STOP_TEXT.split())
_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(word: str) -> str:
    """Crude suffix strip so "counting"/"counted" match "count" (matching is by substring)."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _terms(query: str) -> list[str]:
    return list(dict.fromkeys(w.lower() for w in _WORD_RE.findall(query or "")))


def _search_terms(query: str) -> list[str]:
    """Distinct stemmed query words, minus stopwords (empty when nothing meaningful is left)."""
    words = [w for w in _terms(query) if w not in _STOPWORDS]
    return list(dict.fromkeys(_stem(w) for w in words))


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


def _excerpt(text: str, terms: list[str], size: int = _EXCERPT_CHARS) -> str:
    lower = text.lower()
    positions = [lower.find(t) for t in terms if lower.find(t) >= 0]
    anchor = min(positions) if positions else 0
    start = max(0, anchor - size // 4)
    end = min(len(text), start + size)
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


# -- refs --------------------------------------------------------------------
#
# A ref names one message of one checkpoint: ``<session_id>@<checkpoint_id>#<index>``. The
# checkpoint and index are optional (a bare ``<session_id>`` scopes a search to the session);
# memory provenance stores refs in this form and ``nachos_history_search(ref=...)`` opens them.


def make_ref(session_id: str, checkpoint_id: str | None = None, index: int | None = None) -> str:
    ref = str(session_id)
    if checkpoint_id:
        ref += f"@{checkpoint_id}"
        if index is not None:
            ref += f"#{index}"
    return ref


def parse_ref(ref: str) -> tuple[str, str | None, int | None]:
    """Split a ref into ``(session_id, checkpoint_id, message_index)``; missing parts are None."""
    text = (ref or "").strip()
    index: int | None = None
    head, sep, tail = text.rpartition("#")
    if sep and tail.isdigit():
        text, index = head, int(tail)
    session, at, checkpoint = text.rpartition("@")
    if not at:
        return text, None, None
    return session, checkpoint or None, index if checkpoint else None


# -- windows -----------------------------------------------------------------


def _call_ids(message: dict[str, Any]) -> set[str]:
    return {
        str(c["id"]) for c in message.get("tool_calls") or [] if isinstance(c, dict) and c.get("id")
    }


def _window(messages: list[Any], idx: int, before: int, after: int) -> set[int]:
    """Indexes around ``idx``, closed over tool-call/result pairs.

    A tool result without its assistant call (or a call without its results) is
    unreadable, so whichever half is missing from the plain window is pulled in.
    """
    picked = set(range(max(0, idx - before), min(len(messages), idx + after + 1)))
    for i in list(picked):
        m = messages[i]
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool" and m.get("tool_call_id"):
            for j in range(i - 1, max(-1, i - _PAIR_SCAN), -1):
                prev = messages[j]
                if isinstance(prev, dict) and str(m["tool_call_id"]) in _call_ids(prev):
                    picked.add(j)
                    break
        ids = _call_ids(m) if m.get("role") == "assistant" else set()
        if ids:
            for j in range(i + 1, min(len(messages), i + _PAIR_SCAN)):
                nxt = messages[j]
                is_result = isinstance(nxt, dict) and nxt.get("role") == "tool"
                if is_result and str(nxt.get("tool_call_id")) in ids:
                    picked.add(j)
    return picked


def _clip(text: str, limit: int, terms: list[str] | None = None) -> str:
    if len(text) <= limit:
        return text
    if terms:
        return _excerpt(text, terms, limit)
    return text[:limit].rstrip() + "…"


def _parts(
    messages: list[Any],
    indexes: list[int],
    matched: set[int],
    terms: list[str],
    part_chars: int,
) -> list[dict[str, Any]]:
    out = []
    for i in indexes:
        m = messages[i]
        text = message_text(m) if isinstance(m, dict) else ""
        is_match = i in matched
        out.append(
            {
                "index": i,
                "role": m.get("role") if isinstance(m, dict) else None,
                "match": is_match,
                "text": _clip(
                    text, part_chars * (2 if is_match else 1), terms if is_match else None
                ),
            }
        )
    return out


def _merge_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge hits of one checkpoint whose context windows overlap or touch."""
    merged: list[dict[str, Any]] = []
    by_cp: dict[str, list[dict[str, Any]]] = {}
    for w in windows:  # windows arrive best-first
        group = by_cp.setdefault(w["blob"], [])
        for existing in group:
            if existing["picked"] & w["picked"] or _touching(existing["picked"], w["picked"]):
                existing["picked"] |= w["picked"]
                existing["matched"] |= w["matched"]
                break
        else:
            group.append(w)
            merged.append(w)
    return merged


def _touching(a: set[int], b: set[int]) -> bool:
    return bool(a) and bool(b) and (max(a) + 1 == min(b) or max(b) + 1 == min(a))


def _blob_messages(blob: Path, cache: dict[Path, list[Any]]) -> list[Any]:
    if blob not in cache:
        try:
            with gzip.open(blob, "rb") as f:
                cache[blob] = json.loads(f.read().decode("utf-8")).get("messages", [])
        except (OSError, ValueError) as e:
            logger.debug("Skipping unreadable checkpoint %s: %s", blob, e)
            cache[blob] = []
    return cache[blob]


def _fit_budget(results: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    """Trim context parts (neighbours first) so the combined text stays under ``max_chars``."""
    budget = max_chars
    kept: list[dict[str, Any]] = []
    for r in results:
        size = sum(len(p["text"]) for p in r["context"])
        if size > budget:
            matches = [p for p in r["context"] if p["match"]]
            r["context"] = matches
            r["truncated"] = True
            size = sum(len(p["text"]) for p in matches)
        if size > budget:
            if not kept:  # always return the best hit, clipped
                for p in r["context"]:
                    p["text"] = _clip(p["text"], max(80, budget // max(1, len(r["context"]))))
                kept.append(r)
            break
        budget -= size
        kept.append(r)
    return kept


def _scan(
    rows: list[tuple[Path, dict[str, Any]]], terms: list[str]
) -> tuple[list[dict[str, Any]], dict[str, int], int, float]:
    """One pass over the checkpoints: per distinct message, its term counts and length."""
    seen: set[str] = set()
    cands: list[dict[str, Any]] = []
    df = dict.fromkeys(terms, 0)
    total_len = 0
    for session_dir, row in rows:
        blob = session_dir / str(row.get("blob", ""))
        for i, message in enumerate(_blob_messages(blob, {})):
            if not isinstance(message, dict):
                continue
            text = message_text(message)
            if not text:
                continue
            digest = hashlib.sha1(  # dedupe key only, not a security use
                f"{message.get('role')}\0{text}".encode(), usedforsecurity=False
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            lower = text.lower()
            total_len += len(lower)
            counts = [lower.count(t) for t in terms]
            for n, t in enumerate(terms):
                df[t] += counts[n] > 0
            if any(counts):
                cands.append(
                    {
                        "counts": counts,
                        "length": len(lower),
                        "created": float(row.get("created_at", 0)),
                        "session_dir": session_dir,
                        "row": row,
                        "blob": blob,
                        "index": i,
                    }
                )
    return cands, df, len(seen), total_len / max(1, len(seen))


def _rank(
    cands: list[dict[str, Any]],
    terms: list[str],
    df: dict[str, int],
    n_docs: int,
    avg_len: float,
    min_coverage: float,
) -> list[dict[str, Any]]:
    """IDF-weighted BM25 score + coverage filter; best first, recency breaks ties."""
    idf = [math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5)) for t in terms]
    total_idf = sum(idf)
    ranked = []
    for c in cands:
        covered = sum(w for w, n in zip(idf, c["counts"]) if n)
        coverage = covered / total_idf if total_idf else 0.0
        if coverage < min_coverage:
            continue
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * c["length"] / max(1.0, avg_len))
        c["score"] = sum(w * n * (_BM25_K1 + 1) / (n + norm) for w, n in zip(idf, c["counts"]) if n)
        c["coverage"] = coverage
        ranked.append(c)
    ranked.sort(key=lambda c: (-round(c["score"], 6), -c["created"], c["index"]))
    return ranked


def search_history(
    root: Path,
    query: str,
    *,
    limit: int = 10,
    session_ids: Iterable[str] | None = None,
    max_snapshots: int = 200,
    before: int = 1,
    after: int = 1,
    max_chars: int = 6000,
    min_coverage: float = 0.4,
) -> list[dict[str, Any]]:
    """Return matching messages with surrounding turns, best match first.

    Ranking: BM25 over the scanned messages (IDF computed from those messages, so
    rare words dominate), requiring a hit to cover at least ``min_coverage`` of the
    query's total IDF weight (one common word cannot match a multi-word query); ties
    go to the newer checkpoint. Each distinct message text is returned once,
    attributed to the newest checkpoint holding it. Every result carries the matched
    message plus ``before``/``after`` neighbouring turns (tool call/result pairs are
    kept whole), windows of one checkpoint that overlap are merged, and the combined
    context text is capped at ``max_chars``. ``session_ids`` restricts the search to
    those sessions (None = all; an empty iterable matches nothing).
    """
    terms = _search_terms(query)
    if not terms:
        return []
    rows = _snapshots_newest_first(Path(root), session_ids)[:max_snapshots]
    cands, df, n_docs, avg_len = _scan(rows, terms)
    ranked = _rank(cands, terms, df, n_docs, avg_len, min_coverage)
    windows: list[dict[str, Any]] = []
    cache: dict[Path, list[Any]] = {}
    for c in ranked[: max(limit, 1) * 3]:
        messages = _blob_messages(c["blob"], cache)
        c["picked"] = _window(messages, c["index"], max(0, before), max(0, after))
        c["matched"] = {c["index"]}
        windows.append(c)
    results = [
        _result(w, _blob_messages(w["blob"], cache), terms) for w in _merge_windows(windows)[:limit]
    ]
    return _fit_budget(results, max_chars)


def _result(w: dict[str, Any], messages: list[Any], terms: list[str]) -> dict[str, Any]:
    row = w["row"]
    created = float(row.get("created_at", 0))
    best = w["index"]
    best_msg = messages[best] if best < len(messages) else {}
    text = message_text(best_msg) if isinstance(best_msg, dict) else ""
    session_id, checkpoint_id = row.get("session_id"), row.get("id")
    return {
        "session_id": session_id,
        "checkpoint_id": checkpoint_id,
        "message_index": best,
        "ref": make_ref(str(session_id), str(checkpoint_id), best),
        "checkpoint_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(created)),
        "role": best_msg.get("role") if isinstance(best_msg, dict) else None,
        "score": round(w["score"], 3),
        "coverage": round(w["coverage"], 2),
        "excerpt": _excerpt(text, terms),
        "matched_indexes": sorted(w["matched"]),
        "context": _parts(messages, sorted(w["picked"]), w["matched"], terms, _PART_CHARS),
    }


def open_ref(
    root: Path,
    ref: str,
    *,
    before: int = 2,
    after: int = 2,
    max_chars: int = 6000,
) -> dict[str, Any] | None:
    """Context around the message a full ref (session@checkpoint#index) names, or None."""
    session_id, checkpoint_id, index = parse_ref(ref)
    if not (session_id and checkpoint_id and index is not None):
        return None
    for session_dir, row in _snapshots_newest_first(Path(root), [session_id]):
        if row.get("id") != checkpoint_id:
            continue
        messages = _blob_messages(session_dir / str(row.get("blob", "")), {})
        if not 0 <= index < len(messages):
            return None
        w = {
            "row": row,
            "index": index,
            "score": 0.0,
            "coverage": 1.0,
            "matched": {index},
            "picked": _window(messages, index, max(0, before), max(0, after)),
        }
        result = _result(w, messages, [])
        return _fit_budget([result], max_chars)[0]
    return None


def locate_quote(root: Path, session_id: str, quote: str) -> tuple[str, int] | None:
    """Find ``(checkpoint_id, message_index)`` of the newest checkpoint message containing ``quote``.

    Whitespace and case are normalised; None when no checkpoint of the session has it.
    """
    needle = " ".join((quote or "").lower().split())
    if len(needle) < 8:
        return None
    cache: dict[Path, list[Any]] = {}
    for session_dir, row in _snapshots_newest_first(Path(root), [session_id]):
        for i, m in enumerate(_blob_messages(session_dir / str(row.get("blob", "")), cache)):
            if isinstance(m, dict) and needle in " ".join(message_text(m).lower().split()):
                return str(row.get("id")), i
    return None


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
