"""Handlers for the recall, put, remove and history tools.

Pure functions over a ``MemoryService`` (or the checkpoint store) and the few provider settings
they need; the provider owns dispatch, the primary-context gate and error mapping. Each returns
the JSON string handed back to Hermes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nachos_core.history import open_ref, parse_ref, search_history
from nachos_core.memory import PROFILE, MemoryService
from nachos_core.store.md_store import slugify

from .schemas import clamp_int

_NON_PRIMARY = json.dumps({"skipped": "non-primary agent context"})


def recall(service: MemoryService, args: dict[str, Any], *, abstain_mode: str) -> str:
    key = (args.get("key") or "").strip()
    scope = (args.get("scope") or "").strip() or None
    if key:
        entry = service.get_entry(key, scope)
        if entry is None:
            return json.dumps({"error": f"No entry with key {key!r}"})
        return json.dumps(entry)
    query = (args.get("query") or "").strip()
    raw_queries = args.get("queries")
    queries = [str(q) for q in raw_queries] if isinstance(raw_queries, list) else None
    entity = (args.get("entity") or "").strip()
    if not (query or entity or (queries and any(q.strip() for q in queries))):
        return json.dumps({"error": "Provide 'key', 'query', 'queries', or 'entity'."})
    return json.dumps(
        _search(service, args, query, queries, entity, scope, abstain_mode=abstain_mode)
    )


def _search(
    service: MemoryService,
    args: dict[str, Any],
    query: str,
    queries: list[str] | None,
    entity: str,
    scope: str | None,
    *,
    abstain_mode: str,
) -> dict[str, Any]:
    limit = clamp_int(args.get("limit"), 10, 1, 25)
    include_superseded = bool(args.get("include_superseded", False))
    if abstain_mode in ("hint", "on"):
        return service.search_with_confidence(
            query,
            queries=queries,
            limit=limit,
            scope=scope,
            include_superseded=include_superseded,
            entity=entity or None,
            mode=abstain_mode,
        )
    results = service.search(
        query,
        queries=queries,
        limit=limit,
        scope=scope,
        include_superseded=include_superseded,
        entity=entity or None,
    )
    out: dict[str, Any] = {
        "query": query or (queries[0].strip() if queries else ""),
        "count": len(results),
        "results": results,
    }
    if entity:
        out["entity"] = entity
    return out


def put(service: MemoryService, args: dict[str, Any], *, primary: bool) -> str:
    if not primary:
        return _NON_PRIMARY
    try:
        result = service.put(
            title=args.get("title") or "",
            body=args.get("body") or "",
            summary=args.get("summary") or "",
            category=args.get("category") or "general",
            scope=(args.get("scope") or PROFILE).strip(),
            supersedes=args.get("supersedes"),
        )
    except Exception as e:
        result = {"error": str(e)}
    return json.dumps(result)


def remove(service: MemoryService, args: dict[str, Any], *, primary: bool) -> str:
    """Raises ``KeyError`` for an unknown scope; the provider maps it to an error reply."""
    if not primary:
        return _NON_PRIMARY
    key = (args.get("key") or "").strip()
    if not key and args.get("title"):
        key = slugify(args["title"].strip())
    if not key:
        return json.dumps({"error": "Provide 'key' or 'title'."})
    scope = (args.get("scope") or "").strip() or None
    try:
        removed_from = service.remove(key, scope)
    except KeyError:
        raise
    except Exception as e:
        return json.dumps({"error": str(e)})
    if removed_from is None:
        return json.dumps({"error": f"No entry with key {key!r}"})
    return json.dumps({"removed": key, "scope": removed_from})


def history(
    args: dict[str, Any],
    *,
    checkpoint_root: Path | None,
    cfg: Mapping[str, Any],
    session_ids: Sequence[str],
    redact: Callable[[str], str],
) -> str:
    """Search or open checkpointed conversation history; ``session_ids`` scopes the search."""
    query = (args.get("query") or "").strip()
    ref = (args.get("ref") or "").strip()
    if not query and not ref:
        return json.dumps({"error": "query or ref is required"})
    if not checkpoint_root:
        return json.dumps({"error": "Nachos checkpoint store is not initialized"})
    before = clamp_int(args.get("before"), int(cfg["history_context_before"]), 0, 10)
    after = clamp_int(args.get("after"), int(cfg["history_context_after"]), 0, 10)
    max_chars = max(500, int(cfg["history_max_chars"]))
    session, checkpoint, _index = parse_ref(ref) if ref else ("", None, None)
    if ref and checkpoint:
        results = _open_checkpoint(checkpoint_root, ref, before, after, max_chars)
        if results is None:
            return json.dumps(
                {
                    "error": f"Checkpoint for ref {ref!r} is gone (checkpoints expire after "
                    f"checkpoint_max_age_days); search the session with query instead.",
                }
            )
    elif not query:
        return json.dumps({"error": "ref names no checkpoint message; add a query to search it"})
    else:
        sessions = None
        if session:
            sessions = [session]
        elif (args.get("scope") or "conversation") != "all":
            sessions = list(session_ids)
        results = search_history(
            checkpoint_root,
            query,
            limit=clamp_int(args.get("limit"), 8, 1, 25),
            session_ids=sessions,
            before=before,
            after=after,
            max_chars=max_chars,
        )
    return json.dumps(
        {"query": query, "count": len(results), "results": _redacted(results, redact)}
    )


def _open_checkpoint(
    root: Path, ref: str, before: int, after: int, max_chars: int
) -> list[dict[str, Any]] | None:
    result = open_ref(root, ref, before=before, after=after, max_chars=max_chars)
    return None if result is None else [result]


def _redacted(results: list[dict[str, Any]], redact: Callable[[str], str]) -> list[dict[str, Any]]:
    for row in results:
        row["excerpt"] = redact(row["excerpt"])
        for part in row["context"]:
            part["text"] = redact(part["text"])
    return results
