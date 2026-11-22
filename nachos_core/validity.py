"""Entry validity helpers: timestamp backfill, supersedes normalization, recency tie-break,
and long-superseded hygiene queries.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

from .index import MemoryIndex
from .lexical import LexicalIndex
from .links import resolve_successor
from .quality import qualifier_difference, update_signal
from .store.base import EntryMeta, MemoryStore

# Recency tie-break tolerance on RRF scores: covers exact ties and symmetric (0,1)/(1,0)
# hybrid rank swaps (1/60 + 1/62 - 2/61 ~= 8.8e-6) without overriding wider score gaps.
RECENCY_TIE_EPS = 1e-5
RECENCY_MIN_GAP_SECONDS = 60.0
UPDATE_CANDIDATE_LIMIT = 3
LEXICAL_UPDATE_THRESHOLD = 0.25
SEMANTIC_UPDATE_THRESHOLD = 0.48


def normalize_supersedes(supersedes: str | list[str] | None, self_key: str) -> list[str]:
    """Normalise a key or list of keys, dropping blanks, duplicates and ``self_key``."""
    if not supersedes:
        return []
    raw = [supersedes] if isinstance(supersedes, str) else list(supersedes)
    out: list[str] = []
    for item in raw:
        key = str(item or "").strip()
        if key and key != self_key and key not in out:
            out.append(key)
    return out


def apply_recency_tiebreak(
    items: list[tuple[tuple[str, str], dict[str, Any]]],
    metas_by_scope: Mapping[str, Mapping[str, EntryMeta]],
    *,
    eps: float = RECENCY_TIE_EPS,
    min_gap_seconds: float = RECENCY_MIN_GAP_SECONDS,
) -> list[tuple[tuple[str, str], dict[str, Any]]]:
    """Within near-equal fused score clusters (same match count), prefer newer ``updated_at``.

    Only reorders entries whose ``updated_at`` gap is at least ``min_gap_seconds`` so
    same-session or bulk-loaded entries preserve their original order.
    """
    if len(items) < 2:
        return items

    def _updated(entry: tuple[tuple[str, str], dict[str, Any]]) -> float:
        meta = metas_by_scope[entry[0][0]].get(entry[0][1])
        return meta.updated_at if meta is not None and meta.updated_at is not None else 0.0

    out: list[tuple[tuple[str, str], dict[str, Any]]] = []
    i = 0
    while i < len(items):
        lead_score = items[i][1]["score"]
        lead_matches = len(items[i][1]["match"])
        j = i + 1
        while (
            j < len(items)
            and len(items[j][1]["match"]) == lead_matches
            and lead_score - items[j][1]["score"] <= eps
        ):
            j += 1
        cluster = list(items[i:j])
        if len(cluster) > 1:
            for pos in range(1, len(cluster)):
                cur = cluster[pos]
                cur_ts = _updated(cur)
                k = pos
                while k > 0 and cur_ts - _updated(cluster[k - 1]) >= min_gap_seconds:
                    cluster[k] = cluster[k - 1]
                    k -= 1
                cluster[k] = cur
        out.extend(cluster)
        i = j
    return out


def backfill_scope_timestamps(store: MemoryStore, index: MemoryIndex) -> int:
    """Populate missing ``created_at``/``updated_at`` from changes, usage, or migration time."""
    if store.read_only or index.read_only:
        return 0
    metas = store.all_meta()
    missing = [k for k, m in metas.items() if m.created_at is None or m.updated_at is None]
    if not missing:
        return 0
    change_ts = index.change_timestamps()
    usage = index.usage()
    now = time.time()
    to_fill: dict[str, tuple[float, float, bool]] = {}
    for key in missing:
        if key in change_ts:
            created_at, updated_at = change_ts[key]
        elif key in usage and usage[key].get("first_seen"):
            created_at = updated_at = float(usage[key]["first_seen"])
        else:
            created_at = updated_at = now
        to_fill[key] = (created_at, updated_at, True)
    return store.backfill_timestamps(to_fill)


def active_key(key: str, metas: Mapping[str, EntryMeta]) -> str | None:
    """The active entry ``key`` resolves to; a key unknown to ``metas`` counts as active itself.

    None for a cyclic or dangling ``superseded_by`` chain (see ``links.resolve_successor``).
    """
    return resolve_successor(key, metas) if key in metas else key


def redirect_hits(
    hits: Iterable[tuple[str, float | None, int]],
    metas: Mapping[str, EntryMeta] | None,
    *,
    limit: int,
) -> list[tuple[str, float | None, int]]:
    """Substitute each superseded hit with its active successor (resetting chunk to 0 on redirect)."""
    out: list[tuple[str, float | None, int]] = []
    if metas is None:
        for item in hits:
            out.append(item)
            if len(out) >= limit:
                break
        return out
    seen: set[str] = set()
    for key, sim, chunk in hits:
        resolved = active_key(key, metas)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        out.append((resolved, sim, chunk if resolved == key else 0))
        if len(out) >= limit:
            break
    return out


def _raw_update_candidates(
    lexical_index: LexicalIndex,
    index: MemoryIndex,
    *,
    model_id: str | None,
    vector: list[float] | None,
    title: str,
    summary: str,
    body: str,
    excluded: set[str],
    semantic_threshold: float,
    lexical_threshold: float,
    pool_limit: int,
) -> list[tuple[str, float, tuple[str, ...]]]:
    if vector is not None and model_id is not None:
        raw = index.search(vector, model_id, top_k=pool_limit + len(excluded), head_only=True)
        out: list[tuple[str, float, tuple[str, ...]]] = [
            (k, sim, ()) for k, sim in raw if k not in excluded and sim >= semantic_threshold
        ]
        return out[:pool_limit]
    return lexical_index.similar_documents(
        title=title,
        summary=summary,
        body=body,
        threshold=lexical_threshold,
        exclude=excluded,
        limit=pool_limit,
    )


def possible_updates_for(
    store: MemoryStore,
    lexical_index: LexicalIndex,
    index: MemoryIndex,
    metas: Mapping[str, EntryMeta],
    *,
    model_id: str | None,
    vector: list[float] | None,
    self_key: str,
    title: str,
    summary: str,
    body: str,
    already_superseded: set[str],
    semantic_threshold: float = SEMANTIC_UPDATE_THRESHOLD,
    lexical_threshold: float = LEXICAL_UPDATE_THRESHOLD,
    limit: int = UPDATE_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` active entries ranked as possible older versions of this put."""
    excluded = already_superseded | {self_key} | {k for k, m in metas.items() if m.superseded_by}
    raw = _raw_update_candidates(
        lexical_index,
        index,
        model_id=model_id,
        vector=vector,
        title=title,
        summary=summary,
        body=body,
        excluded=excluded,
        semantic_threshold=semantic_threshold,
        lexical_threshold=lexical_threshold,
        pool_limit=max(limit * 4, 12),
    )
    if not raw:
        return []
    new_text = f"{title}\n{summary}\n{body}"
    scored: list[tuple[float, float, str, dict[str, Any]]] = []
    for key, sim, shared_terms in raw:
        doc = lexical_index.docs.get(key)
        old_body = store.get(key)
        if doc is None or old_body is None:
            continue
        boost, reason = update_signal(
            f"{doc.title}\n{old_body}",
            new_text,
            old_title=doc.title,
            new_title=title,
            shared_terms=shared_terms,
        )
        scored.append(
            (
                sim + boost,
                sim,
                key,
                {
                    "key": key,
                    "title": doc.title,
                    "summary": doc.summary,
                    "similarity": round(sim, 3),
                    "reason": reason,
                },
            )
        )
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [row for *_sort_key, row in scored[:limit]]


def superseded_entries(
    scopes: Mapping[str, tuple[MemoryStore, Mapping[str, EntryMeta]]],
    *,
    older_than_days: float = 30,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Return superseded entries older than ``older_than_days`` across scopes."""
    now = now if now is not None else time.time()
    cutoff = now - older_than_days * 86400
    rows: list[dict[str, Any]] = []
    for scope_name, (store, metas) in scopes.items():
        for key, title, _summary, category in store.list():
            meta = metas.get(key)
            if meta is None or not meta.superseded_by:
                continue
            ts = meta.updated_at if meta.updated_at is not None else meta.created_at
            if ts is None or ts > cutoff:
                continue
            rows.append(
                {
                    "scope": scope_name,
                    "key": key,
                    "title": title,
                    "category": category,
                    "superseded_by": meta.superseded_by,
                    "age_days": round((now - ts) / 86400, 1),
                }
            )
    rows.sort(key=lambda r: (-r["age_days"], r["key"]))
    return rows


def suspicious_supersedes(
    scopes: Mapping[str, tuple[MemoryStore, Mapping[str, EntryMeta]]],
) -> list[dict[str, Any]]:
    """Superseded links whose old and new entries differ in a facet qualifier (report only)."""
    rows: list[dict[str, Any]] = []
    for scope_name, (store, metas) in scopes.items():
        by_key = {k: (title, summary, category) for k, title, summary, category in store.list()}
        for old_key, meta in metas.items():
            new_key = meta.superseded_by
            if not new_key or old_key not in by_key or new_key not in by_key:
                continue
            old_title, old_summary, category = by_key[old_key]
            new_title, new_summary, _ = by_key[new_key]
            reason = qualifier_difference(
                old_title,
                f"{old_title}\n{old_summary}",
                new_title,
                f"{new_title}\n{new_summary}",
            )
            if reason is not None:
                rows.append(
                    {
                        "scope": scope_name,
                        "old_key": old_key,
                        "old_title": old_title,
                        "new_key": new_key,
                        "new_title": new_title,
                        "category": category,
                        "reason": reason,
                    }
                )
    rows.sort(key=lambda r: (r["scope"], r["old_key"]))
    return rows


def select_ranked(
    fused: Mapping[tuple[str, str], dict[str, Any]],
    metas_by_scope: Mapping[str, Mapping[str, EntryMeta]],
    *,
    limit: int,
    include_superseded: bool,
    eps: float = RECENCY_TIE_EPS,
    min_gap_seconds: float = RECENCY_MIN_GAP_SECONDS,
) -> list[tuple[tuple[str, str], dict[str, Any]]]:
    """Redirect superseded entries to their active successor (unless requested) and tie-break."""
    if not fused:
        return []
    if include_superseded:
        items = list(fused.items())
    else:
        resolved: dict[tuple[str, str], dict[str, Any]] = {}
        for (scope_name, key), row in fused.items():
            target = active_key(key, metas_by_scope[scope_name])
            if target is None:
                continue
            dest = (scope_name, target)
            prev = resolved.get(dest)
            if prev is None or (row["score"], len(row["match"])) > (
                prev["score"],
                len(prev["match"]),
            ):
                resolved[dest] = row
        items = list(resolved.items())
    items.sort(key=lambda item: (-item[1]["score"], -len(item[1]["match"])))
    items = apply_recency_tiebreak(items, metas_by_scope, eps=eps, min_gap_seconds=min_gap_seconds)
    return items[:limit]


def unused_entries(
    scopes: Mapping[str, tuple[MemoryStore, MemoryIndex, Mapping[str, EntryMeta]]],
    *,
    older_than_days: float,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Active entries older than ``older_than_days`` (first seen) never recalled or returned."""
    now = now if now is not None else time.time()
    cutoff = now - older_than_days * 86400
    rows: list[dict[str, Any]] = []
    for scope_name, (store, index, metas) in scopes.items():
        usage = index.usage()
        for key, title, _summary, category in store.list():
            if metas.get(key) and metas[key].superseded_by:
                continue
            u = usage.get(key)
            if u is None or u["first_seen"] > cutoff or u["recall_count"] or u["hit_count"]:
                continue
            rows.append(
                {
                    "scope": scope_name,
                    "key": key,
                    "title": title,
                    "category": category,
                    "age_days": round((now - u["first_seen"]) / 86400, 1),
                }
            )
    rows.sort(key=lambda r: -r["age_days"])
    return rows


def scope_status(
    store: MemoryStore,
    index: MemoryIndex,
    metas: Mapping[str, EntryMeta],
    model_id: str | None,
) -> dict[str, int]:
    """Compute entry, supersede, approximate-timestamp, index and usage counts for one scope."""
    entries = store.list()
    usage = index.usage()
    return {
        "entries": len(entries),
        "superseded": sum(1 for m in metas.values() if m.superseded_by),
        "approx_timestamps": sum(1 for m in metas.values() if m.approx_timestamps),
        "changes": index.count_changes(),
        "indexed": index.count(model_id) if model_id else 0,
        "never_used": sum(
            1
            for k, *_ in entries
            if not (metas.get(k) and metas[k].superseded_by)
            and not (usage.get(k, {}).get("recall_count") or usage.get(k, {}).get("hit_count"))
        ),
    }


def format_put_result(
    *,
    key: str,
    title: str,
    category: str,
    scope: str,
    is_update: bool,
    applied_supersedes: list[str],
    similar: list[dict[str, Any]],
    possible_updates: list[dict[str, Any]],
    suggested_links: list[dict[str, Any]] | None = None,
    broken_links: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the ``MemoryService.put`` response dict with duplicate/update/link hints."""
    result: dict[str, Any] = {
        "stored": key,
        "title": title,
        "category": category,
        "scope": scope,
        "updated": is_update,
    }
    if applied_supersedes:
        result["superseded"] = applied_supersedes
    if similar:
        result["possible_duplicates"] = similar
    if possible_updates:
        result["possible_updates"] = possible_updates
        result["suggested_supersedes"] = [u["key"] for u in possible_updates]
        result["note"] = (
            "Similar existing entries are listed in possible_updates. If one is an older "
            "version of the same fact, call put again with supersedes=[key]"
            + (" (or merge and remove the other if it is a duplicate)." if similar else ".")
        )
    elif similar:
        result["note"] = (
            "Similar entries already exist. If this is the same memory, merge "
            "the content into one entry (put with the existing title) and "
            "remove the other."
        )
    if suggested_links:
        result["suggested_links"] = suggested_links
    if broken_links:
        result["broken_links"] = broken_links
    return result


def entry_payload(key: str, scope_name: str, body: str, meta: EntryMeta) -> dict[str, Any]:
    """Build the exact-key recall payload, attaching a ``superseded_by`` note when superseded."""
    out: dict[str, Any] = {"key": key, "scope": scope_name, "body": body}
    if meta.created_at is not None:
        out["created_at"] = meta.created_at
    if meta.updated_at is not None:
        out["updated_at"] = meta.updated_at
    if meta.superseded_by:
        out["superseded_by"] = meta.superseded_by
        out["note"] = (
            f"This entry has been superseded by '{meta.superseded_by}'. "
            "Fetch that key for current information."
        )
    return out


def restore_from_change(
    store: MemoryStore,
    index: MemoryIndex,
    change: Mapping[str, Any],
    current_exists: bool,
) -> str:
    """Apply a revert of ``change`` to ``store``/``index``; returns ``'restored'`` or ``'removed'``."""
    key = str(change["key"])
    prev_superseded = change.get("prev_superseded_by")
    if change["action"] == "supersede" and current_exists:
        store.set_superseded_by(key, prev_superseded)
        return "restored"
    if change["prev_title"] is None:
        if current_exists:
            store.remove(key)
        index.delete_vector(key)
        index.delete_provenance(key)
        return "removed"
    store.put(
        key,
        title=change["prev_title"],
        summary=change["prev_summary"] or "",
        category=change["prev_category"] or "general",
        body=change["prev_body"] or "",
    )
    if prev_superseded != (store.get_meta(key) or EntryMeta()).superseded_by:
        store.set_superseded_by(key, prev_superseded)
    index.delete_vector(key)
    index.delete_provenance(key)  # the earlier content's origin is unknown
    index.record_seen([key])
    return "restored"


def history_rows(
    scopes: Mapping[str, MemoryIndex], *, key: str | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """Collect and sort recent change-log rows across scopes, newest first."""
    rows: list[dict[str, Any]] = []
    for scope_name, index in scopes.items():
        for c in index.changes(key=key, limit=limit):
            preview = " ".join(str(c["prev_summary"] or "").split())
            if len(preview) > 60:
                preview = preview[:59] + "…"
            rows.append(
                {
                    "scope": scope_name,
                    "id": c["id"],
                    "ts": c["ts"],
                    "key": c["key"],
                    "action": c["action"],
                    "session_id": c["session_id"],
                    "preview": preview,
                }
            )
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows[:limit]
