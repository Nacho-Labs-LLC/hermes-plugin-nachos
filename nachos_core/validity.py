"""Entry validity helpers: timestamp backfill, supersedes normalization, recency tie-break,
and long-superseded hygiene queries.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .index import MemoryIndex, dot
from .lexical import LexicalIndex, jaccard
from .quality import core_title_tokens, looks_like_update
from .store.base import EntryMeta, MemoryStore

# Recency tie-break tolerance on RRF scores: covers exact ties and symmetric (0,1)/(1,0)
# hybrid rank swaps (1/60 + 1/62 - 2/61 ~= 8.8e-6) without overriding wider score gaps.
RECENCY_TIE_EPS = 1e-5
RECENCY_MIN_GAP_SECONDS = 60.0


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


def _title_subject_candidates(
    lexical_index: LexicalIndex,
    metas: Mapping[str, EntryMeta],
    title: str,
    exclude_keys: set[str],
) -> list[str]:
    """Active keys in ``lexical_index`` whose core subject title matches ``title``."""
    new_core = core_title_tokens(title)
    if len(new_core) < 2:
        return []
    hits: set[str] = set()
    for term in new_core:
        bucket = lexical_index.label_postings.get(term)
        if bucket:
            hits.update(bucket)
    matched: list[str] = []
    for key in sorted(hits):
        if key in exclude_keys:
            continue
        meta = metas.get(key)
        if meta is not None and meta.superseded_by:
            continue
        doc = lexical_index.docs.get(key)
        if doc is None:
            continue
        old_core = core_title_tokens(doc.title)
        if old_core and old_core == new_core:
            matched.append(key)
    return matched


def suggest_supersedes_for(
    store: MemoryStore,
    lexical_index: LexicalIndex,
    metas: Mapping[str, EntryMeta],
    scope_name: str,
    *,
    self_key: str,
    title: str,
    new_text: str,
    similar: list[dict[str, Any]],
    already_superseded: set[str],
) -> list[str]:
    """Return active keys in ``scope_name`` that read like outdated versions of ``new_text``."""
    excluded = already_superseded | {self_key}
    candidate_keys = [
        dup["key"] for dup in similar if dup["scope"] == scope_name and dup["key"] not in excluded
    ]
    for key in _title_subject_candidates(lexical_index, metas, title, excluded):
        if key not in candidate_keys:
            candidate_keys.append(key)
    if not candidate_keys:
        return []
    suggested: list[str] = []
    for key in candidate_keys:
        meta = metas.get(key)
        if meta is not None and meta.superseded_by:
            continue
        doc = lexical_index.docs.get(key)
        body = store.get(key)
        if doc is None or body is None:
            continue
        old_text = f"{doc.title}\n{body}"
        if looks_like_update(old_text, new_text, old_title=doc.title, new_title=title):
            suggested.append(key)
    return suggested


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


def select_ranked(
    fused: Mapping[tuple[str, str], dict[str, Any]],
    metas_by_scope: Mapping[str, Mapping[str, EntryMeta]],
    *,
    limit: int,
    include_superseded: bool,
    eps: float = RECENCY_TIE_EPS,
    min_gap_seconds: float = RECENCY_MIN_GAP_SECONDS,
) -> list[tuple[tuple[str, str], dict[str, Any]]]:
    """Filter superseded entries (unless requested) and apply recency tie-breaking."""
    if not fused:
        return []
    items = [
        item
        for item in fused.items()
        if include_superseded
        or not (
            (meta := metas_by_scope[item[0][0]].get(item[0][1])) is not None and meta.superseded_by
        )
    ]
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
    suggested_supersedes: list[str],
) -> dict[str, Any]:
    """Assemble the ``MemoryService.put`` response dict with duplicate/supersede hints."""
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
    if suggested_supersedes:
        result["suggested_supersedes"] = suggested_supersedes
        result["note"] = (
            "Similar entries already exist and this looks like an update. If it replaces "
            f"them, re-put with supersedes={suggested_supersedes!r} "
            "(or merge and remove the old entry)."
        )
    elif similar:
        result["note"] = (
            "Similar entries already exist. If this is the same memory, merge "
            "the content into one entry (put with the existing title) and "
            "remove the other."
        )
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


def find_duplicate_pairs(
    scopes: Mapping[str, tuple[MemoryStore, MemoryIndex, Mapping[str, EntryMeta]]],
    *,
    model_id: str | None,
    duplicate_threshold: float,
    lexical_threshold: float,
) -> list[dict[str, Any]]:
    """Compute all active near-duplicate entry pairs across scopes."""
    flat: list[tuple[str, str, str, str]] = []
    vecs: dict[tuple[str, str], list[float]] = {}
    for scope_name, (store, index, metas) in scopes.items():
        flat.extend(
            (scope_name, k, t, sm)
            for k, t, sm, _c in store.list()
            if not (metas.get(k) and metas[k].superseded_by)
        )
        if model_id is not None:
            for key, vec in index.vectors(model_id).items():
                vecs[(scope_name, key)] = vec
    pairs: list[dict[str, Any]] = []
    for i, (sa, ka, ta, sma) in enumerate(flat):
        for sb, kb, tb, smb in flat[i + 1 :]:
            va, vb = vecs.get((sa, ka)), vecs.get((sb, kb))
            if va is not None and vb is not None:
                sim, how = dot(va, vb), "semantic"
                if sim < duplicate_threshold:
                    continue
            else:
                sim, how = jaccard(f"{ta} {sma}", f"{tb} {smb}"), "lexical"
                if sim < lexical_threshold:
                    continue
            pairs.append(
                {
                    "a": {"scope": sa, "key": ka, "title": ta},
                    "b": {"scope": sb, "key": kb, "title": tb},
                    "similarity": round(sim, 3),
                    "match": how,
                }
            )
    pairs.sort(key=lambda p: -p["similarity"])
    return pairs
