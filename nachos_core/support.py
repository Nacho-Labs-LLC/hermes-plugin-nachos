"""Helpers split out of ``MemoryService``: incremental cache refresh and change-log snapshots."""

from __future__ import annotations

from .lexical import LexicalIndex
from .store.base import EntryMeta, MemoryStore

Snapshot = tuple[str, str, str, str]  # (title, summary, category, body)


def refresh_after_put(
    lexical_cache: dict[str, tuple[object, LexicalIndex]],
    meta_cache: dict[str, tuple[object, dict[str, EntryMeta]]],
    store: MemoryStore,
    scope_name: str,
    fp_before: object,
    key: str,
    *,
    title: str,
    summary: str,
    category: str,
) -> bool:
    """Update the cached lexical index and entry metadata for one put.

    Returns False when the caches could not be updated in place (stale, unknown fingerprint,
    multi-line summary): the caller must drop them for the scope.
    """
    cached = lexical_cache.get(scope_name)
    if fp_before is None or cached is None or cached[0] != fp_before or "\n" in summary:
        return False
    stored_body = store.get(key)
    fp_after = store.fingerprint()
    if stored_body is None or fp_after is None:
        return False
    cached[1].upsert(key, title=title, summary=summary, category=category, body=stored_body)
    lexical_cache[scope_name] = (fp_after, cached[1])
    meta_cached = meta_cache.get(scope_name)
    new_meta = store.get_meta(key)
    if meta_cached is not None and meta_cached[0] == fp_before and new_meta is not None:
        meta_cached[1][key] = new_meta
        meta_cache[scope_name] = (fp_after, meta_cached[1])
    else:
        meta_cache.pop(scope_name, None)
    return True


def snapshot(store: MemoryStore, key: str) -> Snapshot | None:
    """(title, summary, category, body) for ``key``, or None if absent."""
    body = store.get(key)
    if body is None:
        return None
    for k, title, summary, category in store.list():
        if k == key:
            return title, summary, category, body
    return None


def same_content(prev: Snapshot | None, new: Snapshot) -> bool:
    if prev is None:
        return False
    if prev == new:
        return True
    # The flat-file store stores the summary as the first line of the body.
    title, summary, category, body = new
    led = summary + ("\n\n" + body if body else "")
    return prev == (title, summary, category, led)
