"""Change-log snapshots: the (title, summary, category, body) tuple of an entry."""

from __future__ import annotations

from .store.base import MemoryStore

Snapshot = tuple[str, str, str, str]  # (title, summary, category, body)


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
