"""A memory scope (store + sidecar index) and the fingerprint-keyed read caches over it."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass

from .index import MemoryIndex
from .lexical import LexicalIndex
from .store.base import EntryMeta, MemoryStore

PROFILE = "profile"
SHARED = "shared"


@dataclass
class Scope:
    name: str
    store: MemoryStore
    index: MemoryIndex

    def close(self) -> None:
        for part in (self.store, self.index):
            close = getattr(part, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()


def select_scopes(scopes: Mapping[str, Scope], name: str | None) -> list[Scope]:
    """The named scope, or every scope when ``name`` is empty; ``KeyError`` for an unknown name."""
    if name:
        if name not in scopes:
            raise KeyError(name)
        return [scopes[name]]
    return list(scopes.values())


class ScopeCaches:
    """Per-scope lexical index and entry metadata, valid while the store fingerprint is unchanged.

    Stores without a fingerprint are never cached. A write made through the service refreshes
    the entries in place (``refresh_after_put``) or drops them (``invalidate``).
    """

    def __init__(self) -> None:
        self.lexical_cache: dict[str, tuple[object, LexicalIndex]] = {}
        self.meta_cache: dict[str, tuple[object, dict[str, EntryMeta]]] = {}

    def meta(self, scope: Scope) -> dict[str, EntryMeta]:
        fp = scope.store.fingerprint()
        if fp is not None:
            cached = self.meta_cache.get(scope.name)
            if cached is not None and cached[0] == fp:
                return cached[1]
        metas = scope.store.all_meta()
        if fp is not None:
            self.meta_cache[scope.name] = (fp, metas)
        return metas

    def lexical(self, scope: Scope) -> LexicalIndex:
        fp = scope.store.fingerprint()
        if fp is not None:
            cached = self.lexical_cache.get(scope.name)
            if cached is not None and cached[0] == fp:
                return cached[1]
        index = LexicalIndex.build(scope.store.list(), scope.store.bodies())
        if fp is not None:
            self.lexical_cache[scope.name] = (fp, index)
        return index

    def titles(self, scope: Scope) -> dict[str, str]:
        return {key: doc.title for key, doc in self.lexical(scope).docs.items()}

    def invalidate(self, scope_name: str) -> None:
        self.lexical_cache.pop(scope_name, None)
        self.meta_cache.pop(scope_name, None)

    def clear(self) -> None:
        self.lexical_cache.clear()
        self.meta_cache.clear()

    def refresh_after_put(
        self,
        scope: Scope,
        fp_before: object,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
    ) -> None:
        """Update the cached lexical index and metadata for one put, or drop them.

        Falls back to ``invalidate`` when the caches cannot be updated in place (stale, unknown
        fingerprint, multi-line summary).
        """
        if not self._update_in_place(
            scope, fp_before, key, title=title, summary=summary, category=category
        ):
            self.invalidate(scope.name)

    def _update_in_place(
        self,
        scope: Scope,
        fp_before: object,
        key: str,
        *,
        title: str,
        summary: str,
        category: str,
    ) -> bool:
        cached = self.lexical_cache.get(scope.name)
        if fp_before is None or cached is None or cached[0] != fp_before or "\n" in summary:
            return False
        stored_body = scope.store.get(key)
        fp_after = scope.store.fingerprint()
        if stored_body is None or fp_after is None:
            return False
        cached[1].upsert(key, title=title, summary=summary, category=category, body=stored_body)
        self.lexical_cache[scope.name] = (fp_after, cached[1])
        meta_cached = self.meta_cache.get(scope.name)
        new_meta = scope.store.get_meta(key)
        if meta_cached is not None and meta_cached[0] == fp_before and new_meta is not None:
            meta_cached[1][key] = new_meta
            self.meta_cache[scope.name] = (fp_after, meta_cached[1])
        else:
            self.meta_cache.pop(scope.name, None)
        return True
