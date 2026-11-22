"""Per-scope link graph and entity index, kept in step with the store.

Links live in the sidecar ``links`` table (forward edges by source; backlinks are an
indexed lookup on the target) and are reconciled against the store fingerprint, so hand
edits and external writers are picked up on the next read. Writes made through
``MemoryService`` update one entry's rows incrementally. Entities are an in-memory
index built lazily and cached by the same fingerprint. Both are derived data: deleting
the sidecar loses nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .index import MemoryIndex
from .links import EntityIndex, parse_links, resolve_successor, resolved_edges
from .store.base import EntryMeta, MemoryStore


class ScopeGraph:
    def __init__(
        self,
        name: str,
        store: MemoryStore,
        index: MemoryIndex,
        metas: Callable[[], Mapping[str, EntryMeta]],
        titles: Callable[[], dict[str, str]],
    ) -> None:
        self.name = name
        self.store = store
        self.index = index
        self.metas = metas  # current entry metadata (superseded_by, ...) of the scope
        self.titles = titles  # key -> title of the scope's entries
        self._links_fp: object = None
        self._entities: tuple[object, EntityIndex] | None = None

    # -- keeping in step ---------------------------------------------------

    def ensure(self) -> None:
        """Reconcile stored links with the store when its fingerprint changed."""
        if self.index.read_only:
            return
        fp = self.store.fingerprint()
        if fp is not None and fp == self._links_fp:
            return
        forward = {key: parse_links(body) for key, body in self.store.bodies().items()}
        self.index.replace_links(forward)
        self._links_fp = fp

    def after_write(self, fp_before: object, key: str, summary: str | None = None) -> None:
        """Update ``key``'s rows after a put/remove/revert whose pre-write fingerprint is given."""
        body = self.store.get(key)
        in_step = fp_before is not None and self._links_fp == fp_before
        if not self.index.read_only:
            if body is None:
                self.index.delete_links(key)
            else:
                self.index.set_links(key, parse_links(body))
            self._links_fp = self.store.fingerprint() if in_step else None
        self._update_entities(fp_before, key, summary, body)

    def _update_entities(
        self, fp_before: object, key: str, summary: str | None, body: str | None
    ) -> None:
        cached = self._entities
        if cached is None or fp_before is None or cached[0] != fp_before:
            self._entities = None
        elif body is None:
            cached[1].remove(key)
            self._entities = (self.store.fingerprint(), cached[1])
        elif summary is None:
            self._entities = None
        else:
            cached[1].upsert(key, summary, body)
            self._entities = (self.store.fingerprint(), cached[1])

    def invalidate(self) -> None:
        self._links_fp = None
        self._entities = None

    def entities(self) -> EntityIndex:
        fp = self.store.fingerprint()
        if fp is not None and self._entities is not None and self._entities[0] == fp:
            return self._entities[1]
        index = EntityIndex.build(self.store.list(), self.store.bodies())
        if fp is not None:
            self._entities = (fp, index)
        return index

    # -- reading -----------------------------------------------------------

    def forward_map(self) -> dict[str, list[str]]:
        self.ensure()
        return self.index.all_links()

    def neighbors(self, key: str) -> dict[str, list[str]]:
        """``links`` / ``backlinks`` (resolved to active entries) and ``broken`` targets of ``key``."""
        self.ensure()
        metas = self.metas()
        raw = self.index.links_from(key)
        out: list[str] = []
        for dst in raw:
            target = resolve_successor(dst, metas)
            if target is not None and target != key and target not in out:
                out.append(target)
        sources = list(self.index.links_to(key))
        for old in _predecessors(key, metas):
            sources.extend(self.index.links_to(old))
        back = sorted({s for s in sources if s != key and _active(s, metas)})
        return {"links": out, "backlinks": back, "broken": [d for d in raw if d not in metas]}

    def resolve(self, key: str) -> str | None:
        return resolve_successor(key, self.metas())

    def link_counts(self) -> dict[str, int]:
        """Distinct linked neighbours (either direction) per active entry that has any."""
        metas = self.metas()
        neighbours: dict[str, set[str]] = {}
        for src, dst in resolved_edges(self.forward_map(), metas):
            if _active(src, metas) and _active(dst, metas):
                neighbours.setdefault(src, set()).add(dst)
                neighbours.setdefault(dst, set()).add(src)
        return {key: len(peers) for key, peers in neighbours.items()}


def _active(key: str, metas: Mapping[str, EntryMeta]) -> bool:
    meta = metas.get(key)
    return meta is not None and not meta.superseded_by


def _predecessors(key: str, metas: Mapping[str, EntryMeta]) -> list[str]:
    return [
        old
        for old, meta in metas.items()
        if meta.superseded_by and old != key and resolve_successor(old, metas) == key
    ]


def describe(keys: list[str], titles: Mapping[str, str], **extra: Any) -> list[dict[str, Any]]:
    """``[{"key", "title", ...extra}]`` rows for a list of keys."""
    return [{"key": k, "title": titles.get(k, k), **extra} for k in keys]
