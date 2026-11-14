"""MemoryService — the harness-agnostic memory API behind the Hermes adapter.

Owns one or two *scopes*, each a store plus its sidecar index:

* ``profile`` — always present; private to the active Hermes profile.
* ``shared``  — optional; one store every profile on the machine can read and
  write explicitly (``scope="shared"``). Shown in the manifest under
  ``shared/<category>``.

On top of the stores it adds:

* **Hybrid recall** — lexical term matching fused with embedding similarity
  (reciprocal-rank fusion). Without an embedder it is lexical only.
* **Duplicate flagging** — a put reports existing entries that look like the
  same memory (embedding similarity, or token overlap when no embedder), so
  the agent can merge instead of accumulating near-copies. It flags; it does
  not block.
* **Change log** — every create/update/remove/revert is recorded with the
  previous content so it can be undone (``revert``).
* **Usage telemetry** — first-seen, recall and query-hit counts per entry, so
  hygiene can find memories nobody ever uses.
* **Redaction** — an injected ``redact(text) -> text`` applied to everything
  written. The Hermes adapter supplies Hermes' own secret redactor.

Embedding is lazy: vectors are computed on put and reconciled (hand edits,
model changes) on the next query. An embedder error
disables semantic matching for the rest of the process and is reported in
``status()``; memory keeps working lexically.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

from .embeddings import Embedder
from .index import MemoryIndex, dot
from .lexical import idf_weights, jaccard, query_terms, score_document, tokens
from .quality import low_signal_reason
from .store.base import Entry, MemoryStore
from .store.md_store import slugify

logger = logging.getLogger(__name__)

PROFILE = "profile"
SHARED = "shared"

_RRF_K = 60
_MIN_VECTOR_SCORE = 0.30
_LEXICAL_DUP_THRESHOLD = 0.6


def embed_text(title: str, summary: str, body: str) -> str:
    """What gets embedded for an entry: label, summary, then the body head."""
    return f"{title}\n{summary}\n{body}"


@dataclass
class Draft:
    """A normalised, gated prospective write (see ``MemoryService.prepare``)."""

    title: str
    summary: str
    category: str
    body: str
    key: str
    scope: str
    error: str | None = None


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


class MemoryService:
    def __init__(
        self,
        scopes: OrderedDict[str, Scope],
        *,
        embedder: Embedder | None = None,
        duplicate_threshold: float = 0.65,
        redact: Callable[[str], str] | None = None,
    ):
        if PROFILE not in scopes:
            raise ValueError("a 'profile' scope is required")
        self.scopes = scopes
        self.embedder = embedder
        self.duplicate_threshold = duplicate_threshold
        self.redact = redact or (lambda text: text)
        self.embedder_error: str | None = None
        self.session_id: str = ""
        self._seen_keys: dict[str, frozenset] = {}

    @property
    def profile(self) -> Scope:
        return self.scopes[PROFILE]

    def close(self) -> None:
        for scope in self.scopes.values():
            scope.close()
        if self.embedder is not None:
            with contextlib.suppress(Exception):
                self.embedder.close()

    # -- embeddings --------------------------------------------------------

    @property
    def semantic_enabled(self) -> bool:
        return self.embedder is not None and self.embedder_error is None

    def _active_embedder(self) -> Embedder | None:
        """The embedder when semantic search is usable right now, else None."""
        return self.embedder if self.semantic_enabled else None

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        embedder = self._active_embedder()
        if embedder is None or not texts:
            return None
        try:
            return embedder.embed(texts)
        except Exception as e:  # any backend failure degrades to lexical
            self.embedder_error = f"{type(e).__name__}: {e}"[:300]
            logger.warning(
                "Nachos embeddings disabled for this session (%s); recall falls "
                "back to lexical matching.",
                self.embedder_error,
            )
            return None

    def sync_index(self, scope: Scope) -> int:
        """Embed entries that are new, edited, or embedded with another model."""
        entries = scope.store.list()
        scope.index.retain(key for key, *_ in entries)
        embedder = self._active_embedder()
        if embedder is None:
            return 0
        bodies = scope.store.bodies()
        docs = {
            key: embed_text(title, summary, bodies.get(key, ""))
            for key, title, summary, _category in entries
        }
        stale = scope.index.stale_keys(docs, embedder.model_id)
        if not stale:
            return 0
        vectors = self._embed([docs[k] for k in stale])
        if vectors is None:
            return 0
        for key, vec in zip(stale, vectors):
            scope.index.upsert(key, docs[key], embedder.model_id, vec)
        return len(stale)

    # -- manifest ----------------------------------------------------------

    def manifest_entries(self) -> list[Entry]:
        rows: list[Entry] = []
        for scope in self.scopes.values():
            entries = scope.store.list()
            keys = frozenset(e[0] for e in entries)
            if self._seen_keys.get(scope.name) != keys:
                scope.index.record_seen(keys)
                self._seen_keys[scope.name] = keys
            if scope.name == PROFILE:
                rows.extend(entries)
            else:
                rows.extend((k, t, s, f"{scope.name}/{c}") for k, t, s, c in entries)
        return rows

    # -- recall ------------------------------------------------------------

    def _scopes_for(self, scope: str | None) -> list[Scope]:
        if scope:
            if scope not in self.scopes:
                raise KeyError(scope)
            return [self.scopes[scope]]
        return list(self.scopes.values())

    def get(self, key: str, scope: str | None = None) -> tuple[str, str] | None:
        """Return (scope_name, body) for an exact key, profile scope first."""
        for s in self._scopes_for(scope):
            body = s.store.get(key)
            if body is not None:
                s.index.record_recall(key)
                return s.name, body
        return None

    def _lexical(self, scope: Scope, query: str) -> list[str]:
        raw, terms = query_terms(query)
        candidates = set(scope.store.search(query))
        for term in sorted(raw | set(terms)):
            candidates.update(scope.store.search(term))
        if not candidates:
            return []
        labels = {k: (t, s) for k, t, s, _c in scope.store.list()}
        bodies = scope.store.bodies()
        label_words = {k: tokens(f"{t} {sm}") for k, (t, sm) in labels.items()}
        doc_words = {k: label_words[k] | tokens(bodies.get(k) or "") for k in labels}
        weights = idf_weights(terms, doc_words)
        needle = query.lower()
        scored = []
        for key in candidates & doc_words.keys():
            haystack = f"{labels[key][0]}\n{labels[key][1]}\n{bodies.get(key) or ''}".lower()
            score = score_document(weights, doc_words[key], label_words[key], needle in haystack)
            if score > 0:
                scored.append((score, key))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [key for _score, key in scored]

    def search(
        self, query: str, *, limit: int = 10, scope: str | None = None
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        targets = self._scopes_for(scope)
        qvec = self._query_vector(targets, query)
        fused: dict[tuple[str, str], dict[str, Any]] = {}
        for s in targets:
            for key, rank, how, similarity in self._ranked_hits(s, query, qvec, limit):
                row = fused.setdefault((s.name, key), {"score": 0.0, "match": []})
                row["score"] += 1.0 / (_RRF_K + rank)
                row["match"].append(how)
                if similarity is not None:
                    row["similarity"] = round(similarity, 3)

        # Ties go to entries matched by both lexical and semantic.
        ranked = sorted(
            fused.items(), key=lambda item: (-item[1]["score"], -len(item[1]["match"]))
        )[:limit]
        return self._hydrate(ranked)

    def _query_vector(self, targets: list[Scope], query: str) -> list[float] | None:
        """Embedding of ``query`` (after syncing the target indexes), or None if lexical-only."""
        if not self.semantic_enabled:
            return None
        for s in targets:
            self.sync_index(s)
        vectors = self._embed([query])
        return vectors[0] if vectors else None

    def _ranked_hits(
        self, s: Scope, query: str, qvec: list[float] | None, limit: int
    ) -> Iterator[tuple[str, int, str, float | None]]:
        """(key, rank, how, similarity) hits for one scope, lexical then semantic."""
        # With an embedder, lexical sits half a step below semantic so a
        # lexical-only rank-0 hit does not tie a semantic-only rank-0 hit.
        offset = 1 if qvec is not None else 0
        for rank, key in enumerate(self._lexical(s, query)[: limit * 3]):
            yield key, rank + offset, "lexical", None
        embedder = self._active_embedder()
        if qvec is not None and embedder is not None:
            hits = s.index.search(qvec, embedder.model_id, top_k=limit * 3)
            for rank, (key, sim) in enumerate(h for h in hits if h[1] >= _MIN_VECTOR_SCORE):
                yield key, rank, "semantic", sim

    def _hydrate(
        self, ranked: list[tuple[tuple[str, str], dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Turn fused ranking rows into result dicts, recording a hit per returned entry."""
        results = []
        for (scope_name, key), row in ranked:
            s = self.scopes[scope_name]
            body = s.store.get(key)
            if body is None:
                continue
            s.index.record_hits([key])
            result = {
                "key": key,
                "scope": scope_name,
                "body": body,
                "match": "+".join(row["match"]),
            }
            if "similarity" in row:
                result["similarity"] = row["similarity"]
            results.append(result)
        return results

    # -- duplicates --------------------------------------------------------

    def find_similar(
        self,
        title: str,
        summary: str,
        *,
        exclude: tuple[str, str] | None = None,
        vector: list[float] | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Existing entries that look like the same memory, most similar first."""
        found: dict[tuple[str, str], dict[str, Any]] = {}
        label = f"{title} {summary}"
        for s in self.scopes.values():
            titles = {k: (t, sm) for k, t, sm, _c in s.store.list()}
            for key, (t, sm) in titles.items():
                if (s.name, key) == exclude:
                    continue
                score = jaccard(label, f"{t} {sm}")
                if score >= _LEXICAL_DUP_THRESHOLD:
                    found[(s.name, key)] = {
                        "key": key,
                        "scope": s.name,
                        "title": t,
                        "similarity": round(score, 3),
                        "match": "lexical",
                    }
            embedder = self._active_embedder()
            if vector is not None and embedder is not None:
                for key, sim in s.index.search(vector, embedder.model_id, top_k=limit + 1):
                    if (s.name, key) == exclude or sim < self.duplicate_threshold:
                        continue
                    prev = found.get((s.name, key))
                    if prev is None or sim > prev["similarity"]:
                        found[(s.name, key)] = {
                            "key": key,
                            "scope": s.name,
                            "title": titles.get(key, (key, ""))[0],
                            "similarity": round(sim, 3),
                            "match": "semantic",
                        }
        return sorted(found.values(), key=lambda d: -d["similarity"])[:limit]

    def duplicate_pairs(self) -> list[dict[str, Any]]:
        """Every pair of entries above the duplicate threshold (audit use)."""
        pairs: list[dict[str, Any]] = []
        for s in self.scopes.values():
            self.sync_index(s)
        flat: list[tuple[str, str, str, str]] = []
        for s in self.scopes.values():
            flat.extend((s.name, k, t, sm) for k, t, sm, _c in s.store.list())
        vecs: dict[tuple[str, str], list[float]] = {}
        embedder = self._active_embedder()
        if embedder is not None:
            for s in self.scopes.values():
                for key, vec in s.index.vectors(embedder.model_id).items():
                    vecs[(s.name, key)] = vec
        for i, (sa, ka, ta, sma) in enumerate(flat):
            for sb, kb, tb, smb in flat[i + 1 :]:
                va, vb = vecs.get((sa, ka)), vecs.get((sb, kb))
                if va is not None and vb is not None:
                    sim, how = dot(va, vb), "semantic"
                    if sim < self.duplicate_threshold:
                        continue
                else:
                    sim, how = jaccard(f"{ta} {sma}", f"{tb} {smb}"), "lexical"
                    if sim < _LEXICAL_DUP_THRESHOLD:
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

    # -- change log --------------------------------------------------------

    @staticmethod
    def _snapshot(scope: Scope, key: str) -> tuple[str, str, str, str] | None:
        """(title, summary, category, body) for ``key``, or None if absent."""
        body = scope.store.get(key)
        if body is None:
            return None
        for k, title, summary, category in scope.store.list():
            if k == key:
                return title, summary, category, body
        return None

    @staticmethod
    def _same(prev: tuple[str, str, str, str] | None, new: tuple[str, str, str, str]) -> bool:
        if prev is None:
            return False
        if prev == new:
            return True
        # The flat-file store stores the summary as the first line of the body.
        title, summary, category, body = new
        led = summary + ("\n\n" + body if body else "")
        return prev == (title, summary, category, led)

    def _log(
        self, scope: Scope, key: str, action: str, prev: tuple[str, str, str, str] | None
    ) -> None:
        try:
            scope.index.record_change(key, action, self.session_id, prev)
        except Exception as e:  # the log must never block a write
            logger.warning("Nachos change log write failed: %s", e)

    def history(
        self, key: str | None = None, scope: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Recent memory changes across scopes, newest first."""
        rows: list[dict[str, Any]] = []
        for s in self._scopes_for(scope):
            for c in s.index.changes(key=key, limit=limit):
                preview = " ".join(str(c["prev_summary"] or "").split())
                if len(preview) > 60:
                    preview = preview[:59] + "…"
                rows.append(
                    {
                        "scope": s.name,
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

    def revert(self, change_id: int, scope: str = PROFILE) -> dict[str, Any]:
        """Restore the state an entry had before change ``change_id``."""
        if scope not in self.scopes:
            return {"error": f"Unknown or disabled scope {scope!r}; available: {list(self.scopes)}"}
        target = self.scopes[scope]
        try:
            change = target.index.get_change(int(change_id))
        except (TypeError, ValueError):
            change = None
        if change is None:
            return {"error": f"No change {change_id!r} in scope {scope!r}."}
        key = change["key"]
        current = self._snapshot(target, key)
        if change["prev_title"] is None:
            # Entry did not exist before the change: undoing means removing it.
            if current is not None:
                target.store.remove(key)
            target.index.delete_vector(key)
            restored = "removed"
        else:
            target.store.put(
                key,
                title=change["prev_title"],
                summary=change["prev_summary"] or "",
                category=change["prev_category"] or "general",
                body=change["prev_body"] or "",
            )
            target.index.delete_vector(key)
            target.index.record_seen([key])
            restored = "restored"
        self._log(target, key, "revert", current)
        return {
            "reverted": change["id"],
            "key": key,
            "scope": scope,
            "undid": change["action"],
            "result": restored,
            "title": change["prev_title"] or key,
        }

    # -- writes ------------------------------------------------------------

    def prepare(
        self,
        *,
        title: str,
        body: str,
        summary: str = "",
        category: str = "general",
        scope: str = PROFILE,
    ) -> Draft:
        """Normalise and gate a prospective write without touching any store.

        The single source of truth for what ``put`` accepts: redaction, summary
        defaulting, key derivation and the low-signal gates all live here, so
        dry-runs (``tools/import_memories.py``) cannot drift from real writes.
        ``Draft.error`` is set when the write would be rejected.
        """
        if scope not in self.scopes:
            return Draft(
                "",
                "",
                "",
                "",
                "",
                scope,
                f"Unknown or disabled scope {scope!r}; available: {list(self.scopes)}",
            )
        title = self.redact((title or "").strip())
        body = self.redact((body or "").strip())
        summary = self.redact((summary or "").strip())
        if not summary:
            summary = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        category = (category or "general").strip() or "general"
        draft = Draft(title, summary, category, body, slugify(title) if title else "", scope)
        draft.error = self._gate(draft)
        return draft

    @staticmethod
    def _gate(draft: Draft) -> str | None:
        if not draft.title:
            return "title is required"
        reason = low_signal_reason(draft.body)
        if reason:
            return f"body is low-signal ({reason}); write real content, not a placeholder."
        reason = low_signal_reason(draft.summary)
        if reason:
            return (
                f"summary is low-signal ({reason}); this is the ONLY "
                "text the agent sees in the manifest for this entry "
                "without recalling it — write a real, keyword-dense "
                "summary."
            )
        return None

    def put(
        self,
        *,
        title: str,
        body: str,
        summary: str = "",
        category: str = "general",
        scope: str = PROFILE,
    ) -> dict[str, Any]:
        draft = self.prepare(
            title=title, body=body, summary=summary, category=category, scope=scope
        )
        if draft.error:
            return {"error": draft.error}
        title, summary, category, body, key = (
            draft.title,
            draft.summary,
            draft.category,
            draft.body,
            draft.key,
        )
        target = self.scopes[scope]
        is_update = target.store.get(key) is not None

        text = embed_text(title, summary, body)
        vector = None
        if self.semantic_enabled:
            for s in self.scopes.values():
                self.sync_index(s)
            vectors = self._embed([text])
            vector = vectors[0] if vectors else None
        similar = self.find_similar(title, summary, exclude=(scope, key), vector=vector)

        prev = self._snapshot(target, key)
        if not self._same(prev, (title, summary, category, body)):
            self._log(target, key, "update" if prev else "create", prev)
        target.store.put(key, title=title, summary=summary, category=category, body=body)
        embedder = self._active_embedder()
        if vector is not None and embedder is not None:
            target.index.upsert(key, text, embedder.model_id, vector)
        target.index.record_seen([key])

        result: dict[str, Any] = {
            "stored": key,
            "title": title,
            "category": category,
            "scope": scope,
            "updated": is_update,
        }
        if similar:
            result["possible_duplicates"] = similar
            result["note"] = (
                "Similar entries already exist. If this is the same memory, merge "
                "the content into one entry (put with the existing title) and "
                "remove the other."
            )
        return result

    def remove(self, key: str, scope: str | None = None) -> str | None:
        """Remove ``key``; returns the scope it was removed from, or None."""
        for s in self._scopes_for(scope):
            if s.store.get(key) is not None:
                self._log(s, key, "remove", self._snapshot(s, key))
                s.store.remove(key)
                s.index.delete(key)
                return s.name
        return None

    # -- hygiene -----------------------------------------------------------

    def unused(self, older_than_days: float, now: float | None = None) -> list[dict[str, Any]]:
        """Entries older than N days (first seen) never recalled or returned."""
        now = now if now is not None else time.time()
        cutoff = now - older_than_days * 86400
        rows: list[dict[str, Any]] = []
        for s in self.scopes.values():
            usage = s.index.usage()
            for key, title, _summary, category in s.store.list():
                u = usage.get(key)
                if u is None or u["first_seen"] > cutoff:
                    continue
                if u["recall_count"] or u["hit_count"]:
                    continue
                rows.append(
                    {
                        "scope": s.name,
                        "key": key,
                        "title": title,
                        "category": category,
                        "age_days": round((now - u["first_seen"]) / 86400, 1),
                    }
                )
        rows.sort(key=lambda r: -r["age_days"])
        return rows

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "embeddings": self.embedder.model_id if self.embedder else "off",
            "embeddings_error": self.embedder_error,
            "scopes": {},
        }
        for s in self.scopes.values():
            entries = s.store.list()
            usage = s.index.usage()
            out["scopes"][s.name] = {
                "entries": len(entries),
                "changes": s.index.count_changes(),
                "indexed": s.index.count(self.embedder.model_id) if self.embedder else 0,
                "never_used": sum(
                    1
                    for k, *_ in entries
                    if not (
                        usage.get(k, {}).get("recall_count") or usage.get(k, {}).get("hit_count")
                    )
                ),
            }
        return out
