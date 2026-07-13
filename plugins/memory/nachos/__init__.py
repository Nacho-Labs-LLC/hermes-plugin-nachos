"""Nachos memory provider — 3-tier manifest assembly.

Replaces always-on full memory injection (which has a hard char ceiling)
with a bounded, scalable assembly:

  TIER 1  MANIFEST  — system_prompt_block(): a never-truncating table of
                      contents (title — summary, grouped by category).
                      Scales with entry COUNT, not entry body size.
  TIER 2  PREFETCH  — prefetch(query): relevance-ranked bodies injected for
                      the upcoming turn (lexical TF-IDF default; semantic
                      drop-in optional). Prefetched entries are marked ► in
                      the manifest.
  TIER 3  RECALL    — nachos_memory_recall tool: pull any entry on demand.

Storage is a local seam (SQLite default / flatfile option). The scorer is
a seam too (lexical default / semantic opt-in). No LLM in the hot path;
periodic summary self-correction is a separate parked cron
(tools/correct_summaries.py).

Config (all under nachos.memory, all optional):
    nachos:
      memory:
        store: sqlite            # sqlite | flatfile
        scorer: lexical          # lexical | semantic
        semantic_provider: nachos  # nachos | sentence-transformers | openai
        prefetch_top_n: 5
        prefetch_char_budget: 1500
        manifest_char_budget: 1200

Activate with:  memory.provider: nachos
And disable built-in full injection:  memory.memory_enabled: false

Slash commands:
  /nachos-memory-status  — entry count, store kind, scorer, last prefetch
  /nachos-memory-list    — render the manifest
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make nachos_core importable when dropped in as a plugin
_PLUGIN_ROOT = Path(__file__).resolve().parents[3]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

try:
    from agent.memory_provider import MemoryProvider  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised outside Hermes
    class MemoryProvider:  # type: ignore[no-redef]
        """Fallback base so the module stays importable outside Hermes."""

        pass

from nachos_core.prefetch import get_scorer  # noqa: E402
from nachos_core.store import get_store  # noqa: E402
from nachos_core.store.md_store import slugify  # noqa: E402
from nachos_core.toc import render_toc  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "store": "sqlite",
    "scorer": "lexical",
    "semantic_provider": "nachos",
    "prefetch_top_n": 5,
    "prefetch_char_budget": 1500,
    "manifest_char_budget": 1200,
}


class NachosMemoryProvider(MemoryProvider):
    """3-tier manifest memory provider (manifest / prefetch / recall)."""

    def __init__(self):
        self._store = None
        self._scorer = None
        self._cfg = dict(_DEFAULTS)
        self._session_id = ""
        self._primary = True
        self._last_prefetched: List[str] = []

    @property
    def name(self) -> str:
        return "nachos"

    def is_available(self) -> bool:
        return True  # local, always ready

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._primary = (kwargs.get("agent_context", "primary") == "primary")
        self._load_config()

        from hermes_constants import get_hermes_home  # type: ignore
        hermes_home = Path(kwargs.get("hermes_home") or get_hermes_home())
        mem_dir = hermes_home / "nachos"
        mem_dir.mkdir(parents=True, exist_ok=True)

        store_kind = self._cfg["store"]
        path = mem_dir / ("memory.md" if store_kind == "flatfile" else "memory.db")

        try:
            self._store = get_store(store_kind, path)
        except Exception as e:
            logger.error("Nachos memory store init failed (%s); disabling.", e)
            self._store = None

        try:
            self._scorer = get_scorer(
                self._cfg["scorer"],
                semantic_backend=self._cfg["semantic_provider"],
            )
        except Exception as e:
            logger.warning("Nachos scorer %r failed (%s); using lexical.",
                           self._cfg["scorer"], e)
            self._scorer = get_scorer("lexical")

        logger.info(
            "Nachos memory provider initialized (session=%s, store=%s, "
            "scorer=%s, primary=%s)",
            self._session_id, store_kind, self._cfg["scorer"], self._primary,
        )

    def _load_config(self) -> None:
        try:
            from hermes_cli.config import cfg_get, load_config
            cfg = load_config()
            for key, default in _DEFAULTS.items():
                v = cfg_get(cfg, "nachos", "memory", key)
                if v is not None:
                    if isinstance(default, int) and isinstance(v, (int, float)):
                        self._cfg[key] = int(v)
                    elif isinstance(default, str) and isinstance(v, str) and v.strip():
                        self._cfg[key] = v.strip()
        except Exception as e:
            logger.debug("Nachos memory config load failed: %s", e)

    # -- TIER 1: manifest --------------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            entries = self._store.list()
        except Exception as e:
            logger.debug("Nachos manifest list failed: %s", e)
            return ""
        if not entries:
            return ""
        return render_toc(
            entries,
            prefetched=set(self._last_prefetched),
            char_budget=self._cfg["manifest_char_budget"],
        )

    # -- TIER 2: prefetch --------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not self._scorer or not (query or "").strip():
            return ""
        try:
            entries = self._store.list()
            if not entries:
                return ""
            # coarse candidate narrowing via store.search, fall back to all
            hits = set(self._store.search(query))
            candidates = [e for e in entries if e[0] in hits] if hits else entries
            keys = self._scorer.rank(
                query, candidates, top_n=self._cfg["prefetch_top_n"]
            )
            self._last_prefetched = list(keys)
            if not keys:
                return ""
            budget = self._cfg["prefetch_char_budget"]
            titles = {e[0]: e[1] for e in entries}
            lines = ["# Recalled memory (relevant to this turn)"]
            used = len(lines[0])
            for key in keys:
                body = self._store.get(key) or ""
                block = f"\n## {titles.get(key, key)}\n{body.strip()}"
                if used + len(block) > budget:
                    break
                lines.append(block)
                used += len(block)
            return "\n".join(lines) if len(lines) > 1 else ""
        except Exception as e:
            logger.debug("Nachos prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        # prefetch is cheap + synchronous; no background queue needed.
        return

    # -- TIER 3: recall + curation tools -----------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, PUT_SCHEMA, REMOVE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._store:
            return json.dumps({"error": "Nachos memory store not available"})
        if tool_name == "nachos_memory_recall":
            return self._recall(args)
        if tool_name == "nachos_memory_put":
            return self._put(args)
        if tool_name == "nachos_memory_remove":
            return self._remove(args)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _recall(self, args: Dict[str, Any]) -> str:
        key = (args.get("key") or "").strip()
        query = (args.get("query") or "").strip()
        if key:
            body = self._store.get(key)
            if body is None:
                return json.dumps({"error": f"No entry with key {key!r}"})
            return json.dumps({"key": key, "body": body})
        if query:
            keys = self._store.search(query)
            results = [{"key": k, "body": self._store.get(k)} for k in keys[:10]]
            return json.dumps({"query": query, "count": len(results),
                               "results": results})
        return json.dumps({"error": "Provide 'key' or 'query'."})

    def _put(self, args: Dict[str, Any]) -> str:
        if not self._primary:
            return json.dumps({"skipped": "non-primary agent context"})
        title = (args.get("title") or "").strip()
        if not title:
            return json.dumps({"error": "title is required"})
        body = (args.get("body") or "").strip()
        # on-edit self-correcting summary (v1: no LLM): caller summary, else
        # derive from first non-blank line of the body.
        summary = (args.get("summary") or "").strip()
        if not summary:
            summary = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        category = (args.get("category") or "general").strip()
        key = slugify(title)
        try:
            self._store.put(key, title=title, summary=summary,
                            category=category, body=body)
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"stored": key, "title": title, "category": category})

    def _remove(self, args: Dict[str, Any]) -> str:
        if not self._primary:
            return json.dumps({"skipped": "non-primary agent context"})
        key = (args.get("key") or "").strip()
        if not key and args.get("title"):
            key = slugify(args["title"].strip())
        if not key:
            return json.dumps({"error": "Provide 'key' or 'title'."})
        try:
            self._store.remove(key)
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"removed": key})

    # -- mirror built-in memory writes -------------------------------------

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Best-effort mirror of built-in memory writes into the store.

        Lets nothing be lost if built-in injection is later disabled.
        Never raises.
        """
        if not self._store or not self._primary:
            return
        try:
            meta = metadata or {}
            content = (content or "").strip()

            if action == "remove":
                # On remove the core passes content="" and puts the entry's
                # identifying text in metadata["old_text"] (see memory_manager
                # notify_memory_tool_write). The caller does NOT know nachos's
                # key, and old_text is often a substring, not the full body.
                # A naive slug-of-content remove() silently no-ops for any
                # entry not written by this hook (e.g. seeded via
                # nachos_memory_put), leaving a zombie that is recalled forever
                # despite being "deleted". Resolve the real key by matching
                # old_text against stored bodies.
                old_text = str(meta.get("old_text") or content or "").strip()
                if not old_text:
                    return
                target_key = None
                # exact slug of the first line (fast path for hook-written entries)
                first_ot = next((ln.strip() for ln in old_text.splitlines() if ln.strip()), "")
                slug_key = slugify(first_ot[:60])
                if self._store.get(slug_key) is not None:
                    target_key = slug_key
                else:
                    # old_text may be a substring OR a superset of the stored
                    # body (the built-in tool's match text and the stored body
                    # can diverge in the tail). Anchor on a short stable prefix
                    # of the first line — memory entries are stored body-first,
                    # so the opening chars are the reliable discriminator.
                    anchor = first_ot[:40]
                    for cand in self._store.search(anchor):
                        body = (self._store.get(cand) or "").strip()
                        if anchor and (anchor in body or body[:40] == anchor):
                            target_key = cand
                            break
                if target_key is not None:
                    self._store.remove(target_key)
                else:
                    logger.debug("Nachos remove: no store entry matches %r", old_text[:40])
                return

            if not content:
                return
            first = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
            title = first[:60] if first else "untitled"
            key = slugify(title)
            # Converge with the nachos-native write path (nachos_memory_put):
            # use the target ("memory"/"user") directly as the category rather
            # than a "builtin-" prefix, so both writers land in the same
            # manifest groups and share the slug-of-first-line key convention.
            # A "builtin-" prefix here forks the store into parallel category
            # trees that never merge, reaccumulating dupes over time.
            category = target or "memory"
            self._store.put(key, title=title, summary=first,
                            category=category, body=content)
        except Exception as e:
            logger.debug("Nachos on_memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        try:
            if self._store and hasattr(self._store, "close"):
                self._store.close()
        except Exception:
            pass

    # -- helpers for slash commands ----------------------------------------

    def _entry_count(self) -> int:
        try:
            return len(self._store.list()) if self._store else 0
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "nachos_memory_recall",
    "description": (
        "Pull full memory entry content on demand. Provide 'key' for an "
        "exact entry (from the manifest), or 'query' to search bodies and "
        "return matching entries. Use when the manifest shows an entry "
        "exists and you need its full text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Exact entry key (slug of title)."},
            "query": {"type": "string", "description": "Substring search over entries."},
        },
    },
}

PUT_SCHEMA = {
    "name": "nachos_memory_put",
    "description": (
        "Add or update a memory entry. key is derived from the title "
        "(slug). If summary is omitted it's taken from the first line of "
        "body — write body summary-first. category groups the entry in the "
        "manifest."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short label (becomes the key)."},
            "summary": {"type": "string", "description": "One keyword-dense line for the manifest. Optional; derived from body[0] if absent."},
            "category": {"type": "string", "description": "Manifest group (e.g. 'deposco', 'user'). Default 'general'."},
            "body": {"type": "string", "description": "Full entry content."},
        },
        "required": ["title", "body"],
    },
}

REMOVE_SCHEMA = {
    "name": "nachos_memory_remove",
    "description": "Delete a memory entry by 'key' or 'title'.",
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Entry key."},
            "title": {"type": "string", "description": "Entry title (slugified to key)."},
        },
    },
}


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def _make_status_handler(provider: NachosMemoryProvider):
    def _status(_raw_args: str) -> str:
        lines = [
            "Nachos memory (3-tier manifest)",
            f"  store          : {provider._cfg['store']}",
            f"  scorer         : {provider._cfg['scorer']}"
            + (f" ({provider._cfg['semantic_provider']})"
               if provider._cfg["scorer"] == "semantic" else ""),
            f"  entries        : {provider._entry_count()}",
            f"  prefetch_top_n : {provider._cfg['prefetch_top_n']}",
            f"  last_prefetched: {', '.join(provider._last_prefetched) or '-'}",
            f"  session        : {provider._session_id or '-'}",
        ]
        return "\n".join(lines)
    return _status


def _make_list_handler(provider: NachosMemoryProvider):
    def _list(_raw_args: str) -> str:
        block = provider.system_prompt_block()
        return block or "Nachos memory is empty."
    return _list


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    provider = NachosMemoryProvider()
    ctx.register_memory_provider(provider)

    ctx.register_command(
        "nachos-memory-status",
        _make_status_handler(provider),
        "Nachos memory status — store, scorer, entry count, last prefetch",
    )
    ctx.register_command(
        "nachos-memory-list",
        _make_list_handler(provider),
        "Render the Nachos memory manifest (table of contents)",
    )
