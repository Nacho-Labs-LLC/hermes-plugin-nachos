"""Nachos memory provider — Hermes adapter over ``nachos_core.memory``.

Two layers:

  MANIFEST — system_prompt_block(): a never-truncating table of contents
             (title — summary, grouped by category). Scales with entry
             COUNT, not body size. The only thing injected unprompted.
  RECALL   — nachos_memory_recall: exact key, or a hybrid lexical +
             semantic query. The agent decides relevance from the manifest
             and asks explicitly.

Plus:

  * nachos_memory_put flags likely duplicates (semantic or lexical) so the
    agent can merge instead of piling up near-copies.
  * Optional shared scope: one store every Hermes profile can read, written
    only when the agent passes scope="shared".
  * Every write passes through Hermes' own secret redactor (agent.redact)
    when it is available.
  * Before every compression (whichever compressor runs) the transcript is
    checkpointed; nachos_history_search finds content that was summarized
    away. Checkpoints expire by age.
  * Per-entry usage telemetry (recalls, query hits) for hygiene.
  * Session lineage, delegation capture, backup_paths for the shared store.
  * Session-end capture into a review inbox (see capture_adapter.py).
  * Change log with revert (slash commands in commands.py).

No LLM in the hot path. Embeddings are optional and local by default; see
nachos_core/embeddings.py for backends.

Activate with ``memory.provider: nachos`` and ``memory.memory_enabled: false``.
Settings live in ``$HERMES_HOME/nachos/config.json`` (``hermes memory setup``);
legacy ``nachos.memory.shared_store*`` keys in config.yaml are still read.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

try:
    from agent.memory_provider import MemoryProvider
except ModuleNotFoundError:  # pragma: no cover - exercised outside Hermes

    class MemoryProvider:  # type: ignore[no-redef]  # fallback shadows the Hermes-only import
        """Fallback base so the module stays importable outside Hermes."""


from nachos_core.embeddings import make_embedder
from nachos_core.history import prune_checkpoints, search_history
from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.memory import PROFILE, SHARED, MemoryService, Scope
from nachos_core.snapshots import SnapshotStore
from nachos_core.store import get_store
from nachos_core.store.md_store import slugify
from nachos_core.toc import render_toc

from . import capture_adapter, commands
from .config import (
    CONFIG_SCHEMA,
    DEFAULTS,
    apply_config,
    backup_paths,
    hermes_redactor,
    shared_dir,
    store_file,
)
from .schemas import (
    HISTORY_SCHEMA,
    PUT_SCHEMA,
    RECALL_SCHEMA,
    REMOVE_SCHEMA,
    clamp_int,
    with_scope,
)

logger = logging.getLogger(__name__)


class NachosMemoryProvider(MemoryProvider):
    """Manifest + recall memory provider with semantic recall and history search."""

    pre_compress_checkpoint_api_version = 2

    def __init__(self):
        self._service: MemoryService | None = None
        self._cfg: dict[str, Any] = dict(DEFAULTS)
        self._session_id = ""
        self._lineage: list[str] = []
        self._primary = True
        self._checkpoint_root: Path | None = None
        self._checkpoint_digests: dict[str, tuple[str, str]] = {}
        self._hermes_home: Path | None = None
        self._redact: Callable[[str], str] = lambda text: text
        self._capture = capture_adapter.CaptureWorker()

    @property
    def name(self) -> str:
        return "nachos"

    def is_available(self) -> bool:
        return True  # local, always ready

    @property
    def _store(self):
        """Profile store (kept for older callers)."""
        return self._service.profile.store if self._service else None

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._lineage = [self._session_id] if self._session_id else []
        self._primary = kwargs.get("agent_context", "primary") == "primary"
        hermes_home = self._resolve_home(kwargs.get("hermes_home"))
        self._hermes_home = hermes_home
        self._load_config(hermes_home)
        mem_dir = hermes_home / "nachos"
        mem_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_root = mem_dir / "snapshots"

        redactor = hermes_redactor() if self._cfg["redact_secrets"] == "on" else None
        self._redact = redactor or (lambda text: text)
        self._service = self._build_service(hermes_home, mem_dir)

        if self._primary:
            self._prune_checkpoints()
            if self._capture_enabled():
                self._start_capture_thread()
        self._log_initialized(redacting=redactor is not None)

    @staticmethod
    def _resolve_home(configured_home: str | None) -> Path:
        if configured_home:
            return Path(configured_home)
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())

    def _build_service(self, hermes_home: Path, mem_dir: Path) -> MemoryService | None:
        """Open the configured scopes; None (memory disabled) if the store cannot be opened."""
        try:
            scopes: OrderedDict[str, Scope] = OrderedDict()
            scopes[PROFILE] = self._open_scope(PROFILE, mem_dir)
            if self._cfg["shared_store"] == "on":
                scopes[SHARED] = self._open_scope(SHARED, shared_dir(self._cfg, hermes_home))
            service = MemoryService(
                scopes,
                embedder=self._make_embedder(),
                duplicate_threshold=float(self._cfg["duplicate_threshold"]),
                redact=self._redact,
            )
            service.session_id = self._session_id
            return service
        except Exception as e:
            logger.error("Nachos memory store init failed (%s); disabling.", e)
            return None

    def _prune_checkpoints(self) -> None:
        root = self._checkpoint_root
        if root is None:
            return
        try:
            removed = prune_checkpoints(root, float(self._cfg["checkpoint_max_age_days"]))
            if removed:
                logger.info("Nachos removed %d expired transcript checkpoints", removed)
        except Exception as e:
            logger.debug("Nachos checkpoint retention failed: %s", e)

    def _log_initialized(self, *, redacting: bool) -> None:
        service = self._service
        logger.info(
            "Nachos memory provider initialized (session=%s, store=%s, scopes=%s, "
            "embeddings=%s, redaction=%s, primary=%s)",
            self._session_id,
            self._cfg["store"],
            ",".join(service.scopes) if service else "-",
            service.embedder.model_id if service and service.embedder else "off",
            "hermes" if redacting else "off",
            self._primary,
        )

    def _open_scope(self, name: str, directory: Path) -> Scope:
        directory.mkdir(parents=True, exist_ok=True)
        path = store_file(directory, self._cfg["store"])
        return Scope(name, get_store(self._cfg["store"], path), MemoryIndex(index_path_for(path)))

    def _make_embedder(self):
        try:
            return make_embedder(
                self._cfg["embeddings"],
                model=self._cfg["embedding_model"],
                dtype=self._cfg["embedding_dtype"],
                base_url=self._cfg["embedding_base_url"],
                api_key_env=self._cfg["embedding_api_key_env"],
            )
        except Exception as e:
            logger.warning("Nachos embeddings unavailable (%s); using lexical recall.", e)
            return None

    def _apply_config(self, values: dict[str, Any]) -> None:
        apply_config(self._cfg, values)

    def _load_config(self, hermes_home: Path) -> None:
        """Load legacy config.yaml settings, then profile-scoped setup values."""
        try:
            from hermes_cli.config import cfg_get, load_config

            cfg = load_config()
            self._apply_config({key: cfg_get(cfg, "nachos", "memory", key) for key in DEFAULTS})
        except Exception as e:
            logger.debug("Nachos memory config load failed: %s", e)

        config_path = hermes_home / "nachos" / "config.json"
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(values, dict):
                self._apply_config(values)
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Nachos setup config load failed: %s", e)

    def get_config_schema(self) -> list[dict[str, Any]]:
        """Expose the local settings to Hermes setup UIs (no secrets)."""
        return [dict(field, default=DEFAULTS[field["key"]]) for field in CONFIG_SCHEMA]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Persist setup values under the active profile without touching config.yaml."""
        selected = {key: values[key] for key in DEFAULTS if key in values}
        self._apply_config(selected)

        config_path = Path(hermes_home) / "nachos" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = config_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self._cfg, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(config_path)

    # -- manifest ----------------------------------------------------------

    def system_prompt_block(self) -> str:
        contract = (
            "# Active durable-memory backend\n"
            "Nachos is the active, authoritative durable-memory provider for this "
            "profile (the built-in general memory is disabled with "
            "`memory.memory_enabled: false`). Facts about the user (preferences, "
            "style, personal details) go to the built-in user profile: the `memory` "
            'tool with `target: "user"`. All other durable knowledge goes to '
            "`nachos_memory_put`; recall with `nachos_memory_recall`, remove with "
            "`nachos_memory_remove`. Never use the built-in `memory` tool with "
            '`target: "memory"`: it is disabled and not shown to you. Persona and '
            "identity belong to SOUL.md and are not memory. Earlier conversation that was "
            "compressed out of THIS conversation can be recovered with "
            "`nachos_history_search`; for past sessions use Hermes' `session_search`."
        )
        if self._service and SHARED in self._service.scopes:
            contract += (
                " Entries under `shared/...` are shared with every Hermes profile "
                'on this machine; write there only with `scope: "shared"` and '
                "only for facts that are not profile-specific."
            )
        if not self._service:
            return contract
        pending = self._pending_inbox_count()
        if pending:
            contract += (
                f"\n{pending} suggested memories await the user's review "
                "(/nachos-memory-inbox). Mention this when natural; never approve or "
                "reject them yourself."
            )
        try:
            entries = self._service.manifest_entries()
        except Exception as e:
            logger.debug("Nachos manifest list failed: %s", e)
            return contract
        if not entries:
            return contract
        manifest = render_toc(entries, char_budget=self._cfg["manifest_char_budget"])
        return f"{contract}\n\n{manifest}" if manifest else contract

    # -- checkpoints -------------------------------------------------------

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Checkpoint the transcript before a fail-closed compression pass."""
        if not self._session_id:
            raise RuntimeError("Nachos checkpoint requires a session id")
        root = getattr(self, "_checkpoint_root", None)
        if root is None:
            raise RuntimeError("Nachos checkpoint store is not initialized")
        digest = hashlib.sha256(
            json.dumps(messages, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        known = self._checkpoint_digests.get(self._session_id)
        if known is not None and known[0] == digest:
            return self._compress_note(known[1])
        meta = SnapshotStore(root, self._session_id).save(
            messages=list(messages),
            reason="pre-compress-checkpoint",
            notes=[f"sha256={digest}"],
        )
        self._checkpoint_digests[self._session_id] = (digest, meta.id)
        return self._compress_note(meta.id)

    @staticmethod
    def _compress_note(checkpoint_id: str) -> str:
        return (
            f"The full pre-compression transcript was saved as Nachos checkpoint "
            f"{checkpoint_id}. The summary does not need to preserve exact details "
            "(commands, paths, error text, numbers, decisions); instead end it with a "
            "one-line pointer saying they can be recovered with the "
            "`nachos_history_search` tool."
        )

    # -- session-end capture -----------------------------------------------

    def _capture_enabled(self) -> bool:
        return (
            self._cfg.get("capture") == "review"
            and self._service is not None
            and self._hermes_home is not None
        )

    @property
    def _inbox_dir(self) -> Path:
        if self._hermes_home is None:
            raise RuntimeError("Nachos provider is not initialized (no Hermes home)")
        return self._hermes_home / "nachos" / "inbox"

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Queue the transcript for background extraction. Fast; no LLM; never raises."""
        try:
            if not self._primary or not self._capture_enabled() or not self._session_id:
                return
            if capture_adapter.queue_session(
                self._inbox_dir, self._session_id, messages, self._redact
            ):
                self._start_capture_thread()
        except Exception as e:
            logger.debug("Nachos session-end capture failed: %s", e)

    _capture_llm = staticmethod(capture_adapter.capture_llm)

    def process_capture(
        self, llm: Callable[[list[dict[str, str]]], str] | None = None
    ) -> dict[str, int]:
        """Process pending capture jobs now (thread-free). Never raises."""
        service = self._service
        if service is None or not self._capture_enabled():
            return dict(capture_adapter.IDLE_STATS)
        try:
            return self._capture.process(
                self._inbox_dir,
                service,
                llm or self._capture_llm,
                capture_adapter.read_user_profile(self._hermes_home),
            )
        except Exception as e:
            logger.debug("Nachos capture processing failed: %s", e)
            return dict(capture_adapter.IDLE_STATS)

    def _start_capture_thread(self) -> None:
        """Start the daemon worker if jobs are pending and none is running."""
        self._capture.start(self._inbox_dir, self.process_capture)

    def _pending_inbox_count(self) -> int:
        try:
            return self._service.profile.index.count_inbox("pending") if self._service else 0
        except Exception:
            return 0

    def _queued_captures(self) -> int:
        """Capture jobs still processing; kicks the worker when there are some."""
        if not self._capture_enabled():
            return 0
        queued = capture_adapter.outstanding_count(self._inbox_dir)
        if queued:
            self._start_capture_thread()
        return queued

    # -- slash-command text (rendering lives in commands.py) ---------------

    def status_text(self) -> str:
        return commands.status_text(
            self._service,
            self._cfg,
            self._checkpoint_root,
            self._session_id,
            self._pending_inbox_count(),
        )

    def changes_text(self, key: str = "") -> str:
        return commands.changes_text(self._service, key)

    def revert_text(self, handle: str) -> str:
        return commands.revert_text(self._service, handle)

    def audit_text(self, older_than_days: float = 30) -> str:
        return commands.audit_text(self._service, older_than_days)

    def inbox_text(self, raw_args: str = "") -> str:
        return commands.inbox_text(self._service, raw_args, self._queued_captures)

    # -- session lineage ---------------------------------------------------

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Track the logical conversation across session id rotations."""
        if not new_session_id:
            return
        if reset:
            self._lineage = [new_session_id]
        else:
            if not self._lineage and parent_session_id:
                self._lineage = [parent_session_id]
            if new_session_id not in self._lineage:
                self._lineage.append(new_session_id)
        self._session_id = new_session_id
        if self._service is not None and hasattr(self._service, "session_id"):
            try:
                self._service.session_id = new_session_id
            except Exception as e:
                logger.debug("Nachos service session_id update failed: %s", e)

    # -- delegation --------------------------------------------------------

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs
    ) -> None:
        """Record a finished subagent task/result so history search can find it. Never raises."""
        try:
            root = getattr(self, "_checkpoint_root", None)
            if not self._primary or root is None or not self._session_id:
                return
            SnapshotStore(root, self._session_id).save(
                messages=[
                    {"role": "user", "content": self._redact(f"[delegated task] {task}")},
                    {"role": "assistant", "content": self._redact(f"[subagent result] {result}")},
                ],
                reason="delegation",
                notes=[f"child_session_id={child_session_id}"],
            )
        except Exception as e:
            logger.debug("Nachos delegation capture failed: %s", e)

    # -- backup ------------------------------------------------------------

    def backup_paths(self) -> list[str]:
        """Shared-store directory when it lives outside HERMES_HOME. No init, no network."""
        return backup_paths()

    # -- tools -------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        shared = bool(self._service and SHARED in self._service.scopes)
        schemas = [RECALL_SCHEMA, PUT_SCHEMA, REMOVE_SCHEMA]
        if shared:
            schemas = [with_scope(s) for s in schemas]
        return [*schemas, HISTORY_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if tool_name == "nachos_history_search":
            return self._history(args)
        service = self._service
        if not service:
            return json.dumps({"error": "Nachos memory store not available"})
        handler = {
            "nachos_memory_recall": self._recall,
            "nachos_memory_put": self._put,
            "nachos_memory_remove": self._remove,
        }.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return handler(service, args)
        except KeyError as e:
            return json.dumps({"error": f"Unknown or disabled scope {e}"})

    def _recall(self, service: MemoryService, args: dict[str, Any]) -> str:
        key = (args.get("key") or "").strip()
        query = (args.get("query") or "").strip()
        scope = (args.get("scope") or "").strip() or None
        if key:
            found = service.get(key, scope)
            if found is None:
                return json.dumps({"error": f"No entry with key {key!r}"})
            return json.dumps({"key": key, "scope": found[0], "body": found[1]})
        if query:
            limit = clamp_int(args.get("limit"), 10, 1, 25)
            results = service.search(query, limit=limit, scope=scope)
            return json.dumps({"query": query, "count": len(results), "results": results})
        return json.dumps({"error": "Provide 'key' or 'query'."})

    def _put(self, service: MemoryService, args: dict[str, Any]) -> str:
        if not self._primary:
            return json.dumps({"skipped": "non-primary agent context"})
        try:
            result = service.put(
                title=args.get("title") or "",
                body=args.get("body") or "",
                summary=args.get("summary") or "",
                category=args.get("category") or "general",
                scope=(args.get("scope") or PROFILE).strip(),
            )
        except Exception as e:
            result = {"error": str(e)}
        return json.dumps(result)

    def _remove(self, service: MemoryService, args: dict[str, Any]) -> str:
        if not self._primary:
            return json.dumps({"skipped": "non-primary agent context"})
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

    def _history(self, args: dict[str, Any]) -> str:
        query = (args.get("query") or "").strip()
        if not query:
            return json.dumps({"error": "query is required"})
        if not self._checkpoint_root:
            return json.dumps({"error": "Nachos checkpoint store is not initialized"})
        sessions = None
        if (args.get("scope") or "conversation") != "all":
            sessions = list(self._lineage) or ([self._session_id] if self._session_id else [])
        results = search_history(
            self._checkpoint_root,
            query,
            limit=clamp_int(args.get("limit"), 8, 1, 25),
            session_ids=sessions,
        )
        for row in results:
            row["excerpt"] = self._redact(row["excerpt"])
        return json.dumps({"query": query, "count": len(results), "results": results})

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            if self._service:
                self._service.close()


def register(ctx) -> None:
    provider = NachosMemoryProvider()
    ctx.register_memory_provider(provider)
    commands.register_commands(ctx, provider)
