"""Nachos context engine — zone-based compaction with snapshots.

The other half of the Nachos plate. Pairs with the nachos memory
plugin under the same brand.

What this plugin does:

  • ZONE-BASED PRESSURE. Instead of a single binary "compress?" check,
    we map utilization to a zone (green/yellow/orange/red/critical) and
    pick a graded action. Most turns can be handled by cheaper actions
    that don't require an LLM call.

      yellow   — prune old tool results (no LLM, ~30-60% savings)
      orange   — sliding window with tool-pair preservation (no LLM)
      red      — sliding + delegate summary to Hermes' built-in
      critical — emergency: aggressive sliding + summary

  • TOOL-PAIR PRESERVATION. Every assistant message with tool_calls and
    its tool_result are dropped together or kept together. Never
    bisected. Hermes' built-in compressor does this too; we just expose
    it as the FIRST CLASS contract.

  • SNAPSHOTS. Before any aggressive/emergency compaction, we take a
    gzipped snapshot of the message list under
    ~/.hermes/<profile>/nachos/snapshots/<session_id>/. Manual snapshots
    via /nachos snapshot are exempt from rotation. Restore via
    /nachos restore <id>.

  • DELEGATE TO HERMES FOR LLM SUMMARIZATION. When red/critical zones
    require summarization, we hand the messages off to Hermes' default
    ContextCompressor. We do NOT try to out-summarize 1,700+ lines of
    mature code. Our value-add is the orchestration: graded action,
    tool-pair safety, snapshots, observability.

Config:

    context:
      engine: nachos

    nachos:
      compaction:
        thresholds:
          proactive_prune: 0.60
          light_compaction: 0.75
          aggressive_compaction: 0.85
          emergency: 0.95
        protect_first_n: 3
        protect_last_n: 6
        delegate_summary_to_hermes: true   # let built-in handle red/critical
      snapshots:
        enabled: true
        keep: 10                           # rotation per-session
        keep_labeled: true                 # /nachos snapshot <label> survives rotation

Slash commands registered by this plugin:
  /nachos-snapshot   — save a manual snapshot (optional label arg)
  /nachos-snapshots  — list current session snapshots newest-first
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make nachos_core importable when dropped in as a plugin
_PLUGIN_ROOT = Path(__file__).resolve().parents[3]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

try:
    from agent.context_engine import ContextEngine  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised outside Hermes
    class ContextEngine:  # type: ignore[no-redef]
        """Fallback base so the module can be imported outside Hermes."""

        pass

from nachos_core.budget import (  # noqa: E402
    BudgetThresholds,
    CompactionDecision,
    calc_budget,
    decide,
)
from nachos_core.compactor import (  # noqa: E402
    drop_old_tool_results,
    slide_window,
)
from nachos_core.snapshots import SnapshotStore  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NachosContextEngine(ContextEngine):
    """Context engine: zone-based compaction + snapshot lifecycle."""

    def __init__(self):
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0
        # Hermes owns both the active capacity and its compression trigger.
        # ``context_length`` remains the host model's nominal window for status;
        # this private value may be lowered for an auxiliary summarizer.
        self._compression_context_limit: Optional[int] = None

        # Compaction params (override via config; sensible defaults)
        self._thresholds = BudgetThresholds()
        self._protect_first_n = 3
        self._protect_last_n = 6
        self._delegate_summary = True   # use Hermes built-in for LLM summaries

        # Snapshots
        self._snapshots_enabled = True
        self._snapshot_keep = 10
        self._snapshot_keep_labeled = True
        self._snapshot_store: Optional[SnapshotStore] = None
        self._session_id: str = ""

        # Hermes' built-in for delegating summarization
        self._hermes_compressor = None

        # Last decision for /nachos status
        self._last_decision = None
        self._last_action_taken = "none"

    @property
    def name(self) -> str:
        return "nachos"

    # -- Lifecycle ---------------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home  # type: ignore

        self._session_id = session_id or ""
        self._load_config()

        if self._snapshots_enabled and self._session_id:
            hermes_home = Path(kwargs.get("hermes_home") or get_hermes_home())
            snap_root = hermes_home / "nachos" / "snapshots"
            self._snapshot_store = SnapshotStore(snap_root, self._session_id)

        # Construct Hermes' built-in compressor for delegated summarization
        if self._delegate_summary:
            try:
                from agent.context_compressor import ContextCompressor
                model = kwargs.get("model") or ""
                self._hermes_compressor = ContextCompressor(
                    model=model,
                    # The delegate is immediately calibrated from Hermes'
                    # budget below; Nachos' zone settings are not its trigger.
                    threshold_percent=0.0,
                    protect_first_n=self._protect_first_n,
                    protect_last_n=max(self._protect_last_n, 20),
                    base_url=kwargs.get("base_url", ""),
                    api_key=kwargs.get("api_key", ""),
                    provider=kwargs.get("provider", ""),
                    api_mode=kwargs.get("api_mode", ""),
                    quiet_mode=True,
                )
                self._sync_hermes_compressor_budget()
            except Exception as e:
                logger.info(
                    "Nachos: Hermes compressor delegation disabled (%s); "
                    "red/critical zones will use sliding-only fallback.", e,
                )
                self._hermes_compressor = None

        logger.info(
            "Nachos context engine started (session=%s, snapshots=%s, "
            "delegate_summary=%s)",
            self._session_id,
            "on" if self._snapshot_store else "off",
            "on" if self._hermes_compressor else "off",
        )

    def _load_config(self) -> None:
        try:
            from hermes_cli.config import cfg_get, load_config
            cfg = load_config()

            t = cfg_get(cfg, "nachos", "compaction", "thresholds") or {}
            if isinstance(t, dict):
                for key, attr in [
                    ("proactive_prune", "proactive_prune"),
                    ("light_compaction", "light_compaction"),
                    ("aggressive_compaction", "aggressive_compaction"),
                    ("emergency", "emergency"),
                ]:
                    v = t.get(key)
                    if isinstance(v, (int, float)) and 0 < v < 1:
                        setattr(self._thresholds, attr, float(v))

            pf = cfg_get(cfg, "nachos", "compaction", "protect_first_n")
            pl = cfg_get(cfg, "nachos", "compaction", "protect_last_n")
            ds = cfg_get(cfg, "nachos", "compaction",
                         "delegate_summary_to_hermes")
            if isinstance(pf, int) and pf >= 0:
                self._protect_first_n = pf
            if isinstance(pl, int) and pl >= 0:
                self._protect_last_n = pl
            if isinstance(ds, bool):
                self._delegate_summary = ds

            se = cfg_get(cfg, "nachos", "snapshots", "enabled")
            sk = cfg_get(cfg, "nachos", "snapshots", "keep")
            skl = cfg_get(cfg, "nachos", "snapshots", "keep_labeled")
            if isinstance(se, bool):
                self._snapshots_enabled = se
            if isinstance(sk, int) and sk > 0:
                self._snapshot_keep = sk
            if isinstance(skl, bool):
                self._snapshot_keep_labeled = skl
        except Exception as e:
            logger.debug("Nachos engine config load failed: %s", e)

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "", api_mode: str = "") -> None:
        """Mirror the nominal host model window; await its budget contract."""
        self.context_length = context_length
        self._compression_context_limit = context_length or None
        self.threshold_tokens = 0
        self.threshold_percent = 0.0
        if self._hermes_compressor is not None:
            try:
                self._hermes_compressor.update_model(
                    model=model, context_length=context_length,
                    base_url=base_url, api_key=api_key,
                    provider=provider, api_mode=api_mode,
                )
            except Exception as e:
                logger.debug("Hermes compressor update_model failed: %s", e)

    def set_compression_budget(
        self,
        context_limit: Optional[int],
        trigger_tokens: Optional[int],
        *,
        reason: str = "",
    ) -> None:
        """Accept Hermes' authoritative working capacity and trigger.

        Zone thresholds only select the action once this host-defined trigger
        has fired; they never derive an alternate compression boundary.
        """
        try:
            parsed_context = int(context_limit) if context_limit is not None else 0
            parsed_trigger = int(trigger_tokens) if trigger_tokens is not None else 0
        except (TypeError, ValueError):
            parsed_context = parsed_trigger = 0
        if parsed_context <= 0 or parsed_trigger <= 0:
            logger.debug("Ignoring invalid Hermes compression budget: %r / %r", context_limit, trigger_tokens)
            return
        self._compression_context_limit = parsed_context
        self.threshold_tokens = min(parsed_trigger, parsed_context)
        self.threshold_percent = (
            self.threshold_tokens / self.context_length
            if self.context_length else 0.0
        )
        self._sync_hermes_compressor_budget()
        logger.info(
            "Nachos compression budget set to trigger %s within %s%s; nominal context remains %s",
            f"{self.threshold_tokens:,}",
            f"{self._compression_context_limit:,}",
            f" ({reason})" if reason else "",
            f"{self.context_length:,}",
        )

    def _effective_context_length(self) -> int:
        return self._compression_context_limit or self.context_length

    def _sync_hermes_compressor_budget(self) -> None:
        """Calibrate the private delegate to the same host-owned budget."""
        if self._hermes_compressor is None:
            return
        effective_context = self._effective_context_length()
        if not effective_context or not self.threshold_tokens:
            return
        update_model = getattr(self._hermes_compressor, "update_model", None)
        if callable(update_model):
            update_model(
                model=getattr(self._hermes_compressor, "model", ""),
                context_length=effective_context,
                base_url=getattr(self._hermes_compressor, "base_url", ""),
                api_key=getattr(self._hermes_compressor, "api_key", ""),
                provider=getattr(self._hermes_compressor, "provider", ""),
                api_mode=getattr(self._hermes_compressor, "api_mode", ""),
            )

        self._hermes_compressor.threshold_tokens = self.threshold_tokens
        self._hermes_compressor.threshold_percent = (
            self.threshold_tokens / effective_context
        )
        summary_ratio = getattr(self._hermes_compressor, "summary_target_ratio", None)
        if isinstance(summary_ratio, (int, float)):
            self._hermes_compressor.tail_token_budget = int(
                self.threshold_tokens * summary_ratio
            )
        if hasattr(self._hermes_compressor, "max_summary_tokens"):
            self._hermes_compressor.max_summary_tokens = min(
                int(effective_context * 0.05), 10_000
            )

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if not isinstance(usage, dict):
            return
        # Same field names Hermes' base compressor uses
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = (
            self.last_prompt_tokens + self.last_completion_tokens
        )

    # -- Decide / compress -------------------------------------------------

    def _decide_for_host_budget(self, tokens: int):
        budget = calc_budget(
            tokens,
            self._effective_context_length(),
            self._thresholds,
        )
        if not self.threshold_tokens or tokens < self.threshold_tokens:
            return budget, CompactionDecision(
                zone="green",
                action="none",
                reason="Below Hermes' compression trigger; no action needed.",
                target_token_reduction=0,
                snapshot_recommended=False,
            )
        decision = decide(budget)
        if decision.action == "none":
            # Hermes has declared pressure before Nachos' first zone. Retain
            # the zone strategy for escalation, but start with its gentlest
            # action rather than creating a second trigger boundary.
            decision = CompactionDecision(
                zone="yellow",
                action="prune",
                reason="Hermes compression trigger reached; pruning first.",
                target_token_reduction=int(tokens * 0.15),
                snapshot_recommended=False,
            )
        return budget, decision

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = (prompt_tokens if prompt_tokens is not None
                  else self.last_prompt_tokens)
        if not self.context_length or not self.threshold_tokens:
            return False
        _, decision = self._decide_for_host_budget(tokens)
        self._last_decision = decision
        return decision.action != "none"

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None,
                 focus_topic: str = None) -> List[Dict[str, Any]]:
        """Pick an action based on the current zone and execute it.

        For yellow/orange we run our own (no LLM). For red/critical we
        delegate to Hermes' built-in summarizer when available, falling
        back to aggressive sliding when not.
        """
        tokens = current_tokens or self.last_prompt_tokens
        budget, decision = self._decide_for_host_budget(tokens)
        self._last_decision = decision
        self._last_action_taken = decision.action

        logger.info(
            "Nachos compact: zone=%s action=%s tokens=%d/%d (%.1f%%) target_drop=%d",
            decision.zone, decision.action, tokens, self._effective_context_length(),
            budget.utilization_ratio * 100, decision.target_token_reduction,
        )

        # Snapshot before destructive actions
        if (decision.snapshot_recommended
                and self._snapshots_enabled
                and self._snapshot_store):
            try:
                self._snapshot_store.save(
                    messages=list(messages),
                    reason=f"pre-compaction-{decision.action}",
                    notes=[decision.reason,
                           f"utilization={budget.utilization_ratio:.2%}"],
                )
                self._snapshot_store.rotate(
                    keep=self._snapshot_keep,
                    keep_labeled=self._snapshot_keep_labeled,
                )
            except Exception as e:
                logger.warning("Pre-compaction snapshot failed: %s", e)

        # Execute action
        if decision.action == "none":
            return messages

        if decision.action == "prune":
            result = drop_old_tool_results(messages, keep_recent=6)
            self.compression_count += 1
            return result.messages

        if decision.action == "light":
            result = slide_window(
                messages,
                protect_head=self._protect_first_n,
                protect_tail=self._protect_last_n,
                target_token_reduction=decision.target_token_reduction,
            )
            self.compression_count += 1
            return result.messages

        # red / critical → delegate to Hermes if available
        if self._hermes_compressor is not None:
            try:
                out = self._hermes_compressor.compress(
                    messages=messages,
                    current_tokens=tokens,
                    focus_topic=focus_topic,
                )
                self.compression_count += 1
                return out
            except Exception as e:
                logger.warning(
                    "Hermes summarization failed (%s); falling back to "
                    "aggressive sliding.", e,
                )

        # Fallback aggressive sliding (no LLM)
        result = slide_window(
            messages,
            protect_head=self._protect_first_n,
            protect_tail=self._protect_last_n,
            target_token_reduction=decision.target_token_reduction,
        )
        self.compression_count += 1
        return result.messages

    # -- Status / observability -------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        base = super().get_status()
        if self._last_decision:
            base["nachos_zone"] = self._last_decision.zone
            base["nachos_last_action"] = self._last_action_taken
            base["nachos_last_decision_reason"] = self._last_decision.reason
        if self._snapshot_store:
            with contextlib.suppress(Exception):
                base["nachos_snapshot_count"] = len(self._snapshot_store.list())
        return base

    # -- Snapshot tools (exposed to agent) --------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SNAPSHOT_SAVE_SCHEMA, SNAPSHOT_LIST_SCHEMA, SNAPSHOT_LOAD_SCHEMA]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        import json
        if name == "nachos_snapshot_save":
            return self._tool_snapshot_save(args, kwargs)
        if name == "nachos_snapshot_list":
            return self._tool_snapshot_list()
        if name == "nachos_snapshot_load":
            return self._tool_snapshot_load(args)
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _tool_snapshot_save(self, args: Dict[str, Any],
                            ctx_kwargs: Dict[str, Any]) -> str:
        import json
        if not self._snapshot_store:
            return json.dumps({"error": "Snapshots not enabled"})
        messages = ctx_kwargs.get("messages") or []
        label = args.get("label")
        try:
            meta = self._snapshot_store.save(
                messages=list(messages),
                reason="manual",
                label=label,
                notes=args.get("notes") or [],
            )
            return json.dumps({"id": meta.id, "label": meta.label,
                               "message_count": meta.message_count})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_snapshot_list(self) -> str:
        import json
        if not self._snapshot_store:
            return json.dumps({"error": "Snapshots not enabled"})
        return json.dumps({"snapshots": self._snapshot_store.list()})

    def _tool_snapshot_load(self, args: Dict[str, Any]) -> str:
        import json
        if not self._snapshot_store:
            return json.dumps({"error": "Snapshots not enabled"})
        snap_id = args.get("id")
        if not snap_id:
            return json.dumps({"error": "id is required"})
        snap = self._snapshot_store.load(snap_id)
        if not snap:
            return json.dumps({"error": f"Snapshot {snap_id} not found"})
        # Return metadata only — restoring the messages is the runtime's
        # job (would need to swap them into the live agent loop). The
        # agent can READ the snapshot to inspect it.
        return json.dumps({
            "id": snap.meta.id,
            "session_id": snap.meta.session_id,
            "created_at": snap.meta.created_at,
            "message_count": snap.meta.message_count,
            "reason": snap.meta.reason,
            "label": snap.meta.label,
            "messages_preview": snap.messages[-5:] if snap.messages else [],
        })


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SNAPSHOT_SAVE_SCHEMA = {
    "name": "nachos_snapshot_save",
    "description": (
        "Save a manual conversation snapshot — message list + metadata "
        "stored at ~/.hermes/<profile>/nachos/snapshots/. Use before "
        "risky tool sequences. Labeled snapshots survive rotation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {"type": "string",
                      "description": "Human-friendly tag (e.g. 'pre-refactor')."},
            "notes": {"type": "array", "items": {"type": "string"},
                      "description": "Free-form notes."},
        },
    },
}

SNAPSHOT_LIST_SCHEMA = {
    "name": "nachos_snapshot_list",
    "description": "List snapshots for the current session, newest first.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SNAPSHOT_LOAD_SCHEMA = {
    "name": "nachos_snapshot_load",
    "description": (
        "Inspect a snapshot's metadata + tail of messages by id. "
        "Does NOT restore — restoration is a runtime concern; use this "
        "tool to verify what's in the snapshot before deciding."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string",
                   "description": "Snapshot id from nachos_snapshot_list."},
        },
        "required": ["id"],
    },
}


# ---------------------------------------------------------------------------
# Slash command handlers (closures over the engine instance)
# ---------------------------------------------------------------------------

def _make_snapshot_handler(engine: NachosContextEngine):
    def _nachos_snapshot(raw_args: str) -> str:
        if not engine._snapshot_store:
            return (
                "Snapshots not enabled or session not started yet.\n"
                "Ensure nachos.snapshots.enabled: true in config and\n"
                "that a session is active (on_session_start has run)."
            )
        label = (raw_args or "").strip() or None
        try:
            meta = engine._snapshot_store.save(
                messages=[],
                reason="manual",
                label=label,
                notes=["manual snapshot via /nachos-snapshot slash command",
                       "no messages captured — call at session boundary for full capture"],
            )
            lines = [f"Snapshot saved: id={meta.id}"]
            if meta.label:
                lines.append(f"  label   : {meta.label}")
            lines.append(f"  msgs    : {meta.message_count} (0 — slash command context)")
            lines.append(f"  reason  : {meta.reason}")
            lines.append("")
            lines.append("Note: message capture requires session boundary.")
            lines.append("Use 'nachos_snapshot_save' tool during a conversation")
            lines.append("turn for a full message capture.")
            return "\n".join(lines)
        except Exception as e:
            return f"Snapshot failed: {e}"

    return _nachos_snapshot


def _make_snapshots_handler(engine: NachosContextEngine):
    def _nachos_snapshots(raw_args: str) -> str:
        if not engine._snapshot_store:
            return (
                "Snapshots not enabled or session not started yet.\n"
                "Ensure nachos.snapshots.enabled: true in config."
            )
        try:
            snapshots = engine._snapshot_store.list()
        except Exception as e:
            return f"Error listing snapshots: {e}"

        if not snapshots:
            return f"No snapshots found for session {engine._session_id or '(unknown)'}."

        import time as _time

        lines = [f"Nachos snapshots — session {engine._session_id or '(unknown)'}"]
        lines.append(f"Total: {len(snapshots)}")
        lines.append("")
        # Header
        lines.append(f"  {'id':<14} {'created':<20} {'reason':<28} {'label':<18} {'msgs':>5}")
        lines.append("  " + "-" * 90)
        for s in snapshots:  # list() returns newest-first
            snap_id = (s.get("id") or "")[:14]
            created_ts = s.get("created_at") or 0
            try:
                created_str = _time.strftime("%Y-%m-%d %H:%M:%S",
                                             _time.gmtime(float(created_ts)))
            except Exception:
                created_str = str(created_ts)[:20]
            reason = (s.get("reason") or "")[:28]
            label = (s.get("label") or "")[:18]
            msgs = s.get("message_count", 0)
            lines.append(f"  {snap_id:<14} {created_str:<20} {reason:<28} {label:<18} {msgs:>5}")

        return "\n".join(lines)

    return _nachos_snapshots


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    engine = NachosContextEngine()
    ctx.register_context_engine(engine)

    ctx.register_command(
        "nachos-snapshot",
        _make_snapshot_handler(engine),
        "Save a manual Nachos snapshot (no messages in slash command context)",
        args_hint="[label]",
    )
    ctx.register_command(
        "nachos-snapshots",
        _make_snapshots_handler(engine),
        "List current session's Nachos snapshots newest-first",
    )
