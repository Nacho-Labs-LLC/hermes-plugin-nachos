"""Nachos memory plugin — Hermes MemoryProvider face of the Nachos
context layer.

What this plugin contributes:
  • A MEMORY MANIFEST in the system prompt — compact pointer list
    summarizing what's in your memory file PLUS extracted durable
    facts, without dumping any of it inline
  • A PROMPT-ASSEMBLER trace recorded as a PromptReport every turn
  • LLM-DRIVEN FACT EXTRACTION at session end and pre-compress —
    distills durable triples (subject/predicate/object) and stores
    them dedup'd at ~/.hermes/<profile>/nachos/facts.jsonl
  • nachos_recall tool — pull full text of any memory entry
  • nachos_facts tool — query extracted facts by predicate/kind
  • nachos_report tool — last PromptReport contribution

Pair with the nachos context engine plugin (v0.2+) for compaction +
snapshots + a system-wide PromptReport.

Config:

    memory:
      provider: nachos
    nachos:
      manifest:
        max_tokens: 400
        recent_topic_count: 5
      extraction:
        enabled: true
        on_session_end: true
        on_pre_compress: false   # opt-in; LLM call per compaction
        min_confidence: 0.6
        max_response_tokens: 2048

Slash commands registered by this plugin:
  /nachos-status   — manifest chars, fact count, extraction state, recent facts
  /nachos-report   — last PromptReport as a formatted table
  /nachos-extract  — informational: explains how extraction works
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

# Make nachos_core + adapters importable when dropped in as a plugin
_PLUGIN_ROOT = Path(__file__).resolve().parents[3]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from agent.memory_provider import MemoryProvider  # noqa: E402

from nachos_core.assembler import AssembleParams, PromptAssembler  # noqa: E402
from nachos_core.extractor import (  # noqa: E402
    ExtractionConfig,
    extract_facts,
)
from nachos_core.manifest import (  # noqa: E402
    ManifestConfig,
    build_manifest,
    render_manifest,
)
from nachos_core.types import PromptReport  # noqa: E402
from adapters.hermes_memory import HermesMemoryReader  # noqa: E402
from adapters.hermes_extractor import (  # noqa: E402
    JsonlFactStore,
    make_hermes_llm_call,
)
from plugins.memory.nachos.migration import (  # noqa: E402
    MigrationSourceError,
    known_sources,
    list_sources,
    migrate_memories,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

NACHOS_RECALL_SCHEMA = {
    "name": "nachos_recall",
    "description": (
        "Pull full content of memory entries flagged by the Nachos manifest. "
        "Use when the manifest in the system prompt mentions something "
        "relevant but you need the actual text. Returns matching entries "
        "by substring match against content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Substring to match against memory content "
                               "(case-insensitive).",
            },
            "kind": {
                "type": "string",
                "description": "Optional kind filter: preference, environment, "
                               "project, general.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 5, max 20).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

NACHOS_FACTS_SCHEMA = {
    "name": "nachos_facts",
    "description": (
        "Query the durable fact store extracted by Nachos. Facts are "
        "(subject, predicate, object) triples distilled from past "
        "conversations. Use when you want a structured view of what's "
        "been observed about the user / projects / decisions over time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Optional substring filter on subject.",
            },
            "predicate": {
                "type": "string",
                "description": "Optional substring filter on predicate.",
            },
            "kind": {
                "type": "string",
                "description": "Optional kind filter (preference, project, "
                               "environment, decision, skill, relationship, "
                               "general).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20, max 100).",
                "default": 20,
            },
        },
    },
}

NACHOS_REPORT_SCHEMA = {
    "name": "nachos_report",
    "description": (
        "Get the last assembled PromptReport — section breakdown, sizes, "
        "and hashes for the most recent system prompt contribution. "
        "Useful when auditing what context the model received this turn."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class NachosMemoryProvider(MemoryProvider):
    """Memory-side of the Nachos context layer."""

    def __init__(self):
        self._reader: Optional[HermesMemoryReader] = None
        self._assembler = PromptAssembler()
        self._manifest_cache: str = ""
        self._last_report: Optional[PromptReport] = None
        self._session_id: str = ""
        self._turn: int = 0
        self._reports_dir: Optional[Path] = None
        self._fact_store: Optional[JsonlFactStore] = None
        self._manifest_config = ManifestConfig()
        self._extraction_config = ExtractionConfig()
        self._extract_on_session_end: bool = True
        self._extract_on_pre_compress: bool = False
        self._extract_thread: Optional[threading.Thread] = None
        self._llm_call = None

    @property
    def name(self) -> str:
        return "nachos"

    def is_available(self) -> bool:
        return True

    # -- Lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home  # type: ignore

        hermes_home = Path(kwargs.get("hermes_home") or get_hermes_home())
        self._reader = HermesMemoryReader(hermes_home=hermes_home)
        self._session_id = session_id or ""
        self._turn = 0

        # Config overrides
        self._load_config()

        # Reports + facts directories
        nachos_dir = hermes_home / "nachos"
        self._reports_dir = nachos_dir / "reports"
        try:
            self._reports_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Could not create reports dir %s: %s",
                           self._reports_dir, e)
            self._reports_dir = None

        self._fact_store = JsonlFactStore(nachos_dir / "facts.jsonl")
        self._extraction_config.default_source_session = self._session_id

        # Wire LLM call for extraction (lazily, may fail if aux client absent)
        try:
            self._llm_call = make_hermes_llm_call()
        except Exception as e:
            logger.info("Nachos extraction disabled (no aux client): %s", e)
            self._llm_call = None

        # Attach reader to fact store so manifest sees extracted-fact counts
        self._reader.set_fact_store(self._fact_store)

        self._refresh_manifest()
        logger.info("Nachos memory provider initialized (session=%s, "
                    "manifest_chars=%d, extraction=%s)",
                    self._session_id, len(self._manifest_cache),
                    "on" if self._llm_call else "off")

    def _load_config(self) -> None:
        try:
            from hermes_cli.config import load_config, cfg_get
            cfg = load_config()

            mt = cfg_get(cfg, "nachos", "manifest", "max_tokens")
            rt = cfg_get(cfg, "nachos", "manifest", "recent_topic_count")
            if isinstance(mt, int) and mt > 0:
                self._manifest_config.max_tokens = mt
            if isinstance(rt, int) and rt >= 0:
                self._manifest_config.recent_topic_count = rt

            ext_se = cfg_get(cfg, "nachos", "extraction", "on_session_end")
            ext_pc = cfg_get(cfg, "nachos", "extraction", "on_pre_compress")
            min_c = cfg_get(cfg, "nachos", "extraction", "min_confidence")
            max_t = cfg_get(cfg, "nachos", "extraction", "max_response_tokens")
            if isinstance(ext_se, bool):
                self._extract_on_session_end = ext_se
            if isinstance(ext_pc, bool):
                self._extract_on_pre_compress = ext_pc
            if isinstance(min_c, (int, float)) and 0 <= min_c <= 1:
                self._extraction_config.min_confidence = float(min_c)
            if isinstance(max_t, int) and max_t > 0:
                self._extraction_config.max_response_tokens = max_t

            ext_enabled = cfg_get(cfg, "nachos", "extraction", "enabled")
            if isinstance(ext_enabled, bool) and not ext_enabled:
                self._extract_on_session_end = False
                self._extract_on_pre_compress = False
        except Exception as e:
            logger.debug("Nachos config load failed (using defaults): %s", e)

    def _refresh_manifest(self) -> None:
        if not self._reader:
            self._manifest_cache = ""
            return
        try:
            manifest = build_manifest(self._reader, self._manifest_config)
            self._manifest_cache = render_manifest(manifest)
        except Exception as e:
            logger.warning("Nachos manifest build failed: %s", e)
            self._manifest_cache = ""

    # -- Prompt contribution -----------------------------------------------

    def system_prompt_block(self) -> str:
        if not self._manifest_cache:
            self._refresh_manifest()
        if not self._manifest_cache:
            return ""

        params = AssembleParams(memory_manifest=self._manifest_cache)
        prompt, report = self._assembler.assemble(params)
        report.session_id = self._session_id
        report.turn = self._turn
        self._last_report = report
        self._persist_report(report)
        return self._manifest_cache

    def _persist_report(self, report: PromptReport) -> None:
        if not self._reports_dir or not self._session_id:
            return
        try:
            session_file = self._reports_dir / f"{self._session_id}.jsonl"
            with session_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(report.to_dict()) + "\n")
        except Exception as e:
            logger.debug("Failed to persist PromptReport: %s", e)

    # -- Tools -------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [NACHOS_RECALL_SCHEMA, NACHOS_FACTS_SCHEMA, NACHOS_REPORT_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any],
                         **kwargs) -> str:
        if tool_name == "nachos_recall":
            return self._tool_recall(args)
        if tool_name == "nachos_facts":
            return self._tool_facts(args)
        if tool_name == "nachos_report":
            return self._tool_report()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _tool_recall(self, args: Dict[str, Any]) -> str:
        if not self._reader:
            return json.dumps({"error": "Reader not initialized"})
        query = (args.get("query") or "").strip().lower()
        if not query:
            return json.dumps({"error": "query is required"})
        kind = args.get("kind") or None
        try:
            limit = int(args.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 20))

        entries = self._reader.list_entries(kind=kind, limit=500)
        matches = [
            {"content": e["content"], "kind": e["kind"], "target": e["target"]}
            for e in entries
            if query in e["content"].lower()
        ][:limit]
        return json.dumps({
            "query": query,
            "matches": matches,
            "count": len(matches),
        })

    def _tool_facts(self, args: Dict[str, Any]) -> str:
        if not self._fact_store:
            return json.dumps({"error": "Fact store not initialized"})
        all_facts = self._fact_store.list_all()
        subj = (args.get("subject") or "").strip().lower()
        pred = (args.get("predicate") or "").strip().lower()
        kind = (args.get("kind") or "").strip().lower() or None
        try:
            limit = int(args.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 100))

        out = []
        for f in all_facts:
            if subj and subj not in f.subject.lower():
                continue
            if pred and pred not in f.predicate.lower():
                continue
            if kind and f.kind != kind:
                continue
            out.append({
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "kind": f.kind,
                "confidence": round(f.confidence, 3),
                "extracted_at": f.extracted_at,
            })
            if len(out) >= limit:
                break

        return json.dumps({"facts": out, "count": len(out),
                           "total_in_store": len(all_facts)})

    def _tool_report(self) -> str:
        if not self._last_report:
            return json.dumps({"error": "No PromptReport recorded yet"})
        return json.dumps(self._last_report.to_dict())

    # -- Hermes lifecycle hooks -------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn = turn_number
        if turn_number > 0 and turn_number % 5 == 0:
            self._refresh_manifest()

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        self._manifest_cache = ""

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Optional fact extraction at compaction boundary.

        Returns empty string — we don't contribute to the compression
        prompt itself; we just take the opportunity to extract facts
        from messages about to be summarized away.
        """
        if (self._extract_on_pre_compress
                and self._llm_call
                and messages):
            self._extract_in_background(list(messages),
                                        reason="pre_compress")
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if (self._extract_on_session_end
                and self._llm_call
                and messages):
            # Run synchronously at session end so we don't lose the work
            # to process exit. Hermes calls this on real boundaries only,
            # not per-turn — synchronous is fine.
            self._extract_now(list(messages), reason="session_end",
                              wait=True)

    def _extract_in_background(self, messages, reason: str) -> None:
        # Drain previous extraction first to keep at most 1 in flight
        if self._extract_thread and self._extract_thread.is_alive():
            return  # silently skip, the previous one is still running

        def _run():
            try:
                self._extract_now(messages, reason=reason, wait=False)
            except Exception as e:
                logger.warning("Background extraction failed: %s", e)

        t = threading.Thread(target=_run, daemon=True,
                             name=f"nachos-extract-{reason}")
        t.start()
        self._extract_thread = t

    def _extract_now(self, messages, reason: str, wait: bool) -> None:
        if not self._fact_store or not self._llm_call:
            return
        try:
            result = extract_facts(messages, self._llm_call,
                                   self._extraction_config)
            if not result.parse_success:
                logger.info("Nachos extraction (%s) — parse failure: %s",
                            reason, result.error or "unknown")
                return
            if not result.facts:
                logger.info("Nachos extraction (%s) — 0 facts kept (raw=%d)",
                            reason, result.raw_count)
                return
            inserted, updated = self._fact_store.upsert(result.facts)
            logger.info(
                "Nachos extraction (%s) — kept=%d raw=%d inserted=%d updated=%d",
                reason, result.kept, result.raw_count, inserted, updated,
            )
            # Manifest is now stale — facts changed
            self._manifest_cache = ""
        except Exception as e:
            logger.warning("Nachos extraction (%s) failed: %s", reason, e)

    def shutdown(self) -> None:
        if self._extract_thread and self._extract_thread.is_alive():
            self._extract_thread.join(timeout=10.0)


# ---------------------------------------------------------------------------
# Slash command handlers (closures over the provider instance)
# ---------------------------------------------------------------------------

def _make_status_handler(provider: NachosMemoryProvider):
    def _nachos_status(raw_args: str) -> str:
        lines = ["Nachos Memory Status", "=" * 40]

        # Manifest
        manifest_chars = len(provider._manifest_cache)
        lines.append(f"manifest_chars : {manifest_chars:,}")

        # Fact store
        if provider._fact_store:
            try:
                all_facts = provider._fact_store.list_all()
                lines.append(f"total_facts    : {len(all_facts):,}")
            except Exception as e:
                lines.append(f"total_facts    : error ({e})")
                all_facts = []
        else:
            lines.append("total_facts    : (not initialized)")
            all_facts = []

        # Session
        lines.append(f"session_id     : {provider._session_id or '(none)'}")
        lines.append(f"current_turn   : {provider._turn}")

        # Extraction state
        extract_on = bool(provider._llm_call)
        lines.append(f"extraction     : {'on' if extract_on else 'off'}")
        lines.append(f"extract_on_end : {'yes' if provider._extract_on_session_end else 'no'}")
        lines.append(f"extract_on_cmp : {'yes' if provider._extract_on_pre_compress else 'no'}")

        # Last report
        if provider._last_report:
            r = provider._last_report
            lines.append(f"last_report    : turn={r.turn} chars={r.total_chars:,}"
                         f" tokens={r.total_tokens or '?'}")
        else:
            lines.append("last_report    : (none yet)")

        # Recent facts (last 5 by extracted_at, descending)
        if all_facts:
            lines.append("")
            lines.append("Recent facts (up to 5):")
            lines.append(f"  {'subject':<22} {'predicate':<20} {'object':<25} {'kind':<12} {'conf'}")
            lines.append("  " + "-" * 85)
            # Sort by extracted_at descending (string ISO timestamps sort correctly)
            sorted_facts = sorted(
                all_facts,
                key=lambda f: getattr(f, "extracted_at", "") or "",
                reverse=True,
            )
            for f in sorted_facts[:5]:
                subj = (f.subject or "")[:22]
                pred = (f.predicate or "")[:20]
                obj = (f.object or "")[:25]
                kind = (f.kind or "")[:12]
                conf = f"{f.confidence:.2f}"
                lines.append(f"  {subj:<22} {pred:<20} {obj:<25} {kind:<12} {conf}")

        return "\n".join(lines)

    return _nachos_status


def _make_report_handler(provider: NachosMemoryProvider):
    def _nachos_report(raw_args: str) -> str:
        want_history = "--history" in (raw_args or "")

        if want_history and provider._reports_dir and provider._session_id:
            session_file = provider._reports_dir / f"{provider._session_id}.jsonl"
            if session_file.exists():
                try:
                    reports = []
                    with session_file.open("r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                try:
                                    reports.append(json.loads(line))
                                except Exception:
                                    pass
                    if reports:
                        lines = [f"Nachos PromptReport history — {len(reports)} turn(s)"]
                        lines.append("=" * 60)
                        for rd in reports[-10:]:  # show last 10 turns
                            turn = rd.get("turn", "?")
                            total_chars = rd.get("total_chars", 0)
                            total_tokens = rd.get("total_tokens", "?")
                            gen_at = rd.get("generated_at", "")[:19]
                            lines.append(f"\nTurn {turn}  ({gen_at})  chars={total_chars:,}  tokens={total_tokens}")
                            sections = rd.get("sections", [])
                            if sections:
                                lines.append(f"  {'section':<22} {'chars':>8}  {'tokens':>7}  hash")
                                lines.append("  " + "-" * 55)
                                for s in sections:
                                    name = (s.get("name") or "")[:22]
                                    sc = s.get("size_chars", 0)
                                    st = s.get("size_tokens", "?")
                                    sh = (s.get("hash") or "")[:8]
                                    lines.append(f"  {name:<22} {sc:>8,}  {str(st):>7}  {sh}...")
                        return "\n".join(lines)
                    else:
                        return "No PromptReport history found for this session."
                except Exception as e:
                    return f"Error reading report history: {e}"
            else:
                return f"No report file found for session {provider._session_id}."

        # Current / last report
        report = provider._last_report
        if not report:
            return "No PromptReport recorded yet. Reports are generated each turn once the provider is active."

        turn_label = f"turn {report.turn}" if report.turn is not None else "(turn unknown)"
        lines = [f"Nachos PromptReport — {turn_label}"]
        lines.append("=" * 60)
        lines.append(f"{'section':<22} {'chars':>8}  {'tokens':>7}  hash")
        lines.append("-" * 55)
        for s in report.sections:
            name = (s.name or "")[:22]
            sc = s.size_chars
            st = s.size_tokens if s.size_tokens is not None else "?"
            sh = (s.hash or "")[:8]
            lines.append(f"{name:<22} {sc:>8,}  {str(st):>7}  {sh}...")
        lines.append("-" * 55)
        total_t = report.total_tokens if report.total_tokens is not None else "?"
        lines.append(f"{'total':<22} {report.total_chars:>8,}  {str(total_t):>7}")
        lines.append(f"\ngenerated_at: {report.generated_at}")
        if report.session_id:
            lines.append(f"session_id  : {report.session_id}")
        lines.append("\nTip: /nachos-report --history  shows all recorded turns this session.")
        return "\n".join(lines)

    return _nachos_report


def _make_extract_handler(provider: NachosMemoryProvider):
    def _nachos_extract(raw_args: str) -> str:
        lines = ["Nachos Fact Extraction"]
        lines.append("=" * 40)

        if not provider._llm_call:
            lines.append("Extraction is currently OFF (no aux LLM client configured).")
            lines.append("Check your config: nachos.extraction.enabled: true")
            lines.append("")
            lines.append("To query existing facts: use the nachos_facts tool.")
            return "\n".join(lines)

        if not provider._extract_on_session_end:
            lines.append("on_session_end extraction is disabled in config.")

        lines.append("Extraction runs automatically at:")
        lines.append("  - /clear (session boundary)")
        lines.append("  - session end (process exit / new session)")
        if provider._extract_on_pre_compress:
            lines.append("  - pre-compress (context compaction, enabled)")
        else:
            lines.append("  - pre-compress (disabled; set on_pre_compress: true to enable)")
        lines.append("")
        lines.append("Extraction needs the full conversation history, which is only")
        lines.append("available at a session boundary — not mid-session via slash command.")
        lines.append("")

        if provider._fact_store:
            try:
                count = len(provider._fact_store.list_all())
                lines.append(f"Current fact store: {count:,} facts")
            except Exception:
                pass
        lines.append("")
        lines.append("To query existing facts: use the nachos_facts tool.")
        lines.append("To force extraction now: end the session with /clear.")
        return "\n".join(lines)

    return _nachos_extract


def _format_source_listing(provider: NachosMemoryProvider) -> str:
    if provider._reader is None:
        return "Nachos is not initialized in this session yet. Start a fresh session and try again."
    statuses = list_sources(provider._reader.hermes_home)
    lines = ["Nachos migration sources"]
    for status in statuses:
        state = "ready" if status.available else ("configured" if status.configured else "not configured")
        implemented = "yes" if status.implemented else "no"
        count = "?" if status.entry_count is None else str(status.entry_count)
        lines.append(
            f"- {status.name}: state={state}; implemented={implemented}; entries={count}; kind={status.kind}"
        )
        lines.append(f"  {status.summary}")
        if status.setup_hint:
            lines.append(f"  hint: {status.setup_hint}")
    return "\n".join(lines)


def _make_migrate_handler(provider: NachosMemoryProvider):
    known = set(known_sources())

    def _nachos_migrate(raw_args: str) -> str:
        if provider._reader is None:
            return "Nachos is not initialized in this session yet. Start a fresh session and try again."
        if provider._llm_call is None:
            return (
                "Nachos extraction is unavailable in this session, so migration cannot run. "
                "Set a working auxiliary provider/model and try again."
            )

        tokens = [token.strip() for token in (raw_args or "").split() if token.strip()]
        if any(token.lower() in {"--list", "list", "--list-sources", "sources"} for token in tokens):
            return _format_source_listing(provider)

        source = "all"
        target = "both"
        dry_run = False
        for token in tokens:
            lower = token.lower()
            if lower in known:
                source = lower
            elif lower in {"memory", "user", "both"}:
                target = lower
            elif lower in {"--dry-run", "dry-run", "dryrun"}:
                dry_run = True
            else:
                return (
                    "Usage: /nachos-migrate [all|builtin|holographic|byterover|hindsight|honcho|mem0|openviking|retaindb|supermemory] "
                    "[memory|user|both] [--dry-run]\n"
                    "Use /nachos-migrate --list to inspect source readiness.\n"
                    f"Unrecognized argument: {token}"
                )

        try:
            report = migrate_memories(
                hermes_home=provider._reader.hermes_home,
                source=source,
                target=target,
                dry_run=dry_run,
            )
        except MigrationSourceError as exc:
            return str(exc)

        provider._manifest_cache = ""
        source_counts = cast(Dict[str, int], report.get("source_counts", {}))
        source_counts_text = ", ".join(
            f"{name}={count}" for name, count in sorted(source_counts.items())
        ) or "-"
        return (
            "Nachos memory migration complete\n"
            f"source         : {report['source']}\n"
            f"target         : {report['target']}\n"
            f"dry_run        : {'yes' if report['dry_run'] else 'no'}\n"
            f"source_entries : {report['source_entry_count']}\n"
            f"source_counts  : {source_counts_text}\n"
            f"batches        : {report['batch_count']}\n"
            f"candidate_facts: {report['candidate_fact_count']}\n"
            f"inserted       : {report['inserted']}\n"
            f"updated        : {report['updated']}\n"
            f"report_path    : {report['report_path']}"
        )

    return _nachos_migrate


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    provider = NachosMemoryProvider()
    ctx.register_memory_provider(provider)

    ctx.register_command(
        "nachos-status",
        _make_status_handler(provider),
        "Nachos context layer status — manifest, facts, extraction state",
    )
    ctx.register_command(
        "nachos-report",
        _make_report_handler(provider),
        "Last Nachos PromptReport (section breakdown, sizes, hashes)",
        args_hint="[--history]",
    )
    ctx.register_command(
        "nachos-extract",
        _make_extract_handler(provider),
        "Info about Nachos fact extraction scheduling",
    )
    ctx.register_command(
        "nachos-migrate",
        _make_migrate_handler(provider),
        "Import Hermes memories from built-in and provider backends into the Nachos fact store",
        args_hint="[source] [memory|user|both] [--dry-run|--list]",
    )
