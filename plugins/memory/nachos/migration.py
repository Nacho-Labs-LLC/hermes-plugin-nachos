from __future__ import annotations

import importlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from adapters.hermes_extractor import JsonlFactStore, make_hermes_llm_call
from adapters.hermes_memory import HermesMemoryReader
from nachos_core.extractor import ExtractionConfig, extract_facts

DEFAULT_BATCH_CHARS = 12000
DEFAULT_BATCH_ENTRIES = 12
DEFAULT_SOURCE_SESSION = "nachos-memory-import"
_TEXT_PRIORITY_KEYS = (
    "content",
    "memory",
    "text",
    "summary",
    "description",
    "note",
    "message",
    "value",
    "body",
)


class MigrationSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceAdapter:
    name: str
    summary: str
    kind: str
    implemented: bool
    supports_target_filter: bool
    requires_network: bool
    loader: Callable[[Path, str], List[Dict[str, str]]] | None = None
    setup_hint: str = ""


@dataclass(frozen=True)
class SourceStatus:
    name: str
    summary: str
    kind: str
    implemented: bool
    supports_target_filter: bool
    requires_network: bool
    configured: bool
    available: bool
    entry_count: int | None
    setup_hint: str


@dataclass(frozen=True)
class SourcePreview:
    configured: bool
    available: bool
    entry_count: int | None


@dataclass(frozen=True)
class LoadResult:
    source: str
    entries: List[Dict[str, str]]
    preview: SourcePreview


def _split_chunks(raw: str) -> List[str]:
    return [chunk.strip() for chunk in raw.split("§") if chunk.strip()]


def _import_provider_module(name: str):
    return importlib.import_module(f"plugins.memory.{name}")


def _load_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            sample = fh.read(2048)
    except Exception:
        return True
    return b"\x00" in sample


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _flatten_strings(value: Any, *, limit: int = 200) -> List[str]:
    results: List[str] = []

    def visit(obj: Any) -> None:
        if len(results) >= limit or obj is None:
            return
        if isinstance(obj, str):
            text = obj.strip()
            if text:
                results.append(text)
            return
        if isinstance(obj, (int, float, bool)):
            results.append(str(obj))
            return
        if isinstance(obj, dict):
            for key in _TEXT_PRIORITY_KEYS:
                if key in obj:
                    visit(obj[key])
            for key, nested in obj.items():
                if key not in _TEXT_PRIORITY_KEYS:
                    visit(nested)
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                visit(item)

    visit(value)
    deduped: List[str] = []
    seen = set()
    for item in results:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped[:limit]


def _extract_json_content(payload: Any) -> str:
    bits = _flatten_strings(payload)
    return "\n\n".join(bit for bit in bits if bit)


def _entry(provider: str, target: str, entry_id: str, content: str) -> Dict[str, str] | None:
    text = (content or "").strip()
    if not text:
        return None
    return {
        "provider": provider,
        "target": target,
        "entry_id": entry_id,
        "content": text,
    }


def _unwrap_results(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("results", "memories", "items", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
    return []


def _load_builtin_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    memories_dir = hermes_home / "memories"
    entries: List[Dict[str, str]] = []
    sources = []
    if target in ("both", "memory"):
        sources.append((memories_dir / "MEMORY.md", "memory"))
    if target in ("both", "user"):
        sources.append((memories_dir / "USER.md", "user"))
    for path, source_target in sources:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        for idx, chunk in enumerate(_split_chunks(raw), 1):
            item = _entry("builtin", source_target, f"builtin:{source_target}:{idx}", chunk)
            if item:
                entries.append(item)
    return entries


def _load_holographic_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    del target
    db_path = hermes_home / "memory_store.db"
    if not db_path.exists():
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT fact_id, content, category, tags, trust_score, created_at, updated_at
            FROM facts
            ORDER BY fact_id ASC
            """
        ).fetchall()
    finally:
        con.close()

    entries: List[Dict[str, str]] = []
    for row in rows:
        content = (row["content"] or "").strip()
        if not content:
            continue
        meta = (
            f"Legacy holographic fact (id={row['fact_id']}, "
            f"category={row['category'] or 'general'}, "
            f"trust={row['trust_score']}, tags={row['tags'] or '-'}, "
            f"created_at={row['created_at']}, updated_at={row['updated_at']})."
        )
        item = _entry(
            "holographic",
            row["category"] or "general",
            f"holographic:{row['fact_id']}",
            f"{meta}\n\n{content}",
        )
        if item:
            entries.append(item)
    return entries


def _load_byterover_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    del target
    root = hermes_home / "byterover"
    if not root.exists():
        return []
    entries: List[Dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_probably_binary(path):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            if path.suffix.lower() == ".json":
                content = _extract_json_content(json.loads(path.read_text(encoding="utf-8")))
            else:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        item = _entry("byterover", "memory", f"byterover:{rel}", content)
        if item:
            entries.append(item)
    return entries


def _load_mem0_config(hermes_home: Path) -> dict:
    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
    }
    config_path = hermes_home / "mem0.json"
    if config_path.exists():
        file_config = _load_json_file(config_path)
        for key in ("api_key", "user_id", "agent_id"):
            value = _stringify_scalar(file_config.get(key))
            if value:
                config[key] = value
    return config


def _make_mem0_client(api_key: str):
    from mem0 import MemoryClient
    return MemoryClient(api_key=api_key)


def _load_mem0_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    del target
    config = _load_mem0_config(hermes_home)
    api_key = config.get("api_key", "").strip()
    if not api_key:
        raise MigrationSourceError("Mem0 API key missing (MEM0_API_KEY or mem0.json api_key).")
    client = _make_mem0_client(api_key)
    rows = _unwrap_results(client.get_all(filters={"user_id": config["user_id"]}))
    entries: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        content = (row.get("memory") or row.get("content") or "").strip()
        entry_id = row.get("id") or row.get("memory_id") or f"{config['user_id']}:{idx}"
        item = _entry("mem0", "user", f"mem0:{entry_id}", content)
        if item:
            entries.append(item)
    return entries


def _make_retaindb_client(api_key: str, base_url: str, project: str):
    module = _import_provider_module("retaindb")
    return module._Client(api_key, base_url, project)


def _resolve_retaindb_project(hermes_home: Path) -> str:
    explicit = os.environ.get("RETAINDB_PROJECT", "").strip()
    if explicit:
        return explicit
    if hermes_home.name and hermes_home.name != ".hermes":
        return f"hermes-{hermes_home.name}"
    return "default"


def _load_retaindb_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    del target
    api_key = os.environ.get("RETAINDB_API_KEY", "").strip()
    if not api_key:
        raise MigrationSourceError("RetainDB API key missing (RETAINDB_API_KEY).")
    base_url = os.environ.get("RETAINDB_BASE_URL", "https://api.retaindb.com").rstrip("/")
    user_id = os.environ.get("RETAINDB_USER_ID", "default").strip() or "default"
    client = _make_retaindb_client(api_key, base_url, _resolve_retaindb_project(hermes_home))
    rows = _unwrap_results(client.get_profile(user_id))
    entries: List[Dict[str, str]] = []
    for idx, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            continue
        content = (row.get("content") or row.get("memory") or "").strip()
        entry_id = row.get("id") or row.get("memory_id") or f"{user_id}:{idx}"
        memory_type = row.get("memory_type") or row.get("type") or "factual"
        item = _entry("retaindb", "user", f"retaindb:{entry_id}", f"RetainDB memory (type={memory_type}).\n\n{content}")
        if item:
            entries.append(item)
    return entries


def _make_openviking_client(endpoint: str, api_key: str, account: str, user: str, agent: str):
    module = _import_provider_module("openviking")
    return module._VikingClient(endpoint, api_key=api_key, account=account, user=user, agent=agent)


def _openviking_result(value: Any) -> Any:
    return value.get("result") if isinstance(value, dict) and "result" in value else value


def _openviking_is_dir(node: Any) -> bool | None:
    if isinstance(node, dict):
        if "isDir" in node:
            return bool(node.get("isDir"))
        if "is_dir" in node:
            return bool(node.get("is_dir"))
        if node.get("type") == "dir":
            return True
        if node.get("type") == "file":
            return False
    return None


def _openviking_child_uri(parent_uri: str, node: Any) -> str:
    if isinstance(node, dict):
        uri = _stringify_scalar(node.get("uri"))
        if uri:
            return uri
        name = _stringify_scalar(node.get("name"))
        if name:
            return f"{parent_uri.rstrip('/')}/{name}"
    return ""


def _openviking_read_content(client: Any, uri: str) -> str:
    result = _openviking_result(client.get("/api/v1/content/read", params={"uri": uri}))
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return (result.get("content") or result.get("text") or "").strip()
    return ""


def _load_openviking_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    del hermes_home, target
    endpoint = os.environ.get("OPENVIKING_ENDPOINT", "http://127.0.0.1:1933").rstrip("/")
    api_key = os.environ.get("OPENVIKING_API_KEY", "")
    account = os.environ.get("OPENVIKING_ACCOUNT", "default")
    user = os.environ.get("OPENVIKING_USER", "default")
    agent = os.environ.get("OPENVIKING_AGENT", "hermes")
    client = _make_openviking_client(endpoint, api_key, account, user, agent)
    root_uri = f"viking://user/{user}/memories"
    queue = [root_uri]
    seen_dirs = set()
    entries: List[Dict[str, str]] = []
    while queue:
        current = queue.pop(0)
        if current in seen_dirs:
            continue
        seen_dirs.add(current)
        children = _openviking_result(client.get("/api/v1/fs/ls", params={"uri": current}))
        if not isinstance(children, list):
            continue
        for child in children:
            uri = _openviking_child_uri(current, child)
            if not uri:
                continue
            is_dir = _openviking_is_dir(child)
            if is_dir is True:
                queue.append(uri)
                continue
            content = _openviking_read_content(client, uri)
            target_name = "user" if "/preferences/" in uri else "memory"
            item = _entry("openviking", target_name, f"openviking:{uri}", content)
            if item:
                entries.append(item)
    return entries


def _make_supermemory_client(api_key: str, timeout: float, container_tag: str, search_mode: str):
    module = _import_provider_module("supermemory")
    return module._SupermemoryClient(api_key, timeout, container_tag, search_mode=search_mode)


def _load_supermemory_config(hermes_home: Path) -> dict:
    config = {
        "container_tag": os.environ.get("SUPERMEMORY_CONTAINER_TAG", "hermes"),
        "search_mode": "hybrid",
        "api_timeout": 15.0,
    }
    config_path = hermes_home / "supermemory.json"
    if config_path.exists():
        file_config = _load_json_file(config_path)
        for key in ("container_tag", "search_mode", "api_timeout"):
            if key in file_config and file_config[key] not in (None, ""):
                config[key] = file_config[key]
    return config


def _load_supermemory_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    del target
    api_key = os.environ.get("SUPERMEMORY_API_KEY", "").strip()
    if not api_key:
        raise MigrationSourceError("Supermemory API key missing (SUPERMEMORY_API_KEY).")
    config = _load_supermemory_config(hermes_home)
    client = _make_supermemory_client(
        api_key,
        float(config.get("api_timeout", 15.0) or 15.0),
        str(config.get("container_tag", "hermes") or "hermes"),
        str(config.get("search_mode", "hybrid") or "hybrid"),
    )
    profile = client.get_profile()
    entries: List[Dict[str, str]] = []
    for idx, bit in enumerate(_flatten_strings(profile.get("static")), 1):
        item = _entry("supermemory", "user", f"supermemory:static:{idx}", bit)
        if item:
            entries.append(item)
    for idx, bit in enumerate(_flatten_strings(profile.get("dynamic")), 1):
        item = _entry("supermemory", "memory", f"supermemory:dynamic:{idx}", bit)
        if item:
            entries.append(item)
    results = profile.get("search_results") or client.search_memories("user preferences project context environment", limit=25)
    for idx, row in enumerate(results, 1):
        if not isinstance(row, dict):
            continue
        content = (row.get("memory") or row.get("content") or "").strip()
        entry_id = row.get("id") or f"result:{idx}"
        item = _entry("supermemory", "memory", f"supermemory:{entry_id}", content)
        if item:
            entries.append(item)
    return entries


def _make_honcho_manager():
    client_module = importlib.import_module("plugins.memory.honcho.client")
    session_module = importlib.import_module("plugins.memory.honcho.session")
    config = client_module.HonchoClientConfig.from_global_config()
    client = client_module.get_honcho_client(config)
    manager = session_module.HonchoSessionManager(honcho=client, config=config, context_tokens=config.context_tokens)
    return config, manager


def _load_honcho_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    del hermes_home, target
    config, manager = _make_honcho_manager()
    if not getattr(config, "enabled", False) or not (getattr(config, "api_key", None) or getattr(config, "base_url", None)):
        raise MigrationSourceError("Honcho is not configured (api key or base URL missing).")
    session_key = DEFAULT_SOURCE_SESSION
    manager.get_or_create(session_key)
    context = manager.get_session_context(session_key, peer="user")
    entries: List[Dict[str, str]] = []
    card_text = (context.get("card") or "").strip()
    if not card_text:
        try:
            card_text = "\n".join(manager.get_peer_card(session_key, peer="user")).strip()
        except Exception:
            card_text = ""
    for entry_id, target_name, text in (
        ("honcho:user-card", "user", card_text),
        ("honcho:user-representation", "user", context.get("representation", "")),
        ("honcho:session-summary", "memory", context.get("summary", "")),
    ):
        item = _entry("honcho", target_name, entry_id, text)
        if item:
            entries.append(item)
    return entries


def _make_hindsight_provider():
    module = _import_provider_module("hindsight")
    provider = module.HindsightMemoryProvider()
    config = module._load_config()
    bank_cfg = (config.get("banks") or {}).get("hermes", {})
    provider._config = config
    provider._mode = config.get("mode", "cloud")
    provider._api_key = config.get("apiKey") or config.get("api_key") or os.environ.get("HINDSIGHT_API_KEY", "")
    provider._api_url = config.get("api_url") or os.environ.get("HINDSIGHT_API_URL", getattr(module, "_DEFAULT_API_URL", ""))
    provider._timeout = float(config.get("timeout") or getattr(module, "_DEFAULT_TIMEOUT", 120))
    provider._idle_timeout = int(config.get("idle_timeout") or os.environ.get("HINDSIGHT_IDLE_TIMEOUT", getattr(module, "_DEFAULT_IDLE_TIMEOUT", 300)))
    provider._bank_id = bank_cfg.get("bankId") or os.environ.get("HINDSIGHT_BANK_ID", "hermes")
    provider._budget = bank_cfg.get("budget") or os.environ.get("HINDSIGHT_BUDGET", "mid")
    provider._llm_base_url = config.get("llm_base_url") or os.environ.get("HINDSIGHT_API_LLM_BASE_URL", "")
    return provider


def _coerce_hindsight_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("answer", "content", "text", "summary"):
            text = _stringify_scalar(value.get(key))
            if text:
                return text
        return _extract_json_content(value)
    if isinstance(value, list):
        return _extract_json_content(value)
    return ""


def _load_hindsight_entries(hermes_home: Path, target: str = "both") -> List[Dict[str, str]]:
    del hermes_home, target
    provider = _make_hindsight_provider()
    entries: List[Dict[str, str]] = []
    prompts = [
        ("profile", "user", "Summarize durable user facts, preferences, identity details, and constraints already stored in this bank."),
        ("project", "memory", "Summarize durable project, environment, and workflow facts already stored in this bank."),
    ]
    for suffix, target_name, query in prompts:
        result = provider._run_hindsight_operation(
            lambda client, q=query: client.areflect(bank_id=provider._bank_id, query=q, budget=provider._budget)
        )
        item = _entry("hindsight", target_name, f"hindsight:{suffix}", _coerce_hindsight_text(result))
        if item:
            entries.append(item)
    return entries


def source_registry() -> Dict[str, SourceAdapter]:
    return {
        "builtin": SourceAdapter("builtin", "Built-in durable memory files: MEMORY.md and USER.md.", "local-files", True, True, False, _load_builtin_entries),
        "holographic": SourceAdapter("holographic", "Local SQLite fact store used by Hermes holographic memory provider.", "local-sqlite", True, False, False, _load_holographic_entries),
        "byterover": SourceAdapter("byterover", "ByteRover local context tree under $HERMES_HOME/byterover.", "cli+vendor-store", True, False, False, _load_byterover_entries),
        "hindsight": SourceAdapter("hindsight", "Hindsight reflective memory snapshot via configured client/backend.", "api-or-local-daemon", True, False, True, _load_hindsight_entries, "Uses a reflective snapshot because Hermes does not expose a raw list-all export path for Hindsight."),
        "honcho": SourceAdapter("honcho", "Honcho peer card / representation / session summary snapshot.", "sdk+api", True, False, True, _load_honcho_entries, "Exports Honcho's user card, representation, and summary snapshot rather than a raw conclusions dump."),
        "mem0": SourceAdapter("mem0", "Mem0 hosted memory platform via MemoryClient.get_all().", "api", True, False, True, _load_mem0_entries),
        "openviking": SourceAdapter("openviking", "OpenViking memory subtree traversal under viking://user/<user>/memories.", "api", True, False, True, _load_openviking_entries),
        "retaindb": SourceAdapter("retaindb", "RetainDB profile-backed memory export via /v1/memory/profile or /v1/memories.", "api", True, False, True, _load_retaindb_entries),
        "supermemory": SourceAdapter("supermemory", "Supermemory profile + search snapshot using the configured container tag.", "api", True, False, True, _load_supermemory_entries, "Exports profile/search snapshots because Hermes does not expose a full list-all read path for Supermemory."),
    }


def known_sources() -> List[str]:
    return ["all", *source_registry().keys()]


def _provider_configured(hermes_home: Path, source: str) -> bool:
    env = os.environ
    if source == "builtin":
        memories_dir = hermes_home / "memories"
        return (memories_dir / "MEMORY.md").exists() or (memories_dir / "USER.md").exists()
    if source == "holographic":
        return (hermes_home / "memory_store.db").exists()
    if source == "byterover":
        return (hermes_home / "byterover").exists() or bool(env.get("BRV_API_KEY"))
    if source == "hindsight":
        return bool(env.get("HINDSIGHT_API_KEY") or env.get("HINDSIGHT_MODE") or (hermes_home / "hindsight" / "config.json").exists())
    if source == "honcho":
        return bool(env.get("HONCHO_API_KEY") or env.get("HONCHO_BASE_URL") or (hermes_home / "honcho.json").exists() or (Path.home() / ".hermes" / "honcho.json").exists() or (Path.home() / ".honcho" / "config.json").exists())
    if source == "mem0":
        return bool(env.get("MEM0_API_KEY") or (hermes_home / "mem0.json").exists())
    if source == "openviking":
        return bool(env.get("OPENVIKING_API_KEY") or env.get("OPENVIKING_ENDPOINT"))
    if source == "retaindb":
        return bool(env.get("RETAINDB_API_KEY"))
    if source == "supermemory":
        return bool(env.get("SUPERMEMORY_API_KEY"))
    return False


def _preview_source(hermes_home: Path, adapter: SourceAdapter, target: str) -> SourcePreview:
    configured = _provider_configured(hermes_home, adapter.name)
    if not configured:
        return SourcePreview(False, False, None)
    if not adapter.implemented or adapter.loader is None:
        return SourcePreview(configured, False, None)
    try:
        entries = adapter.loader(hermes_home, target)
    except Exception:
        return SourcePreview(configured, False, None)
    return SourcePreview(configured, True, len(entries))


def list_sources(hermes_home: Path, target: str = "both") -> List[SourceStatus]:
    statuses: List[SourceStatus] = []
    for adapter in source_registry().values():
        preview = _preview_source(hermes_home, adapter, target)
        statuses.append(SourceStatus(adapter.name, adapter.summary, adapter.kind, adapter.implemented, adapter.supports_target_filter, adapter.requires_network, preview.configured, preview.available, preview.entry_count, adapter.setup_hint))
    return statuses


def _load_from_adapter(hermes_home: Path, adapter: SourceAdapter, target: str) -> LoadResult:
    configured = _provider_configured(hermes_home, adapter.name)
    if not configured:
        raise MigrationSourceError(f"Source '{adapter.name}' is not configured for this Hermes home.")
    if not adapter.implemented or adapter.loader is None:
        raise MigrationSourceError(f"Source '{adapter.name}' is recognized but not implemented yet. {adapter.setup_hint}".strip())
    try:
        entries = adapter.loader(hermes_home, target)
    except MigrationSourceError:
        raise
    except Exception as exc:
        raise MigrationSourceError(f"Failed loading source '{adapter.name}': {exc}") from exc
    return LoadResult(adapter.name, entries, SourcePreview(configured, True, len(entries)))


def _load_entries(hermes_home: Path, source: str, target: str) -> tuple[List[Dict[str, str]], Dict[str, int], List[Dict[str, object]]]:
    registry = source_registry()
    if source == "all":
        selected_names = []
        for name, adapter in registry.items():
            if adapter.implemented and _preview_source(hermes_home, adapter, target).available:
                selected_names.append(name)
    else:
        if source not in registry:
            raise MigrationSourceError(f"Unsupported source: {source}")
        selected_names = [source]

    entries: List[Dict[str, str]] = []
    source_counts: Dict[str, int] = {}
    source_details: List[Dict[str, object]] = []
    for name in selected_names:
        adapter = registry[name]
        result = _load_from_adapter(hermes_home, adapter, target)
        entries.extend(result.entries)
        source_counts[name] = len(result.entries)
        source_details.append({
            "name": name,
            "summary": adapter.summary,
            "kind": adapter.kind,
            "implemented": adapter.implemented,
            "configured": result.preview.configured,
            "available": result.preview.available,
            "entry_count": result.preview.entry_count,
        })
    return entries, source_counts, source_details


def _render_batch_messages(entries: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for entry in entries:
        messages.append({
            "role": "user",
            "content": (
                f"Legacy Hermes memory entry from provider={entry['provider']} target={entry['target']} ({entry['entry_id']}). "
                f"Extract durable facts from this entry only. Do not mention migration metadata in the extracted facts.\n\n{entry['content']}"
            ),
        })
    return messages


def _build_batches(entries: List[Dict[str, str]], *, max_entries: int = DEFAULT_BATCH_ENTRIES, max_chars: int = DEFAULT_BATCH_CHARS) -> List[List[Dict[str, str]]]:
    batches: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    current_chars = 0
    for entry in entries:
        entry_chars = len(entry["content"])
        if current and (len(current) >= max_entries or current_chars + entry_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(entry)
        current_chars += entry_chars
    if current:
        batches.append(current)
    return batches


def migrate_memories(*, hermes_home: Path, source: str = "all", target: str = "both", dry_run: bool = False, source_session: str = DEFAULT_SOURCE_SESSION, max_batch_entries: int = DEFAULT_BATCH_ENTRIES, max_batch_chars: int = DEFAULT_BATCH_CHARS, min_confidence: float = 0.6, max_response_tokens: int = 2048) -> Dict[str, object]:
    reader = HermesMemoryReader(hermes_home=hermes_home)
    llm_call = make_hermes_llm_call()
    nachos_dir = hermes_home / "nachos"
    fact_store = JsonlFactStore(nachos_dir / "facts.jsonl")
    entries, source_counts, source_details = _load_entries(hermes_home, source, target)
    batches = _build_batches(entries, max_entries=max_batch_entries, max_chars=max_batch_chars)
    config = ExtractionConfig(min_confidence=min_confidence, max_response_tokens=max_response_tokens, default_source_session=source_session)

    all_facts = []
    batch_reports = []
    for idx, batch in enumerate(batches, 1):
        result = extract_facts(_render_batch_messages(batch), llm_call, config)
        batch_reports.append({
            "batch": idx,
            "entry_count": len(batch),
            "raw_count": result.raw_count,
            "kept": result.kept,
            "parse_success": result.parse_success,
            "error": result.error,
            "entry_ids": [entry["entry_id"] for entry in batch],
        })
        if result.parse_success and result.facts:
            all_facts.extend(result.facts)

    inserted = updated = 0
    if not dry_run and all_facts:
        inserted, updated = fact_store.upsert(all_facts)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hermes_home": str(hermes_home),
        "source": source,
        "target": target,
        "dry_run": dry_run,
        "source_session": source_session,
        "source_entry_count": len(entries),
        "source_counts": source_counts,
        "source_details": source_details,
        "batch_count": len(batches),
        "candidate_fact_count": len(all_facts),
        "inserted": inserted,
        "updated": updated,
        "reader_targets": [e["target"] for e in reader.list_entries(limit=10)],
        "batch_reports": batch_reports,
        "facts_preview": [asdict(fact) for fact in all_facts[:20]],
    }
    migrations_dir = nachos_dir / "migrations"
    migrations_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "dryrun" if dry_run else "apply"
    report_path = migrations_dir / f"{stamp}-{source}-{target}-{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
