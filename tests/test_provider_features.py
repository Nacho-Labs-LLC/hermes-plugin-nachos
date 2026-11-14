"""Hermes adapter: config, tool surface, shared scope, redaction, history tool."""

import json
import sys
import types

from nachos_core.snapshots import SnapshotStore
from nachos_hermes.memory_provider import NachosMemoryProvider, register


def _provider(home, **cfg):
    cfg.setdefault("embeddings", "off")
    p = NachosMemoryProvider()
    p.save_config(cfg, str(home))
    p.initialize("session-1", hermes_home=str(home))
    return p


def _call(p, tool, **args):
    return json.loads(p.handle_tool_call(tool, args))


def _tool_names(p):
    return [s["name"] for s in p.get_tool_schemas()]


def test_tool_surface_without_shared_scope(tmp_path):
    p = _provider(tmp_path)

    assert _tool_names(p) == [
        "nachos_memory_recall",
        "nachos_memory_put",
        "nachos_memory_remove",
        "nachos_history_search",
    ]
    assert all(
        "scope" not in s["parameters"]["properties"]
        for s in p.get_tool_schemas()
        if s["name"] != "nachos_history_search"
    )


def test_shared_scope_is_visible_across_profiles(tmp_path):
    root = tmp_path / "hermes"
    work = _provider(
        root / "profiles" / "work", shared_store="on", shared_store_path=str(tmp_path / "shared")
    )
    home = _provider(
        root / "profiles" / "home", shared_store="on", shared_store_path=str(tmp_path / "shared")
    )

    stored = _call(
        work,
        "nachos_memory_put",
        title="Preferred editor",
        body="Neovim with lazy.nvim on every machine.",
        scope="shared",
    )
    _call(
        work,
        "nachos_memory_put",
        title="Work VPN",
        body="Connect through the corp WireGuard config.",
    )

    assert stored["scope"] == "shared"
    assert "shared/general" in home.system_prompt_block()
    assert _call(home, "nachos_memory_recall", key="preferred-editor")["scope"] == "shared"
    assert "error" in _call(home, "nachos_memory_recall", key="work-vpn")
    assert "scope" in home.get_tool_schemas()[1]["parameters"]["properties"]


def test_writes_use_the_hermes_redactor_when_available(tmp_path, monkeypatch):
    calls = []

    def redact_sensitive_text(text, *, force=False, file_read=False):
        calls.append((force, file_read))
        return text.replace("ghp_abcdefghijklmnopqrstuvwxyz0123456789", "«redacted:ghp_…»")

    agent = types.ModuleType("agent")
    redact = types.ModuleType("agent.redact")
    redact.redact_sensitive_text = redact_sensitive_text
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.redact", redact)

    p = _provider(tmp_path)
    _call(
        p,
        "nachos_memory_put",
        title="GitHub token location",
        body="Token ghp_abcdefghijklmnopqrstuvwxyz0123456789 lives in the CI vault.",
    )

    body = _call(p, "nachos_memory_recall", key="github-token-location")["body"]
    assert "ghp_abc" not in body
    assert calls
    assert calls[0] == (True, True)


def test_redaction_can_be_disabled(tmp_path, monkeypatch):
    redact = types.ModuleType("agent.redact")
    redact.redact_sensitive_text = lambda _text, **_kw: "REDACTED"
    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.redact", redact)

    p = _provider(tmp_path, redact_secrets="off")
    _call(p, "nachos_memory_put", title="Plain note", body="Nothing secret in this note at all.")

    assert _call(p, "nachos_memory_recall", key="plain-note")["body"].startswith("Nothing")


def test_history_tool_searches_pre_compress_checkpoints(tmp_path):
    p = _provider(tmp_path)
    p.on_pre_compress(
        [
            {
                "role": "user",
                "content": "The migration must run before 2am because of the batch job.",
            },
        ]
    )

    out = _call(p, "nachos_history_search", query="migration batch")

    assert out["count"] == 1
    assert out["results"][0]["session_id"] == "session-1"
    assert "2am" in out["results"][0]["excerpt"]
    assert "error" in _call(p, "nachos_history_search", query="")


def test_expired_checkpoints_are_pruned_on_initialize(tmp_path):
    root = tmp_path / "nachos" / "snapshots"
    store = SnapshotStore(root, "old-session")
    store.save(messages=[{"role": "user", "content": "ancient"}])
    row = json.loads(store.index_file.read_text())
    row["created_at"] = 0
    store.index_file.write_text(json.dumps(row) + "\n")

    _provider(tmp_path, checkpoint_max_age_days=30)

    assert not (root / "old-session").exists()


def test_status_and_audit_render(tmp_path):
    p = _provider(tmp_path)
    _call(
        p,
        "nachos_memory_put",
        title="User prefers dark theme",
        body="User prefers dark theme in every editor and terminal.",
    )
    _call(
        p,
        "nachos_memory_put",
        title="User prefers a dark theme",
        body="User prefers dark theme in all editors and terminals.",
    )

    status = p.status_text()
    audit = p.audit_text(30)

    assert "profile     : 2 entries" in status
    assert "embeddings  : off" in status
    assert "Likely duplicates: 1" in audit


def _msgs(text):
    return [{"role": "user", "content": text}]


def test_lineage_follows_compression_and_resets(tmp_path):
    p = _provider(tmp_path)
    assert p._lineage == ["session-1"]
    p.on_pre_compress(_msgs("alpha secret-plan detail one"))

    p.on_session_switch("session-2", parent_session_id="session-1", reset=False)
    p.on_session_switch("session-2", parent_session_id="session-1")  # idempotent
    assert p._lineage == ["session-1", "session-2"]
    assert p._session_id == "session-2"
    p.on_pre_compress(_msgs("beta secret-plan detail two"))
    # checkpoint dedup is per session
    assert (
        p.on_pre_compress(_msgs("beta secret-plan detail two")).split()[:4]
        == p.on_pre_compress(_msgs("beta secret-plan detail two")).split()[:4]
    )
    assert len(list((tmp_path / "nachos" / "snapshots" / "session-2").glob("*.json.gz"))) == 1

    out = _call(p, "nachos_history_search", query="secret-plan")
    assert out["count"] == 2

    p.on_session_switch("session-3", reset=True)
    assert p._lineage == ["session-3"]
    assert _call(p, "nachos_history_search", query="secret-plan")["count"] == 0
    assert _call(p, "nachos_history_search", query="secret-plan", scope="all")["count"] == 2


def test_on_pre_compress_note_and_dedup(tmp_path):
    p = _provider(tmp_path)
    messages = _msgs("run migrate --fast at 2am")
    note = p.on_pre_compress(messages)
    index = (tmp_path / "nachos" / "snapshots" / "session-1" / "index.jsonl").read_text()
    checkpoint_id = json.loads(index.splitlines()[0])["id"]

    assert checkpoint_id in note
    assert "nachos_history_search" in note
    assert "commands" in note
    assert "error text" in note
    assert p.on_pre_compress(messages) == note  # dedup hit: same note, no new checkpoint
    assert len(index.splitlines()) == 1


def test_on_pre_compress_fails_closed_before_initialize():
    p = NachosMemoryProvider()
    try:
        p.on_pre_compress([])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_backup_paths_shared_off_and_on(tmp_path, monkeypatch):
    home = tmp_path / "home" / ".hermes"
    (home / "nachos").mkdir(parents=True)
    fake = types.ModuleType("hermes_constants")
    fake.get_hermes_home = lambda: home
    monkeypatch.setitem(sys.modules, "hermes_constants", fake)
    cfg = home / "nachos" / "config.json"

    p = NachosMemoryProvider()  # never initialized
    assert p.backup_paths() == []

    outside = tmp_path / "elsewhere" / "shared"
    cfg.write_text(json.dumps({"shared_store": "off", "shared_store_path": str(outside)}))
    assert p.backup_paths() == []

    cfg.write_text(json.dumps({"shared_store": "on", "shared_store_path": str(outside)}))
    assert p.backup_paths() == [str(outside)]

    cfg.write_text(json.dumps({"shared_store": "on", "shared_store_path": str(home / "inner")}))
    assert p.backup_paths() == []


def test_delegation_outcome_is_searchable(tmp_path):
    p = _provider(tmp_path)
    p.on_delegation(
        "audit the flux-capacitor module",
        "found leak in flux.c line 42",
        child_session_id="child-9",
    )
    out = _call(p, "nachos_history_search", query="flux leak")
    assert out["count"] == 1
    assert "subagent result" in out["results"][0]["excerpt"]
    index = (tmp_path / "nachos" / "snapshots" / "session-1" / "index.jsonl").read_text()
    row = json.loads(index.splitlines()[0])
    assert row["reason"] == "delegation"
    assert "child_session_id=child-9" in row["notes"]

    p._primary = False
    p.on_delegation("x", "y")  # skipped, never raises
    assert len(index.splitlines()) == 1


def test_history_and_revert_commands_render(tmp_path):
    p = _provider(tmp_path)
    _call(p, "nachos_memory_put", title="Editor", body="Original editor preference text.")
    _call(p, "nachos_memory_put", title="Editor", body="Replaced editor preference text.")

    commands = {}

    class Ctx:
        def register_memory_provider(self, provider):
            pass

        def register_command(self, name, handler, *_a, **_k):
            commands[name] = handler

    register(Ctx())
    assert {"nachos-memory-history", "nachos-memory-revert"} <= set(commands)

    text = p.changes_text()
    assert "profile:2" in text
    assert "update" in text
    assert "editor" in text
    assert "change log" in p.status_text()

    out = p.revert_text("profile:2")
    assert "Reverted update of editor" in out
    assert "Original editor" in p._service.get("editor")[1]
    assert "Usage" in p.revert_text("garbage")
    assert "No change" in p.revert_text("profile:999")
