import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.memory.nachos.migration import (  # noqa: E402
    MigrationSourceError,
    _load_entries,
    _load_mem0_entries,
    _load_retaindb_entries,
    _load_supermemory_entries,
    _load_honcho_entries,
    _load_hindsight_entries,
    _load_openviking_entries,
    _load_byterover_entries,
    list_sources,
    source_registry,
)


@pytest.fixture()
def fake_hermes_home(tmp_path):
    hermes_home = tmp_path / "fake_hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    (memories / "MEMORY.md").write_text("fact one\n§\nfact two\n", encoding="utf-8")
    (memories / "USER.md").write_text("pref one\n", encoding="utf-8")

    con = sqlite3.connect(hermes_home / "memory_store.db")
    con.execute(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY,
            content TEXT,
            category TEXT,
            tags TEXT,
            trust_score REAL,
            created_at TEXT,
            updated_at TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO facts (fact_id, content, category, tags, trust_score, created_at, updated_at)
        VALUES (1, 'uses uv instead of pip', 'environment', 'tooling', 0.9, '2026-01-01', '2026-01-02')
        """
    )
    con.commit()
    con.close()
    return hermes_home


def test_source_registry_contains_all_supported_backends():
    registry = source_registry()
    assert set(registry) == {
        "builtin",
        "holographic",
        "byterover",
        "hindsight",
        "honcho",
        "mem0",
        "openviking",
        "retaindb",
        "supermemory",
    }
    assert all(adapter.implemented for adapter in registry.values())


def test_list_sources_reports_counts_and_readiness(fake_hermes_home):
    statuses = {status.name: status for status in list_sources(fake_hermes_home)}

    assert statuses["builtin"].available
    assert statuses["builtin"].entry_count == 3
    assert statuses["holographic"].available
    assert statuses["holographic"].entry_count == 1

    assert not statuses["mem0"].available
    assert statuses["mem0"].entry_count is None
    assert statuses["mem0"].implemented


def test_load_entries_all_includes_available_sources_only(fake_hermes_home):
    entries, source_counts, source_details = _load_entries(fake_hermes_home, "all", "both")

    assert len(entries) == 4
    assert source_counts == {"builtin": 3, "holographic": 1}
    assert [detail["name"] for detail in source_details] == ["builtin", "holographic"]


def test_load_entries_rejects_unconfigured_provider(fake_hermes_home):
    with pytest.raises(MigrationSourceError) as exc:
        _load_entries(fake_hermes_home, "mem0", "both")

    assert "not configured" in str(exc.value)


def test_load_byterover_entries_reads_local_tree(fake_hermes_home):
    root = fake_hermes_home / "byterover"
    root.mkdir(parents=True)
    (root / "notes.md").write_text("architectural preference", encoding="utf-8")
    (root / "memory.json").write_text('{"content": "uses uv"}', encoding="utf-8")

    entries = _load_byterover_entries(fake_hermes_home)

    assert [entry["entry_id"] for entry in entries] == [
        "byterover:memory.json",
        "byterover:notes.md",
    ]
    assert entries[0]["content"] == "uses uv"


def test_load_mem0_entries_uses_get_all(fake_hermes_home, monkeypatch):
    (fake_hermes_home / "mem0.json").write_text('{"api_key": "k", "user_id": "nate"}', encoding="utf-8")

    class FakeClient:
        def get_all(self, filters):
            assert filters == {"user_id": "nate"}
            return {"results": [{"id": "1", "memory": "prefers terse responses"}]}

    monkeypatch.setattr(
        "plugins.memory.nachos.migration._make_mem0_client",
        lambda api_key: FakeClient(),
    )

    entries = _load_mem0_entries(fake_hermes_home)
    assert entries == [
        {
            "provider": "mem0",
            "target": "user",
            "entry_id": "mem0:1",
            "content": "prefers terse responses",
        }
    ]


def test_load_retaindb_entries_uses_profile_payload(fake_hermes_home, monkeypatch):
    monkeypatch.setenv("RETAINDB_API_KEY", "token")
    monkeypatch.setenv("RETAINDB_USER_ID", "nate")

    class FakeClient:
        def get_profile(self, user_id):
            assert user_id == "nate"
            return {"memories": [{"id": "m1", "content": "uses tmux", "memory_type": "preference"}]}

    monkeypatch.setattr(
        "plugins.memory.nachos.migration._make_retaindb_client",
        lambda api_key, base_url, project: FakeClient(),
    )

    entries = _load_retaindb_entries(fake_hermes_home)
    assert entries[0]["entry_id"] == "retaindb:m1"
    assert "type=preference" in entries[0]["content"]


def test_load_supermemory_entries_combines_profile_and_search(fake_hermes_home, monkeypatch):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "token")

    class FakeClient:
        def get_profile(self):
            return {
                "static": ["Lives in North GA"],
                "dynamic": ["Working on Hermes memory migration"],
                "search_results": [{"id": "x1", "memory": "Uses dark themes"}],
            }

        def search_memories(self, query, limit):
            return []

    monkeypatch.setattr(
        "plugins.memory.nachos.migration._make_supermemory_client",
        lambda api_key, timeout, container_tag, search_mode: FakeClient(),
    )

    entries = _load_supermemory_entries(fake_hermes_home)
    assert [entry["entry_id"] for entry in entries] == [
        "supermemory:static:1",
        "supermemory:dynamic:1",
        "supermemory:x1",
    ]


def test_load_honcho_entries_exports_snapshot(fake_hermes_home, monkeypatch):
    class FakeConfig:
        enabled = True
        api_key = "token"
        base_url = None

    class FakeManager:
        def get_or_create(self, session_key):
            assert session_key == "nachos-memory-import"

        def get_session_context(self, session_key, peer="user"):
            return {
                "card": "- prefers direct answers",
                "representation": "Direct, terse, pragmatic.",
                "summary": "Building durable systems.",
            }

        def get_peer_card(self, session_key, peer="user"):
            return ["prefers direct answers"]

    monkeypatch.setattr(
        "plugins.memory.nachos.migration._make_honcho_manager",
        lambda: (FakeConfig(), FakeManager()),
    )

    entries = _load_honcho_entries(fake_hermes_home)
    assert [entry["entry_id"] for entry in entries] == [
        "honcho:user-card",
        "honcho:user-representation",
        "honcho:session-summary",
    ]


def test_load_hindsight_entries_exports_reflective_snapshots(fake_hermes_home, monkeypatch):
    class FakeProvider:
        _bank_id = "hermes"
        _budget = "mid"

        def __init__(self):
            self.calls = []

        def _run_hindsight_operation(self, fn):
            self.calls.append(fn)
            if len(self.calls) == 1:
                return {"answer": "User prefers concise responses"}
            return "Project uses uv and pytest"

    monkeypatch.setattr(
        "plugins.memory.nachos.migration._make_hindsight_provider",
        lambda: FakeProvider(),
    )

    entries = _load_hindsight_entries(fake_hermes_home)
    assert [entry["entry_id"] for entry in entries] == ["hindsight:profile", "hindsight:project"]


def test_load_openviking_entries_walks_memory_tree(fake_hermes_home, monkeypatch):
    monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://ov")
    monkeypatch.setenv("OPENVIKING_USER", "nate")

    class FakeClient:
        def get(self, path, params=None):
            uri = params["uri"]
            if path == "/api/v1/fs/ls" and uri == "viking://user/nate/memories":
                return {"result": [{"uri": "viking://user/nate/memories/preferences", "type": "dir"}]}
            if path == "/api/v1/fs/ls" and uri == "viking://user/nate/memories/preferences":
                return {"result": [{"uri": "viking://user/nate/memories/preferences/mem_1.md", "type": "file"}]}
            if path == "/api/v1/content/read":
                return {"result": {"content": "Prefers tmux and uv"}}
            raise AssertionError((path, params))

    monkeypatch.setattr(
        "plugins.memory.nachos.migration._make_openviking_client",
        lambda endpoint, api_key, account, user, agent: FakeClient(),
    )

    entries = _load_openviking_entries(fake_hermes_home)
    assert entries == [
        {
            "provider": "openviking",
            "target": "user",
            "entry_id": "openviking:viking://user/nate/memories/preferences/mem_1.md",
            "content": "Prefers tmux and uv",
        }
    ]
