"""tools/import_memories.py — explicit one-time import from Hermes providers."""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import import_memories

from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.store import get_store
from nachos_core.store.md_store import slugify

D = import_memories.ENTRY_DELIMITER


@pytest.fixture(autouse=True)
def _identity_redactor(monkeypatch):
    monkeypatch.setattr(import_memories, "hermes_redactor", lambda: lambda t: t)


@pytest.fixture
def home(tmp_path):
    mem = tmp_path / "home" / "memories"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text(
        D.join(
            [
                "Project uses uv and pytest for all Python tooling.",
                "Deploys go through the blue green pipeline on Fridays.",
            ]
        ),
        encoding="utf-8",
    )
    (mem / "USER.md").write_text("User prefers concise answers with no emojis.", encoding="utf-8")
    return tmp_path / "home"


def _run(home, tmp_path, *extra, capsys=None, user=True):
    store = tmp_path / "nachos.db"
    flags = ["--include-user-profile"] if user else []
    code = import_memories.main(
        ["builtin", "--hermes-home", str(home), "--path", str(store), "--json", *flags, *extra]
    )
    assert code == 0
    return store, json.loads(capsys.readouterr().out)


def _entries(store):
    s = get_store("sqlite", store)
    return {k: (cat, s.get(k)) for k, _t, _sm, cat in s.list()}


def test_builtin_dry_run_writes_nothing(home, tmp_path, capsys):
    store, out = _run(home, tmp_path, capsys=capsys)
    assert out["dry_run"] is True
    assert len(out["imported"]) == 3
    assert not store.exists()


def test_builtin_dry_run_leaves_existing_store_untouched(home, tmp_path, capsys):
    store = tmp_path / "nachos.db"
    get_store("sqlite", store).put(
        "keep", title="Keep", summary="keep this one", category="x", body="keep body text here"
    )
    _run(home, tmp_path, capsys=capsys)
    assert list(_entries(store)) == ["keep"]


def test_run_imports_with_categories_and_session_id(home, tmp_path, capsys):
    store, out = _run(home, tmp_path, "--run", capsys=capsys)
    entries = _entries(store)
    assert len(out["imported"]) == 3
    assert len(entries) == 3
    cats = {c for c, _b in entries.values()}
    assert cats == {"memory", "user"}
    idx = MemoryIndex(index_path_for(store))
    changes = idx.changes(limit=10)
    assert len(changes) == 3
    assert {c["session_id"] for c in changes} == {"import:builtin"}


def test_low_signal_skipped(home, tmp_path, capsys):
    (home / "memories" / "USER.md").write_text(
        f"nothing{D}Real preference about vim keybindings.", encoding="utf-8"
    )
    for extra in ((), ("--run",)):
        _store, out = _run(home, tmp_path, *extra, capsys=capsys)
        assert [r["title"] for r in out["skipped_low_signal"]] == ["nothing"]
        assert len(out["imported"]) == 3


def test_collision_skipped_unless_overwrite(home, tmp_path, capsys):
    store = tmp_path / "nachos.db"
    s = get_store("sqlite", store)
    title = "Project uses uv and pytest for all Python tooling."
    key = slugify(title)
    s.put(key, title=title, summary="existing different", category="memory", body="Different body.")
    s.close() if hasattr(s, "close") else None

    _, out = _run(home, tmp_path, "--run", capsys=capsys)
    assert [r["key"] for r in out["collisions"]] == [key]
    assert _entries(store)[key][1] == "Different body."

    _, out = _run(home, tmp_path, "--run", "--overwrite", capsys=capsys)
    assert out["collisions"] == []
    assert _entries(store)[key][1].startswith("Project uses uv")


def test_rerun_reports_unchanged(home, tmp_path, capsys):
    _run(home, tmp_path, "--run", capsys=capsys)
    _, out = _run(home, tmp_path, "--run", capsys=capsys)
    assert len(out["unchanged"]) == 3
    assert out["imported"] == []


def test_duplicate_flagged(home, tmp_path, capsys):
    store = tmp_path / "nachos.db"
    get_store("sqlite", store).put(
        "uv-pytest",
        title="uv pytest",
        summary="Project uses uv and pytest for Python tooling",
        category="x",
        body="Project uses uv and pytest for all Python tooling here.",
    )
    _, out = _run(home, tmp_path, "--run", capsys=capsys)
    assert any("uv" in d["title"] for d in out["duplicates"])


def test_redaction_applied(home, tmp_path, capsys, monkeypatch):
    (home / "memories" / "MEMORY.md").write_text(
        "API key is sk-live-123456 for the billing sandbox.", encoding="utf-8"
    )
    monkeypatch.setattr(
        import_memories,
        "hermes_redactor",
        lambda: lambda t: t.replace("sk-live-123456", "[REDACTED]"),
    )
    store, _ = _run(home, tmp_path, "--run", capsys=capsys)
    blob = json.dumps(_entries(store))
    assert "sk-live" not in blob
    assert "REDACTED" in blob


def test_flatfile_store(home, tmp_path):
    store = tmp_path / "memory.md"
    code = import_memories.main(
        [
            "builtin",
            "--hermes-home",
            str(home),
            "--store",
            "flatfile",
            "--path",
            str(store),
            "--run",
            "--json",
            "--include-user-profile",
        ]
    )
    assert code == 0
    assert store.exists()
    assert len(get_store("flatfile", store).list()) == 3


def test_missing_source_is_an_error(tmp_path):
    assert import_memories.main(["holographic", "--hermes-home", str(tmp_path)]) == 2


def test_holographic_import(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    conn = sqlite3.connect(home / "memory_store.db")
    conn.execute(
        "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "content TEXT NOT NULL UNIQUE, category TEXT DEFAULT 'general', "
        "tags TEXT DEFAULT '', trust_score REAL DEFAULT 0.5)"
    )
    conn.executemany(
        "INSERT INTO facts (content, category) VALUES (?, ?)",
        [
            ("User prefers dark themes in every editor.", "user_pref"),
            ("The staging cluster lives in eu-west-1.", None),
        ],
    )
    conn.commit()
    conn.close()
    before = (home / "memory_store.db").read_bytes()
    store = tmp_path / "n.db"
    assert (
        import_memories.main(
            ["holographic", "--hermes-home", str(home), "--path", str(store), "--run", "--json"]
        )
        == 0
    )
    entries = _entries(store)
    assert sorted(c for c, _b in entries.values()) == ["general", "user_pref"]
    assert (home / "memory_store.db").read_bytes() == before


def test_byterover_import(tmp_path):
    home = tmp_path / "h"
    tree = home / "byterover" / ".brv" / "context-tree" / "arch"
    tree.mkdir(parents=True)
    (tree / "auth.md").write_text(
        "---\ntitle: Auth design\n---\nTokens are rotated every hour by the gateway.\n"
    )
    (tree / "plain.md").write_text("# Cache policy\nEntries expire after ten minutes.\n")
    store = tmp_path / "n.db"
    assert (
        import_memories.main(
            ["byterover", "--hermes-home", str(home), "--path", str(store), "--run", "--json"]
        )
        == 0
    )
    entries = _entries(store)
    assert set(entries) == {"auth-design", "cache-policy"}
    assert {c for c, _b in entries.values()} == {"byterover"}


def test_help_lists_unsupported_providers(capsys):
    with pytest.raises(SystemExit):
        import_memories.main(["--help"])
    out = capsys.readouterr().out
    for name in ("hindsight", "honcho", "mem0", "openviking", "retaindb", "supermemory"):
        assert name in out


def test_user_profile_skipped_by_default(home, tmp_path, capsys):
    store, out = _run(home, tmp_path, "--run", capsys=capsys, user=False)
    assert len(out["imported"]) == 2
    assert {c for c, _b in _entries(store).values()} == {"memory"}


def test_include_user_profile_imports_user_category(home, tmp_path, capsys):
    store, out = _run(home, tmp_path, "--run", capsys=capsys)
    assert len(out["imported"]) == 3
    assert "user" in {c for c, _b in _entries(store).values()}
