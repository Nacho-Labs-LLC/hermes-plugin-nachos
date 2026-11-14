"""tools/memory_audit.py — duplicates, unused, purge-category (in-process)."""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import memory_audit

from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.store import get_store


def _seed(kind: str, path: Path) -> None:
    store = get_store(kind, path)
    rows = [
        (
            "editor-prefs",
            "Editor prefs",
            "user prefers vim keybindings in editor",
            "user",
            "Uses vim.",
        ),
        (
            "editor-prefs-two",
            "Editor prefs two",
            "user prefers vim keybindings in editor",
            "user",
            "Vim again.",
        ),
        (
            "dropped-one",
            "Dropped one",
            "archived tool output about deploy",
            "scratch",
            "old output",
        ),
        (
            "dropped-two",
            "Dropped two",
            "archived tool output about build",
            "scratch",
            "old output 2",
        ),
        (
            "deploy-notes",
            "Deploy notes",
            "blue green deploy procedure for prod",
            "ops",
            "Steps for deploy.",
        ),
    ]
    for key, title, summary, category, body in rows:
        store.put(key, title=title, summary=summary, category=category, body=body)
    getattr(store, "close", lambda: None)()


@pytest.fixture(
    params=[("sqlite", "memory.db"), ("flatfile", "memory.md")], ids=["sqlite", "flatfile"]
)
def store(request, tmp_path):
    kind, name = request.param
    path = tmp_path / name
    _seed(kind, path)
    return kind, path


def _run(capsys, kind, path, *argv):
    code = memory_audit.main(["--store", kind, "--path", str(path), "--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_duplicates_lists_pair(store, capsys):
    kind, path = store
    code, out = _run(capsys, kind, path, "duplicates")
    assert code == 0
    keys = {frozenset((p["a"]["key"], p["b"]["key"])) for p in out["duplicates"]}
    assert frozenset(("editor-prefs", "editor-prefs-two")) in keys


def test_unused_respects_age(store, capsys):
    kind, path = store
    idx = MemoryIndex(index_path_for(path))
    idx.record_seen(["editor-prefs", "deploy-notes"])
    idx.record_recall("deploy-notes")
    idx.close()
    # make usage rows 40 days old
    conn = sqlite3.connect(str(index_path_for(path)))
    conn.execute("UPDATE usage SET first_seen = ?", (time.time() - 40 * 86400,))
    conn.commit()
    conn.close()

    _, out = _run(capsys, kind, path, "unused", "--days", "30")
    assert [r["key"] for r in out["unused"]] == ["editor-prefs"]
    _, out = _run(capsys, kind, path, "unused", "--days", "60")
    assert out["unused"] == []


def test_purge_category_dry_run_then_run(store, capsys):
    kind, path = store
    code, out = _run(capsys, kind, path, "purge-category", "scratch")
    assert code == 0
    assert out["dry_run"] is True
    assert out["removed"] == 0
    assert len(out["matched"]) == 2
    assert len(get_store(kind, path).list()) == 5

    _, out = _run(capsys, kind, path, "purge-category", "scratch", "--run")
    assert out["removed"] == 2
    remaining = {k for k, *_ in get_store(kind, path).list()}
    assert remaining == {"editor-prefs", "editor-prefs-two", "deploy-notes"}


def test_missing_store_errors(tmp_path, capsys):
    code = memory_audit.main(["--path", str(tmp_path / "nope.db"), "duplicates"])
    assert code == 2
    assert "no store" in capsys.readouterr().err


def test_human_output(store, capsys):
    kind, path = store
    code = memory_audit.main(["--store", kind, "--path", str(path), "purge-category", "ops"])
    assert code == 0
    assert "Would remove 1 entries" in capsys.readouterr().out


def test_history_and_revert_dry_run_then_run(store, capsys):
    kind, path = store
    _run(capsys, kind, path, "purge-category", "scratch", "--run")

    _, out = _run(capsys, kind, path, "history")
    assert [c["action"] for c in out["changes"]] == ["remove", "remove"]
    _, out = _run(capsys, kind, path, "history", "--key", "dropped-one")
    assert len(out["changes"]) == 1
    change_id = out["changes"][0]["id"]

    code, out = _run(capsys, kind, path, "revert", str(change_id))
    assert code == 0
    assert out["dry_run"] is True
    assert get_store(kind, path).get("dropped-one") is None

    _, out = _run(capsys, kind, path, "revert", str(change_id), "--run")
    assert out["result"] == "restored"
    assert get_store(kind, path).get("dropped-one").endswith("old output")


def test_revert_unknown_id_errors(store):
    kind, path = store
    code = memory_audit.main(["--store", kind, "--path", str(path), "revert", "99"])
    assert code == 2
