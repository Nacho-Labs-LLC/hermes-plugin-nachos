"""Quote grounding at capture, provenance on approval, and surfacing it in reads and searches."""

import json
import sqlite3
import time

import pytest

from nachos_core import capture
from nachos_core.index import MemoryIndex
from nachos_core.snapshots import SnapshotStore
from nachos_hermes import maintenance
from nachos_hermes.memory_provider import NachosMemoryProvider

QUOTE = "Release is cut with make release from the main branch"


def _transcript():
    msgs = []
    for i in range(4):
        msgs.append({"role": "user", "content": f"u{i} " + "a" * 500})
        msgs.append({"role": "assistant", "content": f"r{i} " + "b" * 500})
    msgs[2]["content"] = f"How do we ship? {QUOTE}. Then tag it. " + "a" * 400
    return msgs


def _candidate(title="Release procedure", quote=QUOTE, **extra):
    return {
        "title": title,
        "summary": "release is cut with make release from main",
        "category": "procedure",
        "body": "Run make release on the main branch to cut a release.",
        "reason": "recurring",
        "quote": quote,
        **extra,
    }


@pytest.fixture(autouse=True)
def no_threads(monkeypatch):
    monkeypatch.setattr(NachosMemoryProvider, "_start_capture_thread", lambda _self: None)


def _provider(home, session="sess-1"):
    p = NachosMemoryProvider()
    p.save_config({"embeddings": "off"}, str(home))
    p.initialize(session, hermes_home=str(home), agent_context="primary")
    return p


def _capture(p, *candidates):
    p.on_session_end(_transcript())
    p.process_capture(lambda _messages: json.dumps(list(candidates)))
    return p._service.profile.index


# -- grounding ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("quote", "found"),
    [
        ("make release from the main branch", True),
        ("  MAKE   RELEASE\nfrom the MAIN branch ", True),
        ("Release is cut ... from the main branch", True),  # elision
        ("Release is cut … then tag it", True),  # unicode ellipsis, same message
        ("release is cut with make deploy", False),  # reworded
        ("from the main branch ... Release is cut", False),  # fragments out of order
        ("make", False),  # too short to ground anything
        ("", False),
    ],
)
def test_ground_quote(quote, found):
    messages = capture.trim_messages(_transcript())
    assert (capture.ground_quote(quote, messages) is not None) is found


def test_curly_quotes_and_dashes_are_normalised():
    messages = [{"role": "user", "content": 'it\'s a "quoted" value - really'}]
    curly = (
        "it"
        + chr(0x2019)
        + "s a "
        + chr(0x201C)
        + "quoted"
        + chr(0x201D)
        + " value "
        + chr(0x2014)
        + " really"
    )
    assert capture.ground_quote(curly, messages) == 0


def test_prompt_asks_for_a_verbatim_quote():
    prompt = capture.build_prompt(capture.trim_messages(_transcript()), [])
    assert '"quote"' in prompt[0]["content"]
    assert "VERBATIM" in prompt[0]["content"]


def test_ungrounded_candidates_are_flagged_not_dropped(tmp_path):
    p = _provider(tmp_path)
    index = _capture(
        p,
        _candidate(),
        _candidate("Invented fact", quote="we always ship on fridays at noon sharp"),
        _candidate("No quote at all", quote=""),
    )
    rows = {r["title"]: r for r in index.list_inbox()}

    assert rows["Release procedure"]["grounded"] is True
    assert rows["Invented fact"]["grounded"] is False
    assert rows["No quote at all"]["grounded"] is False
    listing = p.inbox_text("")
    assert "Invented fact [ungrounded]" in listing
    assert "Release procedure [ungrounded]" not in listing
    assert "UNGROUNDED" in p.inbox_text("show 2")
    assert f'evidence: "{QUOTE}"' in p.inbox_text("show 1")


def test_approve_all_skips_ungrounded_but_explicit_id_works(tmp_path):
    p = _provider(tmp_path)
    _capture(p, _candidate(), _candidate("Invented fact", quote="nothing like this was said"))

    out = p.inbox_text("approve all")
    assert "Approved #1" in out
    assert "Skipped #2" in out
    assert p._service.get("invented-fact") is None
    assert "Approved #2" in p.inbox_text("approve 2")


def test_quote_is_redacted_before_it_is_stored(tmp_path):
    p = _provider(tmp_path)
    p._service.redact = lambda text: text.replace("make release", "[R]")
    index = _capture(p, _candidate())
    [row] = index.list_inbox()
    assert row["quote"] == QUOTE.replace("make release", "[R]")
    p.inbox_text("approve 1")
    assert "[R]" in index.get_provenance("release-procedure")["quote"]
    assert "make release" not in index.get_provenance("release-procedure")["quote"]


# -- provenance on approval -----------------------------------------------------


def test_approval_keeps_session_time_quote_and_checkpoint_ref(tmp_path):
    p = _provider(tmp_path)
    meta = SnapshotStore(tmp_path / "nachos" / "snapshots", "sess-1").save(
        messages=_transcript(), reason="pre-compress-checkpoint"
    )
    index = _capture(p, _candidate())
    p.inbox_text("approve 1")

    prov = index.get_provenance("release-procedure")
    assert prov["source"] == "capture"
    assert prov["session_id"] == "sess-1"
    assert prov["quote"] == QUOTE
    assert prov["grounded"] is True
    assert prov["checkpoint_id"] == meta.id
    assert prov["message_index"] == 2
    assert abs(prov["captured_at"] - time.time()) < 60


def test_ungrounded_approval_drops_the_unverified_quote(tmp_path):
    p = _provider(tmp_path)
    index = _capture(p, _candidate(quote="never said this anywhere in session"))
    p.inbox_text("approve 1")
    prov = index.get_provenance("release-procedure")
    assert prov["quote"] == ""
    assert prov["grounded"] is False


def test_manual_save_records_session_and_time_only(tmp_path):
    p = _provider(tmp_path, "sess-manual")
    result = json.loads(
        p.handle_tool_call(
            "nachos_memory_put",
            {"title": "Manual fact", "body": "The cache TTL is sixty seconds in prod."},
        )
    )
    assert result["stored"] == "manual-fact"
    prov = p._service.profile.index.get_provenance("manual-fact")
    assert prov["source"] == "manual"
    assert prov["session_id"] == "sess-manual"
    assert prov["quote"] == ""
    assert prov["checkpoint_id"] is None


def test_resaving_identical_content_keeps_the_capture_record(tmp_path):
    p = _provider(tmp_path)
    index = _capture(p, _candidate())
    p.inbox_text("approve 1")
    entry = json.loads(p.handle_tool_call("nachos_memory_recall", {"key": "release-procedure"}))
    p._service.put(
        title="Release procedure",
        body=entry["body"],
        summary=_candidate()["summary"],
        category="procedure",
    )
    assert index.get_provenance("release-procedure")["source"] == "capture"
    p._service.put(title="Release procedure", body="Changed: run make ship instead of release.")
    assert index.get_provenance("release-procedure")["source"] == "manual"


# -- surfacing ----------------------------------------------------------------


def test_read_and_recall_show_a_compact_source_and_it_opens_the_history(tmp_path):
    p = _provider(tmp_path)
    SnapshotStore(tmp_path / "nachos" / "snapshots", "sess-1").save(
        messages=_transcript(), reason="pre-compress-checkpoint"
    )
    _capture(p, _candidate())
    p.inbox_text("approve 1")

    read = json.loads(p.handle_tool_call("nachos_memory_recall", {"key": "release-procedure"}))
    source = read["source"]
    assert source["session"] == "sess-1"
    assert source["date"] == time.strftime("%Y-%m-%d", time.gmtime())
    assert source["evidence"] == QUOTE
    assert source["via"] == "capture"

    found = json.loads(p.handle_tool_call("nachos_memory_recall", {"query": "make release main"}))
    hit = found["results"][0]["source"]
    assert hit["ref"] == source["ref"]
    assert "evidence" not in hit  # search results stay compact

    opened = json.loads(p.handle_tool_call("nachos_history_search", {"ref": source["ref"]}))
    [result] = opened["results"]
    assert result["message_index"] == 2
    assert QUOTE in " ".join(part["text"] for part in result["context"])
    assert [part["index"] for part in result["context"]] == [1, 2, 3]


def test_bare_session_ref_scopes_a_history_query(tmp_path):
    p = _provider(tmp_path)
    for session, word in (("sess-1", "alpacas"), ("sess-2", "alpacas")):
        SnapshotStore(tmp_path / "nachos" / "snapshots", session).save(
            messages=[{"role": "user", "content": f"{session} keeps {word}"}],
            reason="pre-compress-checkpoint",
        )
    scoped = json.loads(
        p.handle_tool_call("nachos_history_search", {"ref": "sess-2", "query": "alpacas"})
    )
    assert [r["session_id"] for r in scoped["results"]] == ["sess-2"]
    assert "error" in json.loads(p.handle_tool_call("nachos_history_search", {"ref": "sess-2"}))
    assert "error" in json.loads(p.handle_tool_call("nachos_history_search", {}))
    gone = json.loads(p.handle_tool_call("nachos_history_search", {"ref": "sess-2@missing#1"}))
    assert "gone" in gone["error"]


def test_history_context_is_redacted_and_uses_config(tmp_path):
    p = _provider(tmp_path)
    p.save_config(
        {"embeddings": "off", "history_context_before": 0, "history_context_after": 0},
        str(tmp_path),
    )
    p.initialize("sess-1", hermes_home=str(tmp_path), agent_context="primary")
    p._redact = lambda text: text.replace("hunter2", "[X]")
    SnapshotStore(tmp_path / "nachos" / "snapshots", "sess-1").save(
        messages=[
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "the password hunter2 is rotated"},
            {"role": "user", "content": "after"},
        ],
        reason="pre-compress-checkpoint",
    )
    out = json.loads(p.handle_tool_call("nachos_history_search", {"query": "password rotated"}))
    [part] = out["results"][0]["context"]
    assert "[X]" in part["text"]
    assert "hunter2" not in json.dumps(out)


def test_removing_an_entry_removes_its_provenance(tmp_path):
    p = _provider(tmp_path)
    index = _capture(p, _candidate())
    p.inbox_text("approve 1")
    p.handle_tool_call("nachos_memory_remove", {"key": "release-procedure"})
    assert index.get_provenance("release-procedure") is None


# -- migration safety -----------------------------------------------------------


def test_old_index_files_gain_the_new_columns_and_table(tmp_path):
    path = tmp_path / "old.index.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts REAL NOT NULL, session_id TEXT,
            title TEXT NOT NULL, summary TEXT NOT NULL, category TEXT NOT NULL,
            body TEXT NOT NULL, reason TEXT, duplicates_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'pending');
        INSERT INTO inbox (created_ts, session_id, title, summary, category, body)
            VALUES (1.0, 's', 'Old', 'sum', 'c', 'body');
        """
    )
    conn.commit()
    conn.close()

    # Read-only access to the old file works and reports "unknown", never an error.
    ro = MemoryIndex(path, read_only=True)
    [row] = ro.list_inbox()
    assert row["title"] == "Old"
    assert row["grounded"] is None
    assert row["quote"] is None
    assert ro.get_provenance_many(["k"]) == {}
    ro.close()

    index = MemoryIndex(path)
    [row] = index.list_inbox()
    assert row["grounded"] is None
    index.add_inbox(
        session_id="s", title="New", summary="s", category="c", body="b", quote="q", grounded=True
    )
    assert index.list_inbox()[1]["grounded"] is True
    index.set_provenance("k", session_id="s", source="manual")
    assert index.get_provenance("k")["session_id"] == "s"
    index.close()


def test_entries_without_provenance_have_no_source(tmp_path):
    p = _provider(tmp_path)
    p._service.profile.store.put(
        "legacy", title="Legacy", summary="s", category="c", body="Old entry body text."
    )
    read = json.loads(p.handle_tool_call("nachos_memory_recall", {"key": "legacy"}))
    assert "source" not in read


# -- maintenance ------------------------------------------------------------------


def test_maintain_reports_entries_without_provenance_as_info(tmp_path):
    p = _provider(tmp_path)
    p._service.profile.store.put(
        "legacy", title="Legacy", summary="s", category="c", body="Old entry body text."
    )
    p.handle_tool_call(
        "nachos_memory_put", {"title": "Fresh", "body": "A freshly saved fact about caching."}
    )
    p.shutdown()

    report = maintenance._report(maintenance_args(tmp_path))
    result = report["checks"]["provenance"]
    assert result == {
        "count": 1,
        "items": [{"key": "legacy"}],
        "severity": "info",
        "available": True,
    }
    assert "provenance: 1 (info)" in maintenance._human_report(report)


def maintenance_args(home):
    class Args:
        days = 30.0
        profile_home = home
        store = "sqlite"
        check = "provenance"

    return Args
