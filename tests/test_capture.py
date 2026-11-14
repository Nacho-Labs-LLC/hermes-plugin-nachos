"""Session capture: job queue, extraction with a fake LLM, quality gate, dedup."""

import gzip
import json
from collections import OrderedDict

from nachos_core import capture
from nachos_core.index import MemoryIndex
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.store import get_store
from nachos_hermes import capture_adapter


def _service(tmp_path):
    scopes = OrderedDict()
    scopes[PROFILE] = Scope(
        PROFILE,
        get_store("sqlite", tmp_path / "memory.db"),
        MemoryIndex(tmp_path / "memory.index.db"),
    )
    return MemoryService(scopes)


def _messages(n=4, size=500):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"question {i} " + "x" * size})
        out.append({"role": "assistant", "content": f"answer {i} " + "y" * size})
        out.append({"role": "tool", "content": "SECRET TOOL OUTPUT"})
    return out


GOOD = {
    "title": "Preferred editor",
    "summary": "User prefers Neovim everywhere",
    "category": "preference",
    "body": "The user edits all code in Neovim.",
    "reason": "stated preference",
}


def _run(tmp_path, answers):
    svc = _service(tmp_path)
    inbox = tmp_path / "inbox"
    capture.write_job(inbox, "s1", _messages())
    calls = []
    it = iter(answers)

    def llm(messages):
        calls.append(messages)
        return next(it)

    return svc, inbox, llm, calls


def test_trim_drops_tool_text_truncates_and_keeps_recent():
    msgs = _messages(n=100, size=1900)
    trimmed = capture.trim_messages(msgs)
    assert all(m["role"] in ("user", "assistant") for m in trimmed)
    assert all(len(m["content"]) <= capture.MAX_MESSAGE_CHARS for m in trimmed)
    assert sum(len(m["content"]) for m in trimmed) <= capture.MAX_TOTAL_CHARS
    assert trimmed[-1]["content"].startswith("answer 99")


def test_trivial_sessions_write_no_job(tmp_path):
    assert capture.write_job(tmp_path, "s", _messages(n=2)) is None
    assert capture.write_job(tmp_path, "s", _messages(n=3, size=10)) is None
    path = capture.write_job(tmp_path, "s", _messages())
    with gzip.open(path, "rt") as fh:
        data = json.load(fh)
    assert data["session_id"] == "s"
    assert data["messages"]


def test_good_json_creates_inbox_row_and_deletes_job(tmp_path):
    svc, inbox, llm, calls = _run(tmp_path, [json.dumps([GOOD])])
    stats = capture.process_pending(inbox, svc, llm)
    rows = svc.profile.index.list_inbox()
    assert stats["candidates"] == 1
    assert rows[0]["title"] == "Preferred editor"
    assert not capture.has_pending(inbox)
    assert not list((inbox / "processing").glob("*"))
    assert "Neovim" not in calls[0][1]["content"]  # not yet known
    assert "SECRET TOOL OUTPUT" not in calls[0][1]["content"]


def test_fenced_json_is_parsed(tmp_path):
    svc, inbox, llm, _ = _run(tmp_path, ["```json\n" + json.dumps([GOOD]) + "\n```"])
    capture.process_pending(inbox, svc, llm)
    assert svc.profile.index.count_inbox() == 1


def test_malformed_retries_once_then_drops(tmp_path):
    svc, inbox, llm, calls = _run(tmp_path, ["not json", "still {not json"])
    capture.process_pending(inbox, svc, llm)
    assert capture.has_pending(inbox)
    assert len(calls) == 1
    capture.process_pending(inbox, svc, llm)
    assert not capture.has_pending(inbox)
    assert len(calls) == 2
    assert svc.profile.index.count_inbox() == 0


def test_empty_array_is_fine(tmp_path):
    svc, inbox, llm, _ = _run(tmp_path, ["[]"])
    capture.process_pending(inbox, svc, llm)
    assert svc.profile.index.count_inbox() == 0
    assert not capture.has_pending(inbox)


def test_quality_gate_rejects_low_signal(tmp_path):
    bad = dict(GOOD, title="Placeholder", summary="todo", body="nothing")
    svc, inbox, llm, _ = _run(tmp_path, [json.dumps([bad, GOOD])])
    capture.process_pending(inbox, svc, llm)
    assert [r["title"] for r in svc.profile.index.list_inbox()] == ["Preferred editor"]


def test_duplicates_and_updates_are_annotated(tmp_path):
    svc, inbox, llm, _ = _run(tmp_path, [json.dumps([GOOD])])
    svc.put(title="Preferred editor", body="Uses vim.", summary="Vim is the editor")
    svc.put(title="Editor choice", body="Neovim all code", summary="User prefers Neovim everywhere")
    capture.process_pending(inbox, svc, llm)
    dups = svc.profile.index.list_inbox()[0]["duplicates"]
    assert dups[0]["match"] == "update"
    assert dups[0]["key"] == "preferred-editor"
    assert any(d["key"] == "editor-choice" for d in dups)


def test_unavailable_llm_leaves_job_pending(tmp_path):
    svc, inbox, _, _ = _run(tmp_path, [])

    def llm(_m):
        raise capture.CaptureUnavailable("no client")

    capture.process_pending(inbox, svc, llm)
    capture.process_pending(inbox, svc, llm)
    assert capture.has_pending(inbox)


def test_claimed_job_is_not_processed_twice(tmp_path):
    svc, inbox, llm, calls = _run(tmp_path, ["[]"])
    job = capture.pending_jobs(inbox)[0]
    assert capture._claim(job, inbox) is not None
    assert capture._claim(job, inbox) is None
    capture.process_pending(inbox, svc, llm)
    assert calls == []


def test_user_profile_entries_are_shown_as_known_and_not_to_be_proposed(tmp_path):
    svc, inbox, llm, calls = _run(tmp_path, ["[]"])
    capture.process_pending(inbox, svc, llm, user_facts=["Prefers Neovim", "Lives in Oslo"])
    system, user = calls[0][0]["content"], calls[0][1]["content"]
    assert "Already known about the user" in user
    assert "- Prefers Neovim" in user
    assert "- Lives in Oslo" in user
    assert "Do NOT propose facts about the user's preferences" in system
    assert "persona or identity" in system


def test_prompt_without_user_profile_has_no_known_section(tmp_path):
    svc, inbox, llm, calls = _run(tmp_path, ["[]"])
    capture.process_pending(inbox, svc, llm)
    assert "Already known about the user" not in calls[0][1]["content"]
    assert "Do NOT propose facts about the user's preferences" in calls[0][0]["content"]


def test_adapter_reads_user_md_entries(tmp_path):
    assert capture_adapter.read_user_profile(tmp_path) == []
    assert capture_adapter.read_user_profile(None) == []
    mem = tmp_path / "memories"
    mem.mkdir()
    (mem / "USER.md").write_text("Prefers Neovim\n§\nLives in Oslo\n", encoding="utf-8")
    assert capture_adapter.read_user_profile(tmp_path) == ["Prefers Neovim", "Lives in Oslo"]
