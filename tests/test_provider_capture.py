"""Provider side of session capture: hook, inbox command, prompt line, config."""

import gzip
import json

import pytest

from nachos_core import capture
from nachos_hermes.memory_provider import NachosMemoryProvider


@pytest.fixture(autouse=True)
def no_threads(monkeypatch):
    """Keep tests deterministic: record thread starts instead of running them."""
    starts = []
    monkeypatch.setattr(
        NachosMemoryProvider, "_start_capture_thread", lambda _self: starts.append(1)
    )
    return starts


def _provider(home, primary=True, **cfg):
    cfg.setdefault("embeddings", "off")
    p = NachosMemoryProvider()
    p.save_config(cfg, str(home))
    p.initialize(
        "sess-1", hermes_home=str(home), agent_context="primary" if primary else "subagent"
    )
    return p


def _msgs():
    return [
        m
        for i in range(4)
        for m in (
            {"role": "user", "content": f"u{i} " + "a" * 500},
            {"role": "assistant", "content": f"r{i} " + "b" * 500},
        )
    ]


CAND = [
    {
        "title": "Deploy procedure",
        "summary": "Deploy with make release then tag",
        "category": "procedure",
        "body": "Run make release, then git tag.",
        "reason": "recurring",
    }
]


def _job(home):
    return home / "nachos" / "inbox" / "pending" / "sess-1.json.gz"


def _fill(p):
    p.on_session_end(_msgs())
    p.process_capture(lambda _m: json.dumps(CAND))


def test_session_end_writes_job_and_skips_trivial(tmp_path):
    p = _provider(tmp_path)
    p.on_session_end(_msgs()[:2])
    assert not _job(tmp_path).exists()
    p.on_session_end(_msgs())
    assert _job(tmp_path).exists()


def test_non_primary_and_capture_off_write_nothing(tmp_path):
    _provider(tmp_path / "a", primary=False).on_session_end(_msgs())
    _provider(tmp_path / "b", capture="off").on_session_end(_msgs())
    assert not _job(tmp_path / "a").exists()
    assert not _job(tmp_path / "b").exists()


def test_session_end_never_raises(tmp_path):
    p = _provider(tmp_path)
    p.on_session_end(None)
    p.on_session_end([object()])


def test_processing_creates_rows_and_prompt_line(tmp_path):
    p = _provider(tmp_path)
    assert "await the user's review" not in p.system_prompt_block()
    _fill(p)
    assert not _job(tmp_path).exists()
    assert (
        "1 suggested memories await the user's review (/nachos-memory-inbox)."
        in p.system_prompt_block()
    )
    assert "1 pending" in p.status_text()
    listing = p.inbox_text("")
    assert "#1 Deploy procedure" in listing
    assert "make release" in p.inbox_text("show 1")


def test_approve_stores_entry_and_logs_change(tmp_path):
    p = _provider(tmp_path)
    _fill(p)
    out = p.inbox_text("approve 1")
    assert "Approved #1" in out
    assert p._service.get("deploy-procedure")[1].startswith("Run make release")
    assert p._service.history()[0]["key"] == "deploy-procedure"
    assert "await the user's review" not in p.system_prompt_block()


def test_reject_and_approve_all(tmp_path):
    p = _provider(tmp_path)
    _fill(p)
    assert "Rejected #1" in p.inbox_text("reject all")
    assert p._service.get("deploy-procedure") is None
    assert "No suggested" in p.inbox_text("")


def test_approve_reports_put_errors(tmp_path):
    p = _provider(tmp_path)
    _fill(p)
    p._service.profile.index._conn.execute("UPDATE inbox SET body = 'todo'")
    out = p.inbox_text("approve 1")
    assert "Could not approve #1" in out
    assert p._service.profile.index.count_inbox("pending") == 1


def test_capture_off_does_not_process(tmp_path):
    p = _provider(tmp_path, capture="off")
    assert p.process_capture(lambda _m: "[]") == {"jobs": 0, "candidates": 0, "retry": 0}


def test_missing_aux_client_leaves_job_pending(tmp_path):
    p = _provider(tmp_path)
    p.on_session_end(_msgs())
    p.process_capture()  # no agent.auxiliary_client in the test env
    assert capture.has_pending(tmp_path / "nachos" / "inbox")


def test_session_end_starts_the_worker(tmp_path, no_threads):
    p = _provider(tmp_path)
    no_threads.clear()
    p.on_session_end(_msgs()[:2])
    assert no_threads == []
    p.on_session_end(_msgs())
    assert no_threads == [1]


def test_list_never_calls_llm_and_reports_background_work(tmp_path, no_threads, monkeypatch):
    p = _provider(tmp_path)
    p.on_session_end(_msgs())
    no_threads.clear()

    def boom(_m):
        raise AssertionError("llm must not run from the slash command")

    monkeypatch.setattr(NachosMemoryProvider, "_capture_llm", staticmethod(boom))
    out = p.inbox_text("")
    assert "Processing 1 captured session(s) in the background" in out
    assert no_threads == [1]
    assert _job(tmp_path).exists()
    p.inbox_text("show 1")
    p.inbox_text("approve all")
    assert _job(tmp_path).exists()


def test_outage_keeps_job_and_drops_after_five_failures(tmp_path):
    p = _provider(tmp_path)
    p.on_session_end(_msgs())

    def down(_m):
        raise TimeoutError("endpoint down")

    for i in range(1, 5):
        p.process_capture(down)
        assert _job(tmp_path).exists()
        with gzip.open(_job(tmp_path), "rt") as fh:
            assert json.load(fh)["failures"] == i
    p.process_capture(down)
    assert not _job(tmp_path).exists()


def test_success_resets_failure_count(tmp_path):
    p = _provider(tmp_path)
    p.on_session_end(_msgs())

    def down(_m):
        raise RuntimeError("x")

    p.process_capture(down)
    p.process_capture(lambda _m: "garbage")  # malformed: separate retry rule
    with gzip.open(_job(tmp_path), "rt") as fh:
        data = json.load(fh)
    assert data["failures"] == 0
    assert data["attempts"] == 1
