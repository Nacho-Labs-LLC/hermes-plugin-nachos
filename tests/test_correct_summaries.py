"""tools/correct_summaries.py — flag_reason + no-LLM-available CLI path."""

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import correct_summaries as cs

from nachos_core.store import SqliteStore


class TestFlagReason:
    def test_low_signal_summary_flagged(self):
        assert cs.flag_reason("Title", "nothing", "real body content here") is not None

    def test_redundant_summary_flagged(self):
        assert (
            cs.flag_reason("Git workflow", "git workflow", "Branches off 3plp not master.")
            is not None
        )

    def test_stale_summary_flagged(self):
        assert (
            cs.flag_reason(
                "Title",
                "news digest TLDR format",
                "FeatureFlagService wraps ConfigService for per-BU rollout.",
            )
            is not None
        )

    def test_good_summary_not_flagged(self):
        assert (
            cs.flag_reason(
                "Deposco git workflow",
                "worktree plus SSH remote, branches off 3plp",
                "Branches off 3plp not master. Remote is SSH.",
            )
            is None
        )


class TestMainNoLlmPath:
    def test_reports_flagged_entries_without_crashing_when_no_client(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        db = tmp_path / "mem.db"
        store = SqliteStore(str(db))
        store.put(
            "good",
            title="Good entry",
            summary="keyword dense accurate summary of body content",
            category="c",
            body="keyword dense accurate summary of body content, more.",
        )
        store.put(
            "bad",
            title="Bad entry",
            summary="nothing",
            category="c",
            body="real body content that is fine.",
        )
        store.close()

        monkeypatch.setattr(cs, "_resolve_client", lambda: (None, None))
        monkeypatch.setattr(
            sys,
            "argv",
            ["correct_summaries.py", "--store", "sqlite", "--path", str(db)],
        )
        rc = cs.main()
        out = capsys.readouterr().out
        assert rc == 2
        assert "[bad]" in out
        assert "[good]" not in out

    def test_nothing_flagged_returns_zero(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "mem.db"
        store = SqliteStore(str(db))
        store.put(
            "good",
            title="Good entry",
            summary="keyword dense accurate summary of body content",
            category="c",
            body="keyword dense accurate summary of body content, more.",
        )
        store.close()

        monkeypatch.setattr(
            sys,
            "argv",
            ["correct_summaries.py", "--store", "sqlite", "--path", str(db)],
        )
        rc = cs.main()
        out = capsys.readouterr().out
        assert rc == 0
        assert "Nothing flagged" in out
