"""Write-time summary/body quality gate — nachos_core.quality + provider wiring."""

from pathlib import Path

import pytest

from nachos_core.quality import (
    is_redundant_summary,
    low_signal_reason,
    shares_vocabulary,
)
from nachos_hermes.memory_provider import NachosMemoryProvider


class TestLowSignalReason:
    @pytest.mark.parametrize(
        "text", ["", "   ", "nothing", "N/A", "TODO", "placeholder", "-", "x", "a b"]
    )
    def test_flags_low_signal(self, text):
        assert low_signal_reason(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Branches off 3plp not master, uses SSH remote.",
            "FeatureFlagService wraps ConfigService for per-BU rollout toggles.",
        ],
    )
    def test_accepts_real_content(self, text):
        assert low_signal_reason(text) is None

    def test_repeated_single_word_flagged(self):
        assert low_signal_reason("blah blah blah blah") is not None


class TestRedundantSummary:
    def test_summary_equal_to_title_is_redundant(self):
        assert is_redundant_summary("Git Workflow", "git workflow.")

    def test_different_summary_is_fine(self):
        assert not is_redundant_summary("worktree + SSH remote", "Git workflow")


class TestSharesVocabulary:
    def test_overlapping_words_pass(self):
        assert shares_vocabulary("worktree branches off 3plp", "Branches off 3plp, not master.")

    def test_disjoint_words_fail(self):
        assert not shares_vocabulary(
            "news digest TLDR format", "FeatureFlagService wraps ConfigService."
        )


class TestProviderWriteGate:
    def _provider(self, tmp_path: Path) -> NachosMemoryProvider:
        p = NachosMemoryProvider()
        p.initialize("session-1", hermes_home=str(tmp_path))
        return p

    def test_put_rejects_low_signal_summary(self, tmp_path):
        provider = self._provider(tmp_path)
        out = provider.handle_tool_call(
            "nachos_memory_put",
            {
                "title": "Some real title here",
                "summary": "nothing",
                "body": "A real body with enough real content to pass the gate.",
            },
        )
        assert "error" in out
        assert "low-signal" in out

    def test_put_rejects_low_signal_body(self, tmp_path):
        provider = self._provider(tmp_path)
        out = provider.handle_tool_call(
            "nachos_memory_put",
            {
                "title": "Some real title here",
                "body": "todo",
            },
        )
        assert "error" in out
        assert "low-signal" in out

    def test_put_accepts_real_content(self, tmp_path):
        provider = self._provider(tmp_path)
        out = provider.handle_tool_call(
            "nachos_memory_put",
            {
                "title": "Deposco git workflow",
                "body": "Branches off 3plp not master. Remote is SSH, push "
                "username is naterichardson not email.",
            },
        )
        assert "stored" in out

    def test_put_rejects_low_signal_derived_summary(self, tmp_path):
        """When summary is omitted, it derives from body's first line — if
        THAT line is low-signal, the derived summary must still be caught."""
        provider = self._provider(tmp_path)
        out = provider.handle_tool_call(
            "nachos_memory_put",
            {
                "title": "Some real title here",
                "body": "nothing\n\nActually there's real content down here "
                "but the first line, which becomes the summary, isn't.",
            },
        )
        assert "error" in out
        assert "low-signal" in out
