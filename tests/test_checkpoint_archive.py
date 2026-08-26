"""Fail-closed transcript archival tests for the Nachos memory provider."""

from pathlib import Path

from nachos_core.snapshots import SnapshotStore
from nachos_hermes.memory_provider import NachosMemoryProvider


def test_pre_compress_checkpoint_archives_once_for_identical_transcript(tmp_path: Path):
    provider = NachosMemoryProvider()
    provider._session_id = "session-1"
    provider._checkpoint_root = tmp_path
    messages = [
        {"role": "user", "content": "Keep this evidence."},
        {"role": "assistant", "content": "Archived."},
    ]

    first = provider.on_pre_compress(messages)
    second = provider.on_pre_compress(messages)

    snapshots = SnapshotStore(tmp_path, "session-1").list()
    assert first.startswith("checkpoint:")
    assert second == first
    assert len(snapshots) == 1
    assert snapshots[0]["reason"] == "pre-compress-checkpoint"


def test_pre_compress_checkpoint_contract_is_fail_closed_v2():
    assert NachosMemoryProvider.pre_compress_checkpoint_api_version == 2
