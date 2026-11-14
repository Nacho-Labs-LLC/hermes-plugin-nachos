"""Transcript checkpoint search and age-based retention."""

import json
import time

from nachos_core.history import checkpoint_stats, prune_checkpoints, search_history
from nachos_core.snapshots import SnapshotStore


def _checkpoint(root, session, messages, age_days=0.0, label=None):
    store = SnapshotStore(root, session)
    meta = store.save(messages=messages, reason="pre-compress-checkpoint", label=label)
    if age_days:
        rows = [json.loads(line) for line in store.index_file.read_text().splitlines()]
        for row in rows:
            if row["id"] == meta.id:
                row["created_at"] = time.time() - age_days * 86400
        store.index_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return meta


def test_finds_messages_containing_all_terms_across_sessions(tmp_path):
    _checkpoint(
        tmp_path,
        "s1",
        [
            {"role": "user", "content": "the deploy failed with ECONNRESET on host web-3"},
            {"role": "assistant", "content": "retrying the deploy"},
        ],
    )
    _checkpoint(
        tmp_path,
        "s2",
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": '{"cmd": "kubectl rollout restart web"}',
                        }
                    },
                ],
            },
        ],
    )

    deploy = search_history(tmp_path, "deploy ECONNRESET")
    rollout = search_history(tmp_path, "kubectl rollout")

    assert [r["session_id"] for r in deploy] == ["s1"]
    assert "ECONNRESET" in deploy[0]["excerpt"]
    assert rollout[0]["session_id"] == "s2"
    assert search_history(tmp_path, "deploy ECONNRESET", session_ids=["s2"]) == []


def test_overlapping_checkpoints_return_each_message_once(tmp_path):
    early = [{"role": "user", "content": "remember the staging password rotation is monthly"}]
    _checkpoint(tmp_path, "s1", early, age_days=1)
    _checkpoint(tmp_path, "s1", [*early, {"role": "assistant", "content": "noted"}])

    assert len(search_history(tmp_path, "staging rotation")) == 1


def test_prune_removes_only_old_unlabeled_checkpoints(tmp_path):
    msg = [{"role": "user", "content": "hello there"}]
    _checkpoint(tmp_path, "old", msg, age_days=45)
    _checkpoint(tmp_path, "mixed", msg, age_days=45)
    _checkpoint(tmp_path, "mixed", msg, age_days=45, label="keep-me")
    _checkpoint(tmp_path, "mixed", msg)

    assert prune_checkpoints(tmp_path, 30) == 2
    assert not (tmp_path / "old").exists()
    assert len(SnapshotStore(tmp_path, "mixed").list()) == 2
    assert checkpoint_stats(tmp_path)["count"] == 2
    assert prune_checkpoints(tmp_path, 0) == 0


def test_session_ids_scope_the_search(tmp_path):
    _checkpoint(tmp_path, "a", [{"role": "user", "content": "deploy token rotation notes"}])
    _checkpoint(tmp_path, "b", [{"role": "user", "content": "deploy token rotation again"}])
    _checkpoint(tmp_path, "c", [{"role": "user", "content": "deploy token rotation third"}])

    assert len(search_history(tmp_path, "deploy rotation")) == 3
    scoped = search_history(tmp_path, "deploy rotation", session_ids=["a", "b"])
    assert {r["session_id"] for r in scoped} == {"a", "b"}
    assert search_history(tmp_path, "deploy rotation", session_ids=[]) == []
