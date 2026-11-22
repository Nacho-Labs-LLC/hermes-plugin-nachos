"""History search: IDF ranking, context windows, merged windows, char cap and stable refs."""

import json
import time

from nachos_core.history import (
    locate_quote,
    make_ref,
    open_ref,
    parse_ref,
    search_history,
)
from nachos_core.snapshots import SnapshotStore


def _save(root, session, messages, age_days=0.0):
    store = SnapshotStore(root, session)
    meta = store.save(messages=messages, reason="pre-compress-checkpoint")
    if age_days:
        rows = [json.loads(line) for line in store.index_file.read_text().splitlines()]
        for row in rows:
            row["created_at"] = time.time() - age_days * 86400
        store.index_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return meta


def _chat(*texts):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)
    ]


def test_hit_returns_adjacent_turns_labelled_with_role_and_position(tmp_path):
    _save(
        tmp_path,
        "s",
        _chat("setup question", "the zebra config lives in /etc/zebra.conf", "thanks"),
    )
    [hit] = search_history(tmp_path, "zebra config")

    assert hit["message_index"] == 1
    parts = hit["context"]
    assert [(p["index"], p["role"], p["match"]) for p in parts] == [
        (0, "user", False),
        (1, "assistant", True),
        (2, "user", False),
    ]
    assert "/etc/zebra.conf" in parts[1]["text"]


def test_context_size_is_configurable(tmp_path):
    _save(tmp_path, "s", _chat("a0", "a1", "a2", "needle here", "a4", "a5", "a6"))
    wide = search_history(tmp_path, "needle", before=3, after=2)[0]["context"]
    none = search_history(tmp_path, "needle", before=0, after=0)[0]["context"]

    assert [p["index"] for p in wide] == [0, 1, 2, 3, 4, 5]
    assert [p["index"] for p in none] == [3]


def test_tool_call_and_result_stay_together(tmp_path):
    msgs = [
        {"role": "user", "content": "check disk usage"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "function": {"name": "terminal", "arguments": '{"cmd": "df -h"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "/dev/sda1 91% used"},
        {"role": "assistant", "content": "The volume is nearly full."},
        {"role": "user", "content": "ok"},
    ]
    _save(tmp_path, "s", msgs)

    answer = search_history(tmp_path, "volume nearly full", before=0, after=0)[0]
    assert [p["index"] for p in answer["context"]] == [
        3
    ]  # plain window only: nothing depends on a pair

    result = search_history(tmp_path, "volume nearly full", before=1, after=0)[0]
    assert [p["index"] for p in result["context"]] == [1, 2, 3]  # call pulled in with its result

    call = search_history(tmp_path, "df -h terminal", before=0, after=0)[0]
    assert [p["index"] for p in call["context"]] == [1, 2]  # result pulled in with its call


def test_overlapping_windows_merge_into_one_result(tmp_path):
    _save(
        tmp_path,
        "s",
        _chat(
            "x",
            "alpha beta first",
            "mid",
            "alpha beta second",
            "y",
            "far",
            "far2",
            "far3",
            "alpha beta third",
        ),
    )
    results = search_history(tmp_path, "alpha beta")

    assert len(results) == 2
    merged = next(r for r in results if len(r["matched_indexes"]) == 2)
    assert merged["matched_indexes"] == [1, 3]
    assert [p["index"] for p in merged["context"]] == [0, 1, 2, 3, 4]


def test_total_context_chars_are_capped(tmp_path):
    big = "word " * 400
    _save(tmp_path, "s", _chat(big, "target " + big, big))
    _save(tmp_path, "t", _chat(big, "target " + big, big))
    results = search_history(tmp_path, "target", max_chars=1500)

    total = sum(len(p["text"]) for r in results for p in r["context"])
    assert total <= 1500 + 50
    assert results  # the best hit is always returned, clipped


def test_partial_queries_match_and_rare_words_outrank_common_ones(tmp_path):
    _save(
        tmp_path,
        "s",
        _chat(
            "what port is the staging database on",
            "staging staging staging deploy deploy notes",
            "postgres listens on port 6543 in staging",
        ),
    )
    hits = search_history(tmp_path, "which port does postgres use in staging")
    assert hits[0]["message_index"] == 2  # "postgres" is rare: it decides the order


def test_one_common_word_cannot_satisfy_a_multi_word_query(tmp_path):
    msgs = [f"routine deploy note {i}" for i in range(12)] + ["unrelated words entirely"]
    _save(tmp_path, "s", _chat(*msgs))

    assert search_history(tmp_path, "deploy kubernetes ingress certificate") == []
    assert search_history(tmp_path, "the") == []  # only stopwords
    assert search_history(tmp_path, "deploy")  # a single real word still works


def test_ties_go_to_the_newer_checkpoint(tmp_path):
    _save(tmp_path, "old", _chat("rotate the signing keys quarterly"), age_days=10)
    _save(tmp_path, "new", _chat("rotate the signing keys yearly"))
    assert [r["session_id"] for r in search_history(tmp_path, "rotate signing keys")] == [
        "new",
        "old",
    ]


def test_refs_are_stable_and_round_trip(tmp_path):
    meta = _save(tmp_path, "sess:1", _chat("zero", "find the quokka", "two"))
    [hit] = search_history(tmp_path, "quokka")

    assert hit["ref"] == make_ref("sess:1", meta.id, 1)
    assert parse_ref(hit["ref"]) == ("sess:1", meta.id, 1)
    assert parse_ref("sess:1") == ("sess:1", None, None)
    assert parse_ref(f"s@{meta.id}") == ("s", meta.id, None)
    assert hit["session_id"] == "sess:1"
    assert hit["checkpoint_id"] == meta.id
    assert search_history(tmp_path, "quokka")[0]["ref"] == hit["ref"]


def test_open_ref_returns_context_and_none_for_gone_checkpoints(tmp_path):
    _save(tmp_path, "s", _chat("a", "b", "c", "d", "e"))
    ref = make_ref("s", SnapshotStore(tmp_path, "s").list()[0]["id"], 2)

    opened = open_ref(tmp_path, ref, before=1, after=1)
    assert opened is not None
    assert [p["index"] for p in opened["context"]] == [1, 2, 3]
    assert opened["ref"] == ref
    assert open_ref(tmp_path, make_ref("s", "nope", 2)) is None
    assert (
        open_ref(tmp_path, make_ref("s", SnapshotStore(tmp_path, "s").list()[0]["id"], 99)) is None
    )
    assert open_ref(tmp_path, "s") is None


def test_locate_quote_finds_checkpoint_message_with_normalised_text(tmp_path):
    meta = _save(tmp_path, "s", _chat("hello", "We deploy   with  Make Release"))
    assert locate_quote(tmp_path, "s", "deploy with make release") == (meta.id, 1)
    assert locate_quote(tmp_path, "s", "not in there at all") is None
    assert locate_quote(tmp_path, "other", "deploy with make release") is None
    assert locate_quote(tmp_path, "s", "hi") is None  # too short to mean anything
