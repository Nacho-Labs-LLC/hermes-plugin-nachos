"""The recall benchmark harness itself: metric math, fixture integrity, lexical smoke run.

Lexical only, so it stays fast; the benchmark proper (python -m bench) is never run by pytest.
"""

import argparse
import json
from collections import Counter

import pytest

from bench import recall
from nachos_core.lexical import jaccard, query_terms, tokens
from nachos_core.memory import embed_text
from nachos_core.store.md_store import slugify


@pytest.fixture(scope="module")
def dataset():
    return recall.load_dataset()


def _opts(**kw):
    base = {
        "model": "",
        "dtype": "",
        "base_url": "",
        "api_key_env": "",
        "command": "nachos-embeddings",
        "repeat": 1,
    }
    return argparse.Namespace(**{**base, **kw})


class TestMetrics:
    def test_rank_of_any_uses_the_best_placed_expected_key(self):
        assert recall.rank_of(["b", "z"], ["a", "b", "c"]) == 2
        assert recall.rank_of(["c", "a"], ["a", "b", "c"]) == 1

    def test_rank_of_misses_are_none(self):
        assert recall.rank_of(["x"], ["a", "b"]) is None
        assert recall.rank_of(["x"], []) is None

    def test_rank_of_all_needs_every_key_and_uses_the_worst_placed(self):
        assert recall.rank_of(["a", "c"], ["a", "b", "c"], "all") == 3
        assert recall.rank_of(["a", "x"], ["a", "b", "c"], "all") is None

    def test_hit_at(self):
        ranks = [1, 3, None, 6]
        assert recall.hit_at(ranks, 1) == 0.25
        assert recall.hit_at(ranks, 5) == 0.5
        assert recall.hit_at([], 5) == 0.0

    def test_mrr_at_ignores_ranks_past_the_cutoff(self):
        assert recall.mrr_at([1, 2, None, 11], 10) == pytest.approx((1 + 0.5) / 4)
        assert recall.mrr_at([], 10) == 0.0

    def test_percentile_interpolates(self):
        values = [10.0, 20.0, 30.0, 40.0]
        assert recall.percentile(values, 0) == 10.0
        assert recall.percentile(values, 50) == 25.0
        assert recall.percentile(values, 100) == 40.0
        assert recall.percentile([], 95) == 0.0
        assert recall.percentile([7.0], 95) == 7.0

    def test_aggregate_splits_by_category_in_canonical_order(self):
        rows = [
            {"rank": 1, "ms": 1.0, "category": "natural"},
            {"rank": None, "ms": 3.0, "category": "keyword"},
            {"rank": 2, "ms": 2.0, "category": "keyword"},
        ]
        agg = recall.aggregate(rows)
        assert agg["overall"]["n"] == 3
        assert list(agg["by_category"]) == ["keyword", "natural"]
        assert agg["by_category"]["keyword"]["hit@1"] == 0.0
        assert agg["by_category"]["keyword"]["hit@5"] == 0.5
        assert agg["by_category"]["natural"]["mrr@10"] == 1.0


class TestDatasetIntegrity:
    def test_size_targets(self, dataset):
        memories, queries = dataset
        assert 150 <= len(memories) <= 250
        assert len(queries) >= 300

    def test_memory_keys_are_unique(self, dataset):
        keys = [slugify(m["title"]) for m in dataset[0]]
        assert [k for k, n in Counter(keys).items() if n > 1] == []

    def test_query_ids_and_texts_are_unique(self, dataset):
        queries = dataset[1]
        assert len({q["id"] for q in queries}) == len(queries)
        assert len({(q["query"], tuple(q["expect"])) for q in queries}) == len(queries)

    def test_every_expected_key_exists(self, dataset):
        keys = {slugify(m["title"]) for m in dataset[0]}
        for q in dataset[1]:
            assert q["expect"], q["id"]
            assert set(q["expect"]) <= keys, q["id"]

    def test_categories_are_known_and_all_present(self, dataset):
        counts = Counter(q["category"] for q in dataset[1])
        assert set(counts) == set(recall.CATEGORIES)
        assert all(n >= 20 for n in counts.values()), counts

    def test_match_modes(self, dataset):
        for q in dataset[1]:
            assert q.get("match", "any") in ("any", "all"), q["id"]
            if q["category"] == "multi-hop":
                assert q["match"] == "all"
                assert len(q["expect"]) >= 2, q["id"]

    def test_every_memory_passes_the_write_gates(self, dataset, tmp_path):
        svc = recall.build_service(tmp_path, None)
        try:
            for m in dataset[0]:
                draft = svc.prepare(
                    title=m["title"], summary=m["summary"], body=m["body"], category=m["category"]
                )
                assert draft.error is None, (m["title"], draft.error)
        finally:
            svc.close()

    def test_paraphrase_queries_share_no_terms_with_their_answer(self, dataset):
        by_key = {slugify(m["title"]): m for m in dataset[0]}
        for q in dataset[1]:
            if q["category"] != "paraphrase":
                continue
            _raw, terms = query_terms(q["query"])
            for key in q["expect"]:
                m = by_key[key]
                shared = set(terms) & tokens(embed_text(m["title"], m["summary"], m["body"]))
                assert not shared, (q["id"], sorted(shared))

    def test_long_body_answers_sit_past_the_embedded_prefix(self, dataset):
        by_key = {slugify(m["title"]): m for m in dataset[0]}
        for q in dataset[1]:
            if q["category"] != "long-body":
                continue
            _raw, terms = query_terms(q["query"])
            for key in q["expect"]:
                m = by_key[key]
                text = embed_text(m["title"], m["summary"], m["body"])
                assert len(text) > 2000, key
                assert set(terms) & (tokens(text) - tokens(text[:2000])), q["id"]

    def test_outdated_queries_expect_one_entry_of_a_pair(self, dataset):
        memories, queries = dataset
        titles = {slugify(m["title"]): m["title"] for m in memories}
        outdated = [q for q in queries if q["category"] == "outdated"]
        assert len(outdated) >= 20
        for q in outdated:
            assert len(q["expect"]) == 1
            assert (
                titles[q["expect"][0]].endswith("(current)") or "(2026)" in titles[q["expect"][0]]
            )

    def test_near_dup_queries_have_a_sibling_entry(self, dataset):
        memories, queries = dataset
        by_key = {slugify(m["title"]): m for m in memories}
        for q in queries:
            if q["category"] != "near-dup":
                continue
            target = by_key[q["expect"][0]]["title"]
            siblings = [
                k
                for k, o in by_key.items()
                if o["title"] != target and jaccard(o["title"], target) >= 0.3
            ]
            assert siblings, q["id"]


class TestRunner:
    def test_lexical_run_reports_metrics(self, dataset):
        memories, queries = dataset
        subset = [q for q in queries if q["category"] in ("keyword", "multi-hop")][:30]
        run = recall.run_backend("off", _opts(), memories, subset)
        assert run["status"] == "ok"
        assert run["overall"]["n"] == len(subset)
        assert run["overall"]["hit@5"] > 0.5  # keyword queries are easy for lexical recall
        assert set(run["by_category"]) <= {"keyword", "multi-hop"}
        assert run["build_s"] >= 0
        assert json.dumps(run)  # serialisable for --json

    def test_missing_nachos_cli_is_skipped_with_a_reason(self):
        embedder, reason = recall.resolve_embedder("nachos", _opts(command="no-such-cli-xyz"))
        assert embedder is None
        assert "no-such-cli-xyz" in reason

    def test_openai_without_url_is_skipped(self):
        embedder, reason = recall.resolve_embedder("openai", _opts())
        assert embedder is None
        assert "--base-url" in reason

    def test_off_is_lexical_not_skipped(self):
        assert recall.resolve_embedder("off", _opts()) == (None, None)

    def test_cli_json_output(self, capsys):
        assert recall.main(["--category", "near-dup", "--json", "-"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["environment"]["memories"] >= 150
        assert report["runs"][0]["backend"] == "off"

    def test_cli_rejects_unknown_backend(self):
        with pytest.raises(SystemExit):
            recall.parse_args(["--embeddings", "bogus"])
