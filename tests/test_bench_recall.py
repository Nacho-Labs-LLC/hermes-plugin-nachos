"""The recall benchmark harness itself: metric math, fixture integrity, lexical smoke run.

Lexical only, so it stays fast; the benchmark proper (python -m bench) is never run by pytest.
"""

import argparse
import json
import sys
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
        "query_prefix": "",
        "document_prefix": "",
        "no_types": False,
        "chunk_penalty": None,
        "floor": "off",
        "rerank": False,
        "label": "",
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
            if q["category"] == "negative":
                assert q["expect"] == [], q["id"]  # no correct answer exists
                continue
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


class TestAlternateDataDir:
    ADV = recall.DATA_DIR / "adversarial"

    def test_default_data_dir_is_unchanged(self):
        assert recall.parse_args([]).data_dir == recall.DATA_DIR

    def test_updates_are_appended_with_supersedes_by_label(self):
        memories, queries = recall.load_dataset(self.ADV)
        updates = json.loads((self.ADV / "updates.json").read_text(encoding="utf-8"))
        base = json.loads((self.ADV / "memories.json").read_text(encoding="utf-8"))
        assert len(memories) == len(base) + len(updates)
        keys = {slugify(m["title"]) for m in memories}
        linked = [m for m in memories if m.get("supersedes")]
        assert len(linked) == sum(1 for u in updates if u["update_of"])
        assert all(k in keys for m in linked for k in m["supersedes"])
        assert all(k in keys for q in queries for k in q["expect"])

    def test_suggest_eval_reports_by_case(self, capsys):
        args = ["--data-dir", str(self.ADV), "--suggest-eval", "--json", "-"]
        assert recall.main(args) == 0
        run = json.loads(capsys.readouterr().out)["runs"][0]
        assert run["by_case"]["update-nocue"]["n"] >= 1

    def test_no_answer_queries_are_reported_apart(self):
        memories, queries = recall.load_dataset(self.ADV)
        ans = [q for q in queries if q["category"] != "no-answer"][:5]
        no_ans = [q for q in queries if q["category"] == "no-answer"]
        run = recall.run_backend("off", _opts(), memories, ans + no_ans)
        assert run["no_answer"]["n"] == len(no_ans)
        assert run["no_answer"]["n"] >= 30
        assert run["overall"]["n"] == 5
        assert "abstention" in run


class TestNegativeQueries:
    def test_negative_rows_are_reported_apart_from_the_overall_hit_metrics(self):
        rows = [
            {"rank": 1, "ms": 1.0, "category": "keyword", "n_results": 3},
            {"rank": None, "ms": 2.0, "category": "negative", "n_results": 0},
            {"rank": None, "ms": 3.0, "category": "negative", "n_results": 7},
            {"rank": None, "ms": 4.0, "category": "negative", "n_results": 2},
        ]
        agg = recall.aggregate(rows)
        assert agg["overall"]["n"] == 1  # negatives never count as misses
        assert agg["overall"]["hit@1"] == 1.0
        neg = agg["by_category"]["negative"]
        assert neg["n"] == 3
        assert neg["fp@1"] == pytest.approx(
            2 / 3, abs=1e-3
        )  # two of three queries returned something
        assert neg["fp@5"] == pytest.approx(
            (0 + 1.0 + 0.4) / 3, abs=1e-3
        )  # mean filled share of top 5

    def test_negative_queries_exist_and_have_no_expected_key(self, dataset):
        negatives = [q for q in dataset[1] if q["category"] == "negative"]
        assert len(negatives) >= 25
        assert all(q["expect"] == [] for q in negatives)


def test_ranking_flags_are_parsed_into_service_options():
    opts = recall.parse_args(
        ["--chunk-penalty", "0", "--floor", "auto", "--rerank", "--rerank-candidates", "12"]
    )
    assert recall.ranking_options(opts) == {
        "chunk_penalty": 0.0,
        "similarity_floor": "p75",
        "rerank_candidates": 12,
        "rerank_mode": "sum:1",
    }
    assert recall.ranking_options(recall.parse_args([])) == {}


class TestModelOptions:
    def test_model_adds_the_nachos_backend_beside_lexical(self):
        assert recall.parse_args([]).backends == ["off"]
        assert recall.parse_args(["--model", "org/m"]).backends == ["off", "nachos"]
        assert recall.parse_args(["--model", "m", "--embeddings", "nachos"]).backends == ["nachos"]

    def test_model_flags_reach_the_embedder(self):
        opts = recall.parse_args(
            [
                "--command",
                sys.executable,
                "--model",
                "Xenova/all-MiniLM-L6-v2",
                "--pooling",
                "cls",
                "--max-length",
                "128",
                "--dtype",
                "q4",
                "--rerank-max-length",
                "2048",
            ]
        )
        embedder, reason = recall.resolve_embedder("nachos", opts)
        assert reason is None
        assert embedder.model_id == "nachos:Xenova/all-MiniLM-L6-v2:q4"
        assert embedder._model_args() == [
            "--model",
            "Xenova/all-MiniLM-L6-v2",
            "--dtype",
            "q4",
            "--pooling",
            "cls",
            "--max-length",
            "128",
        ]
        assert embedder.rerank_max_length == 2048

    def test_label_names_the_model_run_but_not_the_lexical_baseline(self):
        opts = recall.parse_args(["--model", "org/m", "--label", "mine"])
        assert recall._run_name("off", opts) == "off"
        assert recall._run_name("nachos", opts) == "mine"
        single = recall.parse_args(["--embeddings", "off", "--label", "base"])
        assert recall._run_name("off", single) == "base"
