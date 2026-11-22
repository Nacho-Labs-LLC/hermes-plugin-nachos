"""Tests for per-model calibration priors, cold-start blending, search confidence/abstention,
and possible_updates cold-start safeguards.
"""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import pytest

from bench import recall
from nachos_core.calibration import (
    CALIBRATION_FULL_HEADS,
    assess_confidence,
    blend_profiles,
    prior_profile,
)
from nachos_core.embeddings import Embedder
from nachos_core.index import MemoryIndex
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.ranking import normalize_queries, update_threshold_for
from nachos_core.store import get_store
from nachos_hermes.config import DEFAULTS, apply_config
from nachos_hermes.memory_provider import NachosMemoryProvider
from nachos_hermes.schemas import RECALL_SCHEMA


class _FixedModelEmbedder(Embedder):
    """Deterministic test embedder that mimics a known model family's cosine baseline."""

    def __init__(self, model_id: str, vectors_by_tag: dict[str, list[float]]) -> None:
        self.model_id = model_id
        self._vectors = vectors_by_tag

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            low = text.lower()
            matched = next((v for tag, v in self._vectors.items() if tag in low), None)
            out.append(list(matched) if matched is not None else [1.0, 0.0])
        return out


def _make_service(tmp_path: Path, embedder: Embedder | None = None) -> MemoryService:
    scope = Scope(
        PROFILE,
        get_store("sqlite", tmp_path / "memory.db"),
        MemoryIndex(tmp_path / "idx.db"),
    )
    return MemoryService(OrderedDict([(PROFILE, scope)]), embedder=embedder)


class TestCalibrationPriorsAndBlending:
    @pytest.mark.parametrize(
        "model_id",
        [
            "nachos:default",
            "nachos:Xenova/all-MiniLM-L6-v2:q8",
            "st:all-MiniLM-L6-v2",
            "nachos:Xenova/e5-small-v2:q8:dp='passage: '",
            "nachos:Xenova/bge-small-en-v1.5:q8:pooling=cls",
            "nachos:Snowflake/snowflake-arctic-embed-s:q8:pooling=cls",
            "nachos:Snowflake/snowflake-arctic-embed-m-v2.0:q8:pooling=cls",
            "nachos:Alibaba-NLP/gte-modernbert-base:q8:pooling=cls",
            "openai:http://localhost:11434/v1:nomic-embed-text-v1.5",
        ],
    )
    def test_known_model_families_have_complete_priors(self, model_id: str) -> None:
        prof = prior_profile(model_id)
        assert prof is not None
        for key in ("p50", "p75", "p90", "p95", "p99", "p99.5", "mean", "std"):
            assert key in prof
        assert prof["p50"] <= prof["p75"] <= prof["p95"] <= prof["p99"] <= prof["p99.5"]
        assert prof["std"] > 0

    def test_unknown_model_has_no_prior(self) -> None:
        assert prior_profile("custom-unknown-embedder-v1") is None
        assert prior_profile("") is None

    def test_arctic_v2_matches_before_arctic_v1(self) -> None:
        v1 = prior_profile("Snowflake/snowflake-arctic-embed-s")
        v2 = prior_profile("Snowflake/snowflake-arctic-embed-m-v2.0")
        assert v1 is not None
        assert v2 is not None
        assert v1["p50"] > 0.5
        assert v2["p50"] < 0.25

    def test_blend_profiles_transitions_smoothly_from_prior_to_empirical(self) -> None:
        prior = {"p50": 0.50, "p99": 0.66, "mean": 0.50, "std": 0.05}
        emp = {"p50": 0.60, "p99": 0.76, "mean": 0.60, "std": 0.07}
        assert blend_profiles(prior, None, 10) == prior
        assert blend_profiles(None, emp, 40) == emp
        assert blend_profiles(None, None, 10) is None
        mid = blend_profiles(prior, emp, 50, full_heads=100)
        assert mid is not None
        assert mid["p50"] == pytest.approx(0.55)
        assert mid["p99"] == pytest.approx(0.71)
        full = blend_profiles(prior, emp, CALIBRATION_FULL_HEADS, full_heads=CALIBRATION_FULL_HEADS)
        assert full == emp

    def test_index_uses_prior_below_30_and_blends_above_30(self, tmp_path: Path) -> None:
        index = MemoryIndex(tmp_path / "idx.db")
        model = "nachos:Alibaba-NLP/gte-modernbert-base:q8:pooling=cls"
        prior = prior_profile(model)
        assert prior is not None
        try:
            # Cold start (5 vectors < 30): uses prior directly.
            for i in range(5):
                index.upsert(f"k{i}", f"t{i}", model, [1.0, float(i) * 0.01])
            cold = index.similarity_profile(model)
            assert cold == prior
            # Cold-start possible_updates immediately gets gte's p99 prior (~0.667), not 0.48!
            assert update_threshold_for(index, model, 0.48) == pytest.approx(prior["p99"])

            # At 40 entries, blends 40% empirical + 60% prior.
            rnd = random.Random(7)  # noqa: S311 - deterministic test data
            for i in range(5, 40):
                ang = rnd.uniform(0, math.pi)
                index.upsert(f"k{i}", f"t{i}", model, [math.cos(ang), math.sin(ang)])
            blended = index.similarity_profile(model)
            assert blended is not None
            assert blended != prior
        finally:
            index.close()


class TestConfidenceAndAbstention:
    def test_assess_confidence_empty_lexical_semantic_and_rerank(self) -> None:
        empty = assess_confidence(has_results=False, lexical_coverage=0.0)
        assert empty["confident"] is False
        assert empty["confidence"] == "none"

        # Lexical strong match vs weak partial overlap.
        lex_hi = assess_confidence(has_results=True, lexical_coverage=0.95, lexical_margin=0.6)
        assert lex_hi["confident"] is True
        assert lex_hi["confidence"] == "high"
        lex_lo = assess_confidence(has_results=True, lexical_coverage=0.2, lexical_margin=0.1)
        assert lex_lo["confident"] is False
        assert lex_lo["confidence"] == "low"

        # Semantic calibrated z-score: same cosine 0.62 is strong for MiniLM, weak for gte.
        minilm = prior_profile("Xenova/all-MiniLM-L6-v2")
        gte = prior_profile("Alibaba-NLP/gte-modernbert-base")
        res_mini = assess_confidence(
            has_results=True,
            lexical_coverage=0.4,
            semantic_top1=0.62,
            semantic_topk=0.20,
            profile=minilm,
            agreement=True,
        )
        res_gte = assess_confidence(
            has_results=True,
            lexical_coverage=0.1,
            semantic_top1=0.55,
            semantic_topk=0.52,
            profile=gte,
            agreement=False,
        )
        assert res_mini["confident"] is True
        assert res_mini["confidence"] == "high"
        assert res_gte["confident"] is False
        assert res_gte["confidence"] == "low"

        # Rerank score lifts a borderline hit when confident.
        lifted = assess_confidence(
            has_results=True,
            lexical_coverage=0.0,
            semantic_top1=0.30,
            semantic_topk=0.22,
            profile=minilm,
            agreement=False,
            rerank_score=0.92,
        )
        assert lifted["confident"] is True
        assert lifted["signals"]["rerank"] == 0.92

    def test_search_with_confidence_abstains_on_miss_and_keeps_low_conf_candidates(
        self, tmp_path: Path
    ) -> None:
        svc = _make_service(tmp_path)
        try:
            svc.put(
                title="Harbor billing service",
                summary="Go service that issues invoices and handles payment webhooks",
                body="Harbor lives in billing/harbor. Go 1.23, Postgres 16, gRPC API.",
            )
            svc.put(
                title="Lumen analytics dashboard",
                summary="React and TypeScript dashboard for product metrics",
                body="Lumen uses Vite, React 18 and visx charts for weekly retention.",
            )

            # Strong match -> confident=True, no abstention flag.
            hit = svc.search_with_confidence("harbor invoices")
            assert hit["confident"] is True
            assert hit["confidence"] == "high"
            assert "no_confident_match" not in hit
            assert hit["results"][0]["key"] == "harbor-billing-service"

            # Completely off-topic -> empty results + no_confident_match.
            miss = svc.search_with_confidence("sourdough starter feeding schedule")
            assert miss["confident"] is False
            assert miss["confidence"] == "none"
            assert miss["no_confident_match"] is True
            assert miss["results"] == []

            # Near-miss: mentions Harbor, but asks for 5 unstored terms -> abstains while
            # still returning Harbor as a low-confidence candidate!
            near = svc.search_with_confidence("Harbor PDF font licensing kerning table")
            assert near["confident"] is False
            assert near["confidence"] == "low"
            assert near["no_confident_match"] is True
            assert "low-confidence" in near["note"]
            assert near["results"][0]["key"] == "harbor-billing-service"
        finally:
            svc.close()

    def test_redirected_old_wording_query_stays_confident(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            svc.put(
                title="Payment processor v1",
                summary="Billing runs through OldPay merchant accounts",
                body="All card charges route through OldPay merchant accounts.",
            )
            svc.put(
                title="Billing processor",
                summary="Payments settle via Adyen platform",
                body="All merchant settlements now run on Adyen.",
                supersedes="payment-processor-v1",
            )
            res = svc.search_with_confidence("OldPay merchant")
            assert res["results"][0]["key"] == "billing-processor"
            assert res["confident"] is True
            assert res["signals"]["lexical_coverage"] >= 0.9
        finally:
            svc.close()

    def test_possible_updates_cold_start_uses_model_prior(self, tmp_path: Path) -> None:
        # Cosine between [1, 0] and [0.55, 0.835] is 0.55: above uncalibrated 0.48,
        # but BELOW gte-modernbert's p99 prior (0.6667), so distinct put is NOT falsely flagged!
        emb = _FixedModelEmbedder(
            "nachos:Alibaba-NLP/gte-modernbert-base:q8:pooling=cls",
            {
                "alpha": [1.0, 0.0],
                "beta": [0.55, math.sqrt(1.0 - 0.55**2)],
                "alpha-v2": [0.95, math.sqrt(1.0 - 0.95**2)],
            },
        )
        svc = _make_service(tmp_path, embedder=emb)
        try:
            svc.put(title="Alpha config", body="Alpha database service runs on port 5432.")
            distinct = svc.put(title="Beta schedule", body="Beta weekly standup meeting notes.")
            assert "possible_updates" not in distinct

            update = svc.put(title="Alpha-v2 config", body="Alpha-v2 database runs on port 6432.")
            assert [u["key"] for u in update["possible_updates"]] == ["alpha-config"]
        finally:
            svc.close()

    def test_provider_recall_tool_and_config(self, tmp_path: Path) -> None:
        cfg = dict(DEFAULTS)
        assert cfg["abstention"] == "on"
        apply_config(cfg, {"abstention": "off"})
        assert cfg["abstention"] == "off"
        apply_config(cfg, {"abstention": "bogus"})
        assert cfg["abstention"] == "off"

        provider = NachosMemoryProvider()
        provider.initialize("sess-abstain", hermes_home=str(tmp_path))
        try:
            assert "no_confident_match" in provider.system_prompt_block()
            assert "no_confident_match" in RECALL_SCHEMA["description"]

            provider.handle_tool_call(
                "nachos_memory_put",
                {"title": "Harbor billing", "body": "Harbor handles invoices and card webhooks."},
            )
            hit = json.loads(
                provider.handle_tool_call("nachos_memory_recall", {"query": "harbor invoices"})
            )
            assert hit["confident"] is True
            assert hit["confidence"] == "high"
            assert "no_confident_match" not in hit

            miss = json.loads(
                provider.handle_tool_call("nachos_memory_recall", {"query": "sourdough recipe"})
            )
            assert miss["confident"] is False
            assert miss["no_confident_match"] is True
            assert "note" in miss

            # Opt-out returns legacy minimal payload.
            provider._cfg["abstention"] = "off"
            legacy = json.loads(
                provider.handle_tool_call("nachos_memory_recall", {"query": "harbor invoices"})
            )
            assert "confident" not in legacy
            assert legacy["count"] == 1
        finally:
            provider.shutdown()

    def test_bench_aggregate_reports_abstention_metrics(self) -> None:
        rows = [
            {"rank": 1, "ms": 1.0, "category": "keyword", "expect": ["a"], "confident": True},
            {"rank": 2, "ms": 1.0, "category": "natural", "expect": ["b"], "confident": False},
            {
                "rank": None,
                "ms": 1.0,
                "category": "negative",
                "kind": "",
                "expect": [],
                "n_results": 2,
                "confident": False,
            },
            {
                "rank": None,
                "ms": 1.0,
                "category": "negative",
                "kind": "near-miss",
                "expect": [],
                "n_results": 3,
                "confident": False,
            },
            {
                "rank": None,
                "ms": 1.0,
                "category": "negative",
                "kind": "near-miss",
                "expect": [],
                "n_results": 3,
                "confident": True,
            },
        ]
        agg = recall.aggregate(rows)
        ab = agg["abstention"]
        assert ab["abstained_unanswerable"] == 2
        assert ab["n_unanswerable"] == 3
        assert ab["abstained_answerable"] == 1
        assert ab["precision"] == pytest.approx(2 / 3, abs=1e-3)
        assert ab["recall"] == pytest.approx(2 / 3, abs=1e-3)
        assert ab["standard_recall"] == 1.0
        assert ab["near_miss_recall"] == 0.5
        assert ab["confident_hit@1"] == 0.5


class _CountingEmbedder(_FixedModelEmbedder):
    """Records every embed batch call so tests can verify single-batch multi-query embedding."""

    def __init__(self, model_id: str, vectors_by_tag: dict[str, list[float]]) -> None:
        super().__init__(model_id, vectors_by_tag)
        self.calls: list[list[str]] = []

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return super()._embed_batch(texts)


class TestMultiQueryRecall:
    def test_normalize_queries_dedupes_strips_and_caps_at_four(self) -> None:
        assert normalize_queries("  foo ", ["FOO", "", "bar", "baz", "qux", "extra"]) == [
            "foo",
            "bar",
            "baz",
            "qux",
        ]
        assert normalize_queries("", ["  alpha  ", "beta"]) == ["alpha", "beta"]
        assert normalize_queries("", None) == []

    def test_multi_query_batches_embeddings_once_and_fuses_best_similarity_and_confidence(
        self, tmp_path: Path
    ) -> None:
        emb = _CountingEmbedder(
            "nachos:Xenova/all-MiniLM-L6-v2:q8",
            {
                "alembic": [1.0, 0.0, 0.0],
                "schema migration": [0.95, 0.0, math.sqrt(1.0 - 0.95**2)],
                "table changes": [0.15, 0.10, math.sqrt(1.0 - 0.15**2 - 0.10**2)],
                "redis": [0.0, 1.0, 0.0],
            },
        )
        svc = _make_service(tmp_path, embedder=emb)
        try:
            svc.put(
                title="Database migrations",
                summary="Decision: migrations run through Alembic before deploy",
                body="All schema migrations are authored and executed via Alembic.",
            )
            svc.put(
                title="Redis cache",
                summary="Redis stores short-lived session state",
                body="Redis 7 handles rate limiting and session tokens.",
            )
            emb.calls.clear()

            # Single vague query alone has weak similarity and no lexical match -> low confidence.
            weak = svc.search_with_confidence("table changes")
            assert weak["confident"] is False
            assert len(emb.calls) == 1

            emb.calls.clear()
            # Three variants fused in ONE embed call -> deduped hit, best similarity, confident!
            multi = svc.search_with_confidence(
                queries=[
                    "table changes",
                    "schema migration workflow",
                    "Decision: migrations run through Alembic",
                ]
            )
            assert len(emb.calls) == 1
            assert len(emb.calls[0]) == 3
            assert multi["queries"] == [
                "table changes",
                "schema migration workflow",
                "Decision: migrations run through Alembic",
            ]
            assert multi["confident"] is True
            top = multi["results"][0]
            assert top["key"] == "database-migrations"
            assert top["match"] == "lexical+semantic"
            assert top["similarity"] == pytest.approx(1.0, abs=1e-3)
            # Deduplicated: database-migrations appears once even though all 3 queries hit it.
            assert [r["key"] for r in multi["results"]].count("database-migrations") == 1
        finally:
            svc.close()

    def test_provider_recall_accepts_queries_list_with_or_without_query(
        self, tmp_path: Path
    ) -> None:
        provider = NachosMemoryProvider()
        provider.initialize("sess-mq", hermes_home=str(tmp_path))
        try:
            assert "queries" in RECALL_SCHEMA["parameters"]["properties"]
            assert "Alembic" in RECALL_SCHEMA["description"]
            provider.handle_tool_call(
                "nachos_memory_put",
                {
                    "title": "Migration workflow",
                    "summary": "Decision: migrations run through Alembic",
                    "body": "Run alembic upgrade head before deploying API containers.",
                },
            )
            res = json.loads(
                provider.handle_tool_call(
                    "nachos_memory_recall",
                    {
                        "queries": [
                            "how do table changes ship",
                            "Decision: migrations run through Alembic",
                        ]
                    },
                )
            )
            assert res["confident"] is True
            assert res["results"][0]["key"] == "migration-workflow"
            assert len(res["queries"]) == 2
        finally:
            provider.shutdown()

    def test_bench_multi_query_smoke(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        out = tmp_path / "mq.json"
        rc = recall.main(
            [
                "--embeddings",
                "off",
                "--multi-query",
                "--json",
                str(out),
            ]
        )
        assert rc == 0
        captured = capsys.readouterr().out
        assert "Exact 3-query groups" in captured
        report = json.loads(out.read_text(encoding="utf-8"))
        run = report["runs"][0]
        assert run["status"] == "ok"
        assert run["triples"]["n"] == 29
        assert run["triples"]["3-query"]["hit@1"] > run["triples"]["1-query"]["hit@1"]
        assert run["all_groups"]["n"] >= 100
