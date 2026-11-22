"""Ranking round: calibrated floor and duplicate threshold, chunk penalty, rerank stage."""

import math
import random

import pytest

from bench.recall import build_service
from nachos_core.chunking import embed_chunks
from nachos_core.embeddings import Embedder
from nachos_core.index import MemoryIndex
from nachos_core.ranking import blend_order, rerank_passage
from nachos_hermes.config import DEFAULTS, apply_config
from nachos_hermes.memory_provider import NachosMemoryProvider


def _unit(angle):
    return [math.cos(angle), math.sin(angle)]


class AngleEmbedder(Embedder):
    """2-d vectors: text 'a<deg>' embeds at that angle, so cosine = cos(angle difference)."""

    model_id = "angle-v1"

    def __init__(self):
        self.reranked = []

    def _embed_batch(self, texts):
        out = []
        for text in texts:
            tag = next((w for w in text.split() if w[:1] == "a" and w[1:].isdigit()), "a0")
            out.append(_unit(math.radians(float(tag[1:]))))
        return out


def _fill(svc, count=40, seed=3):
    rnd = random.Random(seed)  # noqa: S311 - deterministic test data
    for i in range(count):
        deg = rnd.uniform(0, 180)
        svc.put(title=f"Filler {i}", body=f"filler entry number {i} a{deg:.2f} unrelated words")


class TestIndexCalibration:
    def test_profile_needs_enough_vectors(self, tmp_path):
        index = MemoryIndex(tmp_path / "i.db")
        for i in range(10):
            index.upsert(f"k{i}", "t", "m", [1.0, float(i)])
        assert index.similarity_profile("m") is None

    def test_profile_is_cached_by_model_and_resampled_after_20_percent_drift(self, tmp_path):
        index = MemoryIndex(tmp_path / "i.db")
        rnd = random.Random(1)  # noqa: S311 - deterministic test data
        for i in range(40):
            index.upsert(f"k{i}", "t", "m", _unit(rnd.uniform(0, 3)))
        first = index.similarity_profile("m")
        assert first is not None
        assert first["p50"] <= first["p90"] <= first["p99"]
        row = index._conn.execute("SELECT head_count FROM calibration WHERE model='m'").fetchone()
        assert row[0] == 40
        for i in range(40, 46):  # +15 percent: cached profile is reused
            index.upsert(f"k{i}", "t", "m", _unit(rnd.uniform(0, 3)))
        assert index.similarity_profile("m") == first
        assert index._conn.execute("SELECT head_count FROM calibration").fetchone()[0] == 40
        for i in range(46, 52):  # +30 percent in total: resampled
            index.upsert(f"k{i}", "t", "m", _unit(rnd.uniform(0, 3)))
        assert index.similarity_profile("m") is not None
        assert index._conn.execute("SELECT head_count FROM calibration").fetchone()[0] == 52

    def test_chunk_penalty_requires_a_chunk_to_beat_the_head(self, tmp_path):
        index = MemoryIndex(tmp_path / "i.db")
        head, chunk = _unit(0.5), _unit(0.0)  # query at angle 0: the chunk is the closer one
        index.upsert_chunks("k", ["h", "c"], "m", [head, chunk])
        assert index.search_chunks(_unit(0.0), "m")[0][2] == 1
        assert index.search_chunks(_unit(0.0), "m", chunk_penalty=0.2)[0][2] == 0
        score = index.search_chunks(_unit(0.0), "m", chunk_penalty=0.2)[0][1]
        assert score == pytest.approx(math.cos(0.5))


class TestServiceRanking:
    def test_calibrated_floor_drops_weak_semantic_hits(self, tmp_path):
        svc = build_service(tmp_path, AngleEmbedder())
        _fill(svc)
        svc.put(title="Target", body="the answer entry a10")
        query = "a10 something"
        assert "target" in {r["key"] for r in svc.search(query, limit=40)}
        svc.similarity_floor = "p90"
        profile = svc.profile.index.similarity_profile("angle-v1")
        results = svc.search(query, limit=40)
        assert results[0]["key"] == "target"
        assert all(r["similarity"] >= profile["p90"] for r in results if "similarity" in r)
        assert len(results) < 40

    def test_floor_is_skipped_for_small_stores(self, tmp_path):
        svc = build_service(tmp_path, AngleEmbedder())
        svc.similarity_floor = "p99"
        svc.put(title="Only", body="the single entry a170")
        assert [r["key"] for r in svc.search("a0 question")] == ["only"]  # too few to calibrate

    def test_duplicate_threshold_is_raised_to_the_model_percentile_only_when_calibrating(
        self, tmp_path
    ):
        svc = build_service(tmp_path, AngleEmbedder())
        _fill(svc, 60)
        scope = svc.profile
        assert svc._duplicate_threshold_for(scope, "angle-v1") == svc.duplicate_threshold
        svc.calibrate_duplicates = True
        profile = scope.index.similarity_profile("angle-v1")
        assert profile["p99"] > 0.65  # 2-d angles in 0-180 degrees: neighbours are very close
        assert svc._duplicate_threshold_for(scope, "angle-v1") == pytest.approx(profile["p99"])
        svc.duplicate_threshold = 2.0
        assert svc._duplicate_threshold_for(scope, "angle-v1") == 2.0  # never lowered

    def test_chunk_penalty_is_applied_in_search(self, tmp_path):
        svc = build_service(
            tmp_path,
            AngleEmbedder(),
        )
        long_body = "intro a90 text. " + ("filler words here. " * 120) + "tail a0 answer"
        svc.put(title="Long entry", body=long_body)
        hit = svc.search("a0 query")[0]
        assert hit["key"] == "long-entry"
        scope = svc.profile
        plain = scope.index.search_chunks(_unit(0.0), "angle-v1")[0]
        assert plain[2] > 0  # without a penalty the tail chunk is the best match
        svc.chunk_penalty = 2.0
        penalised = scope.index.search_chunks(_unit(0.0), "angle-v1", chunk_penalty=2.0)[0]
        assert penalised[2] == 0


class RerankEmbedder(AngleEmbedder):
    supports_rerank = True

    def __init__(self, scores=None, fail=False):
        super().__init__()
        self.scores = scores
        self.fail = fail
        self.calls = []

    def rerank(self, query, passages):
        self.calls.append((query, passages))
        if self.fail:
            raise RuntimeError("worker died")
        if self.scores is not None:
            return self.scores(passages)
        return [float(p.count("best")) for p in passages]


class TestRerank:
    def _service(self, tmp_path, emb, **kw):
        svc = build_service(tmp_path, emb)
        for i, deg in enumerate((0, 5, 10, 15)):
            word = "best" if i == 3 else "plain"
            svc.put(title=f"Doc {i}", summary=f"summary {i}", body=f"{word} body a{deg}")
        for key, value in kw.items():
            setattr(svc, key, value)
        return svc

    def test_pure_rerank_order_overrides_the_fused_order(self, tmp_path):
        emb = RerankEmbedder()
        svc = self._service(tmp_path, emb, rerank_mode="rerank")
        results = svc.search("a0 question", limit=4)
        assert results[0]["key"] == "doc-3"  # the only passage containing "best"
        assert results[0]["rerank"] == 1.0
        assert len(emb.calls) == 1
        assert "Doc 3\nsummary 3\nbest body a15" in emb.calls[0][1]

    def test_blend_keeps_a_strong_fused_rank_in_play(self, tmp_path):
        emb = RerankEmbedder(scores=lambda ps: [0.5 if "plain" in p else 0.51 for p in ps])
        svc = self._service(tmp_path, emb, rerank_mode="blend:10")
        fused_first = build_service(tmp_path / "x", AngleEmbedder())
        for i, deg in enumerate((0, 5, 10, 15)):
            fused_first.put(title=f"Doc {i}", summary=f"summary {i}", body=f"x body a{deg}")
        assert fused_first.search("a0 question")[0]["key"] == "doc-0"
        assert svc.search("a0 question")[0]["key"] in {"doc-0", "doc-3"}

    def test_only_the_top_candidates_are_sent(self, tmp_path):
        emb = RerankEmbedder()
        svc = self._service(tmp_path, emb, rerank_candidates=2)
        svc.search("a0 question", limit=1)
        assert len(emb.calls[0][1]) == 2

    def test_a_failing_reranker_keeps_the_fused_result_and_is_disabled_once(self, tmp_path, caplog):
        emb = RerankEmbedder(fail=True)
        svc = self._service(tmp_path, emb)
        with caplog.at_level("WARNING", logger="nachos_core.memory"):
            first = svc.search("a0 question")
            second = svc.search("a0 question")
        assert [r["key"] for r in first] == [r["key"] for r in second]
        assert len(emb.calls) == 1
        assert sum("reranking disabled" in r.message for r in caplog.records) == 1

    def test_embedders_without_rerank_are_untouched(self, tmp_path):
        svc = self._service(tmp_path, AngleEmbedder())
        assert "rerank" not in svc.search("a0 question")[0]


class TestRankingHelpers:
    def test_blend_order_modes(self):
        keys = ["a", "b", "c"]
        assert blend_order(keys, [0.1, 0.9, 0.5], "rerank") == ["b", "c", "a"]
        assert blend_order(keys, [0.1, 0.9, 0.5], "blend:1")[0] == "b"
        assert blend_order(keys, [0.5, 0.5, 0.5], "blend:10") == keys  # ties keep fused order
        assert blend_order(keys, [0.5, 0.5, 0.5], "sum:1") == keys

    def test_sum_blend_lets_a_confident_reranker_overturn_but_not_a_lukewarm_one(self):
        keys = ["a", "b", "c"]
        assert blend_order(keys, [0.05, 0.95, 0.1], "sum:1")[0] == "b"  # confident: b wins
        assert blend_order(keys, [0.30, 0.31, 0.2], "sum:1")[0] == "a"  # near-tie: fused rank holds
        with pytest.raises(ValueError, match="one rerank score"):
            blend_order(keys, [1.0])

    def test_rerank_passage_uses_the_best_chunk_text(self):
        body = "start of body. " + ("padding words. " * 200) + "needle sentence at the end."
        head = rerank_passage("T", "S", body, 0)
        assert head.startswith("T\nS\nstart of body.")
        last = len(embed_chunks("T", "S", body)) - 1
        tail = rerank_passage("T", "S", body, last)
        assert "needle sentence" in tail
        assert tail.count("T\n") == 1  # the chunk's own title prefix is not repeated
        assert len(tail) <= 1500


class TestConfig:
    def test_rerank_and_calibration_settings_validate_their_choices(self):
        cfg = dict(DEFAULTS)
        apply_config(
            cfg,
            {
                "rerank": "nachos",
                "similarity_floor": "auto",
                "rerank_candidates": 12,
                "rerank_model": " some/model ",
            },
        )
        assert (cfg["rerank"], cfg["similarity_floor"]) == ("nachos", "auto")
        assert cfg["rerank_candidates"] == 12
        assert cfg["rerank_model"] == "some/model"
        apply_config(
            cfg, {"rerank": "bedrock", "similarity_floor": "p10", "duplicate_calibration": "x"}
        )
        assert cfg["rerank"] == "nachos"  # invalid values are ignored
        assert cfg["similarity_floor"] == "auto"
        assert cfg["duplicate_calibration"] == "auto"

    def test_a_pinned_duplicate_threshold_is_never_calibrated(self):
        provider = NachosMemoryProvider()
        assert provider._calibrate_duplicates() is True  # defaults: auto at 0.65
        provider._cfg["duplicate_threshold"] = 0.5
        assert provider._calibrate_duplicates() is False
        provider._cfg["duplicate_threshold"] = 0.65
        provider._cfg["duplicate_calibration"] = "off"
        assert provider._calibrate_duplicates() is False
