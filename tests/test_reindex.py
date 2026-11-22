"""Safe model switching: old vectors kept until done, bounded steps, lexical meanwhile."""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from pathlib import Path

from nachos_core import reindex
from nachos_core.embeddings import Embedder
from nachos_core.index import MemoryIndex, _pack
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.store import get_store

_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


class WordEmbedder(Embedder):
    """Counts embed calls; one dimension per known word so similarity is predictable."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.calls = 0
        self.texts = 0

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        return [[float(w in t.lower()) + 0.01 for w in _WORDS] for t in texts]


def _service(tmp_path: Path, embedder: Embedder | None, **kwargs: float) -> MemoryService:
    d = tmp_path / PROFILE
    d.mkdir(parents=True, exist_ok=True)
    scope = Scope(PROFILE, get_store("sqlite", d / "memory.db"), MemoryIndex(d / "memory.index.db"))
    return MemoryService(OrderedDict([(PROFILE, scope)]), embedder=embedder, **kwargs)


def _fill(service: MemoryService, count: int) -> None:
    for i in range(count):
        word = _WORDS[i % len(_WORDS)]
        service.put(
            title=f"Entry {i} about {word}",
            body=f"The {word} procedure number {i} is documented here in detail.",
            summary=f"{word} note {i}",
        )


class TestBoundedStepsAndLexicalFallback:
    def test_small_stale_sets_are_embedded_inline(self, tmp_path: Path) -> None:
        service = _service(tmp_path, WordEmbedder("model-a"))
        _fill(service, 5)
        assert service.model_status()["scopes"][PROFILE]["pending"] == 0
        assert not service.reindexing

    def test_switch_keeps_old_vectors_and_recalls_lexically_until_done(
        self, tmp_path: Path
    ) -> None:
        count = reindex.BULK_ENTRIES + 12
        old = WordEmbedder("model-a")
        service = _service(tmp_path, old, reindex_budget_s=0.0)
        _fill(service, count)
        index = service.profile.index
        assert index.model_counts() == {"model-a": count}
        service.close()

        new = WordEmbedder("model-b")
        service = _service(tmp_path, new, reindex_budget_s=0.0)
        hits = service.search("alpha procedure")  # first recall after the switch
        assert hits, "lexical fallback must still answer"
        assert all(h["match"] == "lexical" for h in hits)
        assert service.reindexing
        status = service.model_status()
        assert status["state"] == "reindexing"
        scope = status["scopes"][PROFILE]
        assert scope["reindexing"] is True
        assert 0 < scope["embedded"] < count
        assert scope["other_models"] == {"model-a": count}  # old vectors kept
        # One bounded step per call: a zero budget still runs exactly one batch.
        assert new.texts >= reindex.FIRST_BATCH_TEXTS  # a zero budget still runs one batch

    def test_steps_complete_the_switch_then_purge_and_recalibrate(self, tmp_path: Path) -> None:
        count = reindex.BULK_ENTRIES + 10
        service = _service(tmp_path, WordEmbedder("model-a"), reindex_budget_s=0.0)
        _fill(service, count)
        service.close()

        service = _service(tmp_path, WordEmbedder("model-b"), reindex_budget_s=0.0)
        steps = 0
        while service.reindexing or steps == 0:
            service.search("bravo")
            steps += 1
            assert steps < 60
        assert steps > 1, "a zero budget must spread the work over several recalls"
        scope = service.model_status()["scopes"][PROFILE]
        assert scope["embedded"] == count
        assert scope["pending"] == 0
        assert scope["other_models"] == {}  # old model dropped only now
        assert scope["calibration"] is not None  # recalibrated for the new model
        hits = service.search("alpha procedure")
        assert any("semantic" in h["match"] for h in hits)

    def test_explicit_reindex_runs_to_completion(self, tmp_path: Path) -> None:
        count = reindex.BULK_ENTRIES + 8
        service = _service(tmp_path, WordEmbedder("model-a"))
        _fill(service, count)
        service.close()
        service = _service(tmp_path, WordEmbedder("model-b"), reindex_budget_s=0.0)
        status = service.reindex()
        assert status["state"] == "ready"
        assert status["scopes"][PROFILE]["embedded"] == count
        assert not service.reindexing

    def test_switching_back_before_completion_loses_nothing(self, tmp_path: Path) -> None:
        count = reindex.BULK_ENTRIES + 6
        first = _service(tmp_path, WordEmbedder("model-a"))
        _fill(first, count)
        first.close()
        mid = _service(tmp_path, WordEmbedder("model-b"), reindex_budget_s=0.0)
        mid.search("alpha")  # starts the switch but does not finish it
        assert mid.reindexing
        mid.close()
        back = _service(tmp_path, WordEmbedder("model-a"))
        back.search("alpha")
        scope = back.model_status()["scopes"][PROFILE]
        # model-a's vectors were never touched: nothing to re-embed.
        assert scope["pending"] == 0
        assert not scope["reindexing"]

    def test_writes_during_a_switch_stay_lexical_and_are_embedded_later(
        self, tmp_path: Path
    ) -> None:
        count = reindex.BULK_ENTRIES + 6
        first = _service(tmp_path, WordEmbedder("model-a"))
        _fill(first, count)
        first.close()
        service = _service(tmp_path, WordEmbedder("model-b"), reindex_budget_s=0.0)
        result = service.put(
            title="Brand new golf note",
            body="Golf details are documented here for the team.",
            summary="golf note for the team",
        )
        assert result["stored"] == "brand-new-golf-note"
        assert service.reindexing
        service.reindex()
        assert service.model_status()["scopes"][PROFILE]["pending"] == 0

    def test_embedder_failure_degrades_to_lexical_without_losing_old_vectors(
        self, tmp_path: Path
    ) -> None:
        count = reindex.BULK_ENTRIES + 6
        first = _service(tmp_path, WordEmbedder("model-a"))
        _fill(first, count)
        first.close()

        class Broken(WordEmbedder):
            def _embed_batch(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
                raise RuntimeError("model missing")

        service = _service(tmp_path, Broken("model-b"))
        hits = service.search("alpha procedure")
        assert hits
        assert all(h["match"] == "lexical" for h in hits)
        status = service.model_status()
        assert status["state"] == "error"
        assert status["scopes"][PROFILE]["other_models"] == {"model-a": count}


class TestIndexModelAwareness:
    def test_vectors_table_migrates_to_a_per_model_primary_key(self, tmp_path: Path) -> None:
        path = tmp_path / "old.index.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE vectors (
                key TEXT NOT NULL, chunk_no INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL,
                PRIMARY KEY (key, chunk_no)
            );
            """
        )
        conn.execute("INSERT INTO vectors VALUES ('a', 0, 'h', 'm1', ?)", (_pack([1.0, 0.0]),))
        conn.commit()
        conn.close()
        index = MemoryIndex(path)
        assert index.model_counts() == {"m1": 1}
        index.upsert("a", "text", "m2", [0.0, 1.0])  # same key, other model: both kept
        assert index.model_counts() == {"m1": 1, "m2": 1}
        assert index.purge_other_models("m2") == 1
        assert index.model_counts() == {"m2": 1}

    def test_stale_keys_ignore_other_models(self) -> None:
        index = MemoryIndex(":memory:")
        index.upsert("a", "text", "m1", [1.0, 0.0])
        assert index.stale_keys({"a": "text"}, "m1") == []
        assert index.stale_keys({"a": "text"}, "m2") == ["a"]
        index.upsert("a", "text", "m2", [0.0, 1.0])
        assert index.stale_keys({"a": "text"}, "m2") == []
        assert index.vectors("m1")  # both models coexist
        assert index.vectors("m2")


class TestEmbedStale:
    def test_stops_when_the_backend_fails(self) -> None:
        index = MemoryIndex(":memory:")
        docs = {f"k{i}": [f"text {i}"] for i in range(40)}
        done = reindex.embed_stale(lambda _texts: None, index, "m", docs, list(docs), budget_s=1.0)
        assert done == 0
        assert index.model_counts() == {}

    def test_small_sets_ignore_the_budget(self) -> None:
        index = MemoryIndex(":memory:")
        docs = {f"k{i}": [f"text {i}"] for i in range(5)}
        done = reindex.embed_stale(
            lambda texts: [[1.0, 0.0] for _ in texts], index, "m", docs, list(docs), budget_s=0.0
        )
        assert done == 5
