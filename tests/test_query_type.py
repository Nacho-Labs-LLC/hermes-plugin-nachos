"""MemoryService sends the right input type: queries as "query", stored text as "document"."""

from bench.recall import build_service
from nachos_core.embeddings import Embedder
from nachos_core.recall import MIN_LEXICAL_COVERAGE as _MIN_LEXICAL_COVERAGE
from nachos_hermes.config import DEFAULTS, apply_config


class TypeRecorder(Embedder):
    model_id = "types-v1"

    def __init__(self):
        self.calls = []

    def _embed_batch(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def _embed_typed(self, texts, input_type):
        self.calls.append((input_type, len(texts)))
        return self._embed_batch(texts)


def test_put_embeds_documents_and_search_embeds_a_query(tmp_path):
    rec = TypeRecorder()
    svc = build_service(tmp_path, rec)
    svc.put(title="Deploy notes", body="Use the blue pipeline to ship.")
    svc.put(title="Cache notes", body="Redis holds sessions.")
    assert {kind for kind, _ in rec.calls} == {"document"}
    rec.calls.clear()
    svc.search("how do we ship")
    assert rec.calls == [("query", 1)]


def test_reindex_after_a_model_change_uses_documents(tmp_path):
    rec = TypeRecorder()
    svc = build_service(tmp_path, rec)
    svc.put(title="Deploy notes", body="Use the blue pipeline to ship.")
    rec.model_id = "types-v2"
    rec.calls.clear()
    svc.search("ship")
    assert rec.calls == [("document", 1), ("query", 1)]


def test_prefix_config_keeps_meaningful_spaces():
    cfg = dict(DEFAULTS)
    apply_config(cfg, {"embedding_query_prefix": "query: ", "embedding_model": " m "})
    assert cfg["embedding_query_prefix"] == "query: "
    assert cfg["embedding_model"] == "m"


class _Versioned(TypeRecorder):
    """Learns a richer model id once warmed up, like CliEmbedder after its ready line."""

    def __init__(self):
        super().__init__()
        self.model_id = "v-base"

    def warm_up(self):
        self.model_id = "v-base:pooling=cls"


def test_index_is_keyed_by_the_warmed_up_model_id_and_re_embeds_old_vectors(tmp_path):
    rec = _Versioned()
    svc = build_service(tmp_path, rec)
    svc.put(title="Deploy notes", body="Use the blue pipeline to ship.")
    scope = svc.profile
    assert scope.index.count("v-base:pooling=cls") == 1
    assert scope.index.count("v-base") == 0
    # Vectors stored under the old (pre-upgrade) id are stale and get re-embedded.
    stale = svc.profile.store.list()[0][0]
    scope.index.upsert(stale, "x", "v-base", [1.0, 0.0])
    scope.index.delete_vector(stale)
    rec.calls.clear()
    svc.sync_index(scope)
    assert scope.index.count("v-base:pooling=cls") == 1


def test_hybrid_search_ignores_weak_lexical_matches_but_lexical_only_keeps_them(tmp_path):
    assert 0 < _MIN_LEXICAL_COVERAGE < 1
    hybrid = build_service(tmp_path / "h", TypeRecorder())
    lexical = build_service(tmp_path / "l", None)
    for svc in (hybrid, lexical):
        svc.put(title="Cache layer", body="Redis keeps hot keys, unrelated words, something else.")
        svc.put(title="Release train", body="Alpha beta gamma delta epsilon zeta service.")
    query = "service alpha unrelated words something else"  # release-train covers 2 of 6 terms
    assert "release-train" in {r["key"] for r in lexical.search(query)}
    matches = {r["key"]: r["match"] for r in hybrid.search(query)}
    assert "lexical" in matches["cache-layer"]
    assert "lexical" not in matches.get("release-train", "")


def test_low_cosine_semantic_hits_are_not_dropped_by_an_absolute_floor(tmp_path):
    class Orthogonal(TypeRecorder):
        def _embed_batch(self, texts):  # every query is nearly orthogonal to every document
            return [[0.05, 1.0] if t.startswith("zzz") else [1.0, 0.0] for t in texts]

    svc = build_service(tmp_path, Orthogonal())
    svc.put(title="Deploy notes", body="Use the blue pipeline to ship.")
    results = svc.search("zzz")
    assert [r["key"] for r in results] == ["deploy-notes"]
    assert results[0]["similarity"] < 0.1
