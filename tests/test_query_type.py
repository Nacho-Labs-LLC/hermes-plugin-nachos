"""MemoryService sends the right input type: queries as "query", stored text as "document"."""

from bench.recall import build_service
from nachos_core.embeddings import Embedder
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
