"""SemanticScorer — opt-in embedding-based prefetch ranking.

Driver-agnostic over three backends, selected by name:
  * 'nachos'                — nachos-embeddings / mcp-semantic-search MCP
                              (recommended; shells out, no bundled model)
  * 'sentence-transformers' — local model (lazy import)
  * 'openai'                — text-embedding-3 hosted (lazy import)

HOT-PATH DISCIPLINE: this file bundles NO dependency. Every backend import
is lazy and guarded. If the chosen backend is unavailable or errors, rank()
FALLS BACK to LexicalScorer rather than crashing — prefetch must never
break because semantic recall is misconfigured.

Same contract as LexicalScorer: rank(query, candidates, top_n) -> [key].
"""

from __future__ import annotations

import logging
import math
import subprocess
from collections.abc import Sequence
from typing import List, Optional

from .prefetch import LexicalScorer, Scorer
from .store.base import Entry

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("nachos", "sentence-transformers", "openai")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticScorer(Scorer):
    """Embedding cosine-similarity ranking with lexical fallback."""

    def __init__(self, backend: str = "nachos", *, model: Optional[str] = None):
        self.backend = (backend or "nachos").strip().lower()
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"Unknown semantic backend {backend!r}. "
                f"Use one of {_VALID_BACKENDS}."
            )
        self.model = model
        self._fallback = LexicalScorer()
        # cache the sentence-transformers model instance across calls
        self._st_model = None

    # -- public contract ---------------------------------------------------

    def rank(
        self,
        query: str,
        candidates: List[Entry],
        top_n: int = 5,
    ) -> List[str]:
        q = (query or "").strip()
        if not q or not candidates:
            return []
        try:
            texts = [f"{title} {summary}" for (_k, title, summary, _c) in candidates]
            vectors = self._embed([q] + texts)
            if not vectors or len(vectors) != len(texts) + 1:
                raise RuntimeError("embedding backend returned wrong shape")
            qv, doc_vs = vectors[0], vectors[1:]
            scored = []
            for (entry, dv) in zip(candidates, doc_vs):
                sim = _cosine(qv, dv)
                if sim > 0:
                    scored.append((sim, entry[0]))
            scored.sort(key=lambda t: -t[0])
            return [key for (_s, key) in scored[:top_n]]
        except Exception as e:
            logger.warning(
                "Nachos semantic backend %r unavailable (%s) — falling back "
                "to lexical scorer for this prefetch.",
                self.backend, e,
            )
            return self._fallback.rank(query, candidates, top_n)

    # -- backends (all lazy) -----------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "sentence-transformers":
            return self._embed_sentence_transformers(texts)
        if self.backend == "openai":
            return self._embed_openai(texts)
        return self._embed_nachos(texts)

    def _embed_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        try:
            from sentence_transformers import SentenceTransformer  # lazy
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed "
                "(pip install sentence-transformers)"
            ) from e
        if self._st_model is None:
            self._st_model = SentenceTransformer(
                self.model or "all-MiniLM-L6-v2"
            )
        return [list(map(float, v)) for v in self._st_model.encode(texts)]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        try:
            from openai import OpenAI  # lazy
        except ImportError as e:
            raise RuntimeError("openai not installed (pip install openai)") from e
        client = OpenAI()
        resp = client.embeddings.create(
            model=self.model or "text-embedding-3-small",
            input=texts,
        )
        return [list(d.embedding) for d in resp.data]

    def _embed_nachos(self, texts: List[str]) -> List[List[float]]:
        """Shell out to the nachos-embeddings CLI / mcp-semantic-search.

        We do NOT import a model here — the whole point of the nachos
        backend is that embedding lives in the separate published service.
        Expects a CLI that reads newline-delimited texts on stdin and emits
        one JSON array of floats per line on stdout.
        """
        import json
        cmd = ["nachos-embeddings", "embed", "--stdin", "--json"]
        proc = subprocess.run(
            cmd,
            input="\n".join(t.replace("\n", " ") for t in texts),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"nachos-embeddings failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[:200]}"
            )
        vectors = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line:
                vectors.append([float(x) for x in json.loads(line)])
        return vectors
