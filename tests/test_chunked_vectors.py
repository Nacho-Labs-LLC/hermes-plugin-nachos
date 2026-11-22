"""Tests for chunked body vectors: boundaries, max scoring, reconciliation, removal, migration."""

from __future__ import annotations

import sqlite3
from array import array
from collections import OrderedDict
from pathlib import Path

from nachos_core.embeddings import Embedder
from nachos_core.index import MemoryIndex
from nachos_core.memory import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNKS_PER_ENTRY,
    PROFILE,
    MemoryService,
    Scope,
    chunk_body,
    embed_chunks,
    embed_text,
)
from nachos_core.store import get_store

_TOPICS = [
    {"falcon", "raptor", "hawk", "kestrel"},
    {"postgres", "database", "sql", "pgbouncer"},
    {"kubernetes", "helm", "cluster", "ingress"},
    {"billing", "invoice", "stripe", "payout"},
]


class TopicEmbedder(Embedder):
    """Deterministic concept embedder that truncates like real clients (2000 chars)."""

    def __init__(self, model_id: str = "topic-v1") -> None:
        self.model_id = model_id
        self.embedded_texts: list[str] = []

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        out: list[list[float]] = []
        for text in texts:
            words = {w.strip(".,:;!?()").lower() for w in text.split()}
            out.append([float(len(words & group)) + 0.01 for group in _TOPICS])
        return out


def _make_service(tmp_path: Path, embedder: Embedder | None = None) -> MemoryService:
    d = tmp_path / PROFILE
    d.mkdir(parents=True, exist_ok=True)
    scope = Scope(PROFILE, get_store("sqlite", d / "memory.db"), MemoryIndex(d / "memory.index.db"))
    return MemoryService(
        OrderedDict([(PROFILE, scope)]), embedder=embedder, duplicate_threshold=0.9
    )


def _long_body(tail_sentence: str, *, filler_sentences: int = 45) -> str:
    filler = " ".join(
        f"Routine operational log paragraph sentence number {i} covering generic systems."
        for i in range(filler_sentences)
    )
    assert len(filler) > 2200
    return f"{filler} {tail_sentence}"


class TestChunkingBoundaries:
    def test_short_body_produces_only_head_vector(self) -> None:
        chunks = embed_chunks("Title", "Short summary", "A brief body under one chunk.")
        assert chunks == [embed_text("Title", "Short summary", "A brief body under one chunk.")]

    def test_paragraph_boundaries_and_overlap_with_title_prefix(self) -> None:
        paras = [f"Paragraph {i}. " + ("alpha beta gamma delta " * 14) for i in range(8)]
        body = "\n\n".join(paras)
        assert len(body) > CHUNK_SIZE * 2

        chunks = embed_chunks("Runbook title", "One-line summary", body)
        assert len(chunks) >= 3
        assert chunks[0] == embed_text("Runbook title", "One-line summary", body)
        for extra in chunks[1:]:
            assert extra.startswith("Runbook title\n")
            piece = extra[len("Runbook title\n") :]
            assert len(piece) <= CHUNK_SIZE + 50
            # Snapped cleanly on paragraph/word boundaries (starts with Paragraph or word boundary).
            assert not piece.startswith(" ")

        # Adjacent body chunks overlap meaningfully and the tail is covered.
        body_parts = chunk_body(body)
        assert paras[-1].strip() in body_parts[-1]
        for prev, nxt in zip(body_parts, body_parts[1:]):
            shared_tail = prev[-CHUNK_OVERLAP:]
            assert any(word in nxt[: CHUNK_OVERLAP + 100] for word in shared_tail.split()[-5:])

    def test_single_paragraph_sentence_boundaries(self) -> None:
        tail = "The kestrel raptor nests near the north tower."
        body = _long_body(tail)
        chunks = embed_chunks("Field notes", "Operational notes", body)
        assert len(chunks) >= 3
        assert any(tail in c for c in chunks[1:])
        for extra in chunks[1:]:
            assert extra.startswith("Field notes\n")
            piece = extra[len("Field notes\n") :]
            # Should not slice mid-word at the start of an overlapped chunk.
            first_word = piece.split()[0]
            assert first_word in body.split()

    def test_unbroken_string_hard_splits_and_respects_cap(self) -> None:
        huge = "x" * (CHUNK_SIZE * (MAX_CHUNKS_PER_ENTRY + 5))
        chunks = embed_chunks("Blob", "Huge blob", huge)
        assert len(chunks) == MAX_CHUNKS_PER_ENTRY
        assert chunks[0] == embed_text("Blob", "Huge blob", huge)


class TestChunkedScoringAndDuplicates:
    def test_semantic_search_finds_tail_concept_past_2000_chars(self, tmp_path: Path) -> None:
        emb = TopicEmbedder()
        svc = _make_service(tmp_path, embedder=emb)
        try:
            svc.put(
                title="Long incident report",
                summary="Summary of routine platform maintenance.",
                body=_long_body("Final root cause involved falcon and kestrel telemetry."),
            )
            svc.put(
                title="Database tuning",
                summary="Postgres and PgBouncer pool configuration.",
                body="Primary database runs managed Postgres with SQL connection pooling.",
            )
            assert svc.profile.index.count(emb.model_id) == 2
            assert svc.profile.index.vector_count(emb.model_id) >= 4

            # Query uses paraphrase ("raptor hawk") sharing NO lexical tokens with the entry,
            # so only the tail chunk vector can match "falcon kestrel".
            results = svc.search("raptor hawk")
            assert results[0]["key"] == "long-incident-report"
            assert "semantic" in results[0]["match"]
        finally:
            svc.close()

    def test_duplicate_detection_uses_head_vectors_only(self, tmp_path: Path) -> None:
        emb = TopicEmbedder()
        svc = _make_service(tmp_path, embedder=emb)
        try:
            # Two entries with totally different heads (postgres vs kubernetes) but identical
            # billing-heavy tails past 2000 chars: must NOT be flagged as duplicates.
            svc.put(
                title="Postgres architecture",
                summary="Managed Postgres database SQL pool.",
                body=_long_body("Billing invoice stripe payout audit appendix."),
            )
            res = svc.put(
                title="Kubernetes topology",
                summary="Cluster helm ingress routing setup.",
                body=_long_body("Billing invoice stripe payout audit appendix."),
            )
            assert "possible_duplicates" not in res
            assert svc.duplicate_pairs() == []
        finally:
            svc.close()


class TestChunkReconciliationAndRemoval:
    def test_body_edit_reembeds_only_edited_entry_and_prunes_shrunk_chunks(
        self, tmp_path: Path
    ) -> None:
        emb = TopicEmbedder()
        svc = _make_service(tmp_path, embedder=emb)
        try:
            svc.put(
                title="Unchanged long doc",
                summary="Stable reference guide.",
                body=_long_body("Static postgres sql appendix."),
            )
            svc.put(
                title="Editable long doc",
                summary="Mutable operations manual.",
                body=_long_body("Original billing stripe appendix."),
            )
            initial_chunks = svc.profile.index.vector_count(emb.model_id)
            assert initial_chunks >= 6

            # Direct store edit behind MemoryService's back: edit only the second entry.
            emb.embedded_texts.clear()
            svc.profile.store.put(
                "editable-long-doc",
                title="Editable long doc",
                summary="Mutable operations manual.",
                category="general",
                body=_long_body("Updated falcon kestrel appendix."),
            )
            reembedded = svc.sync_index(svc.profile)
            assert reembedded == 1
            assert all("Unchanged long doc" not in t for t in emb.embedded_texts)
            assert any("Editable long doc" in t for t in emb.embedded_texts)

            # Shrinking the edited entry to a short body prunes its extra chunks.
            emb.embedded_texts.clear()
            svc.profile.store.put(
                "editable-long-doc",
                title="Editable long doc",
                summary="Mutable operations manual.",
                category="general",
                body="Now a short single-chunk body about kubernetes.",
            )
            assert svc.sync_index(svc.profile) == 1
            assert len(emb.embedded_texts) == 1
            assert svc.profile.index.vector_count(emb.model_id) < initial_chunks
            assert svc.profile.index.count(emb.model_id) == 2
        finally:
            svc.close()

    def test_remove_deletes_all_chunks(self, tmp_path: Path) -> None:
        emb = TopicEmbedder()
        svc = _make_service(tmp_path, embedder=emb)
        try:
            svc.put(
                title="Removable long doc",
                summary="Temporary long guide.",
                body=_long_body("Tail content about stripe invoices."),
            )
            assert svc.profile.index.vector_count(emb.model_id) >= 3
            assert svc.remove("removable-long-doc") == PROFILE
            assert svc.profile.index.count() == 0
            assert svc.profile.index.vector_count() == 0
        finally:
            svc.close()


class TestVectorsSchemaMigration:
    def _create_legacy_index_db(self, db_path: Path) -> None:
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE vectors (
                key          TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                model        TEXT NOT NULL,
                vec          BLOB NOT NULL
            );
            CREATE TABLE usage (
                key          TEXT PRIMARY KEY,
                first_seen   REAL NOT NULL,
                recall_count INTEGER NOT NULL DEFAULT 0,
                hit_count    INTEGER NOT NULL DEFAULT 0,
                last_used    REAL
            );
            CREATE TABLE changes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            REAL NOT NULL,
                key           TEXT NOT NULL,
                action        TEXT NOT NULL,
                session_id    TEXT,
                prev_title    TEXT,
                prev_summary  TEXT,
                prev_category TEXT,
                prev_body     TEXT
            );
            CREATE TABLE inbox (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_ts      REAL NOT NULL,
                session_id      TEXT,
                title           TEXT NOT NULL,
                summary         TEXT NOT NULL,
                category        TEXT NOT NULL,
                body            TEXT NOT NULL,
                reason          TEXT,
                duplicates_json TEXT NOT NULL DEFAULT '[]',
                status          TEXT NOT NULL DEFAULT 'pending'
            );
            """
        )
        blob = array("f", [1.0, 0.0, 0.0, 0.0]).tobytes()
        conn.execute(
            "INSERT INTO vectors VALUES (?, ?, ?, ?)", ("legacy-doc", "oldhash", "topic-v1", blob)
        )
        conn.execute(
            "INSERT INTO usage VALUES (?, ?, ?, ?, ?)", ("legacy-doc", 1000.0, 2, 5, 1005.0)
        )
        conn.execute(
            "INSERT INTO changes (ts, key, action, session_id) VALUES (?, ?, ?, ?)",
            (1000.0, "legacy-doc", "create", "sess-1"),
        )
        conn.execute(
            "INSERT INTO inbox (created_ts, session_id, title, summary, category, body) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (1000.0, "sess-1", "Pending", "Pending summary", "general", "Pending body content."),
        )
        conn.commit()
        conn.close()

    def test_read_only_open_handles_unmigrated_legacy_schema_without_writing(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "memory.index.db"
        self._create_legacy_index_db(db_path)
        before_bytes = db_path.read_bytes()

        ro = MemoryIndex(db_path, read_only=True)
        try:
            assert ro.count("topic-v1") == 1
            assert "legacy-doc" in ro.vectors("topic-v1")
            assert ro.usage()["legacy-doc"]["hit_count"] == 5
        finally:
            ro.close()
        assert db_path.read_bytes() == before_bytes

    def test_rw_open_migrates_legacy_schema_preserves_other_tables_and_rebuilds(
        self, tmp_path: Path
    ) -> None:
        d = tmp_path / PROFILE
        d.mkdir(parents=True, exist_ok=True)
        db_path = d / "memory.index.db"
        self._create_legacy_index_db(db_path)

        store = get_store("sqlite", d / "memory.db")
        store.put(
            "legacy-doc",
            title="Legacy doc",
            summary="Pre-migration long entry.",
            category="general",
            body=_long_body("Tail note mentions falcon and kestrel telemetry."),
        )

        emb = TopicEmbedder()
        svc = MemoryService(
            OrderedDict([(PROFILE, Scope(PROFILE, store, MemoryIndex(db_path)))]), embedder=emb
        )
        try:
            idx = svc.profile.index
            cols = {r[1] for r in idx._conn.execute("PRAGMA table_info(vectors)").fetchall()}
            assert "chunk_no" in cols
            # Non-vector tables survived migration intact.
            assert idx.usage()["legacy-doc"]["hit_count"] == 5
            assert idx.count_changes() == 1
            assert idx.count_inbox() == 1
            # Old vectors were dropped and lazy sync automatically rebuilds chunked vectors.
            assert idx.count() == 0
            assert svc.sync_index(svc.profile) == 1
            assert idx.count(emb.model_id) == 1
            assert idx.vector_count(emb.model_id) >= 3
            assert svc.search("raptor hawk")[0]["key"] == "legacy-doc"
        finally:
            svc.close()
