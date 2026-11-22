"""Tests for per-scope lexical caching: ranking equivalence, invalidation, and scaling."""

from __future__ import annotations

import os
import sqlite3
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

from bench import recall
from nachos_core.index import MemoryIndex
from nachos_core.lexical import idf_weights, query_terms, score_document, tokens
from nachos_core.memory import PROFILE, SHARED, MemoryService, Scope
from nachos_core.store import Entry, MemoryStore, get_store

_TOOLS_DIR = str(Path(__file__).resolve().parents[1] / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import import_memories  # noqa: E402


def _uncached_lexical(scope: Scope, query: str) -> list[str]:
    """Reference implementation matching the pre-cache MemoryService._lexical path."""
    raw, terms = query_terms(query)
    candidates = set(scope.store.search(query))
    for term in sorted(raw | set(terms)):
        candidates.update(scope.store.search(term))
    if not candidates:
        return []
    labels = {k: (t, s) for k, t, s, _c in scope.store.list()}
    bodies = scope.store.bodies()
    label_words = {k: tokens(f"{t} {sm}") for k, (t, sm) in labels.items()}
    doc_words = {k: label_words[k] | tokens(bodies.get(k) or "") for k in labels}
    weights = idf_weights(terms, doc_words)
    needle = query.lower()
    scored = []
    for key in candidates & doc_words.keys():
        body = bodies.get(key) or ""
        haystack = f"{labels[key][0]}\n{labels[key][1]}\n{body}".lower()
        score = score_document(weights, doc_words[key], label_words[key], needle in haystack)
        if score > 0:
            scored.append((score, key))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [key for _score, key in scored]


def _make_service(tmp_path: Path, *, kind: str = "sqlite", shared: bool = False) -> MemoryService:
    scopes: OrderedDict[str, Scope] = OrderedDict()
    for name in (PROFILE, SHARED) if shared else (PROFILE,):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("memory.md" if kind == "flatfile" else "memory.db")
        scopes[name] = Scope(
            name, get_store(kind, path), MemoryIndex(directory / "memory.index.db")
        )
    return MemoryService(scopes)


def _keys(svc: MemoryService, query: str, *, limit: int = 10) -> list[str]:
    return [r["key"] for r in svc.search(query, limit=limit)]


class TestRankingEquivalence:
    def test_full_bench_query_set_matches_uncached_ranking(self, tmp_path: Path) -> None:
        memories, queries = recall.load_dataset()
        svc = recall.build_service(tmp_path, None)
        try:
            recall.load_memories(svc, memories)
            # Warm cache once and confirm the exact same cached index is reused for all queries.
            first_cached = svc._lexical(svc.profile, queries[0]["query"])
            cached_index = svc._lexical_cache[PROFILE][1]
            assert first_cached == _uncached_lexical(svc.profile, queries[0]["query"])

            for q in queries[1:]:
                expected = _uncached_lexical(svc.profile, q["query"])
                actual = svc._lexical(svc.profile, q["query"])
                assert actual == expected, (q["id"], q["query"])

            assert svc._lexical_cache[PROFILE][1] is cached_index
        finally:
            svc.close()

    @pytest.mark.parametrize("kind", ["sqlite", "flatfile"])
    def test_ies_plural_stemming_matches_uncached_ranking(self, tmp_path: Path, kind: str) -> None:
        svc = _make_service(tmp_path, kind=kind)
        try:
            svc.put(
                title="Retention rules",
                summary="Retention rules for audit archives.",
                body="All regional offices enforce strict retention policies every quarter.",
            )
            svc.put(
                title="Security policy",
                summary="Access control policy for production.",
                body="Production access follows the least-privilege policy document.",
            )
            for query in ("policy", "policies", "policy retention", "policies quarterly"):
                assert svc._lexical(svc.profile, query) == _uncached_lexical(svc.profile, query)
        finally:
            svc.close()

    def test_cached_index_stores_term_sets_and_doc_frequencies(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            svc.put(
                title="Deploy pipelines",
                summary="Automated canary rollouts",
                body="Canary deployments validate metrics before promoting containers.",
            )
            svc.put(
                title="Container registry",
                summary="Private image mirror",
                body="Containers are mirrored into the internal registry nightly.",
            )
            svc.search("canary")
            index = svc._lexical_cache[PROFILE][1]
            doc = index.docs["deploy-pipelines"]
            assert "pipeline" in doc.title_words
            assert "canary" in doc.summary_words
            assert "container" in doc.body_words
            assert index.doc_freq["container"] == 2
            assert index.doc_freq["canary"] == 1
        finally:
            svc.close()


class TestCacheInvalidation:
    @pytest.mark.parametrize("kind", ["sqlite", "flatfile"])
    def test_cache_reused_without_store_reads(
        self, tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc = _make_service(tmp_path, kind=kind)
        try:
            svc.put(
                title="Postgres cluster",
                summary="Managed Postgres 16 primary and replicas",
                body="Primary database uses PgBouncer in transaction pooling mode.",
            )
            svc.search("postgres")

            calls = {"list": 0, "bodies": 0}
            orig_list = svc.profile.store.list
            orig_bodies = svc.profile.store.bodies

            def _counting_list():
                calls["list"] += 1
                return orig_list()

            def _counting_bodies():
                calls["bodies"] += 1
                return orig_bodies()

            monkeypatch.setattr(svc.profile.store, "list", _counting_list)
            monkeypatch.setattr(svc.profile.store, "bodies", _counting_bodies)

            assert _keys(svc, "pgbouncer") == ["postgres-cluster"]
            assert _keys(svc, "replicas") == ["postgres-cluster"]
            assert calls == {"list": 0, "bodies": 0}
        finally:
            svc.close()

    @pytest.mark.parametrize("kind", ["sqlite", "flatfile"])
    def test_invalidation_after_put_create_and_update(self, tmp_path: Path, kind: str) -> None:
        svc = _make_service(tmp_path, kind=kind)
        try:
            svc.put(
                title="Staging deploy",
                summary="Blue green deploy script",
                body="Staging deploys run the blue green rollout script.",
            )
            assert _keys(svc, "staging") == ["staging-deploy"]
            assert _keys(svc, "grafana") == []

            # Create invalidates cache.
            svc.put(
                title="Monitoring stack",
                summary="Prometheus and Grafana dashboards",
                body="Alerting dashboards live in Grafana backed by Prometheus.",
            )
            assert _keys(svc, "grafana") == ["monitoring-stack"]

            # Update invalidates cache (old body token disappears, new token appears).
            svc.put(
                title="Monitoring stack",
                summary="Datadog metrics and traces",
                body="Alerting dashboards migrated entirely to Datadog monitors.",
            )
            assert _keys(svc, "grafana") == []
            assert _keys(svc, "datadog") == ["monitoring-stack"]
        finally:
            svc.close()

    @pytest.mark.parametrize("kind", ["sqlite", "flatfile"])
    def test_invalidation_after_remove(self, tmp_path: Path, kind: str) -> None:
        svc = _make_service(tmp_path, kind=kind)
        try:
            svc.put(
                title="Redis cache",
                summary="In-memory session cache on port 6379",
                body="Redis stores ephemeral rate-limit buckets and session tokens.",
            )
            assert _keys(svc, "redis") == ["redis-cache"]

            assert svc.remove("redis-cache") == PROFILE
            assert _keys(svc, "redis") == []
        finally:
            svc.close()

    @pytest.mark.parametrize("kind", ["sqlite", "flatfile"])
    def test_invalidation_after_revert(self, tmp_path: Path, kind: str) -> None:
        svc = _make_service(tmp_path, kind=kind)
        try:
            svc.put(
                title="Queue broker",
                summary="RabbitMQ message broker configuration",
                body="Background jobs publish to RabbitMQ exchanges with dead-letter queues.",
            )
            svc.put(
                title="Queue broker",
                summary="Kafka event streaming cluster",
                body="Background jobs publish to Kafka topics with three partitions.",
            )
            assert _keys(svc, "kafka") == ["queue-broker"]
            assert _keys(svc, "rabbitmq") == []

            # Revert update -> restores RabbitMQ, drops Kafka.
            update_id = svc.history(key="queue-broker")[0]["id"]
            svc.revert(update_id, PROFILE)
            assert _keys(svc, "rabbitmq") == ["queue-broker"]
            assert _keys(svc, "kafka") == []

            # Remove then revert remove -> entry reappears.
            svc.remove("queue-broker")
            assert _keys(svc, "rabbitmq") == []
            remove_id = svc.history(key="queue-broker")[0]["id"]
            svc.revert(remove_id, PROFILE)
            assert _keys(svc, "rabbitmq") == ["queue-broker"]
        finally:
            svc.close()

    def test_invalidation_after_import(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path)
        try:
            svc.put(
                title="Existing note",
                summary="Initial workspace note",
                body="Baseline content before running memory import.",
            )
            assert _keys(svc, "kestrel") == []

            items = [
                import_memories.Item(
                    title="Kestrel build farm",
                    body="Kestrel build farm runs Linux ARM64 runners for nightly wheels.",
                    category="infra",
                    origin="MEMORY.md",
                )
            ]
            report = import_memories.run_import(
                svc, items, run=True, overwrite=False, category=None
            )
            assert len(report["imported"]) == 1
            assert _keys(svc, "kestrel") == ["kestrel-build-farm"]
        finally:
            svc.close()

    def test_invalidation_after_external_flatfile_edit(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path, kind="flatfile")
        md_path = tmp_path / PROFILE / "memory.md"
        try:
            svc.put(
                title="VPN gateway",
                summary="WireGuard gateway for internal hosts",
                body="Connect to internal hosts through the WireGuard bastion gateway.",
            )
            assert _keys(svc, "wireguard") == ["vpn-gateway"]
            assert _keys(svc, "tailscale") == []

            # Simulate an external hand edit modifying the entry and adding a new one.
            updated = (
                "# general\n\n"
                "## VPN gateway\n"
                "Tailscale mesh replaces legacy tunnels.\n\n"
                "Connect to internal hosts through the Tailscale mesh network.\n\n"
                "## Backup schedule\n"
                "Nightly borgbackup snapshots to offsite vault.\n"
            )
            md_path.write_text(updated, encoding="utf-8")
            st = md_path.stat()
            os.utime(md_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

            assert _keys(svc, "wireguard") == []
            assert _keys(svc, "tailscale") == ["vpn-gateway"]
            assert _keys(svc, "borgbackup") == ["backup-schedule"]

            # External deletion also invalidates the cache.
            md_path.unlink()
            assert _keys(svc, "tailscale") == []
        finally:
            svc.close()

    def test_invalidation_after_external_sqlite_edit(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path, kind="sqlite")
        db_path = tmp_path / PROFILE / "memory.db"
        try:
            svc.put(
                title="Search backend",
                summary="Legacy Elasticsearch cluster",
                body="Full-text logs are indexed in Elasticsearch cluster alpha.",
            )
            assert _keys(svc, "elasticsearch") == ["search-backend"]
            assert _keys(svc, "opensearch") == []

            # Another process / connection updates and inserts directly in SQLite.
            ext = sqlite3.connect(str(db_path))
            try:
                ext.execute(
                    "UPDATE entries SET summary = ?, body = ? WHERE key = ?",
                    (
                        "Migrated OpenSearch cluster",
                        "Full-text logs are indexed in OpenSearch cluster beta.",
                        "search-backend",
                    ),
                )
                ext.execute(
                    "INSERT INTO entries (key, title, summary, category, body) VALUES (?, ?, ?, ?, ?)",
                    (
                        "external-note",
                        "External note",
                        "Added by another process",
                        "general",
                        "External writer added ClickHouse telemetry pipeline details.",
                    ),
                )
                ext.commit()
            finally:
                ext.close()

            assert _keys(svc, "elasticsearch") == []
            assert _keys(svc, "opensearch") == ["search-backend"]
            assert _keys(svc, "clickhouse") == ["external-note"]
        finally:
            svc.close()

    def test_scope_invalidation_is_isolated(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path, shared=True)
        try:
            svc.put(
                title="Profile tool",
                summary="Local profile helper",
                body="Profile scope uses ripgrep for code search.",
                scope=PROFILE,
            )
            svc.put(
                title="Shared tool",
                summary="Machine-wide shared helper",
                body="Shared scope uses fd-find for file lookup.",
                scope=SHARED,
            )
            svc.search("ripgrep")
            shared_index = svc._lexical_cache[SHARED][1]

            svc.put(
                title="Profile editor",
                summary="Local editor preference",
                body="Profile scope uses helix editor for quick edits.",
                scope=PROFILE,
            )
            assert PROFILE not in svc._lexical_cache
            assert svc._lexical_cache[SHARED][1] is shared_index
            assert _keys(svc, "helix") == ["profile-editor"]
            assert svc._lexical_cache[SHARED][1] is shared_index
        finally:
            svc.close()

    def test_store_without_fingerprint_skips_caching(self, tmp_path: Path) -> None:
        class _DictStore(MemoryStore):
            def __init__(self) -> None:
                self._rows: dict[str, tuple[str, str, str, str]] = {}

            def list(self) -> list[Entry]:
                return [(k, t, s, c) for k, (t, s, c, _b) in sorted(self._rows.items())]

            def get(self, key: str) -> str | None:
                row = self._rows.get(key)
                return row[3] if row else None

            def search(self, query: str) -> list[str]:
                q = query.strip().lower()
                return [
                    k for k, (t, s, _c, b) in self._rows.items() if q in f"{t}\n{s}\n{b}".lower()
                ]

            def put(self, key: str, *, title: str, summary: str, category: str, body: str) -> None:
                self._rows[key] = (title, summary, category, body)

            def remove(self, key: str) -> None:
                self._rows.pop(key, None)

        store = _DictStore()
        scope = Scope(PROFILE, store, MemoryIndex(tmp_path / "memory.index.db"))
        svc = MemoryService(OrderedDict([(PROFILE, scope)]))
        try:
            store.put(
                "one", title="One", summary="Alpha entry", category="general", body="Alpha details"
            )
            assert _keys(svc, "alpha") == ["one"]
            assert PROFILE not in svc._lexical_cache

            store.put(
                "two", title="Two", summary="Beta entry", category="general", body="Beta details"
            )
            assert _keys(svc, "beta") == ["two"]
        finally:
            svc.close()


class TestHydrateBatching:
    def test_search_records_hits_across_results_and_scopes(self, tmp_path: Path) -> None:
        svc = _make_service(tmp_path, shared=True)
        try:
            svc.put(
                title="Deploy alpha",
                summary="Kubernetes rollout alpha",
                body="Kubernetes deployment guide for alpha cluster.",
                scope=PROFILE,
            )
            svc.put(
                title="Deploy beta",
                summary="Kubernetes rollout beta",
                body="Kubernetes deployment guide for beta cluster.",
                scope=PROFILE,
            )
            svc.put(
                title="Deploy shared",
                summary="Kubernetes rollout shared",
                body="Kubernetes deployment guide for shared cluster.",
                scope=SHARED,
            )

            results = svc.search("kubernetes")
            assert len(results) == 3

            prof_usage = svc.scopes[PROFILE].index.usage()
            shared_usage = svc.scopes[SHARED].index.usage()
            assert prof_usage["deploy-alpha"]["hit_count"] == 1
            assert prof_usage["deploy-beta"]["hit_count"] == 1
            assert shared_usage["deploy-shared"]["hit_count"] == 1
        finally:
            svc.close()
