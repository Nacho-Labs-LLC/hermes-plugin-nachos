"""Cross-feature integration tests for Round 4 on naterchrdsn/memory-next."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

from nachos_core.embeddings import Embedder
from nachos_core.index import MemoryIndex
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.reconciliation.acquire import SourceSpec
from nachos_core.reconciliation.apply import apply_plan
from nachos_core.reconciliation.approval import approve_proposals
from nachos_core.reconciliation.boundary import SourceBoundary
from nachos_core.reconciliation.export import export_plan
from nachos_core.reconciliation.findings import generate_findings
from nachos_core.reconciliation.inventory import build_inventory
from nachos_core.reconciliation.packets import build_packet
from nachos_core.reconciliation.review import new_session
from nachos_core.reconciliation.signals import collect_signals
from nachos_core.store import SqliteStore
from nachos_hermes import maintenance

AT = "2026-11-22T12:00:00+00:00"
DB = "nachos/memory.db"


class _SwitchableEmbedder(Embedder):
    def __init__(self, model_id: str, dim: int) -> None:
        self.model_id = model_id
        self.dimension = dim
        self.calls: list[list[str]] = []

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for idx, _t in enumerate(texts):
            ang = 0.2 * (idx + 1)
            vec = [0.0] * self.dimension
            vec[0] = math.cos(ang)
            vec[1] = math.sin(ang)
            out.append(vec)
        return out


def _open_service(home: Path, embedder: Embedder | None = None) -> MemoryService:
    (home / "nachos").mkdir(parents=True, exist_ok=True)
    scope = Scope(
        PROFILE,
        SqliteStore(home / DB),
        MemoryIndex(home / "nachos" / "memory.index.db"),
    )
    return MemoryService(OrderedDict([(PROFILE, scope)]), embedder=embedder)


def _approve_all(packet: dict[str, Any]) -> dict[str, Any]:
    session = new_session(packet)
    ids = [r["proposal_id"] for r in session["reviews"]]
    approved = approve_proposals(session, ids, observed_at=AT)
    return export_plan(approved)


def test_provenance_and_links_survive_reconcile_merge_and_supersede(tmp_path: Path) -> None:
    svc = _open_service(tmp_path)
    try:
        svc.put(
            title="Postgres backup schedule",
            summary="Nightly pg_dump at 02:00 to s3",
            category="ops",
            body="Nightly pg_dump at 02:00 UTC to the s3 backups bucket.\nRetention is 30 days.",
            key="pg-backup",
            provenance={
                "source": "capture",
                "session_id": "sess-1",
                "checkpoint_id": "cp-1",
                "message_index": 4,
                "quote": "Nightly pg_dump runs at 02:00 UTC to s3 backups bucket.",
            },
        )
        svc.put(
            title="Postgres backup schedule",
            summary="Nightly pg_dump at 02:00 to s3",
            category="ops",
            body="Nightly pg_dump at 02:00 UTC to the s3 backups bucket.",
            key="pg-backup-copy",
            provenance={
                "source": "capture",
                "session_id": "sess-2",
                "checkpoint_id": "cp-2",
                "message_index": 7,
                "quote": "Backup job writes to s3 backups bucket every night.",
            },
        )
        svc.put(
            title="Disaster recovery runbook",
            summary="Restore database from nightly backup",
            category="ops",
            body="Follow [[pg-backup]] when restoring primary Postgres cluster.",
            key="dr-runbook",
        )
        svc.put(
            title="Legacy cache note",
            summary="Mentions missing entry",
            category="ops",
            body="See [[nonexistent-service]] for legacy cache TTLs and sizing.",
            key="legacy-cache",
        )
        # Ensure deterministic timestamps: pg-backup-copy is newer than pg-backup.
        conn = svc.profile.store._conn
        conn.execute("UPDATE entries SET updated_at = 100.0 WHERE key = 'pg-backup'")
        conn.execute("UPDATE entries SET updated_at = 200.0 WHERE key = 'pg-backup-copy'")
        conn.commit()

        signals = collect_signals(svc)
        assert any(
            b["key"] == "legacy-cache" and b["target"] == "nonexistent-service"
            for b in signals.broken_links
        )
        assert "pg-backup-copy" in signals.provenance

        boundary = SourceBoundary(tmp_path, (DB,))
        profile = {"id": "p", "root": str(tmp_path)}
        spec = SourceSpec("sqlite")
        inventory = build_inventory(boundary, profile, spec)
        gen = generate_findings(inventory, signals, observed_at=AT)

        # Findings include provenance source in evidence AND broken-link rewrite proposal.
        merge_finding = next(f for f in gen.findings if f["type"] == "duplicate")
        assert any("sess-2" in ev["observation"] for ev in merge_finding["evidence"])
        broken_prop = next(
            p
            for p in gen.proposals
            if p["kind"] == "rewrite" and p["changes"][0]["before"]["key"] == "legacy-cache"
        )
        assert "[[nonexistent-service]]" not in broken_prop["changes"][0]["after"]["body"]

        packet = build_packet(inventory, findings=gen.findings, proposals=gen.proposals)
        plan = _approve_all(packet)
        report = apply_plan(plan, svc, boundary, profile, spec, run=True, allow_remove=True)
        assert report["status"] == "applied"

        # 1. Capture provenance survives reconcile merge on the survivor.
        survivor = svc.get_entry("pg-backup-copy")
        assert survivor is not None
        assert survivor["source"]["via"] == "capture"
        assert survivor["source"]["session"] == "sess-2"
        assert "s3 backups bucket" in survivor["source"]["evidence"]

        # 2. Incoming link [[pg-backup]] was rewritten to [[pg-backup-copy]] (zero broken links!).
        runbook = svc.get_entry("dr-runbook")
        assert runbook is not None
        assert "[[pg-backup-copy]]" in runbook["body"]
        assert svc.link_report()["broken_links"] == []

        # 3. Provenance survives supersede AND links follow supersedes to active successor.
        svc.put(
            title="Postgres backup v2",
            summary="Hourly WAL archiving to GCS",
            category="ops",
            body="WAL archiving runs continuously to GCS bucket.",
            key="pg-backup-v2",
            supersedes="pg-backup-copy",
            provenance={
                "source": "capture",
                "session_id": "sess-3",
                "quote": "Moved backups to GCS.",
            },
        )
        old_after_sup = svc.get_entry("pg-backup-copy")
        assert old_after_sup is not None
        assert old_after_sup["source"]["session"] == "sess-2"
        runbook_after_sup = svc.get_entry("dr-runbook")
        assert runbook_after_sup is not None
        assert [lnk["key"] for lnk in runbook_after_sup["links"]] == ["pg-backup-v2"]
    finally:
        svc.close()


def test_abstention_multi_query_related_and_entity_filter_work_together(tmp_path: Path) -> None:
    svc = _open_service(tmp_path)
    try:
        svc.put(
            title="Harbor billing service",
            summary="Go service issuing invoices on harbor.internal",
            body="Harbor runs on harbor.internal and depends on [[pg-primary]] for invoice storage.",
            key="harbor-billing",
            provenance={
                "source": "capture",
                "session_id": "sess-h",
                "quote": "Harbor runs on harbor.internal.",
            },
        )
        svc.put(
            title="Postgres primary",
            summary="Primary database cluster",
            body="Postgres 16 primary stores billing and ledger tables.",
            key="pg-primary",
        )
        svc.put(
            title="Lumen dashboard",
            summary="React dashboard for metrics",
            body="Lumen renders weekly charts on lumen.internal.",
            key="lumen-dash",
        )

        # Multi-query + entity filter + related expansion + confidence + provenance!
        res = svc.search_with_confidence(
            queries=["invoice processor", "harbor invoices"],
            entity="harbor.internal",
            mode="hint",
        )
        assert res["confident"] is True
        assert res["entity"] == "harbor.internal"
        assert len(res["queries"]) == 2
        assert [r["key"] for r in res["results"]] == ["harbor-billing"]
        assert res["results"][0]["source"]["session"] == "sess-h"
        assert [rel["key"] for rel in res["results"][0]["related"]] == ["pg-primary"]

        # Entity-only lookup returns confidence fields + related too.
        ent = svc.search_with_confidence(entity="harbor.internal", mode="hint")
        assert ent["confident"] is True
        assert ent["entity"] == "harbor.internal"
        assert [r["key"] for r in ent["results"]] == ["harbor-billing"]
        assert [rel["key"] for rel in ent["results"][0]["related"]] == ["pg-primary"]
    finally:
        svc.close()


def test_model_switch_reembeds_chunked_vectors_without_affecting_links_or_provenance(
    tmp_path: Path,
) -> None:
    long_body = "Harbor runs on harbor.internal and links to [[pg-primary]].\n\n" + (
        "Detailed operational runbook paragraph about billing retries and settlement windows. " * 25
    )
    emb_a = _SwitchableEmbedder("test:model-a", dim=4)
    svc_a = _open_service(tmp_path, embedder=emb_a)
    try:
        svc_a.put(
            title="Harbor runbook",
            summary="Long billing runbook",
            body=long_body,
            key="harbor-runbook",
            provenance={
                "source": "capture",
                "session_id": "sess-m",
                "quote": "Harbor billing runbook.",
            },
        )
        svc_a.put(
            title="Postgres primary",
            summary="Primary DB",
            body="Postgres 16 primary cluster on db-01.internal.",
            key="pg-primary",
        )
        svc_a.sync_index(svc_a.profile)
        chunks_a = svc_a.profile.index._conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE key = 'harbor-runbook' AND model = 'test:model-a'"
        ).fetchone()[0]
        assert chunks_a >= 2  # head + body chunks
    finally:
        svc_a.close()

    # Switch to model-b (different dimension!) and re-embed.
    emb_b = _SwitchableEmbedder("test:model-b", dim=8)
    svc_b = _open_service(tmp_path, embedder=emb_b)
    try:
        svc_b.sync_index(svc_b.profile)
        assert not svc_b.reindexing
        assert svc_b.profile.index.model_counts() == {"test:model-b": 2}
        chunks_b = svc_b.profile.index._conn.execute(
            "SELECT COUNT(*) FROM vectors WHERE key = 'harbor-runbook' AND model = 'test:model-b'"
        ).fetchone()[0]
        assert chunks_b == chunks_a
        # Links, entities, and provenance are completely intact after model switch!
        entry = svc_b.get_entry("harbor-runbook")
        assert entry is not None
        assert [lnk["key"] for lnk in entry["links"]] == ["pg-primary"]
        assert entry["source"]["session"] == "sess-m"
        assert [r["key"] for r in svc_b.search(entity="harbor.internal")] == ["harbor-runbook"]
    finally:
        svc_b.close()


def test_maintain_report_includes_links_provenance_and_reconciliation_without_duplication(
    tmp_path: Path, capsys: Any
) -> None:
    svc = _open_service(tmp_path)
    try:
        svc.put(
            title="Deploy notes",
            summary="Deploy notes",
            body="Blue green deploys flip load balancer weights after smoke tests pass. See [[missing-doc]].",
            key="deploy-notes",
        )
    finally:
        svc.close()

    rc = maintenance.main(["--profile-home", str(tmp_path), "--json", "all"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    checks = report["checks"]
    assert {"links", "provenance", "reconciliation", "summaries", "duplicates"} <= checks.keys()
    assert len(checks) == len(set(checks))
    assert checks["links"]["broken_links"] == 1
    assert checks["provenance"]["available"] is True
    assert checks["reconciliation"]["count"] >= 1
    assert "by_kind" in checks["reconciliation"]
