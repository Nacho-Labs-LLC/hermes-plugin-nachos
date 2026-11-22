"""Tests for timestamps, store migrations, supersedes/revert, manifest hiding, recall filtering,
recency tie-break, put-time suggestion heuristic, and audit/maintain reporting.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import pytest

from nachos_core.index import MemoryIndex, index_path_for
from nachos_core.memory import PROFILE, SHARED, MemoryService, Scope
from nachos_core.quality import looks_like_update, qualifier_difference, update_signal
from nachos_core.ranking import update_threshold_for
from nachos_core.store import MDStore, SqliteStore, get_store
from nachos_core.toc import render_toc
from nachos_hermes import commands
from nachos_hermes.memory_provider import NachosMemoryProvider
from nachos_hermes.schemas import PUT_SCHEMA


def _make_service(tmp_path: Path, *, kind: str = "sqlite", shared: bool = False) -> MemoryService:
    ext = "memory.md" if kind == "flatfile" else "memory.db"
    names = (PROFILE, SHARED) if shared else (PROFILE,)
    return MemoryService(
        OrderedDict(
            (n, Scope(n, get_store(kind, tmp_path / n / ext), MemoryIndex(tmp_path / n / "idx.db")))
            for n in names
        )
    )


def _create_legacy_sqlite_store(db_path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE entries (
            key      TEXT PRIMARY KEY,
            title    TEXT NOT NULL,
            summary  TEXT NOT NULL,
            category TEXT NOT NULL,
            body     TEXT NOT NULL
        );
        CREATE INDEX idx_entries_category ON entries(category);
        """
    )
    conn.executemany("INSERT INTO entries VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def _create_legacy_index_without_prev_superseded(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE vectors (
            key TEXT PRIMARY KEY, content_hash TEXT NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL
        );
        CREATE TABLE usage (
            key TEXT PRIMARY KEY, first_seen REAL NOT NULL,
            recall_count INTEGER NOT NULL DEFAULT 0, hit_count INTEGER NOT NULL DEFAULT 0,
            last_used REAL
        );
        CREATE TABLE changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, key TEXT NOT NULL,
            action TEXT NOT NULL, session_id TEXT, prev_title TEXT, prev_summary TEXT,
            prev_category TEXT, prev_body TEXT
        );
        CREATE TABLE inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts REAL NOT NULL, session_id TEXT,
            title TEXT NOT NULL, summary TEXT NOT NULL, category TEXT NOT NULL, body TEXT NOT NULL,
            reason TEXT, duplicates_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'pending'
        );
        """
    )
    conn.execute(
        "INSERT INTO changes (ts, key, action, session_id, prev_title, prev_summary, prev_category, prev_body) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1_700_000_000.0, "from-changes", "create", "s1", None, None, None, None),
    )
    conn.execute(
        "INSERT INTO changes (ts, key, action, session_id, prev_title, prev_summary, prev_category, prev_body) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1_700_000_500.0, "from-changes", "update", "s2", "From changes", "s", "general", "b"),
    )
    conn.execute(
        "INSERT INTO usage (key, first_seen, recall_count, hit_count) VALUES (?, ?, 0, 0)",
        ("from-usage", 1_710_000_000.0),
    )
    conn.commit()
    conn.close()


class TestStoreMigrationsAndTimestamps:
    def test_sqlite_read_only_on_unmigrated_db_does_not_write(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        idx_path = index_path_for(db_path)
        _create_legacy_sqlite_store(
            db_path,
            [("from-changes", "From changes", "Summary line one", "general", "Full body content")],
        )
        _create_legacy_index_without_prev_superseded(idx_path)

        before_db = (db_path.stat().st_mtime_ns, db_path.read_bytes())
        before_idx = (idx_path.stat().st_mtime_ns, idx_path.read_bytes())

        store = SqliteStore(db_path, read_only=True)
        index = MemoryIndex(idx_path, read_only=True)
        svc = MemoryService(OrderedDict([(PROFILE, Scope(PROFILE, store, index))]))
        try:
            assert [k for k, *_ in store.list()] == ["from-changes"]
            meta = store.get_meta("from-changes")
            assert meta is not None
            assert meta.created_at is None
            assert meta.superseded_by is None
            assert index.changes()[0]["prev_superseded_by"] is None
            assert svc.superseded(30) == []
        finally:
            svc.close()

        assert (db_path.stat().st_mtime_ns, db_path.read_bytes()) == before_db
        assert (idx_path.stat().st_mtime_ns, idx_path.read_bytes()) == before_idx

    def test_sqlite_migration_backfills_from_changes_usage_and_now(self, tmp_path: Path) -> None:
        db_path = tmp_path / "memory.db"
        idx_path = index_path_for(db_path)
        _create_legacy_sqlite_store(
            db_path,
            [
                ("from-changes", "From changes", "Tracked in changes", "general", "Body one here."),
                ("from-usage", "From usage", "Tracked in usage", "general", "Body two here."),
                ("from-now", "From now", "Untracked entry", "general", "Body three here."),
            ],
        )
        _create_legacy_index_without_prev_superseded(idx_path)

        t_before = time.time()
        svc = MemoryService(
            OrderedDict([(PROFILE, Scope(PROFILE, SqliteStore(db_path), MemoryIndex(idx_path)))])
        )
        try:
            metas = svc.profile.store.all_meta()
            assert metas["from-changes"].created_at == pytest.approx(1_700_000_000.0)
            assert metas["from-changes"].updated_at == pytest.approx(1_700_000_500.0)
            assert metas["from-changes"].approx_timestamps is True

            assert metas["from-usage"].created_at == pytest.approx(1_710_000_000.0)
            assert metas["from-usage"].updated_at == pytest.approx(1_710_000_000.0)
            assert metas["from-usage"].approx_timestamps is True

            assert metas["from-now"].created_at is not None
            assert metas["from-now"].created_at >= t_before - 1.0
            assert metas["from-now"].approx_timestamps is True

            status = svc.status()["scopes"][PROFILE]
            assert status["approx_timestamps"] == 3

            # A newly written entry gets exact (non-approx) timestamps.
            svc.put(title="Fresh entry", body="Fresh entry created after migration.")
            fresh_meta = svc.profile.store.get_meta("fresh-entry")
            assert fresh_meta is not None
            assert fresh_meta.approx_timestamps is False
            assert fresh_meta.created_at == fresh_meta.updated_at

            # Updating preserves created_at and advances updated_at.
            created = fresh_meta.created_at
            time.sleep(0.01)
            svc.put(title="Fresh entry", body="Fresh entry updated with new details.")
            updated_meta = svc.profile.store.get_meta("fresh-entry")
            assert updated_meta is not None
            assert updated_meta.created_at == created
            assert updated_meta.updated_at is not None
            assert updated_meta.updated_at > created
        finally:
            svc.close()

    def test_flatfile_old_and_new_format_roundtrip(self, tmp_path: Path) -> None:
        md_path = tmp_path / "memory.md"
        legacy_md = (
            "# ops\n\n"
            "## Legacy entry\n"
            "Legacy summary line without comments.\n\n"
            "More details about the legacy entry.\n"
        )
        md_path.write_text(legacy_md, encoding="utf-8")
        before = (md_path.stat().st_mtime_ns, md_path.read_bytes())

        # Read-only open parses old format unchanged without writing.
        ro = MDStore(md_path, read_only=True)
        assert ro.list() == [
            ("legacy-entry", "Legacy entry", "Legacy summary line without comments.", "ops")
        ]
        assert ro.get_meta("legacy-entry") is not None
        assert ro.get_meta("legacy-entry").created_at is None
        assert (md_path.stat().st_mtime_ns, md_path.read_bytes()) == before

        # RW MemoryService backfills approximate timestamps and keeps body/summary clean.
        idx = MemoryIndex(index_path_for(md_path))
        svc = MemoryService(OrderedDict([(PROFILE, Scope(PROFILE, MDStore(md_path), idx))]))
        try:
            raw_text = md_path.read_text(encoding="utf-8")
            assert "<!-- created: ~" in raw_text
            assert "<!--" not in (svc.profile.store.get("legacy-entry") or "")
            assert svc.status()["scopes"][PROFILE]["approx_timestamps"] == 1

            svc.put(
                title="Current entry",
                summary="Replaces legacy entry with updated details.",
                body="Current entry replaces legacy entry completely.",
                category="ops",
                supersedes="legacy-entry",
            )
        finally:
            svc.close()

        # Re-open from disk and verify metadata round-trip + hand-edited short date parsing.
        reopened = MDStore(md_path)
        leg_meta = reopened.get_meta("legacy-entry")
        cur_meta = reopened.get_meta("current-entry")
        assert leg_meta is not None
        assert leg_meta.superseded_by == "current-entry"
        assert cur_meta is not None
        assert cur_meta.superseded_by is None
        assert cur_meta.approx_timestamps is False
        assert "<!--" not in (reopened.get("legacy-entry") or "")

        # Hand-edited short ISO date without time parses cleanly.
        hand_md = (
            "# general\n\n"
            "## Hand edited\n"
            "<!-- created: 2025-03-10 | updated: 2025-04-01 | superseded_by: current-entry -->\n"
            "Hand-edited entry summary line.\n"
        )
        md_path.write_text(hand_md, encoding="utf-8")
        hm = MDStore(md_path).get_meta("hand-edited")
        assert hm is not None
        assert hm.created_at is not None
        assert hm.updated_at is not None
        assert hm.updated_at > hm.created_at
        assert hm.superseded_by == "current-entry"


class TestSupersedesAndRevert:
    @pytest.mark.parametrize("kind", ["sqlite", "flatfile"])
    def test_supersede_hides_from_manifest_and_search_but_exact_key_returns_note(
        self, tmp_path: Path, kind: str
    ) -> None:
        svc = _make_service(tmp_path, kind=kind)
        try:
            svc.put(
                title="Deploy tooling (legacy)",
                summary="Deploys use Spinnaker pipelines",
                body="Deploys run through Spinnaker pipelines in ops.",
                category="ops",
            )
            res = svc.put(
                title="Deploy tooling (current)",
                summary="Deploys use the ferrule command; Spinnaker is retired",
                body="Deploys now use the ferrule CLI command; Spinnaker is retired.",
                category="ops",
                supersedes="deploy-tooling-legacy",
            )
            assert res["superseded"] == ["deploy-tooling-legacy"]

            # Old entry still exists on disk and exact fetch returns content + superseded_by note.
            entry = svc.get_entry("deploy-tooling-legacy")
            assert entry is not None
            assert "Spinnaker" in entry["body"]
            assert entry["superseded_by"] == "deploy-tooling-current"
            assert "deploy-tooling-current" in entry["note"]

            # Manifest hides superseded entry and render_toc shows count footer.
            manifest_keys = [k for k, *_ in svc.manifest_entries()]
            assert manifest_keys == ["deploy-tooling-current"]
            assert svc.superseded_count() == 1
            toc = render_toc(svc.manifest_entries(), superseded_count=svc.superseded_count())
            assert "Deploy tooling (current)" in toc
            assert "Deploy tooling (legacy)" not in toc
            assert "1 superseded entry hidden." in toc

            # Default search redirects superseded hits to their current successor;
            # include_superseded=True returns the superseded entry directly.
            default_keys = [r["key"] for r in svc.search("spinnaker deploy pipelines")]
            assert "deploy-tooling-legacy" not in default_keys
            assert default_keys[0] == "deploy-tooling-current"
            assert default_keys.count("deploy-tooling-current") == 1
            all_hits = svc.search("spinnaker deploy pipelines", include_superseded=True)
            by_key = {r["key"]: r for r in all_hits}
            assert "deploy-tooling-legacy" in by_key
            assert by_key["deploy-tooling-legacy"]["superseded_by"] == "deploy-tooling-current"
        finally:
            svc.close()

    def test_search_redirect_follows_chains_avoids_cycles_and_deduplicates(
        self, tmp_path: Path
    ) -> None:
        svc = _make_service(tmp_path)
        try:
            # Chain A -> B -> C where C no longer mentions A's old wording.
            svc.put(
                title="Payment gateway v1",
                summary="Billing runs through OldPay processor",
                body="All card charges route through OldPay merchant accounts.",
            )
            svc.put(
                title="Payment gateway v2",
                summary="Billing migrated to MidPay gateway",
                body="Card billing switched to MidPay API endpoints.",
                supersedes="payment-gateway-v1",
            )
            svc.put(
                title="Billing processor",
                summary="Payments settle via Adyen platform",
                body="All merchant settlements now run on Adyen.",
                supersedes="payment-gateway-v2",
            )

            # Query using only v1's old wording redirects across the chain to the current entry.
            old_word_hits = svc.search("OldPay merchant")
            assert [r["key"] for r in old_word_hits] == ["billing-processor"]

            # Query matching all three entries returns the active successor exactly once.
            broad_hits = svc.search("OldPay MidPay Adyen")
            assert [r["key"] for r in broad_hits] == ["billing-processor"]

            # include_superseded=True skips redirect and returns archived entries directly.
            raw_hits = [r["key"] for r in svc.search("OldPay merchant", include_superseded=True)]
            assert "payment-gateway-v1" in raw_hits

            # Cycle (X -> Y -> X) terminates cleanly and drops both from default search.
            svc.put(title="Cycle alpha", body="Zirconium cycle alpha entry text.")
            svc.put(
                title="Cycle beta",
                body="Zirconium cycle beta entry text.",
                supersedes="cycle-alpha",
            )
            svc.profile.store.set_superseded_by("cycle-beta", "cycle-alpha")
            assert svc.search("zirconium cycle") == []

            # Dangling superseded_by (pointing to a removed key) drops cleanly without error.
            svc.put(title="Orphan old", body="Yttrium orphan old entry text.")
            svc.put(
                title="Orphan next",
                body="Yttrium orphan next entry text.",
                supersedes="orphan-old",
            )
            svc.remove("orphan-next")
            assert svc.search("yttrium orphan") == []
        finally:
            svc.close()

    @pytest.mark.parametrize("kind", ["sqlite", "flatfile"])
    def test_supersede_logged_and_revert_round_trip(self, tmp_path: Path, kind: str) -> None:
        svc = _make_service(tmp_path, kind=kind)
        try:
            svc.put(
                title="Chat tool (old)",
                summary="Engineering chat runs on Mattermost",
                body="Engineering chat runs on self-hosted Mattermost.",
            )
            svc.put(
                title="Chat tool (current)",
                summary="Engineering chat moved from Mattermost to Slack",
                body="Engineering chat moved from Mattermost to Slack workspace.",
                supersedes=["chat-tool-old"],
            )

            hist = svc.history()
            actions = [h["action"] for h in hist]
            assert "supersede" in actions
            assert "create" in actions
            assert "supersede" in commands.changes_text(svc)

            # Revert the supersede -> old entry is active again in manifest and search.
            sup_change = next(h for h in hist if h["action"] == "supersede")
            rev = svc.revert(sup_change["id"], PROFILE)
            assert rev["undid"] == "supersede"
            assert rev["result"] == "restored"
            assert svc.profile.store.get_meta("chat-tool-old").superseded_by is None
            assert "chat-tool-old" in [k for k, *_ in svc.manifest_entries()]
            assert "chat-tool-old" in [r["key"] for r in svc.search("self-hosted mattermost")]

            # Revert the revert -> superseded again!
            rev_change = svc.history(key="chat-tool-old")[0]
            assert rev_change["action"] == "revert"
            svc.revert(rev_change["id"], PROFILE)
            assert svc.profile.store.get_meta("chat-tool-old").superseded_by == "chat-tool-current"
            assert "chat-tool-old" not in [k for k, *_ in svc.manifest_entries()]
        finally:
            svc.close()


class TestSuggestionHeuristic:
    @pytest.mark.parametrize(
        ("old_text", "new_text"),
        [
            ("Logs are kept for 7 days", "Logs are kept for 30 days"),
            ("Services use Python 3.9", "Services standardise on Python 3.12"),
            ("Daily standup is at 09:30", "Daily standup is at 10:15"),
            ("Backups live at /var/backups/v1", "Backups live at /var/backups/v2"),
            ("Chat runs on Mattermost", "Chat moved to Slack"),
            ("Auth uses legacy cookies", "Auth now uses signed JWTs"),
            ("Deploys use Spinnaker", "Spinnaker is retired; use ferrule"),
            ("Passwords stored in Vaultline", "Passwords are no longer in Vaultline"),
            ("Flags in LaunchLite", "Beacon replaced LaunchLite"),
        ],
    )
    def test_positive_update_cases(self, old_text: str, new_text: str) -> None:
        assert looks_like_update(old_text, new_text) is True

    @pytest.mark.parametrize(
        ("old_text", "new_text"),
        [
            (
                "Postgres runs on port 5432 in production",
                "Production Postgres listens on port 5432",
            ),
            (
                "Team prefers dark theme in editors and terminals",
                "User prefers dark colour theme across editors and terminals",
            ),
            (
                "Deploy script sits at ops/deploy.sh",
                "Run ops/deploy.sh for staging rollouts",
            ),
        ],
    )
    def test_negative_pure_duplicate_cases(self, old_text: str, new_text: str) -> None:
        assert looks_like_update(old_text, new_text) is False

    def test_update_signal_reasons_and_boost_without_title_gate(self) -> None:
        # Changed numeric value gives explicit diff reason and positive boost.
        boost, reason = update_signal(
            "Postgres runs on port 5432 in production",
            "Primary database server listens on port 6432 in production",
            old_title="Postgres primary host",
            new_title="Primary database server",
        )
        assert boost > 0
        assert reason == "changed: 5432 -> 6432"

        # Negation cue gives explicit cue reason even with a completely different title.
        boost2, reason2 = update_signal(
            "Errors are tracked in Sentry",
            "We no longer use Sentry for error reporting",
            old_title="Error tracker",
            new_title="Observability stack change",
        )
        assert boost2 > 0
        assert reason2 == "new text says no longer"

        # No cue or fact diff still returns zero boost and a neutral reason (never a veto).
        boost3, reason3 = update_signal(
            "Postgres runs on port 5432 in production",
            "Production Postgres listens on port 5432",
            shared_terms=("postgres", "production"),
        )
        assert boost3 == 0.0
        assert reason3 == "shared terms: postgres, production"

    def test_put_returns_possible_updates_and_alias_without_auto_applying(
        self, tmp_path: Path
    ) -> None:
        svc = _make_service(tmp_path)
        try:
            svc.put(
                title="Postgres primary host",
                summary="Primary Postgres database runs on port 5432 at db-01.internal",
                body="Production Postgres database listens on port 5432 at db-01.internal.",
            )
            # Retitled update still surfaces the old entry in possible_updates (no title gate).
            res = svc.put(
                title="Primary database server",
                summary="Production Postgres database listens on port 6432 at db-01.internal",
                body="Production Postgres database now listens on port 6432 at db-01.internal.",
            )
            updates = res["possible_updates"]
            assert len(updates) >= 1
            top = updates[0]
            assert top["key"] == "postgres-primary-host"
            assert top["title"] == "Postgres primary host"
            assert "5432" in top["summary"]
            assert isinstance(top["similarity"], float)
            assert top["reason"] == "changed: 5432 -> 6432"
            # One-release backward-compatibility alias stays populated.
            assert res["suggested_supersedes"] == [u["key"] for u in updates]
            assert "possible_updates" in res["note"]
            assert "supersedes=[key]" in res["note"]
            # Must NOT auto-apply supersedes.
            assert svc.profile.store.get_meta("postgres-primary-host").superseded_by is None

            # Agent follows the contract: re-put with supersedes=[key] archives the old entry.
            res_apply = svc.put(
                title="Primary database server",
                summary="Production Postgres database listens on port 6432 at db-01.internal",
                body="Production Postgres database now listens on port 6432 at db-01.internal.",
                supersedes=[top["key"]],
            )
            assert res_apply["superseded"] == ["postgres-primary-host"]
            assert svc.profile.store.get_meta("postgres-primary-host").superseded_by == (
                "primary-database-server"
            )
            assert "possible_updates" not in res_apply
        finally:
            svc.close()

    def test_put_possible_updates_uses_head_vectors_when_embedder_active(
        self, tmp_path: Path
    ) -> None:
        class _StubEmbedder:
            model_id = "stub-embed"
            dimension = 2

            def warm_up(self) -> None:
                pass

            def embed(self, texts: list[str], _input_type: str = "document") -> list[list[float]]:
                out: list[list[float]] = []
                for text in texts:
                    low = text.lower()
                    if any(w in low for w in ("billing", "stripe", "adyen", "payment")):
                        out.append([1.0, 0.0])
                    else:
                        out.append([0.0, 1.0])
                return out

            def close(self) -> None:
                pass

        store = get_store("sqlite", tmp_path / "memory.db")
        index = MemoryIndex(tmp_path / "idx.db")
        svc = MemoryService(
            OrderedDict([(PROFILE, Scope(PROFILE, store, index))]),
            embedder=_StubEmbedder(),  # type: ignore[arg-type]
        )
        try:
            svc.put(
                title="Stripe gateway",
                summary="Card charges use Stripe accounts",
                body="Card charges go through Stripe.",
            )
            svc.put(
                title="Office Wi-Fi",
                summary="Guest wireless network credentials",
                body="Guest wireless network rotates monthly.",
            )
            # Zero shared lexical words, completely retitled, no update cue:
            # head vectors still surface the old billing entry in possible_updates!
            res = svc.put(
                title="Adyen settlement",
                summary="Merchant payment settlement uses Adyen",
                body="Merchant payment settlement runs on Adyen.",
            )
            updates = res["possible_updates"]
            assert [u["key"] for u in updates] == ["stripe-gateway"]
            assert updates[0]["similarity"] == pytest.approx(1.0)
            assert updates[0]["reason"] == "similar topic"
        finally:
            svc.close()

    def test_qualifier_difference_surfaces_in_possible_updates_reason(self, tmp_path: Path) -> None:
        assert (
            qualifier_difference(
                "Database connection details: staging",
                "Staging Postgres runs on port 5432.",
                "Database connection details: production",
                "Production Postgres runs on port 5433.",
            )
            == "different qualifier: staging vs production"
        )
        assert (
            qualifier_difference(
                "Backup schedule",
                "Nightly Postgres backups run at 02:00 UTC.",
                "Backup schedule for analytics",
                "Nightly analytics warehouse backups run at 04:00 UTC.",
            )
            == "different qualifier: for analytics (only in new)"
        )
        assert (
            qualifier_difference(
                "Build guide",
                "Build steps for the desktop application.",
                "Build guide for macOS",
                "macOS build steps using Xcode.",
            )
            == "different qualifier: for macOS (only in new)"
        )
        # Legitimate updates without conflicting qualifiers return None.
        assert (
            qualifier_difference(
                "Deploy tooling (legacy)",
                "Deploys use Spinnaker pipelines.",
                "Deploy tooling (current)",
                "Deploys now use the ferrule CLI.",
            )
            is None
        )

        svc = _make_service(tmp_path)
        try:
            svc.put(
                title="Database connection details: staging",
                summary="Staging Postgres runs on port 5432 at stage-pg-01",
                body="Staging Postgres runs on port 5432 at stage-pg-01.",
            )
            res = svc.put(
                title="Database connection details: production",
                summary="Production Postgres runs on port 5433 at prod-pg-01",
                body="Production Postgres runs on port 5433 at prod-pg-01.",
            )
            updates = res["possible_updates"]
            assert updates[0]["key"] == "database-connection-details-staging"
            assert "different qualifier: staging vs production" in updates[0]["reason"]
        finally:
            svc.close()

    def test_possible_updates_uses_calibrated_floor_when_available(self, tmp_path: Path) -> None:
        index = MemoryIndex(tmp_path / "idx.db")
        try:
            # Too few vectors (<30): no calibration, fallback base threshold kept.
            for i in range(5):
                index.upsert(f"k{i}", f"t{i}", "m", [1.0, float(i) * 0.01])
            assert update_threshold_for(index, "m", 0.48) == pytest.approx(0.48)

            # With >=30 tightly clustered vectors (high p99 cosine ~0.99+), calibrated floor rises.
            for i in range(5, 35):
                index.upsert(f"k{i}", f"t{i}", "m", [1.0, float(i) * 0.01])
            profile = index.similarity_profile("m")
            assert profile is not None
            assert update_threshold_for(index, "m", 0.48) == pytest.approx(profile["p99"])
            assert update_threshold_for(index, "m", 0.48, calibrate=False) == pytest.approx(0.48)
        finally:
            index.close()


class TestRecencyTieBreak:
    def test_recency_breaks_ties_only_above_min_gap_and_never_overrides_clear_winner(
        self, tmp_path: Path
    ) -> None:
        svc = _make_service(tmp_path, shared=True)
        try:
            # Two identical-score entries across scopes written moments apart (< 60s):
            # same-session entries must keep their original order (profile first).
            svc.put(
                title="Standup schedule",
                summary="Daily engineering standup schedule details",
                body="Daily engineering standup runs in the morning.",
                scope=PROFILE,
            )
            time.sleep(0.02)
            svc.put(
                title="Standup schedule",
                summary="Daily engineering standup schedule details",
                body="Daily engineering standup runs in the morning.",
                scope=SHARED,
            )
            same_session = svc.search("daily engineering standup")
            assert [h["scope"] for h in same_session] == [PROFILE, SHARED]

            # Once the profile entry is older by > 60s, the newer shared entry wins the tie.
            svc.scopes[PROFILE].store.set_superseded_by(
                "standup-schedule", None, updated_at=time.time() - 120.0
            )
            aged = svc.search("daily engineering standup")
            assert [h["scope"] for h in aged] == [SHARED, PROFILE]

            # Clear score winner (matches all 3 query words) still beats a newer weak match.
            svc.put(
                title="Strong match target",
                summary="Kestrel falcon raptor telemetry",
                body="Kestrel falcon raptor telemetry pipeline details.",
                scope=PROFILE,
            )
            svc.scopes[PROFILE].store.set_superseded_by(
                "strong-match-target", None, updated_at=time.time() - 3600.0
            )
            svc.put(
                title="Newer weak note",
                summary="Generic telemetry overview",
                body="Generic telemetry dashboard note without bird terms.",
                scope=PROFILE,
            )
            strong_hits = svc.search("kestrel falcon telemetry")
            assert strong_hits[0]["key"] == "strong-match-target"
        finally:
            svc.close()

    def test_meta_cache_reused_across_searches_and_invalidated_on_writes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        svc = _make_service(tmp_path)
        try:
            svc.put(title="Alpha note", body="Alpha telemetry pipeline configuration.")
            svc.search("alpha")

            calls = 0
            orig = svc.profile.store.all_meta

            def _counting_all_meta():
                nonlocal calls
                calls += 1
                return orig()

            monkeypatch.setattr(svc.profile.store, "all_meta", _counting_all_meta)
            assert [r["key"] for r in svc.search("alpha")] == ["alpha-note"]
            assert [r["key"] for r in svc.search("telemetry")] == ["alpha-note"]
            assert calls == 0

            # External/direct supersede invalidates meta cache via fingerprint.
            svc.put(
                title="Alpha note v2",
                body="Alpha telemetry pipeline updated configuration.",
                supersedes="alpha-note",
            )
            assert [r["key"] for r in svc.search("alpha")] == ["alpha-note-v2"]
            assert calls >= 1
        finally:
            svc.close()


class TestSurfacesAndProvider:
    def test_provider_tool_calls_and_manifest_contract(self, tmp_path: Path) -> None:
        provider = NachosMemoryProvider()
        provider.initialize("sess-1", hermes_home=str(tmp_path))
        try:
            prompt = provider.system_prompt_block()
            assert "supersedes" in prompt
            assert "possible_updates" in prompt
            assert "possible_updates" in PUT_SCHEMA["description"]
            assert "supersedes=[key]" in PUT_SCHEMA["description"]

            put1 = json.loads(
                provider.handle_tool_call(
                    "nachos_memory_put",
                    {
                        "title": "Python standard 2024",
                        "body": "Services standardise on Python 3.9 across repos.",
                    },
                )
            )
            assert put1["stored"] == "python-standard-2024"

            put2 = json.loads(
                provider.handle_tool_call(
                    "nachos_memory_put",
                    {
                        "title": "Python standard current",
                        "body": "Python 3.12 is now the standard; 3.9 is end of life.",
                        "supersedes": ["python-standard-2024"],
                    },
                )
            )
            assert put2["superseded"] == ["python-standard-2024"]

            prompt_after = provider.system_prompt_block()
            assert "1 superseded entry hidden." in prompt_after
            assert "Python standard 2024" not in prompt_after

            exact = json.loads(
                provider.handle_tool_call("nachos_memory_recall", {"key": "python-standard-2024"})
            )
            assert exact["superseded_by"] == "python-standard-current"
            assert "python-standard-current" in exact["note"]

            q_default = json.loads(
                provider.handle_tool_call("nachos_memory_recall", {"query": "python 3.9 standard"})
            )
            assert "python-standard-2024" not in [r["key"] for r in q_default["results"]]

            q_all = json.loads(
                provider.handle_tool_call(
                    "nachos_memory_recall",
                    {"query": "python 3.9 standard", "include_superseded": True},
                )
            )
            assert "python-standard-2024" in [r["key"] for r in q_all["results"]]
        finally:
            provider.shutdown()

    def test_audit_and_maintain_report_long_superseded_without_modifying_files(
        self, tmp_path: Path
    ) -> None:
        home = tmp_path / "profile"
        db_path = home / "nachos" / "memory.db"
        svc = MemoryService(
            OrderedDict(
                [
                    (
                        PROFILE,
                        Scope(PROFILE, SqliteStore(db_path), MemoryIndex(index_path_for(db_path))),
                    )
                ]
            )
        )
        try:
            svc.put(title="Old policy", body="Old retention window was 7 days.")
            svc.put(
                title="New policy",
                body="Retention window is now 30 days.",
                supersedes="old-policy",
            )
            # Backdate old-policy's updated_at by 45 days.
            old_ts = time.time() - 45 * 86400
            svc.profile.store.set_superseded_by("old-policy", "new-policy", updated_at=old_ts)

            # Add a suspicious supersede across conflicting qualifiers (staging -> production)
            # and a one-sided "for analytics" qualifier.
            svc.put(
                title="Database host: staging",
                body="Staging Postgres runs at stage-db.internal.",
            )
            svc.put(
                title="Database host: production",
                body="Production Postgres runs at prod-db.internal.",
                supersedes="database-host-staging",
            )
            svc.put(
                title="Backup schedule",
                body="Core backups run nightly at 01:00 UTC.",
            )
            svc.put(
                title="Backup schedule for analytics",
                body="Analytics warehouse backups run nightly at 04:00 UTC.",
                supersedes="backup-schedule",
            )

            audit = commands.audit_text(svc, older_than_days=30)
            assert "Superseded > 30 days ago" in audit
            assert "old-policy" in audit
            assert "new-policy" in audit
            assert "Suspicious supersedes: 2" in audit
            assert "database-host-staging" in audit
            assert "different qualifier: staging vs production" in audit
            assert "different qualifier: for analytics (only in new)" in audit
        finally:
            svc.close()

        before = {
            str(p.relative_to(home)): (p.stat().st_mtime_ns, p.read_bytes())
            for p in home.rglob("*")
            if p.is_file()
        }
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "nachos_hermes.maintenance",
                "--profile-home",
                str(home),
                "--json",
                "all",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)
        sup = report["checks"]["superseded"]
        assert sup["count"] == 1
        assert sup["items"][0]["key"] == "old-policy"
        assert sup["items"][0]["superseded_by"] == "new-policy"
        susp = report["checks"]["suspicious_supersedes"]
        assert susp["count"] == 2
        susp_by_old = {item["old_key"]: item for item in susp["items"]}
        assert "staging vs production" in susp_by_old["database-host-staging"]["reason"]
        assert "for analytics" in susp_by_old["backup-schedule"]["reason"]
        assert "old-policy" not in susp_by_old
        after = {
            str(p.relative_to(home)): (p.stat().st_mtime_ns, p.read_bytes())
            for p in home.rglob("*")
            if p.is_file()
        }
        assert after == before
