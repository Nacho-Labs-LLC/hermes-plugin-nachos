"""Linked memory: [[key]] links, backlinks, related expansion, entities, manifest and maintain."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import ClassVar

import pytest

from bench import link_eval, recall
from nachos_core.index import MemoryIndex
from nachos_core.lexical import LexicalIndex
from nachos_core.links import (
    EntityIndex,
    extract_entities,
    lexical_similarities,
    name_vocabulary,
    parse_links,
    resolve_successor,
    suggest_links,
)
from nachos_core.memory import PROFILE, MemoryService, Scope
from nachos_core.store import get_store
from nachos_core.store.base import EntryMeta
from nachos_core.toc import render_toc
from nachos_hermes.memory_provider import NachosMemoryProvider


@pytest.fixture(params=["sqlite", "flatfile"])
def svc(request, tmp_path):
    ext = "memory.md" if request.param == "flatfile" else "memory.db"
    scope = Scope(PROFILE, get_store(request.param, tmp_path / ext), MemoryIndex(tmp_path / "i.db"))
    service = MemoryService(OrderedDict([(PROFILE, scope)]))
    yield service
    service.close()


def put(svc, title, body, **kw):
    out = svc.put(title=title, body=body, summary=kw.pop("summary", ""), category="ops", **kw)
    assert "error" not in out, out
    return out


def stored_links(svc):
    return svc.profile.index.all_links()


# -- parsing -----------------------------------------------------------------


class TestParse:
    def test_keys_titles_aliases_in_order_without_duplicates(self):
        body = "See [[Postgres primary host]], [[redis-cache|the cache]] and [[postgres-primary-host]]."
        assert parse_links(body) == ["postgres-primary-host", "redis-cache"]

    def test_code_is_ignored(self):
        body = "Real [[a-b]].\n```\n[[fenced]]\n```\nand `[[inline]]` and [[]] and [[ ]]"
        assert parse_links(body) == ["a-b"]

    def test_resolve_successor_follows_chain_and_survives_cycles(self):
        metas = {
            "a": EntryMeta(superseded_by="b"),
            "b": EntryMeta(superseded_by="c"),
            "c": EntryMeta(),
            "x": EntryMeta(superseded_by="y"),
            "y": EntryMeta(superseded_by="x"),
        }
        assert resolve_successor("a", metas) == "c"
        assert resolve_successor("c", metas) == "c"
        assert resolve_successor("x", metas) is None
        assert resolve_successor("missing", metas) is None


# -- entities ----------------------------------------------------------------


class TestEntities:
    TEXT = (
        "Primary is db-prim-02.internal (config /etc/pg/pg.conf), ask @ines. Alerts use PagerDuty "
        "and Kafka. Better Stack opens the page; billing at secret/billing. Run dbt_hourly, "
        "see no-reply@lumenfold.io. Self-hosted, 24/7, and/or Mo-Fr."
    )

    def test_kinds(self):
        found = extract_entities(self.TEXT)
        assert found["db-prim-02.internal"] == "host"
        assert "db-prim-02" not in found  # part of the host
        assert found["dbt_hourly"] == "id"
        assert found["@ines"] == "handle"
        assert found["/etc/pg/pg.conf"] == "path"
        assert found["secret/billing"] == "path"
        assert found["pagerduty"] == "camel"
        assert found["better stack"] == "multi"
        assert found["kafka"] == "proper"
        assert found["lumenfold.io"] == "domain"  # bare domain: weak

    def test_precision_noise_is_not_extracted(self):
        found = extract_entities(self.TEXT)
        for noise in ("no-reply", "self-hosted", "24/7", "and/or", "mo-fr", "primary", "alerts"):
            assert noise not in found
        assert "[[x]]" not in extract_entities("link [[Hidden Handle]] here")

    def test_sentence_initial_words_are_not_proper_names(self):
        assert "restart" not in extract_entities("Restart the box. Never Panic.")
        assert "database" not in extract_entities("Check the logs. Database is down.")

    def test_index_is_incremental(self):
        index = EntityIndex()
        index.upsert("a", "s", "uses db-prim-01 and @ines")
        index.upsert("b", "s", "also db-prim-01")
        assert index.entries_for("DB-PRIM-01") == ["a", "b"]
        index.upsert("a", "s", "nothing here")
        assert index.entries_for("db-prim-01") == ["b"]
        index.remove("b")
        assert index.entries_for("db-prim-01") == []

    def test_project_name_vocabulary_skips_common_words(self):
        index = EntityIndex()
        index.upsert("p", "s", "the production line, always in production")
        titles = {
            "a": "Nimbus billing API",
            "b": "Nimbus limits",
            "c": "Marrow search",
            "d": "Production notes",
        }
        vocab = name_vocabulary(titles, index.lowercase)
        assert vocab == {"marrow": "c"}  # Nimbus is shared by two titles; production is common


class TestSuggest:
    def build(self):
        titles = {
            "pg": "Postgres primary host",
            "web": "Staging web",
            "jobs": "Job runner",
        }
        bodies = {
            "pg": "Primary is db-prim-02.internal on port 5432, owner @ines.",
            "web": "Staging runs at stg.lumenfold.io behind basic auth. Ask Ines about access.",
            "jobs": "Runs on db-prim-02.internal cron. Ask Ines. Uses Redis.",
        }
        index = EntityIndex.build(
            [(k, t, b, "c") for k, t, b in zip(titles, titles.values(), bodies.values())], bodies
        )
        return index, titles

    def test_strong_shared_entity_suggests_and_weak_alone_does_not(self):
        index, titles = self.build()
        got = suggest_links(
            self_key="new",
            title="Failover plan",
            summary="s",
            body="Fail over db-prim-02.internal after asking Ines.",
            index=index,
            titles=titles,
            active=set(titles),
        )
        assert [g["key"] for g in got] == ["jobs", "pg"]  # jobs also shares a name
        assert got[1]["shared"] == ["db-prim-02.internal"]
        weak_only = suggest_links(
            self_key="new",
            title="Rota",
            summary="s",
            body="Ask Ines about it.",
            index=index,
            titles=titles,
            active=set(titles),
        )
        assert weak_only == []

    def test_skip_and_generic_entities_are_excluded(self):
        index, titles = self.build()
        kwargs = {
            "self_key": "new",
            "title": "Failover plan",
            "summary": "short summary line",
            "body": "Fail over db-prim-02.internal.",
            "index": index,
            "titles": titles,
            "active": set(titles),
        }
        assert [g["key"] for g in suggest_links(**kwargs, skip={"pg"})] == ["jobs"]
        for i in range(6):  # now db-prim-02.internal is in 8 of 9 entries: generic, no signal
            index.upsert(f"x{i}", "s", "db-prim-02.internal")
            kwargs["titles"] = {**kwargs["titles"], f"x{i}": f"Other {i}"}
        kwargs["active"] = set(kwargs["titles"])
        assert suggest_links(**kwargs) == []

    def test_entries_sharing_a_leading_project_name_are_peers(self):
        index = EntityIndex()
        titles = {"a": "Nimbus billing API", "b": "Quillbot notes"}
        got = suggest_links(
            self_key="c",
            title="Nimbus rate limits",
            summary="s",
            body="600 requests per minute.",
            index=index,
            titles=titles,
            active=set(titles),
        )
        assert [g["key"] for g in got] == ["a"]


# -- sidecar links, backlinks, sync ------------------------------------------


class TestLinkStorage:
    def test_put_parses_links_into_the_sidecar(self, svc):
        put(svc, "Hub", "Start here. Details in [[Spoke One]] and [[spoke-two]].")
        put(svc, "Spoke One", "First spoke holds the retry settings for the hub.")
        assert stored_links(svc) == {"hub": ["spoke-one", "spoke-two"]}
        entry = svc.get_entry("spoke-one")
        assert entry["backlinks"] == [{"key": "hub", "title": "Hub"}]
        assert "links" not in entry
        hub = svc.get_entry("hub")
        assert hub["links"] == [{"key": "spoke-one", "title": "Spoke One"}]
        assert hub["broken_links"] == ["spoke-two"]

    def test_edit_updates_and_removes_links(self, svc):
        put(svc, "Hub", "Start here with [[a-one]] and [[a-two]] for the full picture.")
        put(svc, "Hub", "Start here with [[a-two]] only for the full picture now.")
        assert stored_links(svc) == {"hub": ["a-two"]}
        put(svc, "Hub", "Start here and no links remain in this entry at all.")
        assert stored_links(svc) == {}

    def test_delete_drops_outgoing_and_leaves_incoming_broken(self, svc):
        put(svc, "Hub", "Start with [[Spoke]] for the retry settings and more.")
        put(svc, "Spoke", "The spoke keeps retry settings and points back to [[hub]].")
        assert svc.get_entry("hub")["links"][0]["key"] == "spoke"
        svc.remove("spoke")
        assert stored_links(svc) == {"hub": ["spoke"]}
        hub = svc.get_entry("hub")
        assert hub["broken_links"] == ["spoke"]
        assert "links" not in hub

    def test_revert_restores_links(self, svc):
        put(svc, "Hub", "Start with [[Spoke]] for the retry settings and more.")
        put(svc, "Spoke", "The spoke keeps retry settings for the hub entry.")
        put(svc, "Hub", "Start with nothing now but plain prose about the hub.")
        assert stored_links(svc) == {}
        change = next(c for c in svc.history("hub") if c["action"] == "update")
        svc.revert(change["id"])
        assert stored_links(svc) == {"hub": ["spoke"]}

    def test_links_follow_supersedes_to_the_successor(self, svc):
        put(svc, "Hub", "Read [[Retry settings]] before changing anything here.")
        put(svc, "Retry settings", "Three retries with a ten second backoff for jobs.")
        put(
            svc,
            "Retry policy",
            "Five retries with exponential backoff.",
            supersedes="retry-settings",
        )
        hub = svc.get_entry("hub")
        assert hub["links"] == [{"key": "retry-policy", "title": "Retry policy"}]
        assert "broken_links" not in hub
        assert svc.get_entry("retry-policy")["backlinks"] == [{"key": "hub", "title": "Hub"}]

    def test_external_edit_is_reconciled(self, svc):
        put(svc, "Hub", "Start with [[Spoke]] for the retry settings and more.")
        assert svc.get_entry("hub")["broken_links"] == ["spoke"]
        svc.profile.store.put(
            "spoke",
            title="Spoke",
            summary="s",
            category="ops",
            body="added behind the service's back",
        )
        assert svc.get_entry("hub")["links"][0]["key"] == "spoke"

    def test_put_reports_broken_links_and_never_fails(self, svc):
        out = put(svc, "Hub", "Start with [[Nowhere]] and [[Spoke]] for the retry settings.")
        assert out["broken_links"] == ["nowhere", "spoke"]
        assert "suggested_links" not in out


# -- recall -------------------------------------------------------------------


@pytest.fixture
def linked(svc):
    put(
        svc,
        "Harbor billing",
        "Harbor invoices run nightly. See [[Harbor webhooks]] and [[Harbor retries]].",
    )
    put(svc, "Harbor webhooks", "Webhook endpoint for card events, signed with a shared secret.")
    put(svc, "Harbor retries", "Retry settings for failed webhook deliveries, five attempts.")
    put(
        svc,
        "Cold start notes",
        "Unrelated note about a cold start problem, see [[Harbor billing]].",
    )
    return svc


class TestRelated:
    def test_related_lists_one_hop_neighbours(self, linked):
        res = linked.search("invoices nightly")
        top = res[0]
        assert top["key"] == "harbor-billing"
        assert [(r["key"], r["relation"]) for r in top["related"]] == [
            ("harbor-webhooks", "links"),
            ("harbor-retries", "links"),
            ("cold-start-notes", "linked_from"),
        ]
        assert top["related"][0]["title"] == "Harbor webhooks"

    def test_related_is_deduped_against_hits_and_capped(self, linked):
        res = linked.search("harbor")
        shown = {r["key"] for r in res}
        listed = [x["key"] for r in res for x in r.get("related", [])]
        assert not shown & set(listed)
        assert len(listed) == len(set(listed))
        linked.related_per_hit = 1
        res = linked.search("invoices nightly")
        assert len(res[0]["related"]) == 1
        linked.related_max = 0
        assert all("related" not in r for r in linked.search("invoices nightly"))

    def test_related_is_deterministic_and_can_be_disabled(self, linked):
        runs = [json.dumps(linked.search("harbor"), sort_keys=True) for _ in range(3)]
        assert len(set(runs)) == 1
        linked.related_per_hit = 0
        assert all("related" not in r for r in linked.search("harbor"))

    def test_superseded_neighbours_resolve_and_vanished_ones_are_skipped(self, linked):
        put(linked, "Harbor retry policy", "Seven attempts now.", supersedes="harbor-retries")
        keys = [x["key"] for x in linked.search("invoices nightly")[0]["related"]]
        assert "harbor-retries" not in keys
        assert "harbor-retry-policy" in keys
        linked.remove("harbor-webhooks")
        keys = [x["key"] for x in linked.search("invoices nightly")[0]["related"]]
        assert "harbor-webhooks" not in keys

    def test_no_links_means_no_related_key(self, svc):
        put(svc, "Plain one", "Nothing links anywhere from this entry at all.")
        assert "related" not in svc.search("plain")[0]


class TestEntityLookup:
    def test_entity_filter_with_query_and_lookup_without(self, svc):
        put(svc, "Primary db", "Primary runs on db-prim-02.internal with nightly backups enabled.")
        put(svc, "Backups", "Nightly backups copy db-prim-02.internal to cold storage buckets.")
        put(svc, "Other", "Nightly reports for finance, nothing about the database hosts.")
        got = svc.search("nightly", entity="DB-PRIM-02.internal")
        assert {r["key"] for r in got} == {"primary-db", "backups"}
        listed = svc.search("", entity="db-prim-02.internal")
        assert [r["match"] for r in listed] == ["entity", "entity"]
        assert [r["key"] for r in listed] == ["backups", "primary-db"]

    def test_entity_falls_back_to_a_whole_word_match(self, svc):
        put(svc, "Notes", "We talked to the vendor named zebra about invoices today.")
        put(svc, "Others", "Zebrafish are unrelated to anything in this memory store.")
        assert [r["key"] for r in svc.search("", entity="zebra")] == ["notes"]
        assert svc.search("", entity="nothing-here") == []

    def test_entity_lookup_skips_superseded_unless_asked(self, svc):
        put(svc, "Old host", "The box is db-old-01 and runs the nightly dump job.")
        put(
            svc,
            "New host",
            "The box is db-old-01 renamed, same hardware as before.",
            supersedes="old-host",
        )
        assert [r["key"] for r in svc.search("", entity="db-old-01")] == ["new-host"]
        both = svc.search("", entity="db-old-01", include_superseded=True)
        assert {r["key"] for r in both} == {"old-host", "new-host"}


class TestSuggestionsOnPut:
    def test_put_returns_suggested_links_beside_possible_updates(self, svc):
        put(svc, "Primary db", "Primary runs on db-prim-02.internal, port 5432, owner team data.")
        out = put(
            svc,
            "Failover runbook",
            "To fail over, promote the replica of db-prim-02.internal first.",
        )
        assert [s["key"] for s in out["suggested_links"]] == ["primary-db"]
        assert out["suggested_links"][0]["shared"] == ["db-prim-02.internal"]

    def test_explicit_link_or_possible_update_is_not_suggested_again(self, svc):
        put(svc, "Primary db", "Primary runs on db-prim-02.internal, port 5432, owner team data.")
        out = put(
            svc,
            "Failover runbook",
            "Promote the replica of db-prim-02.internal; see [[Primary db]].",
        )
        assert "suggested_links" not in out


# -- manifest -----------------------------------------------------------------


class TestManifest:
    ENTRIES: ClassVar = [
        ("a", "Alpha", "first summary", "ops"),
        ("b", "Beta", "second summary", "ops"),
        ("c", "Gamma", "third summary", "ops"),
    ]

    def test_counts_appear_when_they_fit(self):
        text = render_toc(self.ENTRIES, char_budget=1000, link_counts={"a": 2, "b": 1})
        assert "- Alpha — first summary [2 links]" in text
        assert "- Beta — second summary [1 link]" in text
        assert "Gamma — third summary\n" in text + "\n"

    def test_counts_are_dropped_when_they_would_break_the_budget(self):
        plain = render_toc(self.ENTRIES, char_budget=None)
        assert render_toc(self.ENTRIES, char_budget=len(plain), link_counts={"a": 2}) == plain
        assert "link" not in plain
        snug = render_toc(self.ENTRIES, char_budget=140)  # summaries shortened to fit
        assert len(snug) <= 140
        assert "\u2026" in snug
        assert render_toc(self.ENTRIES, char_budget=len(snug), link_counts={"a": 2}) == snug
        assert "[2 links]" in render_toc(
            self.ENTRIES, char_budget=len(snug) + 20, link_counts={"a": 2}
        )

    def test_without_counts_nothing_changes(self):
        assert render_toc(self.ENTRIES, link_counts={}) == render_toc(self.ENTRIES)

    def test_service_link_counts(self, linked):
        counts = linked.link_counts()
        assert counts["harbor-billing"] == 3
        assert counts["harbor-webhooks"] == 1
        assert counts["cold-start-notes"] == 1


# -- provider tool -------------------------------------------------------------


class TestProviderTool:
    def provider(self, tmp_path):
        p = NachosMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        return p

    def test_recall_entity_and_key_views(self, tmp_path):
        p = self.provider(tmp_path)
        call = p.handle_tool_call
        call(
            "nachos_memory_put",
            {"title": "Hub", "body": "Hub for db-prim-02.internal. See [[Spoke]] soon."},
        )
        call(
            "nachos_memory_put",
            {"title": "Spoke", "body": "Spoke about db-prim-02.internal backups."},
        )
        by_key = json.loads(call("nachos_memory_recall", {"key": "hub"}))
        assert by_key["links"][0]["key"] == "spoke"
        by_entity = json.loads(call("nachos_memory_recall", {"entity": "db-prim-02.internal"}))
        assert by_entity["entity"] == "db-prim-02.internal"
        assert {r["key"] for r in by_entity["results"]} == {"hub", "spoke"}
        assert "error" in json.loads(call("nachos_memory_recall", {}))
        block = p.system_prompt_block()
        assert "[1 link]" in block
        p.shutdown()

    def test_related_per_hit_config_zero_turns_expansion_off(self, tmp_path):
        p = self.provider(tmp_path)
        p.handle_tool_call(
            "nachos_memory_put",
            {"title": "Hub", "body": "Hub entry that points to [[Spoke]] here."},
        )
        p.handle_tool_call(
            "nachos_memory_put",
            {"title": "Spoke", "body": "Spoke entry with its own details here."},
        )
        assert (
            "related"
            in json.loads(p.handle_tool_call("nachos_memory_recall", {"query": "hub"}))["results"][
                0
            ]
        )
        p._service.related_per_hit = 0
        assert (
            "related"
            not in json.loads(p.handle_tool_call("nachos_memory_recall", {"query": "hub"}))[
                "results"
            ][0]
        )
        p.shutdown()


# -- maintain ------------------------------------------------------------------


class TestMaintain:
    def seed(self, tmp_path):
        path = tmp_path / "nachos" / "memory.db"
        service = MemoryService(
            OrderedDict(
                [
                    (
                        PROFILE,
                        Scope(
                            PROFILE,
                            get_store("sqlite", path),
                            MemoryIndex(path.with_name("memory.index.db")),
                        ),
                    )
                ]
            )
        )
        put(
            service,
            "Nimbus billing",
            "Nimbus invoices run on db-prim-02.internal nightly, see [[Gone entry]].",
        )
        put(
            service,
            "Nimbus limits",
            "Limits for requests on db-prim-02.internal and secret/billing keys.",
        )
        put(
            service,
            "Backups",
            "Nimbus backup dumps from db-prim-02.internal land in secret/billing too.",
        )
        put(service, "Quiet", "Unrelated entry about the office plants and watering schedule.")
        put(service, "Marrow search", "Search cluster notes with nothing special about it at all.")
        put(service, "Reader one", "The Marrow cluster feeds dashboards read by finance people.")
        put(
            service,
            "Reader two",
            "Our Marrow cluster needs a disk upgrade before the next quarter.",
        )
        service.close()
        return tmp_path

    def test_links_check_reports_broken_orphans_and_unlinked(self, tmp_path):
        home = self.seed(tmp_path)
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "nachos_hermes.maintenance",
                "--profile-home",
                str(home),
                "--json",
                "links",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        links = json.loads(out.stdout)["checks"]["links"]
        by_type = {}
        for item in links["items"]:
            by_type.setdefault(item["type"], []).append(item)
        assert by_type["broken_link"] == [
            {"type": "broken_link", "key": "nimbus-billing", "target": "gone-entry"}
        ]
        assert [h["key"] for h in by_type["orphan_hub"]] == ["marrow-search"]
        assert by_type["orphan_hub"][0]["mentioned_by"] == ["reader-one", "reader-two"]
        pairs = {(i["a"], i["b"]) for i in by_type["unlinked_related"]}
        assert ("backups", "nimbus-limits") in pairs
        assert links["count"] == len(links["items"])
        text = subprocess.run(
            [
                sys.executable,
                "-m",
                "nachos_hermes.maintenance",
                "--profile-home",
                str(home),
                "links",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "links: " in text


# -- bench ---------------------------------------------------------------------


def test_link_eval_scores_suggestions_against_labelled_pairs(tmp_path):
    def mem(title, summary, body, **extra):
        return {"title": title, "summary": summary, "category": "ops", "body": body, **extra}

    host = "db-prim-02.internal"
    memories = [
        mem(
            "Primary db", "database host", f"Primary runs on {host} port 5432 for the billing team."
        ),
        mem(
            "Failover runbook",
            "replica promotion",
            f"Promote the replica of {host} before windows.",
        ),
        mem(
            "Weekly report", "finance numbers", f"Weekly numbers are pulled from {host} by finance."
        ),
        mem(
            "Failover runbook v2",
            "replica promotion steps",
            f"Promote the replica of {host} before windows, with new steps.",
            supersedes=["failover-runbook"],
        ),
    ]
    labelled = [{"a": "primary-db", "b": "failover-runbook"}]
    svc = recall.build_service(tmp_path, None)
    try:
        scored = link_eval._score_links(svc, memories, labelled)
    finally:
        svc.close()
    assert scored["recall"] == 1.0  # primary-db / failover-runbook found
    assert scored["fp"] == 3  # report vs db, runbook and v2 are unlabelled
    assert scored["precision"] < 1.0
    assert scored["neutral"] >= 0
    assert {fp["key"] for fp in scored["false_positives"]} == {
        "weekly-report",
        "failover-runbook-v2",
    }


def test_link_eval_cli_runs_on_the_adversarial_set(capsys):
    args = ["--data-dir", str(recall.DATA_DIR / "adversarial"), "--link-eval", "--json", "-"]
    assert recall.main(args) == 0
    run = json.loads(capsys.readouterr().out)["runs"][0]
    assert run["labelled_pairs"] > 40
    assert 0.0 <= run["precision"] <= 1.0
    assert Path(recall.DATA_DIR / "adversarial" / "links.json").exists()


# -- similarity path and project-name precision --------------------------------


class TestSimilarityPath:
    N = 60

    def sims(self, hot_z: dict[str, float], n: int = N):
        """Similarities with a spread (mean ~0.2, sd ~0.1) and chosen outliers at given z."""
        base = [(f"e{i}", 0.2 + 0.1 * ((i * 7) % 11 - 5) / 3.2) for i in range(n)]
        values = [v for _k, v in base]
        mean = sum(values) / n
        sd = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
        return [(k, mean + hot_z[k] * sd if k in hot_z else v) for k, v in base]

    def suggest(self, lexical, embedded=(), found_entities=None, n=None, **kw):
        titles = {f"e{i}": f"Entry {i}" for i in range(n or self.N)}
        index = EntityIndex()
        for key, text in (found_entities or {}).items():
            index.upsert(key, "s", text)
        return suggest_links(
            self_key="new",
            title="New thing",
            summary="s",
            body=kw.pop("body", "plain prose without any entity"),
            index=index,
            titles=titles,
            active=set(titles),
            lexical=lexical,
            embedded=embedded,
            **kw,
        )

    def test_word_overlap_alone_links_an_outlier_and_gives_a_reason(self):
        got = self.suggest(self.sims({"e3": 6.0}))
        assert [g["key"] for g in got] == ["e3"]
        assert got[0]["reason"] == "similar wording"

    def test_a_moderate_similarity_needs_a_shared_entity(self):
        lexical = self.sims({"e3": 4.0})
        assert self.suggest(lexical) == []
        got = self.suggest(
            lexical,
            body="mentions the db-lonely-01 box",
            found_entities={"e3": "runs on db-lonely-01"},
        )
        assert [g["key"] for g in got] == ["e3"]
        assert got[0]["reason"] == "similar wording and shares db-lonely-01"

    def test_embeddings_add_their_z_score_to_the_word_score(self):
        lexical = self.sims({"e4": 3.8})
        assert self.suggest(lexical) == []
        got = self.suggest(lexical, embedded=self.sims({"e4": 5.0}))
        assert got[0]["key"] == "e4"
        assert got[0]["reason"] == "similar meaning and wording"

    def test_small_stores_and_caps(self):
        assert self.suggest(self.sims({"e3": 6.0}, n=10)) == []
        hot = {f"e{i}": 15.0 for i in range(6)}
        assert len(self.suggest(self.sims(hot, n=200), n=200)) == 3

    def test_entity_only_rows_have_a_reason(self):
        got = self.suggest(
            (),
            body="see db-lonely-01",
            found_entities={"e1": "db-lonely-01 here"},
        )
        assert got[0]["reason"] == "shares db-lonely-01"

    def test_lexical_similarities_rank_shared_rare_words_first(self):
        entries = [(f"k{i}", f"Title {i}", f"filler summary {i}", "c") for i in range(25)]
        bodies = {f"k{i}": f"common words only here number {i}" for i in range(25)}
        bodies["k1"] = "webhook retries for the zeppelin gateway"
        entries.append(("new", "Gateway", "gateway notes", "c"))
        bodies["new"] = "zeppelin gateway webhook notes"
        index = LexicalIndex.build(entries, bodies)
        ranked = sorted(
            lexical_similarities("new", index, set(bodies) - {"new"}), key=lambda r: -r[1]
        )
        assert ranked[0][0] == "k1"
        assert ranked[0][1] > 0.2
        assert all(sim == 0.0 for _k, sim in ranked[1:])

    def test_service_suggests_by_wording_and_caps_at_three(self, svc):
        for i in range(70):
            put(
                svc,
                f"Filler entry {i}",
                f"Unrelated content number {i} about assorted topics here.",
            )
        for i in range(2):
            put(svc, f"Gateway part {i}", "Zeppelin gateway webhook retries and signing details.")
        out = put(
            svc, "Gateway overview", "Overview of the zeppelin gateway webhook retries and signing."
        )
        sugg = out["suggested_links"]
        assert 1 <= len(sugg) <= 3
        assert all(s["key"].startswith("gateway-part") for s in sugg)
        assert all(s["reason"] for s in sugg)


class TestProjectNamePrecision:
    def test_generic_capitalised_nouns_are_not_project_names(self):
        titles = {"a": "Incident on-call rotation", "b": "Meeting notes", "c": "Marrow search"}
        assert name_vocabulary(titles) == {"marrow": "c"}

    def test_a_name_must_be_used_by_another_entry_to_link_its_owner(self):
        index = EntityIndex()
        titles = {"o": "Marrow search", "x": "Reader one"}
        index.upsert("x", "s", "plain prose about something else")
        args = {
            "self_key": "new",
            "title": "Notes",
            "summary": "s",
            "body": "The Marrow cluster is slow.",
            "index": index,
            "titles": titles,
            "active": set(titles),
        }
        assert suggest_links(**args) == []  # only the new entry mentions it
        index.upsert("x", "s", "The Marrow cluster needs a disk upgrade.")
        assert [g["key"] for g in suggest_links(**args)] == ["o"]
