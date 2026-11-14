"""MemoryService: hybrid recall, duplicate flagging, usage, scopes, redaction."""

from collections import OrderedDict

import pytest

from nachos_core.embeddings import Embedder
from nachos_core.index import MemoryIndex
from nachos_core.memory import PROFILE, SHARED, MemoryService, Scope
from nachos_core.store import get_store

# Words in the same group land on the same axis, so paraphrases embed close
# together while sharing no tokens: a stand-in for a real embedding model.
_CONCEPTS = [
    {"car", "automobile", "vehicle", "sedan"},
    {"dog", "puppy", "canine", "hound"},
    {"deploy", "release", "ship", "rollout"},
    {"database", "postgres", "sql", "db"},
    {"color", "colour", "palette", "theme"},
]


class ConceptEmbedder(Embedder):
    def __init__(self, model_id="concept-v1"):
        self.model_id = model_id
        self.calls = 0

    def _embed_batch(self, texts):
        self.calls += 1
        out = []
        for text in texts:
            words = {w.strip(".,:;!?()").lower() for w in text.split()}
            out.append([float(len(words & group)) + 0.01 for group in _CONCEPTS])
        return out


class BrokenEmbedder(Embedder):
    model_id = "broken"

    def _embed_batch(self, _texts):
        raise ConnectionError("embedding server down")


def _scope(name, tmp_path, kind="sqlite"):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("memory.md" if kind == "flatfile" else "memory.db")
    return Scope(name, get_store(kind, path), MemoryIndex(directory / "memory.index.db"))


def _service(tmp_path, *, embedder=None, shared=False, kind="sqlite", redact=None):
    scopes = OrderedDict()
    scopes[PROFILE] = _scope(PROFILE, tmp_path, kind)
    if shared:
        scopes[SHARED] = _scope(SHARED, tmp_path, kind)
    return MemoryService(scopes, embedder=embedder, duplicate_threshold=0.9, redact=redact)


def _put(svc, title, body, **kw):
    result = svc.put(title=title, body=body, **kw)
    assert "error" not in result, result
    return result


class TestRecall:
    def test_semantic_recall_finds_a_paraphrase_with_no_shared_words(self, tmp_path):
        svc = _service(tmp_path, embedder=ConceptEmbedder())
        _put(svc, "Vehicle notes", "The family sedan needs new tyres before winter.")
        _put(svc, "Pet notes", "The puppy is allergic to chicken based food.")

        results = svc.search("automobile maintenance")

        assert results[0]["key"] == "vehicle-notes"
        assert "semantic" in results[0]["match"]

    def test_lexical_recall_matches_terms_that_are_not_adjacent(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Staging deploy", "Staging rollouts use the blue/green script in ops/.")
        _put(svc, "Coffee order", "Large oat flat white, no sugar, from the corner cafe.")

        results = svc.search("blue staging script")

        assert [r["key"] for r in results] == ["staging-deploy"]
        assert results[0]["match"] == "lexical"

    def test_lexical_recall_ignores_substring_only_matches(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Token policy", "Access tokens are short-lived and rotated hourly.")
        _put(svc, "Coffee order", "Large oat flat white, no sugar, from the corner cafe.")

        assert svc.search("live") == []
        assert svc.search("tok") == []

    def test_lexical_recall_weights_rare_terms_over_common_ones(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Coffee order", "The user wants an oat flat white every morning.")
        _put(svc, "Commute", "The user cycles to the office most days.")
        _put(svc, "Pet allergy", "The user's dog cannot eat chicken kibble.")
        _put(svc, "Editor setup", "The user runs neovim with a dark colour scheme.")

        results = svc.search("user editor")

        assert results[0]["key"] == "editor-setup"

    @pytest.mark.parametrize("kind", ["sqlite", "flatfile"])
    def test_index_reconciles_entries_written_behind_its_back(self, tmp_path, kind):
        svc = _service(tmp_path, embedder=ConceptEmbedder(), kind=kind)
        svc.profile.store.put(
            "hand-edited",
            title="Hand edited",
            summary="Our hound",
            category="general",
            body="Our hound sleeps all day.",
        )

        results = svc.search("canine")

        assert results
        assert results[0]["key"] == "hand-edited"
        assert svc.profile.index.count("concept-v1") == 1

    def test_model_change_re_embeds_everything(self, tmp_path):
        svc = _service(tmp_path, embedder=ConceptEmbedder("concept-v1"))
        _put(svc, "Database choice", "We standardised on postgres for every service.")
        svc.embedder = ConceptEmbedder("concept-v2")

        assert svc.sync_index(svc.profile) == 1
        assert svc.profile.index.count("concept-v2") == 1

    def test_embedder_failure_degrades_to_lexical_and_is_reported(self, tmp_path):
        svc = _service(tmp_path, embedder=BrokenEmbedder())
        result = _put(svc, "Staging deploy", "Staging rollouts use the blue/green script.")

        assert result["stored"] == "staging-deploy"
        assert [r["key"] for r in svc.search("staging script")] == ["staging-deploy"]
        assert "embedding server down" in svc.status()["embeddings_error"]


class TestDuplicates:
    def test_put_flags_a_semantic_duplicate_but_still_stores(self, tmp_path):
        svc = _service(tmp_path, embedder=ConceptEmbedder())
        _put(svc, "Release process", "Every release goes through the rollout checklist.")

        result = _put(svc, "How we ship", "To ship, deploy via the rollout runbook.")

        assert result["stored"] == "how-we-ship"
        assert [d["key"] for d in result["possible_duplicates"]] == ["release-process"]
        assert result["possible_duplicates"][0]["match"] == "semantic"
        assert "merge" in result["note"]

    def test_put_flags_a_lexical_duplicate_without_embeddings(self, tmp_path):
        svc = _service(tmp_path)
        _put(
            svc, "User prefers dark theme", "User prefers dark theme in every editor and terminal."
        )

        result = _put(
            svc,
            "User prefers a dark theme",
            "User prefers dark theme in all editors and terminals.",
        )

        assert result["possible_duplicates"][0]["key"] == "user-prefers-dark-theme"
        assert result["possible_duplicates"][0]["match"] == "lexical"

    def test_updating_the_same_entry_is_not_a_duplicate(self, tmp_path):
        svc = _service(tmp_path, embedder=ConceptEmbedder())
        _put(svc, "Release process", "Every release goes through the rollout checklist.")

        result = _put(
            svc, "Release process", "Every release goes through the rollout checklist v2."
        )

        assert result["updated"] is True
        assert "possible_duplicates" not in result

    def test_audit_lists_duplicate_pairs(self, tmp_path):
        svc = _service(tmp_path, embedder=ConceptEmbedder())
        _put(svc, "Release process", "Every release goes through the rollout checklist.")
        _put(svc, "How we ship", "To ship, deploy via the rollout runbook.")
        _put(svc, "Pet notes", "The puppy is allergic to chicken based food.")

        pairs = svc.duplicate_pairs()

        assert len(pairs) == 1
        assert {pairs[0]["a"]["key"], pairs[0]["b"]["key"]} == {"release-process", "how-we-ship"}


class TestUsage:
    def test_recall_and_hits_are_counted(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Staging deploy", "Staging rollouts use the blue/green script.")

        svc.get("staging-deploy")
        svc.search("staging")

        usage = svc.profile.index.usage()["staging-deploy"]
        assert usage["recall_count"] == 1
        assert usage["hit_count"] == 1

    def test_unused_lists_only_old_never_recalled_entries(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Staging deploy", "Staging rollouts use the blue/green script.")
        _put(svc, "Coffee order", "Large oat flat white, no sugar, from the corner cafe.")
        svc.get("staging-deploy")
        later = svc.profile.index.usage()["coffee-order"]["first_seen"] + 40 * 86400

        assert [r["key"] for r in svc.unused(30, now=later)] == ["coffee-order"]
        assert svc.unused(60, now=later) == []

    def test_remove_drops_vector_and_usage(self, tmp_path):
        svc = _service(tmp_path, embedder=ConceptEmbedder())
        _put(svc, "Pet notes", "The puppy is allergic to chicken based food.")

        assert svc.remove("pet-notes") == PROFILE
        assert svc.profile.index.count() == 0
        assert svc.profile.index.usage() == {}
        assert svc.remove("pet-notes") is None


class TestScopes:
    def test_shared_entries_are_namespaced_in_the_manifest(self, tmp_path):
        svc = _service(tmp_path, shared=True)
        _put(svc, "Home network", "The NAS lives at 10.0.0.5 and serves media.", category="infra")
        _put(
            svc,
            "Editor",
            "Neovim with the lazy.nvim plugin manager.",
            category="tools",
            scope=SHARED,
        )

        categories = {(key, cat) for key, _t, _s, cat in svc.manifest_entries()}

        assert ("home-network", "infra") in categories
        assert ("editor", "shared/tools") in categories

    def test_get_prefers_profile_then_falls_back_to_shared(self, tmp_path):
        svc = _service(tmp_path, shared=True)
        _put(svc, "Editor", "Shared: Neovim with lazy.nvim everywhere.", scope=SHARED)
        assert svc.get("editor") == (SHARED, "Shared: Neovim with lazy.nvim everywhere.")

        _put(svc, "Editor", "Profile: VS Code for this work profile.")
        assert svc.get("editor")[0] == PROFILE

    def test_writing_to_a_disabled_scope_is_an_error(self, tmp_path):
        svc = _service(tmp_path)
        result = svc.put(title="Editor", body="Neovim with lazy.nvim everywhere.", scope=SHARED)
        assert "error" in result


def test_every_write_passes_through_the_redactor(tmp_path):
    svc = _service(tmp_path, redact=lambda text: text.replace("sk-live-123456", "«redacted»"))
    _put(svc, "API access", "Use key sk-live-123456 for the billing API sandbox.")

    assert "sk-live" not in svc.get("api-access")[1]


class TestChangeLog:
    def _body(self, svc, key="note"):
        got = svc.get(key)
        return got[1] if got else None

    def test_update_then_revert_restores_old_body(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Note", "The original body content of the note.", summary="original summary text")
        _put(svc, "Note", "Clobbered body content replaced it.", summary="new summary text")

        change = svc.history(key="note")[0]
        assert change["action"] == "update"
        res = svc.revert(change["id"], PROFILE)

        assert res["result"] == "restored"
        assert self._body(svc) == "The original body content of the note."
        titles = {k: s for k, _t, s, _c in svc.profile.store.list()}
        assert titles["note"] == "original summary text"

    def test_remove_then_revert_restores_entry(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Note", "Body that must survive removal.", category="user")
        svc.remove("note")
        assert self._body(svc) is None

        change = svc.history()[0]
        assert change["action"] == "remove"
        svc.revert(change["id"], PROFILE)

        assert self._body(svc) == "Body that must survive removal."
        assert svc.profile.store.list()[0][3] == "user"

    def test_create_then_revert_removes_it(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Note", "Freshly created body content.")
        change = svc.history()[0]
        assert change["action"] == "create"

        res = svc.revert(change["id"], PROFILE)

        assert res["result"] == "removed"
        assert self._body(svc) is None

    def test_revert_of_a_revert(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Note", "First version of the body text.")
        _put(svc, "Note", "Second version of the body text.")
        svc.revert(svc.history()[0]["id"], PROFILE)
        assert self._body(svc) == "First version of the body text."

        last = svc.history()[0]
        assert last["action"] == "revert"
        svc.revert(last["id"], PROFILE)

        assert self._body(svc) == "Second version of the body text."

    def test_unchanged_put_records_nothing(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Note", "Stable body content here.", summary="stable summary text")
        _put(svc, "Note", "Stable body content here.", summary="stable summary text")

        assert [c["action"] for c in svc.history()] == ["create"]

    def test_session_id_is_recorded(self, tmp_path):
        svc = _service(tmp_path)
        svc.session_id = "sess-42"
        _put(svc, "Note", "Body content for the session test.")

        assert svc.history()[0]["session_id"] == "sess-42"

    def test_changes_survive_delete_and_retain(self, tmp_path):
        svc = _service(tmp_path)
        _put(svc, "Note", "Body content that will be removed.")
        svc.remove("note")  # calls index.delete
        svc.profile.index.retain([])
        svc.sync_index(svc.profile)

        actions = [c["action"] for c in svc.history()]
        assert actions == ["remove", "create"]

    def test_revert_reindexes_vector(self, tmp_path):
        svc = _service(tmp_path, embedder=ConceptEmbedder())
        _put(svc, "Note", "The sedan is parked outside.")
        _put(svc, "Note", "The puppy sleeps inside.")
        svc.revert(svc.history()[0]["id"], PROFILE)
        assert svc.profile.index.count() == 0

        svc.sync_index(svc.profile)
        assert svc.profile.index.count() == 1

    def test_unknown_change_is_an_error(self, tmp_path):
        svc = _service(tmp_path)
        assert "error" in svc.revert(999, PROFILE)
        assert "error" in svc.revert(1, "nope")

    def test_history_spans_scopes(self, tmp_path):
        svc = _service(tmp_path, shared=True)
        _put(svc, "Note", "Profile body content.")
        _put(svc, "Other", "Shared body content.", scope=SHARED)

        assert {c["scope"] for c in svc.history()} == {PROFILE, SHARED}

    def test_unchanged_put_records_nothing_on_flatfile(self, tmp_path):
        svc = _service(tmp_path, kind="flatfile")
        _put(svc, "Note", "Stable body content here.", summary="stable summary text")
        _put(svc, "Note", "Stable body content here.", summary="stable summary text")

        assert [c["action"] for c in svc.history()] == ["create"]
