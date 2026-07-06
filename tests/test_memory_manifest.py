"""Unit tests for the nachos_core memory-manifest spine (v1).

Covers: store seam (SqliteStore + MDStore, behaviorally parametrized),
MDStore hand-edit tolerance, LexicalScorer TF-IDF ranking, and the
render_toc never-truncate invariant.

Run:
    cd ~/DEV/hermes-plugin-nachos
    python -m pytest tests/test_memory_manifest.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nachos_core.store import SqliteStore, MDStore, get_store
from nachos_core.store.md_store import slugify
from nachos_core.prefetch import LexicalScorer, get_scorer
from nachos_core.toc import render_toc, build_toc


# ---------------------------------------------------------------------------
# Store — parametrized across both drivers
# ---------------------------------------------------------------------------

@pytest.fixture(params=["sqlite", "flatfile"])
def store(request, tmp_path):
    if request.param == "sqlite":
        s = SqliteStore(str(tmp_path / "mem.db"))
    else:
        s = MDStore(str(tmp_path / "mem.md"))
    yield s


def _put(store, key, title, summary, category, body):
    # Canonical contract: the key is slugify(title). Both drivers honor it
    # (MDStore derives keys from titles on read; sqlite stores any key but
    # the provider always uses the slug). Tests pass a hint key for
    # readability but we normalize to the real contract here.
    real_key = slugify(title)
    store.put(real_key, title=title, summary=summary, category=category, body=body)
    return real_key


class TestStoreContract:
    def test_put_get_roundtrip(self, store):
        k = _put(store, "git-wf", "Git workflow", "worktree + SSH remote",
                 "deposco", "worktree + SSH remote\n\nBranches off 3plp.")
        body = store.get(k)
        assert body is not None
        assert "worktree + SSH remote" in body
        assert "Branches off 3plp." in body

    def test_get_missing_returns_none(self, store):
        assert store.get("nope") is None

    def test_list_shape(self, store):
        _put(store, "a", "Alpha", "first", "cat1", "first\nbody a")
        _put(store, "b", "Beta", "second", "cat2", "second\nbody b")
        rows = store.list()
        assert len(rows) == 2
        for row in rows:
            assert len(row) == 4  # (key, title, summary, category)
        keys = {r[0] for r in rows}
        assert keys == {"alpha", "beta"}

    def test_list_sorted_by_category_then_title(self, store):
        _put(store, "z", "Zeta", "s", "bbb", "s\nx")
        _put(store, "a", "Apple", "s", "aaa", "s\nx")
        _put(store, "m", "Mango", "s", "aaa", "s\nx")
        rows = store.list()
        cats = [r[3] for r in rows]
        assert cats == sorted(cats)
        # within aaa, titles sorted
        aaa_titles = [r[1] for r in rows if r[3] == "aaa"]
        assert aaa_titles == ["Apple", "Mango"]

    def test_search_hits_body_and_labels(self, store):
        k1 = _put(store, "e1", "Deposco git", "worktree branches",
                  "deposco", "worktree branches\nRemote is SSH.")
        k2 = _put(store, "e2", "News digest", "TLDR format",
                  "prefs", "TLDR format\nOne consolidated email.")
        assert k1 in store.search("SSH")          # body hit
        assert k1 in store.search("worktree")     # summary hit
        assert k2 in store.search("digest")       # title hit
        assert store.search("nonexistentxyz") == []

    def test_search_empty_query(self, store):
        _put(store, "e1", "T", "s", "c", "s\nb")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_put_replaces(self, store):
        k = _put(store, "k", "Title", "old summary", "cat", "old summary\nold")
        _put(store, "k", "Title", "new summary", "cat", "new summary\nnew")
        assert len(store.list()) == 1
        assert "new" in store.get(k)
        assert "old" not in store.get(k)

    def test_remove(self, store):
        k = _put(store, "k", "T", "s", "c", "s\nbody")
        store.remove(k)
        assert store.get(k) is None
        assert store.list() == []

    def test_remove_missing_is_noop(self, store):
        store.remove("ghost")  # must not raise
        assert store.list() == []

    def test_summary_in_list_matches_put(self, store):
        k = _put(store, "k", "Title", "keyword dense line", "cat",
                 "keyword dense line\n\nmore body")
        row = [r for r in store.list() if r[0] == k][0]
        assert row[2] == "keyword dense line"

    def test_multiple_categories_isolated(self, store):
        _put(store, "a", "A", "sa", "cat1", "sa\nx")
        _put(store, "b", "B", "sb", "cat2", "sb\ny")
        _put(store, "c", "C", "sc", "cat1", "sc\nz")
        cats = {r[3] for r in store.list()}
        assert cats == {"cat1", "cat2"}


# ---------------------------------------------------------------------------
# get_store factory
# ---------------------------------------------------------------------------

class TestGetStore:
    def test_sqlite_default(self, tmp_path):
        assert isinstance(get_store("sqlite", tmp_path / "m.db"), SqliteStore)

    def test_flatfile(self, tmp_path):
        assert isinstance(get_store("flatfile", tmp_path / "m.md"), MDStore)
        assert isinstance(get_store("md", tmp_path / "m2.md"), MDStore)

    def test_unknown_raises(self, tmp_path):
        with pytest.raises(ValueError):
            get_store("postgres", tmp_path / "x")


# ---------------------------------------------------------------------------
# MDStore hand-edit tolerance + on-disk format
# ---------------------------------------------------------------------------

class TestMDStoreHandEdit:
    def test_slugify(self):
        assert slugify("Deposco Git Workflow") == "deposco-git-workflow"
        assert slugify("  Weird!! Chars??  ") == "weird-chars"
        assert slugify("") == "untitled"

    def test_reads_hand_written_file(self, tmp_path):
        p = tmp_path / "mem.md"
        p.write_text(
            "# deposco\n\n"
            "## Git workflow\n"
            "worktree + SSH remote\n"
            "Branches off 3plp not master.\n\n"
            "## Feature flags\n"
            "FeatureFlagService wraps ConfigService.\n\n"
            "# prefs\n\n"
            "## News digest\n"
            "TLDR format, one email.\n",
            encoding="utf-8",
        )
        s = MDStore(str(p))
        rows = s.list()
        assert len(rows) == 3
        keys = {r[0] for r in rows}
        assert keys == {"git-workflow", "feature-flags", "news-digest"}

    def test_summary_is_first_line(self, tmp_path):
        p = tmp_path / "mem.md"
        p.write_text(
            "# c\n\n## Entry\nFIRST LINE is the summary.\nsecond line.\n",
            encoding="utf-8",
        )
        s = MDStore(str(p))
        row = s.list()[0]
        assert row[2] == "FIRST LINE is the summary."

    def test_category_from_h1_grouping(self, tmp_path):
        p = tmp_path / "mem.md"
        p.write_text(
            "# alpha\n\n## One\nbody one\n\n## Two\nbody two\n"
            "# beta\n\n## Three\nbody three\n",
            encoding="utf-8",
        )
        s = MDStore(str(p))
        cat_by_key = {r[0]: r[3] for r in s.list()}
        assert cat_by_key["one"] == "alpha"
        assert cat_by_key["two"] == "alpha"
        assert cat_by_key["three"] == "beta"

    def test_text_before_first_h1_is_general(self, tmp_path):
        p = tmp_path / "mem.md"
        p.write_text("## Orphan\nno category above me\n", encoding="utf-8")
        s = MDStore(str(p))
        row = s.list()[0]
        assert row[3] == "general"

    def test_roundtrip_stable(self, tmp_path):
        p = tmp_path / "mem.md"
        s = MDStore(str(p))
        k = _put(s, "k", "Title", "the summary", "cat", "the summary\n\nrest")
        s2 = MDStore(str(p))
        assert s2.get(k).startswith("the summary")
        row = s2.list()[0]
        assert row[1] == "Title"
        assert row[2] == "the summary"
        assert row[3] == "cat"

    def test_put_leads_body_with_summary(self, tmp_path):
        # body that doesn't already start with summary gets it prepended
        p = tmp_path / "mem.md"
        s = MDStore(str(p))
        k = _put(s, "k", "T", "lead line", "c", "some body without the lead")
        body = s.get(k)
        assert body.splitlines()[0] == "lead line"


# ---------------------------------------------------------------------------
# LexicalScorer — TF-IDF ranking
# ---------------------------------------------------------------------------

class TestLexicalScorer:
    def _cands(self):
        return [
            ("git", "Deposco git", "worktree branches SSH remote", "deposco"),
            ("flags", "Feature flags", "FeatureFlagService config", "deposco"),
            ("news", "News digest", "TLDR email format", "prefs"),
            ("angular", "Angular regression", "zoneless ngClass codemirror", "deposco"),
        ]

    def test_ranks_matching_entry_first(self):
        sc = LexicalScorer()
        out = sc.rank("worktree branches", self._cands(), top_n=5)
        assert out[0] == "git"

    def test_rare_term_wins(self):
        # "codemirror" is unique to the angular entry
        sc = LexicalScorer()
        out = sc.rank("codemirror", self._cands(), top_n=5)
        assert out[0] == "angular"

    def test_top_n_respected(self):
        sc = LexicalScorer()
        # a query term shared by several -> multiple hits, cap to 2
        cands = [
            ("a", "Deposco one", "deposco stuff", "c"),
            ("b", "Deposco two", "deposco things", "c"),
            ("c", "Deposco three", "deposco items", "c"),
        ]
        out = sc.rank("deposco", cands, top_n=2)
        assert len(out) == 2

    def test_no_match_returns_empty(self):
        sc = LexicalScorer()
        assert sc.rank("zzzznomatch", self._cands(), top_n=5) == []

    def test_empty_query_returns_empty(self):
        sc = LexicalScorer()
        assert sc.rank("", self._cands(), top_n=5) == []

    def test_empty_candidates_returns_empty(self):
        sc = LexicalScorer()
        assert sc.rank("anything", [], top_n=5) == []

    def test_get_scorer_lexical(self):
        assert isinstance(get_scorer("lexical"), LexicalScorer)
        assert isinstance(get_scorer(""), LexicalScorer)

    def test_get_scorer_semantic_returns_scorer(self):
        # semantic is implemented in phase 2; selecting it must not raise.
        # It falls back to lexical at rank() time if the backend is absent.
        from nachos_core.semantic import SemanticScorer
        s = get_scorer("semantic")
        assert isinstance(s, SemanticScorer)
        # backend absent in test env -> graceful lexical fallback, no crash
        out = s.rank("codemirror", [
            ("git", "Deposco git", "worktree SSH", "dep"),
            ("ng", "Angular regression", "zoneless codemirror", "dep"),
        ], top_n=2)
        assert out[0] == "ng"

    def test_get_scorer_unknown_raises(self):
        with pytest.raises(ValueError):
            get_scorer("magic")


class TestSemanticScorer:
    def test_invalid_backend_raises(self):
        from nachos_core.semantic import SemanticScorer
        with pytest.raises(ValueError):
            SemanticScorer(backend="pinecone")

    def test_falls_back_to_lexical_when_backend_absent(self):
        # nachos-embeddings CLI not present in test env -> lexical fallback
        from nachos_core.semantic import SemanticScorer
        s = SemanticScorer(backend="nachos")
        cands = [
            ("a", "Deposco git", "worktree branches SSH", "dep"),
            ("b", "News digest", "TLDR email format", "pref"),
        ]
        assert s.rank("worktree", cands, top_n=2)[0] == "a"

    def test_empty_query_returns_empty(self):
        from nachos_core.semantic import SemanticScorer
        s = SemanticScorer(backend="nachos")
        assert s.rank("", [("a", "T", "s", "c")], top_n=2) == []


# ---------------------------------------------------------------------------
# render_toc — never-truncate invariant
# ---------------------------------------------------------------------------

class TestRenderToc:
    def _many(self, n):
        return [
            (f"k{i}", f"Title {i}", f"summary number {i} " * 5, f"cat{i % 3}")
            for i in range(n)
        ]

    def test_empty_returns_empty_string(self):
        assert render_toc([]) == ""

    def test_all_entries_present(self):
        entries = self._many(50)
        out = render_toc(entries)
        for i in range(50):
            assert f"Title {i}" in out

    def test_grouped_by_category(self):
        entries = [
            ("a", "A", "sa", "zcat"),
            ("b", "B", "sb", "acat"),
        ]
        out = render_toc(entries)
        # acat header appears before zcat (sorted)
        assert out.index("## acat") < out.index("## zcat")

    def test_prefetch_marker(self):
        entries = [("a", "Alpha", "sa", "c"), ("b", "Beta", "sb", "c")]
        out = render_toc(entries, prefetched={"a"})
        alpha_line = [l for l in out.splitlines() if "Alpha" in l][0]
        beta_line = [l for l in out.splitlines() if "Beta" in l][0]
        assert "\u25ba" in alpha_line
        assert "\u25ba" not in beta_line

    def test_over_budget_shortens_not_omits(self):
        entries = self._many(40)   # long summaries, many entries
        tight = 800
        out = render_toc(entries, char_budget=tight)
        # every entry title still present despite tight budget
        for i in range(40):
            assert f"Title {i}" in out

    def test_over_budget_shrinks_summaries(self):
        entries = self._many(40)
        full = render_toc(entries)
        squeezed = render_toc(entries, char_budget=800)
        assert len(squeezed) < len(full)

    def test_build_toc_sorts(self):
        rows = build_toc([("z", "Z", "s", "b"), ("a", "A", "s", "a")])
        assert rows[0][3] == "a"
