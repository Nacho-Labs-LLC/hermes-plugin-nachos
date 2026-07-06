# Nachos Memory Manifest — Spec (Vehicle A)

Status: accepted, ready to build
Owner: Nate
Last updated: 2026-07-02

## TL;DR

The built-in Hermes memory has a hard always-on injection ceiling
(MEMORY.md 8K, USER.md 4K). We hit it constantly and prune valuable facts
to make room. This is not a storage problem — it's an *assembly* problem:
full file content is dumped into every system prompt, so the budget is
fixed while knowledge grows. The fix is a three-tier assembly strategy —
always-on **manifest** (bounded table-of-contents) + per-turn **prefetch**
(relevance-scored slice) + on-demand **recall** — replacing full injection.
Underneath, a small **store seam** (SQLite default, flatfile option) and a
**scorer seam** (lexical default, semantic drop-in) give users choice
without bloating the hot path. It becomes the third layer of the nachos
plate (context-engine + optimizer + **memory-manifest** + policy) and lets
the agent drop full MEMORY.md injection and use one coherent system.

## Core principle (why this exists)

"Lightweight, but with choices" — the same ethos that produced
nachos-embeddings + the semantic-search MCP. We are NOT competing in the
memory space. We offer a lean memory layer we actually use, with opt-in
seams so users pick their store and their recall strategy. Lean floor,
optional upgrades, no lock-in.

## What the user gets

- No char ceiling. Store grows without bound; only a bounded manifest + a
  relevant slice ever enter the prompt.
- One memory system, not two. Built-in full injection OFF; no redundant
  built-in + holographic overlap.
- Same curation workflow. Entries hand-edited; the agent owns the manifest
  and keeps summaries honest.
- Choice at two seams: store (sqlite | flatfile), scorer (lexical | semantic).
- Principle-adherent: local-first, no new deps in the package, no LLM in
  the hot path, fully dogfooded.

## The three tiers (the assembly layer — this is the IP)

| Tier | Cadence | Budget | Mechanism | Injection surface |
|---|---|---|---|---|
| 1. Manifest | every turn | bounded (~1K) | full table-of-contents: `title — one-line summary`, grouped by category; scales with entry COUNT not size; NEVER truncates | `system_prompt_block()` |
| 2. Prefetch | every turn, auto | small (~1.5K, top 3-5) | score entry bodies vs incoming message; inject top matches | `prefetch(query)` |
| 3. Recall | on demand | n/a | tool: pull entry by key or search | tool schema + `handle_tool_call` |

Manifest-never-truncates is the invariant that makes recall NON-probabilistic:
the agent always sees the full index and can choose to pull any drawer.
Prefetch is the safety net that surfaces bodies proactively.

## Self-correcting summaries

Summaries are authored (keyword-dense, written FOR retrieval) AND
self-healing. Two triggers:

- **On-edit (in v1, free):** whenever the agent edits an entry body, it
  re-checks the one-line summary still holds and rewrites if drifted.
  Deterministic, no LLM, happens in-loop.
- **Periodic (parked companion cron, NOT package weight):** a separate
  script (reflection-pattern) reads entry bodies on a cadence (M-F, cheap
  model), regenerates summaries that rotted without an edit, writes back.
  Lives OUTSIDE the plugin runtime — zero hot-path cost. Wired up only if
  on-edit proves insufficient (for hand-curated memory, rarely will).

## Store seam (choice #1 — local, enumerable only)

5-method synchronous interface; manifest/prefetch/recall sit ABOVE it:

```
list()        -> [(key, title, summary, category)]   # feeds manifest (tier 1)
get(key)      -> body                                 # feeds recall (tier 3)
search(q)     -> [key, ...]                            # feeds prefetch (tier 2)
put(key, ...) / remove(key)                            # curation
```

Drivers (~100 lines each):
- **SqliteStore** — DEFAULT. stdlib (no dep), fast indexed search at scale.
- **MDStore** — flatfile, hand-editable titled-MD sections, grep-able.

Config: `nachos.memory.store: sqlite | flatfile`

HARD RULE — local & enumerable only. The manifest needs cheap
"list all titles+summaries"; the interface is synchronous by design so it
CANNOT express a network/async backend. Anyone wanting cloud/vector picks a
core provider (mem0/honcho/holographic). This rule is what keeps the seam
from metastasizing into a bad reimplementation of the 8-provider menu.

## Scorer seam (choice #2 — prefetch ranking strategy)

The tier-2 scorer is pluggable behind the seam:

- **lexical** — DEFAULT. keyword/TF-IDF overlap. zero-dep, synchronous,
  ships v1. The lean floor.
- **semantic** — opt-in. delegates to an external embedding/semantic-search
  service. The plugin embeds NOTHING itself — no bundled model, no dep in
  the package. It calls out.

Semantic is itself driver-agnostic (choice within the choice):
- nachos-embeddings / mcp-semantic-search — RECOMMENDED, our IP, this is the
  salvage role for nachos-embeddings post-archive.
- Support the most popular alternatives so users aren't locked to us
  (candidates to confirm at build: a local option like sentence-transformers/
  Ollama embeddings, and a hosted option like OpenAI text-embedding-3).

Config: `nachos.memory.scorer: lexical | semantic`
        `nachos.memory.semantic.provider: nachos | <popular alternatives>`

Why this respects the hot path: lexical default is sync/dumb/zero-dep.
Semantic delegation is opt-in and served by an already-published service,
never bundled — same discipline as parking the periodic corrector in a cron.

## Host-abstraction feasibility (checked against source)

- Only `MemoryProvider.system_prompt_block()` injects into the system
  prompt. NO `on_prompt_build` hook (VALID_HOOKS confirmed). Manifest MUST
  ship as a MemoryProvider.
- `prefetch(query)` / `queue_prefetch(query)` exist — tier 2 has a home.
- `get_tool_schemas` / `handle_tool_call` — tier 3 + curation tools home.
- One-active-provider limit: activating nachos-memory consumes the single
  slot. holographic dropped (compositional half unused by us). Built-in
  full injection disabled via `memory.memory_enabled: false`.

## Layer / component map

```
hermes-plugin-nachos/
  nachos_core/
    manifest.py        REUSE (strip memory coupling) — build/render manifest
    store/             NEW — 5-method interface + SqliteStore + MDStore
    prefetch.py        NEW — scorer seam: LexicalScorer (+ SemanticScorer adapter)
    KEEP: budget, compactor, snapshots, PromptReport types
    DROP: extractor, dedup, old memory adapters (LLM-triple extraction OUT)
  plugins/
    memory/nachos/     REBUILT — MemoryProvider: manifest + prefetch + recall + curation
    context_engine/nachos/  unchanged (+ PromptReport recording)
    nachos-optimizer/  (separate track — tool-output compression + observability)
  tools/
    correct_summaries.py  NEW companion cron script (parked; periodic corrector)
  docs/
    memory-manifest-spec.md  (this file)
```

## Scope cuts (explicitly OUT of v1)

- LLM-driven fact extraction / triples. Curation stays manual.
- Cloud / vector / network stores. Local-enumerable only (hard rule above).
- Periodic corrector wired live — script exists, parked until needed.
- Vehicle B upstream work (below) — earned after A is dogfooded.

## Vehicle B — upstream hermes-agent contributions (the real prizes)

Two ideas from this design are too big/valuable to bury in our package;
they belong upstream where they improve ALL providers, not just ours:

1. **Manifest assembly mode** — a core prompt-assembly capability any
   memory provider can opt into (bounded manifest + prefetch + recall),
   rather than nachos reimplementing storage. This is the missing
   assembly-strategy layer (there's no on_prompt_build hook today).
2. **Living session summaries** — periodic self-correcting summaries of a
   session's content, owned by the context engine (which owns the message
   window). The thing Nate wishes Hermes session recall did. Periodic
   self-correction is the HEADLINE here (vs a memory-hygiene nicety in A).

Sequencing: prove the self-correcting-summary + manifest patterns small on
memory (Vehicle A), dogfood for weeks, then propose B upstream with real
"running in my daily driver" evidence. Matches the extraction playbook:
build for harness #1, extract the contract, generalize.

## Why not the alternatives

| Option | Why not |
|---|---|
| Raise char limit | Band-aid; violates "structural fixes over limit-raises." |
| Keep built-in as-is | The ceiling IS the problem; no tiering path exists. |
| Keep holographic | SQLite fine now, but compositional half unused; runs alongside built-in = redundancy smell. |
| Generic lightweight provider | Dead middle — no edge over built-in except taste. |
| Manifest in context engine | Impossible — no system-prompt injection surface. |
| Pluggable store over ANY backend | Leaks: manifest needs cheap enumerate; cloud/vector can't. Bound to local. |
| Bundle embeddings INTO provider | Heavy dep + async in hot path. Breaks lean. Delegate via seam instead. |
| Wrap existing providers (meta-provider) | One-active-slot limit fights it; only works over local stores. → Vehicle B. |

## Known tradeoff (accepted, and mitigated)

Always-on injection guarantees never-forget. Manifest makes recall
*probabilistic* IN THEORY. Mitigated to near-zero by: (1) manifest never
truncates — full index always visible; (2) prefetch surfaces bodies; (3)
self-correcting summaries keep labels matching drawers. Residual failure:
a summary that doesn't mention what it should — a curation-quality problem
we control, not an algorithmic gamble. Accepted because full injection
ALREADY stopped being perfect the moment we started pruning at the ceiling.

## Roadmap

1. Spec (this doc) — done.
2. nachos_core: store interface + SqliteStore + MDStore + LexicalScorer +
   manifest render. Unit tests, no host imports.
3. SemanticScorer adapter (delegates to MCP; nachos-embeddings default).
4. Rebuild plugins/memory/nachos as the 3-tier MemoryProvider.
5. Live-wire against real ~/.hermes (symlink → discover → load → init →
   exercise; verify RESOLVED config; disable built-in injection surgically;
   confirm rollback).
6. Migrate existing MEMORY.md/USER.md into the store; dogfood.
7. Write parked correct_summaries.py companion cron.
8. Soak. If it holds, propose Vehicle B upstream.

## Open questions (for build phase, all low-regret)

- Prefetch budget + top-N: start 3-5 / ~1.5K, tune on feel.
- Lexical scorer: dumb overlap vs TF-IDF weighting. Lean TF-IDF (still
  zero-dep, rarer terms score higher, better with our keyword-dense entries).
- Which "popular" semantic providers to support beyond nachos: confirm the
  1-2 highest-value (likely one local, one hosted).
- Category axis: reuse existing MEMORY.md headers as manifest groups. Lean yes.
- Summary source: authored one-liner, self-corrected (not derived first-line).

## Phase-1 build notes (frozen interfaces — phase 2 builds against these)

Phase 1 (nachos_core spine) is BUILT and green: 48 new tests, 93 total
passing (only pre-existing doomed-module collection errors remain). Files:
`nachos_core/store/{__init__,base,sqlite_store,md_store}.py`,
`nachos_core/prefetch.py`, `nachos_core/toc.py`,
`tests/test_memory_manifest.py`.

FROZEN CONTRACTS:
- `MemoryStore` ABC (store/base.py): `list() -> [(key,title,summary,category)]`,
  `get(key) -> str|None`, `search(query) -> [key]`,
  `put(key, *, title, summary, category, body)`, `remove(key)`.
  Entry tuple = (key, title, summary, category); body fetched via get().
- `get_store(kind, path)`: 'sqlite'(default) | 'flatfile'.
- **KEY CONTRACT (important):** the canonical key is `slugify(title)`.
  MDStore derives keys from titles on read (hand-written files have no key
  field); SqliteStore stores any key but the provider MUST always pass
  `slugify(title)` so the two drivers behave identically. `slugify` lives
  in `nachos_core/store/md_store.py`.
- `Scorer` ABC (prefetch.py): `rank(query, candidates, top_n=5) -> [key]`.
  `LexicalScorer` = hand-rolled TF-IDF over title+summary (body excluded
  from ranking text to keep hot path light; store.search already matched
  bodies). `get_scorer(name)`: 'lexical'(default); 'semantic' raises
  NotImplementedError until the phase-2 MCP adapter lands.
- `toc.py`: `build_toc(entries)` (sort by category,title),
  `render_toc(entries, *, prefetched=set(), char_budget=None)`. NEVER omits
  an entry; over budget it shortens summaries (caps 120→0) then title-only.
  Prefetched keys marked with ► .
- New manifest lives in `toc.py`, NOT the legacy `manifest.py` (which is
  deleted in phase 2 alongside extractor/dedup/adapters + their tests
  test_extractor.py/test_migration.py, currently failing to collect because
  they import the host `agent` module).

## Decision log

- T1: archive the nachos HARNESS; harvest IP. Confirmed all portable
  shared/ packages already extracted; harness safe to archive.
- T2: drop nachos MEMORY plugin, keep context+policy. Proposed relocating
  manifest + PromptReport to context engine.
- T3 (reversal): manifest CANNOT ride context engine — no system-prompt
  injection surface. Slated manifest for DROP as redundant with injection.
- T4 (re-open): user surfaced the char-ceiling pain. Reframed: problem is
  always-on INJECTION not storage. Manifest is the FIX, as a MemoryProvider.
- T5: fold 3-tier memory-manifest in as nachos's third layer. Prefetch IN
  for v1; LLM extraction OUT. Reversible.
- T6: usage barely changes — same curation, agent owns manifest+summaries.
  Dropped pinned/managed tiers as over-structure. Focus: less-probabilistic
  recall via summary quality + manifest-never-truncates.
- T7: living self-correcting summaries (the thing user wishes session recall
  did). On-edit trigger in v1; periodic self-correction as parked companion
  cron (reflection pattern), NOT package weight. Full periodic vision →
  Vehicle B session summaries.
- T8: store seam — support drop-in stores bounded to LOCAL enumerable
  (SQLite default, flatfile option). Not YAGNI: "choice" is the product
  value (nachos-embeddings ethos). SQLite is stdlib so no-dep preserved.
- T9: scorer seam — semantic recall as opt-in prefetch scorer delegating to
  existing semantic-search MCP (salvages nachos-embeddings). Lexical default.
- T10: semantic scorer itself driver-agnostic — nachos recommended, support
  most popular alternatives (one local + one hosted) so users aren't locked in.
