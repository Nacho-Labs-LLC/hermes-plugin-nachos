# nachos

The context + prompt-optimization layer for [Hermes Agent](https://hermes-agent.nousresearch.com).

Nachos is a plate: independent layers you turn on as you need them. You
don't have to use all of them — they're there, easily activated, and not
bloated. This repo currently ships three layers:

```
┌───────────────────────────────────────────────────────────┐
│  memory-manifest   3-tier assembly: manifest + prefetch +  │  MemoryProvider
│                    recall. Kills the always-on injection    │
│                    char ceiling. Store + scorer seams.      │
├───────────────────────────────────────────────────────────┤
│  context-engine    zone-based compaction, tool-pair-safe    │  ContextEngine
│                    sliding, conversation snapshots.         │
├───────────────────────────────────────────────────────────┤
│  policy            YAML tool-gating with hot reload,        │  plugin + hook
│                    failure-open.                            │
└───────────────────────────────────────────────────────────┘
```

Nachos does **not** compete on memory *storage* — the value is the
assembly *shape* and the *choices* (which store, which recall strategy).
Bring your own backend.

---

## Layer 1 — memory-manifest (`plugins/memory/nachos`)

The built-in Hermes memory injects the full text of your memory files into
every system prompt. That has a hard char ceiling: as knowledge grows you
prune valuable facts to make room. Nachos replaces full injection with a
bounded, scalable **3-tier assembly**:

| Tier | When | What |
|------|------|------|
| **manifest** | every turn | A never-truncating table of contents — `title — summary`, grouped by category. Scales with entry *count* (one line each), not entry body *size*. The ceiling is gone. |
| **prefetch** | every turn | The most relevant entry *bodies* for the incoming message, ranked and injected under a small budget. Marked `►` in the manifest. |
| **recall** | on demand | `nachos_memory_recall` pulls any entry in full when you need it. |

You always see the full index; you fetch the drawer when the label
matches. Curation stays manual (`nachos_memory_put` / `nachos_memory_remove`);
summaries self-correct on edit. There is **no LLM call in the hot path** —
periodic summary correction is a separate, opt-in cron.

### Two seams, lean defaults, no required deps

**Store** — where entries live (local & enumerable only):

| `nachos.memory.store` | Backend | Notes |
|-----------------------|---------|-------|
| `sqlite` (default) | stdlib `sqlite3` | fast, indexed, zero dep |
| `flatfile` | titled-markdown | hand-editable, grep-able |

**Scorer** — how prefetch ranks:

| `nachos.memory.scorer` | Strategy | Notes |
|------------------------|----------|-------|
| `lexical` (default) | hand-rolled TF-IDF | zero dep, synchronous |
| `semantic` | embeddings (cosine) | opt-in; falls back to lexical if the backend is absent |

Semantic is itself driver-agnostic via `nachos.memory.semantic_provider`:
`nachos` (default — [nachos-embeddings](https://github.com/Nacho-Labs-LLC/nachos-embeddings)
MCP, recommended), `sentence-transformers` (local), or `openai`
(`text-embedding-3`). All backend imports are lazy — the package bundles
no model and has no required dependency.

> **Semantic install note:** the optional backend must be installed into
> the venv that runs Hermes (the hermes-agent checkout's `.venv`), NOT the
> plugin repo's venv — the provider executes under the host's interpreter.
> `sentence-transformers` needs `<hermes-agent>/.venv/bin/pip install
> sentence-transformers`. If the backend isn't importable there, prefetch
> silently falls back to lexical (a WARNING is logged per prefetch).

### Config

```yaml
memory:
  provider: nachos
  memory_enabled: false   # disable built-in full injection — nachos owns it now
nachos:
  memory:
    store: sqlite                 # sqlite | flatfile
    scorer: lexical               # lexical | semantic
    semantic_provider: nachos     # nachos | sentence-transformers | openai
    prefetch_top_n: 5
    prefetch_char_budget: 1500
    manifest_char_budget: 1200
```

Slash commands: `/nachos-memory-status`, `/nachos-memory-list`.

Periodic summary self-correction (parked companion cron, out of the hot
path — dry-run by default, prints + warns on expensive resolved models):

```bash
python tools/correct_summaries.py            # dry-run
python tools/correct_summaries.py --run      # write corrected summaries
```

---

## Layer 2 — context-engine (`plugins/context_engine/nachos`)

Zone-based context pressure instead of a single binary "compress?" check.
Most turns are handled by cheap, LLM-free actions:

| Zone | Action |
|------|--------|
| yellow | prune old tool results (no LLM) |
| orange | sliding window, tool-pair preserving (no LLM) |
| red | slide + delegate summary to Hermes' built-in compressor |
| critical | aggressive slide + summary |

Before any destructive compaction it takes a gzipped **conversation
snapshot** (Hermes' built-in checkpoints are filesystem-only). Enable with
`context.engine: nachos`.

**Install note:** Hermes' context-engine loader scans the *bundled*
`hermes-agent/plugins/context_engine/` directory only — not
`~/.hermes/plugins/`. Symlink this layer into your hermes-agent checkout:

```bash
ln -sfn "$PWD/plugins/context_engine/nachos" \
  <hermes-agent-repo>/plugins/context_engine/nachos
```

---

## Layer 3 — policy (`plugins/nachos-policy`)

YAML-based tool-call gating with priority-ordered rules, hot reload, and a
failure-open guarantee (policy bugs never silently kill tool execution).
Default-deny available; ships allow-all so enabling breaks nothing. See
`plugins/nachos-policy/` for the rule schema and examples. Enable with
`nachos.layers.policy: true`.

---

## Install

```bash
git clone https://github.com/Nacho-Labs-LLC/hermes-plugin-nachos.git
# memory provider — user plugin dir is scanned:
ln -sfn "$PWD/hermes-plugin-nachos/plugins/memory/nachos" ~/.hermes/plugins/nachos
# context engine — must live in the hermes-agent checkout (see Layer 2)
```

Then set the config keys for whichever layers you want. Each layer is
independent — activate one, two, or all three.

## Design

Full architecture, decision log, and the upstream roadmap live in
[`docs/memory-manifest-spec.md`](docs/memory-manifest-spec.md).

## Tests

```bash
python -m pytest tests/ -q
```

Pure stdlib + pytest — `nachos_core` has no host or third-party
dependency. Host wiring lives entirely in the plugin entry points.

## License

MIT
