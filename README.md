# Nachos for Hermes Agent

Nachos is a durable-memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com). It keeps what the agent should remember across sessions (projects, environment, decisions, procedures) in a small local store, and puts a short table of contents of that store in the system prompt. The agent pulls full entries when it needs them.

- **Local-first and private.** Memory lives in files under your Hermes home. Nothing leaves the machine unless you point embeddings at a remote endpoint. Secrets are redacted on write.
- **Free to run.** No LLM call happens on the request path. The only optional LLM use is a background pass that suggests memories after a session, and you choose its model.
- **Low dependency.** Lexical recall works with the Python standard library alone. Semantic recall is an optional add-on (a small local model, an OpenAI-compatible endpoint, or `sentence-transformers`).
- **Multi-layered.** Hybrid recall, supersede tracking, links, confidence signals, history search, a review inbox, offline maintenance and reconciliation, and undo. See [How it works](#how-it-works).

Nachos replaces Hermes' built-in general memory (`MEMORY.md`) only:

| Hermes layer | Lives in | Holds | Managed by |
| --- | --- | --- | --- |
| Soul | `SOUL.md` | Persona, tone | You; Nachos leaves it alone |
| User | `USER.md` (built-in `memory` tool, `target: "user"`) | Facts about you | Hermes (stays on) |
| Memory | Nachos store (was `MEMORY.md`) | Projects, environment, decisions, procedures, history | Nachos |

The agent is told this split in the system prompt, and session capture skips anything that belongs to the soul or user layers.

## Quick start

Install into the Python environment that runs Hermes:

```bash
python -m pip install "git+https://github.com/Nacho-Labs-LLC/hermes-plugin-nachos.git@v0.7.0"
```

Alternatively, keep the repository root at `$HERMES_HOME/plugins/nachos/`.

Enable it, then start a new session:

```bash
hermes config set memory.provider nachos
hermes config set memory.memory_enabled false   # turns off only the built-in MEMORY.md; leave memory.user_profile_enabled on
```

If you have an existing `MEMORY.md`, import it first ([Migrate](#migrate)). With no further setup you get lexical recall. For semantic recall install the local embedding CLI (Node.js >= 20.9, version `>= 0.6.0`):

```bash
npm i -g @nacho-labs/nachos-embeddings
```

`embeddings: auto` finds it on `PATH`. Settings are in `hermes memory setup` or `$HERMES_HOME/nachos/config.json` (per profile); see [Configuration](#configuration).

## How it works

Each layer is optional or automatic as noted; the agent-facing tools are listed under [Tools](#tools).

1. **Manifest.** The system prompt carries a `title - summary` line for every active entry, grouped by category. Over `manifest_char_budget`, summaries shorten and titles stay; entries are never dropped. The agent calls `nachos_memory_recall` when something looks relevant.
2. **Hybrid recall.** Keyword search (IDF-weighted; title and summary count more than body) is fused with embedding similarity by reciprocal-rank fusion. Long entries are embedded in overlapping chunks. Without embeddings, or after an embedding error, recall is lexical. Optional: a cross-encoder reranker (`rerank`).
3. **Writes and updates.** `nachos_memory_put` redacts secrets, rejects low-signal text, and flags likely duplicates. When a new entry looks like an updated version of an old one it lists `possible_updates`; the agent then writes the new entry with `supersedes`. A superseded entry leaves the manifest and default search but stays readable by key and reversible.
4. **Links and entities.** Write `[[key]]` or `[[Entry Title]]` in a body to link entries. Recall can attach one-hop `related` entries, `put` suggests links from shared identifiers, hosts, paths and names (no LLM), and `nachos_memory_recall entity=...` filters by entity.
5. **Confidence and multi-query.** Query results carry `confident` / `confidence` hints, and `no_confident_match` when nothing fits (`abstention`). The agent can pass up to four rephrasings in `queries`; they share one embedding call and are fused.
6. **History and provenance.** Before each compression Hermes' transcript is checkpointed, and `nachos_history_search` recovers details compression removed (ranked, with surrounding turns and a stable `ref`). Each entry remembers where it came from (session, time, and a verified supporting quote for captured entries), shown as `source` on recall.
7. **Capture inbox.** After a session of 3 or more user messages, a background worker asks an auxiliary LLM for up to 5 durable facts and queues them as suggestions. Each needs a verbatim supporting quote, checked against the transcript; unverified ones are marked `[ungrounded]`. Nothing enters memory until you approve it with `/nachos-memory-inbox`.
8. **Maintain.** `nachos-maintain` is an offline, report-only audit: duplicates, unused or long-superseded entries, weak summaries, broken links, missing provenance, capture backlog. See [memory maintenance](docs/memory-maintenance.md).
9. **Reconciliation.** `nachos-reconcile` and the `nachos_memory_reconcile` tool turn audit findings into proposals (merge, supersede, unlink, rewrite, delete, choose one) that you approve before anything is applied. See [reconciliation](docs/reconciliation.md).
10. **Undo.** Every create, update, supersede, remove and revert is logged (newest 1000 kept). `/nachos-memory-history` lists changes and `/nachos-memory-revert <scope:id>` undoes one.

Internals (stores, index, ranking, links, capture): [docs/architecture.md](docs/architecture.md).

## Migrate

Nachos never copies other memory automatically. Import once with:

```bash
python tools/import_memories.py <source>            # dry run
python tools/import_memories.py <source> --run      # write
```

`<source>` is `builtin` (`MEMORY.md`; add `--include-user-profile` to also import `USER.md` as category `user`), `holographic` or `byterover`. Other flags: `--category`, `--hermes-home`, `--source-path`, `--store`, `--path`, `--overwrite`, `--json`. The import reads the source only and uses the normal write path (quality gate, redaction, duplicate flagging, change log). Entries whose key already exists with different content are skipped unless `--overwrite`. Server-backed providers (hindsight, honcho, mem0, openviking, retaindb, supermemory) are not supported; `--help` explains importing a manual export.

## Configuration

Run `hermes memory setup` or edit `$HERMES_HOME/nachos/config.json`. Settings are per profile. Values that are the wrong type or outside the allowed set are ignored and the default is used.

| Key | Default | Meaning |
| --- | --- | --- |
| `store` | `sqlite` | Store format: `sqlite` or hand-editable `flatfile` (`memory.md`). |
| `manifest_char_budget` | `1200` | Soft size target for the manifest. |
| `embeddings` | `auto` | Backend: `auto`, `nachos`, `openai`, `sentence-transformers` or `off` ([backends](docs/models.md#backends)). |
| `embedding_model` | blank | Model id. Blank = backend default (MiniLM for `nachos`). For `nachos`, any Hugging Face ONNX encoder or a local directory path. |
| `embedding_dtype` | blank | Weights precision for `nachos`, for example `q8` (about 3x faster, cosine about 0.99 vs fp32). Blank = fp32. Re-embeds on change. |
| `embedding_pooling` | blank | `mean` or `cls` for `nachos`. Blank = the model's profile. Re-embeds on change. |
| `embedding_max_length` | `0` | Truncate inputs to this many tokens for `nachos` (`0` = model default; can only lower the limit). Re-embeds on change. |
| `embedding_base_url` | blank | OpenAI-compatible base URL. |
| `embedding_api_key_env` | blank | Name of the environment variable holding the API key. |
| `embedding_query_prefix` | blank | Text prepended to queries, for example `query: ` (E5) or `search_query: ` (nomic). Spaces are kept. For `nachos`, blank keeps the model's own prefix and `none` disables it; other backends have no built-in prefix. |
| `embedding_document_prefix` | blank | Text prepended to stored entries, for example `passage: ` or `search_document: `. Same blank / `none` rules. Re-embeds on change. |
| `duplicate_threshold` | `0.65` | Similarity (above 0, up to 1) at which a new entry is flagged as a likely duplicate. |
| `duplicate_calibration` | `auto` | `auto` raises the default threshold to the model's own random-pair 99th percentile when that is higher; a threshold other than `0.65` is used as given. `off` always uses `duplicate_threshold`. |
| `similarity_floor` | `off` | `auto` drops semantic hits below the model's random-pair 75th percentile. |
| `abstention` | `hint` | `hint` attaches confidence fields and `no_confident_match` as information; `on` also tells the agent to retry with rephrasings; `off` omits them. |
| `rerank` | `off` | `nachos` re-scores the top fused results on every recall (about 0.2-0.3 s extra; needs the `nachos` backend); `borderline` re-scores only borderline-confidence queries; `off` disables. |
| `rerank_model` | blank | Cross-encoder id. Blank = `jinaai/jina-reranker-v1-tiny-en`. |
| `rerank_dtype` | blank | Cross-encoder precision, for example `q8`. |
| `rerank_max_length` | `0` | Cross-encoder input tokens (`0` = model default: 1024 for the jina reranker, 512 otherwise). |
| `rerank_candidates` | `25` | How many top fused results the full reranker re-scores. |
| `reindex_budget_ms` | `250` | Time one recall or write may spend re-embedding after a model change ([switching models](docs/models.md#switching-models)). |
| `related_per_hit` | `3` | Linked entries attached as `related` to each recall result (at most 8 per query). `0` turns it off. |
| `shared_store` | `off` | `on` adds a scope shared by every profile on the machine. |
| `shared_store_path` | blank | Shared store directory. Blank = `<hermes root>/nachos-shared`. |
| `redact_secrets` | `on` | Redact secrets in everything written to memory (`on` or `off`). |
| `checkpoint_max_age_days` | `30` | Delete transcript checkpoints older than this; `0` keeps them. |
| `history_context_before` | `1` | `nachos_history_search`: turns returned before each match. |
| `history_context_after` | `1` | Turns returned after each match. |
| `history_max_chars` | `6000` | Cap on the text one `nachos_history_search` call returns. |
| `capture` | `review` | `review` suggests entries from finished sessions into the inbox; `off` disables. |

Two settings outside this file affect cost: `auxiliary.nachos_capture.model` (session capture) and `auxiliary.nachos_summary.model` (`tools/correct_summaries.py`) in Hermes' `config.yaml`. If unset, Hermes may fall back to your main model, so pin a cheap one.

## Choosing a model

Name the model you want; there are no presets. Blank `embedding_model` means MiniLM through `nachos-embeddings`. Example:

```json
{ "embeddings": "nachos",
  "embedding_model": "Alibaba-NLP/gte-modernbert-base",
  "embedding_dtype": "q8" }
```

Measured on the synthetic benchmark (208 entries, 393 answerable queries, reranker off, `q8`; p50 is median search latency on one idle 16-thread CPU and varies by machine):

| `embedding_model` (dtype q8) | Download | Dims | hit@1 / hit@5 / MRR | Adversarial hit@1 | p50 | Note |
| --- | --- | --- | --- | --- | --- | --- |
| *(`embeddings: off`)* | none | - | 0.695 / 0.812 / 0.748 | 0.975 | 0.2 ms | Lexical only. |
| blank, or `Xenova/all-MiniLM-L6-v2` | 23 MB | 384 | 0.758 / 0.924 / 0.832 | 0.988 | 8.2 ms | Default; smallest and fastest. |
| `Alibaba-NLP/gte-modernbert-base` | 150 MB | 768 | **0.773 / 0.949 / 0.850** | **1.000** | 20.5 ms | Best measured; about twice the embedding time, 768-dim index. |
| `Snowflake/snowflake-arctic-embed-s` | 34 MB | 384 | 0.753 / 0.911 / 0.824 | 0.988 | 9.3 ms | Ties MiniLM here. |
| `Xenova/bge-small-en-v1.5` | 34 MB | 384 | 0.728 / 0.916 / 0.807 | 0.988 | 16.0 ms | Did not beat MiniLM here. |
| `Xenova/e5-small-v2` | 34 MB | 384 | 0.753 / 0.898 / 0.818 | 0.975 | 15.2 ms | Did not beat MiniLM here. |
| `nomic-ai/nomic-embed-text-v1.5` | 137 MB | 768 | 0.748 / 0.924 / 0.825 | 1.000 | 92.4 ms | Several times slower (measured on a busier machine). |

Differences under about 0.01 hit@1 (4 queries) are not significant on this fixture, and the fixture is synthetic: measure on your own machine with `python -m bench --model <id> --dtype q8`. Method, metrics and limits: [docs/benchmark.md](docs/benchmark.md). Changing a model re-embeds safely in the background; see [docs/models.md](docs/models.md).

## Tools

| Name | Purpose |
| --- | --- |
| `nachos_memory_recall` | Fetch an entry by `key` (also returns `superseded_by`, `links`, `backlinks`, `broken_links`, and `source` with the `evidence` quote), or search with `query`, `queries` (up to 4 rephrasings), and/or `entity`; `limit`, `include_superseded`. Search results include `related`, `source`, and with `abstention` `hint` or `on`, `confident` and `no_confident_match`. |
| `nachos_memory_put` | Add or update an entry: `title`, `body`, optional `summary`, `category`, `supersedes`. Rejects low-signal text; returns `possible_duplicates`, `possible_updates` (alias `suggested_supersedes`), `suggested_links` and `broken_links`. |
| `nachos_memory_remove` | Delete an entry by `key` or `title`. |
| `nachos_memory_reconcile` | Review, approve and apply cleanup proposals: `action` is `review`, `status`, `approve` or `apply`. Anything that writes is primary-agent only. |
| `nachos_history_search` | Recover details compression removed from the current conversation, including subagent results: `query`, `ref`, `scope` (`conversation` or `all`), `limit`, `before`, `after`. For past sessions use Hermes' `session_search`. |

With `shared_store: on`, recall, put and remove also take `scope: "shared"`; the default is the private profile scope. The provider follows session lineage across compression, `/resume` and `/branch`, so history search covers the whole logical conversation.

## Slash commands

| Command | Purpose |
| --- | --- |
| `/nachos-memory-status` | Store, embedding backend, entries per scope, re-embedding progress, checkpoints, capture queue. |
| `/nachos-memory-list` | Render the current manifest. |
| `/nachos-memory-audit [days]` | Likely duplicates, entries not recalled in N days, long-superseded entries (default 30). |
| `/nachos-memory-history [key]` | Recent changes (create, update, supersede, remove, revert; all scopes), each with a `scope:id` handle. |
| `/nachos-memory-revert <scope:id>` | Undo a change; a revert can itself be reverted. |
| `/nachos-memory-inbox [show <id> \| approve <id\|all> \| reject <id\|all>]` | Review entries suggested from past sessions. `approve all` skips `[ungrounded]` ones. |

## Command-line tools

| Command | Purpose |
| --- | --- |
| `nachos-maintain --profile-home H [--store S] [--days N] [--json] <check>` | Offline report-only audit. Checks: `inbox`, `summaries`, `duplicates`, `unused`, `superseded`, `suspicious_supersedes`, `provenance`, `links`, `reconciliation`, `all`. Run it from the Hermes environment's `bin/`; nothing is scheduled automatically. |
| `nachos-reconcile <inventory\|packet\|session\|export\|validate\|apply\|viewer\|skill\|cron-evidence>` | Offline review and apply workflow ([reconciliation](docs/reconciliation.md)). |
| `nachos-models` (`python -m nachos_hermes.models`) `list\|status\|reindex` | Effective model settings and re-embedding ([models](docs/models.md#switching-models)). |
| `tools/memory_audit.py`, `tools/correct_summaries.py`, `tools/import_memories.py` | Source-checkout scripts: audit/purge/history/revert with explicit paths, LLM summary correction, import ([maintenance](docs/memory-maintenance.md#source-checkout-tools)). |

## Privacy and data locations

Everything is under `$HERMES_HOME/nachos/` (per profile):

| Path | Contents |
| --- | --- |
| `memory.db` or `memory.md` | The store, the source of truth. |
| `memory.index.db` | Derived data and logs: vectors, usage counts, links, provenance, change log, inbox. Deleting it loses the change log and usage counts, never entries. |
| `snapshots/` | Gzipped transcript checkpoints for history search. Raw transcripts, like Hermes' own session DB; pruned by `checkpoint_max_age_days`. |
| `inbox/` | Queued capture jobs. |
| `reconcile/` | Review folders written by `nachos_memory_reconcile` (mode 0600). |
| `config.json` | Your settings. |

- Writes to memory are redacted with Hermes' secret redactor; checkpoints are not.
- Nothing is sent over the network by default. Network use is limited to: the first download of an embedding or reranker model (`nachos-embeddings`; `--offline` forbids it), an OpenAI-compatible endpoint if you configure one, and the auxiliary LLM you choose for capture (and `correct_summaries.py`).
- With `shared_store: on` and a path outside `$HERMES_HOME`, `hermes backup` includes the shared store.

## Troubleshooting

- **No semantic recall.** Run `/nachos-memory-status` for the active backend. `auto` needs `nachos-embeddings` on `PATH`, `sentence-transformers`, or `embedding_base_url` plus `embedding_model`. A backend that failed at runtime stays lexical until the next process start.
- **Recall is lexical right after changing a model.** Expected: re-embedding runs in steps. Check progress with `/nachos-memory-status`, or finish it with `python -m nachos_hermes.models reindex --profile-home <home>`.
- **Reranking or query/document types seem inactive.** The `nachos-embeddings` CLI is probably older than 0.6.0 and Nachos has fallen back to basic embeddings. Upgrade with `npm i -g @nacho-labs/nachos-embeddings` (`nachos-embeddings models` also needs 0.6.0).
- **A setting has no effect.** Invalid values are silently ignored in favour of the default; compare with the table above and `python -m nachos_hermes.models list`.
- **Capture suggestions never appear.** Capture needs `capture: review`, a session of 3 or more user messages, and a working auxiliary model. Failed jobs stay queued and are dropped after 5 consecutive failures.
- **`skipped: non-primary agent context`.** Writes from delegated or scheduled sessions are refused; hand the work to a primary session.
- **A hand edit to `memory.md` or `memory.db` is picked up on the next recall;** no rebuild command is needed.

## Upgrading

0.7.0 migrates stores and indexes in place on first open. Back up `$HERMES_HOME/nachos/` first: downgrading to 0.6.x against a migrated `memory.index.db` is not supported. Details and rollback steps: [CHANGELOG.md](CHANGELOG.md#070).

## Development

```bash
uv sync --group dev            # pinned dev tools, package installed editable
bash scripts/check.sh          # ruff, vulture, xenon, pylint duplicate-code, mypy, pytest
bash scripts/install-hooks.sh  # run the gate before every commit
```

`test_memory_entry_point_loads_a_callable` reads installed entry points, so install the package first (`uv sync` or `pip install .`). Design: [docs/architecture.md](docs/architecture.md). Benchmarks: [docs/benchmark.md](docs/benchmark.md).

## License

MIT
