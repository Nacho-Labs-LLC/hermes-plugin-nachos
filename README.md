# Nachos for Hermes Agent

Nachos is a local-first durable-memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com). The system prompt carries a small manifest (a table of contents) of what the agent remembers, and the agent pulls full entries on demand. It replaces Hermes' built-in general memory (MEMORY.md); the user profile and SOUL.md stay as they are. Recall combines keyword and semantic search, likely duplicates are flagged on write, edits are logged and revertible, finished sessions suggest memories into a review inbox, and details lost to compression stay searchable. Secrets are redacted on write. No LLM call happens in the request path.

## How Nachos fits

Hermes has three layers. Nachos replaces only the third.

| Layer | Lives in | Holds | Managed by |
| --- | --- | --- | --- |
| Soul | `SOUL.md` | Who the agent is: persona, tone | You; Nachos leaves it alone |
| User | `USER.md` (built-in `memory` tool, `target: "user"`) | Facts about you: preferences, style, personal details | Hermes (kept on) |
| Memory | Nachos store (was `MEMORY.md`) | Projects, environment, decisions, procedures, history | Nachos |

The agent is told this split in the system prompt, and session capture skips anything that belongs to the soul or user layers.

## Install

Into the Python environment that runs Hermes:

```bash
python -m pip install "git+https://github.com/Nacho-Labs-LLC/hermes-plugin-nachos.git@v0.6.0"
```

Catalog alternative: keep the repository root at `$HERMES_HOME/plugins/nachos/`.

## Enable

```bash
hermes config set memory.provider nachos
hermes config set memory.memory_enabled false
```

This disables only the built-in general memory (MEMORY.md); import it first if you have one (below). Leave `memory.user_profile_enabled` on (the default): the user profile stays in use. Start a new session afterwards.

## Migrate

Nachos never copies other memory automatically. Import once with `python tools/import_memories.py <source>`, where source is `builtin` (MEMORY.md; add `--include-user-profile` to also import USER.md as category `user`, which Hermes keeps using anyway), `holographic` or `byterover` (`--category`, `--hermes-home`, `--source-path`, `--json`). It is a dry-run until `--run`, only reads the source, and uses the normal write path (quality gate, redaction, duplicate flagging, change log). Key collisions with different content are skipped unless `--overwrite`. Server-backed providers (hindsight, honcho, mem0, openviking, retaindb, supermemory) are unsupported; `--help` explains importing a manual export.

## Configuration

Run `hermes memory setup` or edit `$HERMES_HOME/nachos/config.json`. Settings are per profile.

| Key | Default | Meaning |
| --- | --- | --- |
| `store` | `sqlite` | Store format: `sqlite` or hand-editable `flatfile`. |
| `manifest_char_budget` | `1200` | Soft size target for the manifest; summaries shorten, entries are never dropped. |
| `embeddings` | `auto` | Backend: `auto`, `nachos`, `openai`, `sentence-transformers` or `off`. |
| `embedding_model` | blank | Model id (blank = backend default, MiniLM for `nachos`; see [Choosing a model](#choosing-a-model)). For the `nachos` backend any Hugging Face ONNX encoder or a local directory path. |
| `embedding_dtype` | blank | Precision for the `nachos` backend, e.g. `q8` (about 3x faster, cosine about 0.99 vs fp32). Blank = backend default. Changing it re-embeds. |
| `embedding_pooling` | blank | `mean` or `cls` for the `nachos` backend. Blank = the model's built-in profile. Changing it re-embeds. |
| `embedding_max_length` | `0` | Truncate inputs to this many tokens for the `nachos` backend (`0` = model default; can only lower the tokenizer limit). Changing it re-embeds. |
| `embedding_base_url` | blank | OpenAI-compatible base URL. |
| `embedding_api_key_env` | blank | Name of the env var holding the API key. |
| `embedding_query_prefix` | blank | Prepended to search queries, e.g. `query: ` (E5) or `search_query: ` (nomic). Spaces are kept. For the `nachos` backend blank keeps the model's built-in prefix and `none` disables it; the `openai` and `sentence-transformers` backends have no built-in (blank = no prefix). |
| `embedding_document_prefix` | blank | Prepended to stored entries, e.g. `passage: ` (E5) or `search_document: ` (nomic); same blank / `none` rules. Changing it re-embeds. |
| `duplicate_threshold` | `0.65` | Similarity (0-1) above which a new entry is flagged as a likely duplicate. With `duplicate_calibration: auto` this is a minimum: it is raised to the embedding model's own random-pair 99th percentile when that is higher. Any value other than `0.65` is used exactly as given. |
| `duplicate_calibration` | `auto` | `auto` calibrates the default threshold per embedding model (using offline per-model priors below 30 entries and blending into the store's own random-pair distribution from 30 to 100 entries); `off` uses `duplicate_threshold` as is. |
| `similarity_floor` | `off` | `auto` drops semantic hits below the model's own 75th percentile of random-pair similarity (using priors below 30 entries and blending into empirical vectors from 30 to 100 entries). |
| `abstention` | `hint` | `hint` (default) attaches per-query confidence metadata (`confident`, `confidence`, `confidence_score`, `signals`) and `no_confident_match: true` as informational hints while keeping candidates listed (no automatic retry instruction in the tool description); `on` adds retry-with-rephrasings guidance to `nachos_memory_recall`; `off` returns only the raw result list. |
| `rerank` | `off` | `nachos` re-scores the top fused results on every recall with a cross-encoder served by `nachos-embeddings` (needs the `nachos` backend and a CLI with `--rerank-model`; adds ~0.2–0.3 s per recall); `borderline` re-scores only borderline-confidence queries (top 5 candidates, ~10–13% of queries) to verify abstention at near-zero median latency; `off` disables reranking. |
| `rerank_model` | blank | Cross-encoder id (blank = `jinaai/jina-reranker-v1-tiny-en`, about 35 MB at q8). |
| `rerank_dtype` | blank | Cross-encoder precision, e.g. `q8`. |
| `rerank_max_length` | `0` | Cross-encoder input length in tokens (`0` = the model's default: 1024 for the jina reranker, 512 otherwise). |
| `rerank_candidates` | `25` | How many top fused results the reranker re-scores. |
| `reindex_budget_ms` | `250` | After an embedding model change, how long one recall or write may spend re-embedding (see [Switching models](#switching-models)). |
| `shared_store` | `off` | `on` adds a memory scope shared by every profile on the machine. |
| `shared_store_path` | blank | Shared store directory (blank = `<hermes root>/nachos-shared`). |
| `redact_secrets` | `on` | Redact secrets in everything written to memory. |
| `checkpoint_max_age_days` | `30` | Delete pre-compression transcript checkpoints older than this; `0` keeps them. |
| `history_context_before` | `1` | `nachos_history_search`: turns returned before each matched message (a tool call and its result are always kept together). |
| `history_context_after` | `1` | Turns returned after each matched message. |
| `history_max_chars` | `6000` | Cap on the total text one `nachos_history_search` call returns. |
| `capture` | `review` | `review` suggests memories from finished sessions into a review inbox; `off` disables. |

## Embeddings

Semantic recall is optional; without it recall is lexical. A failing backend drops to lexical for the rest of the process.

- `auto` picks the first available: `nachos-embeddings` on PATH, `sentence-transformers`, then an OpenAI-compatible URL (needs `embedding_base_url` and `embedding_model`).
- The `nachos` backend keeps one `nachos-embeddings` worker alive (`npm i -g @nacho-labs/nachos-embeddings`). It needs 0.6.0 or newer; older versions fall back automatically to one process per call. The model-override settings below (`embedding_pooling`, `embedding_max_length`, prefixes, `rerank_max_length`) need a CLI release that has `--pooling`, `--max-length` and `--rerank-max-length`; settings you leave blank are never sent, so an older CLI keeps working with the others.

```bash
npm i -g @nacho-labs/nachos-embeddings   # local model, found on PATH by `auto`
```

```json
{ "embeddings": "openai",
  "embedding_base_url": "http://localhost:11434/v1",
  "embedding_model": "nomic-embed-text" }
```

Any OpenAI-compatible `/embeddings` endpoint works (Ollama above); for a hosted one also set `embedding_api_key_env`. `{ "embeddings": "off" }` disables semantic recall.

### Choosing a model

Name the model you want; there are no presets. Blank `embedding_model` means the backend default (MiniLM through `nachos-embeddings`). For example, the best measured model:

```json
{ "embeddings": "nachos",
  "embedding_model": "Alibaba-NLP/gte-modernbert-base",
  "embedding_dtype": "q8" }
```

Our benchmark (synthetic fixtures in `bench/data`: 208 memories / 393 answerable queries, plus a held-out adversarial set; reranker off; `q8`; p50 is median search latency on one idle 16-thread CPU, so it varies with the machine):

| `embedding_model` (dtype q8) | Download | Dims | hit@1 / hit@5 / MRR | Adversarial hit@1 | p50 | Note |
| --- | --- | --- | --- | --- | --- | --- |
| *(`embeddings: off`)* | none | - | 0.695 / 0.812 / 0.748 | 0.975 | 0.2 ms | Lexical only. |
| blank, or `Xenova/all-MiniLM-L6-v2` | 23 MB | 384 | 0.758 / 0.924 / 0.832 | 0.988 | 8.2 ms | Default; smallest and fastest. |
| `Alibaba-NLP/gte-modernbert-base` | 150 MB | 768 | **0.773 / 0.949 / 0.850** | **1.000** | 20.5 ms | Best measured; about 2x the embedding time and a 768-dim index. |
| `Snowflake/snowflake-arctic-embed-s` | 34 MB | 384 | 0.753 / 0.911 / 0.824 | 0.988 | 9.3 ms | Retrieval-tuned; ties MiniLM here. |
| `Xenova/bge-small-en-v1.5` | 34 MB | 384 | 0.728 / 0.916 / 0.807 | 0.988 | 16.0 ms | Did not beat MiniLM here. |
| `Xenova/e5-small-v2` | 34 MB | 384 | 0.753 / 0.898 / 0.818 | 0.975 | 15.2 ms | Did not beat MiniLM here. |
| `nomic-ai/nomic-embed-text-v1.5` | 137 MB | 768 | 0.748 / 0.924 / 0.825 | 1.000 | 92.4 ms | Several times slower (p50 measured on a busier machine). |

Differences under about 0.01 hit@1 (4 queries) are not significant on this fixture. The default `embedding_dtype` is blank (fp32, about 4x the download); `q8` is what was measured. Measure any model on your own machine with `python -m bench --model <id> --dtype q8` (see [docs/benchmark.md](docs/benchmark.md)).

Changing `embedding_model` (or dtype) re-embeds safely: old vectors are kept until the new model has covered every entry, and recall stays lexical meanwhile (see [Switching models](#switching-models)). For the `nachos` backend `embedding_model` accepts any Hugging Face ONNX encoder, or a local directory (`./models/my-model`, fully offline); `python -m nachos_hermes.models list` shows the effective settings and `nachos-embeddings models` the models it ships profiles for.

**Overrides.** Built-in per-model profiles (pooling, query/document prefixes) are only defaults. Set `embedding_pooling`, `embedding_max_length`, `embedding_query_prefix` and `embedding_document_prefix` to override them for the `nachos` backend (prefix: blank = the model's own, `none` = no prefix). The reranker takes `rerank_model`, `rerank_dtype` and `rerank_max_length` the same way. The embedding overrides except the query prefix change stored vectors, so changing one re-embeds (below); reranker settings never do. (Vectors are L2-normalized by the index, so the CLI's `--no-normalize` has no use here and is not exposed.)

### Switching models

Changing any embedding setting (model, dtype, pooling, max length, document prefix) puts the profile's vectors in a different space. Nachos switches safely:

- **Old vectors are kept** until the new model has embedded every entry, so a crash or a second change loses nothing and switching back is instant. Only then are the old vectors dropped and the new model's similarity thresholds (`duplicate_calibration`, `similarity_floor`) recalibrated from the new vectors.
- **Recall never waits for the whole index.** A few edited entries are embedded inline as before. More than 24 stale entries (a model switch, a first run on an existing store) are re-embedded in small, time-sized batches: about `reindex_budget_ms` (default 250 ms) per recall or write, plus the entry in flight (a long entry is up to 16 chunks). Measured on the bench fixture switching MiniLM to gte-modernbert (208 entries): the first recall used to block for 17.2 s; now recalls during the switch take a median of 314 ms (max 1.8 s) and the switch completes after 43 recalls.
- **Lexical meanwhile.** Until a scope is fully re-embedded, recall, duplicate checks and update hints use lexical matching only; partially embedded rankings would mix two vector spaces.
- **Status.** `/nachos-memory-status` shows `re-embedding: profile 40/208 ...` while it runs. From a shell (read-only, no model loaded):

```bash
python -m nachos_hermes.models status  --profile-home ~/.hermes/profiles/default   # coverage per model, needs_reindex
python -m nachos_hermes.models reindex --profile-home ~/.hermes/profiles/default   # finish now (loads the model)
python -m nachos_hermes.models list    [--profile-home H]                          # effective settings, auxiliary LLMs
```

### Language models Nachos can call

These are chat models, not embedding models, and use Hermes' own `auxiliary` settings in `config.yaml`; Nachos does not change their semantics. Both can cost money if left unset (Hermes may fall back to your main model), so pin a cheap one.

| Setting | Used by | Notes |
| --- | --- | --- |
| `auxiliary.nachos_capture.model` | Session-end capture (`capture: review`) | Suggests memories into the review inbox. |
| `auxiliary.nachos_summary.model` | `tools/correct_summaries.py` | Writes replacement summaries; dry-run still calls the model. |

`python -m nachos_hermes.models list` prints both next to the embedding settings, with their current values when Hermes' config is readable.

## Tools

| Name | Purpose |
| --- | --- |
| `nachos_memory_recall` | Fetch an entry by `key` (returns a `superseded_by` note if superseded), or search with `query` and/or up to 4 rephrasings in `queries` (batched in one embedding call and fused with RRF; `limit`, optional `include_superseded`). With `abstention: hint` (default) or `on`, weak searches flag `no_confident_match: true` while keeping low-confidence candidates visible. |
| `nachos_memory_put` | Add or update an entry (`title`, `body`, optional `summary`, `category`, `supersedes`); rejects low-signal text, lists `possible_duplicates` and `suggested_supersedes`. |
| `nachos_memory_remove` | Delete an entry by `key` or `title`. |
| `nachos_history_search` | Recover details compression removed from the current conversation, including subagent results (`query`, `scope`: `conversation` or `all`, `limit`, `before`/`after` context turns). Hits are ranked by IDF-weighted word match (no need for every word; one common word cannot match a multi-word query), newest first on ties, and come back with the neighbouring turns labelled by role and position. Every result has a stable `ref` (`<session>@<checkpoint>#<index>`); pass `ref` (or a memory entry's `source.ref`) to reopen that spot. For past sessions use Hermes' `session_search`. |

With `shared_store: on`, recall, put and remove also take `scope: "shared"`; the default is the private profile scope. The provider follows session lineage across compression, `/resume` and `/branch`, so history search covers the whole logical conversation.

## Slash commands

| Command | Purpose |
| --- | --- |
| `/nachos-memory-status` | Store, embedding backend, entries per scope, checkpoints, capture queue. |
| `/nachos-memory-list` | Render the current manifest. |
| `/nachos-memory-audit [days]` | Likely duplicates, entries not recalled in N days, and long-superseded entries (default 30). |
| `/nachos-memory-history [key]` | Recent memory changes (create/update/supersede/remove/revert, all scopes), each with a `scope:id` handle. |
| `/nachos-memory-revert <scope:id>` | Undo a change; a revert can itself be reverted. |
| `/nachos-memory-inbox [show <id> \| approve <id\|all> \| reject <id\|all>]` | Review memories suggested from past sessions. |

## Session capture

With `capture: review`, a finished session (3+ user messages, enough text) queues a trimmed transcript in `$HERMES_HOME/nachos/inbox/pending/`; no LLM runs at exit. A background worker later asks an auxiliary LLM for up to 5 durable facts, applies the quality gate and redaction, and stores them as suggestions. The extractor must give each suggestion a verbatim supporting quote; Nachos checks it against the transcript (whitespace/case-insensitive, `...` elisions allowed) and flags suggestions whose quote it cannot find as `[ungrounded]`. They are kept for you to judge, but `approve all` skips them (approve by id). Nothing enters memory until you run `/nachos-memory-inbox approve`; the agent is told how many await review but cannot approve them.

### Provenance

Every saved entry remembers where it came from, in a `provenance` table of the index (so both store backends work, entries saved earlier simply have none, and older index files migrate in place). A manual save records the session and time. An approved suggestion also keeps its verified supporting quote (redacted like everything else) and, when the quote can be found in a transcript checkpoint, a `<session>@<checkpoint>#<index>` ref. `nachos_memory_recall` shows a compact `source` (`session`, `date`, `ref`; `via: capture`; `ungrounded` when approved unverified) on search results and also the `evidence` quote on an exact-key read; pass `source.ref` to `nachos_history_search` to reopen the original context while the checkpoint still exists (`checkpoint_max_age_days`). `nachos-maintain provenance` lists entries with no recorded origin, as information only.

The model comes from `auxiliary.nachos_capture.model` in `config.yaml`. **Cost warning:** if it is unset, Hermes may fall back to your main model, so pin a cheap one. If the call fails, jobs stay queued (dropped after 5 consecutive failures, or one retry on malformed output).

## Maintenance

The wheel/pip package includes `nachos-maintain`, an **offline, report-only**
command with explicit profile selection:

```bash
/path/to/venv/bin/nachos-maintain --profile-home /absolute/path/to/profile --json all
# Independently schedulable: inbox | summaries | duplicates | unused | superseded | suspicious_supersedes | provenance | all
/path/to/venv/bin/nachos-maintain --profile-home /absolute/path/to/profile --days 90 unused
```

No LLM/network, embeddings, memory edits, automatic approvals, deletions, or
merges. `--store flatfile` selects `memory.md`; default is SQLite. Existing
rollback-mode store/index files are opened read-only (`mode=ro` + `query_only`);
WAL-mode databases (and any `-wal` sidecar) are copied into a private temporary
directory before auditing so original files and sidecars are never created,
checkpointed, or modified. Missing stores fail, and missing usage history is
reported as unknown. Findings are review hints: old/unused does not mean false.
See [memory maintenance](docs/memory-maintenance.md) for the JSON contract,
separate daily/weekly/monthly cron recipes, summary review/correction, human
consolidation, safety boundaries, and recovery limitations. Nothing is scheduled
automatically.

Existing **source-checkout** tools remain available with `--store {sqlite,flatfile}`
and `--path` (default `$HERMES_HOME/nachos/memory.db`):

- `tools/memory_audit.py` (`--embeddings`, `--json`): `duplicates`, `unused [--days N]`, `purge-category NAME`, `history [--key K] [--limit N]`, `revert ID`. No content changes unless `--run`, but opening these legacy tools can initialize store/index schema; use `nachos-maintain` for strictly report-only audits.
- `tools/correct_summaries.py [--limit N] [--run]`: generates replacement summaries with `auxiliary.nachos_summary.model`. **Dry-run still calls an LLM**; `--run` writes directly, outside the service change log/redactor. Keep this separately approved, profile-scoped and cost-pinned, not a default cron action.
- The change log (previous content of every service create, update, remove and revert; newest 1000 kept) lives in the sidecar `memory.index.db`.

## Privacy

Data lives under `$HERMES_HOME/nachos/` (store, `memory.index.db`, `snapshots/`, `inbox/`). Memory writes are redacted; checkpoints are raw transcripts, like Hermes' own session DB. With `shared_store: on` and a path outside `$HERMES_HOME`, `hermes backup` includes the shared store.

## Development

```bash
uv sync --group dev            # pinned dev tools, package installed editable
bash scripts/check.sh          # ruff, vulture, xenon, pylint duplicate-code, pytest
bash scripts/install-hooks.sh  # run the gate before every commit
```

`test_memory_entry_point_loads_a_callable` reads installed entry points, so install the package first (`uv sync` or `pip install .`). Design notes: [docs/architecture.md](docs/architecture.md). Changes: [CHANGELOG.md](CHANGELOG.md).

## License

MIT
