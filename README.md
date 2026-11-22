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
| `embedding_model` | blank | Model id (blank = backend default). |
| `embedding_dtype` | blank | Precision for the `nachos` backend, e.g. `q8` (about 3x faster, cosine about 0.99 vs fp32). Blank = backend default. Changing it re-embeds. |
| `embedding_base_url` | blank | OpenAI-compatible base URL. |
| `embedding_api_key_env` | blank | Name of the env var holding the API key. |
| `embedding_query_prefix` | blank | Prepended to search queries for the `openai` and `sentence-transformers` backends, e.g. `query: ` (E5) or `search_query: ` (nomic). Spaces are kept. The `nachos` backend applies its own per-model prefixes. Changing it re-embeds. |
| `embedding_document_prefix` | blank | Prepended to stored entries for the same backends, e.g. `passage: ` (E5) or `search_document: ` (nomic). Changing it re-embeds. |
| `duplicate_threshold` | `0.65` | Similarity (0-1) above which a new entry is flagged as a likely duplicate. With `duplicate_calibration: auto` this is a minimum: it is raised to the embedding model's own random-pair 99th percentile when that is higher. Any value other than `0.65` is used exactly as given. |
| `duplicate_calibration` | `auto` | `auto` calibrates the default threshold per embedding model (cosine ranges differ a lot between models); `off` uses `duplicate_threshold` as is. |
| `similarity_floor` | `off` | `auto` drops semantic hits below the model's own 75th percentile of random-pair similarity, calibrated from the stored vectors. |
| `rerank` | `off` | `nachos` re-scores the top fused results with a cross-encoder served by `nachos-embeddings` (needs the `nachos` backend and a CLI with `--rerank-model`; silently ignored otherwise). Adds roughly 0.3 s per recall. |
| `rerank_model` | blank | Cross-encoder id (blank = `jinaai/jina-reranker-v1-tiny-en`, about 35 MB at q8). |
| `rerank_dtype` | blank | Cross-encoder precision, e.g. `q8`. |
| `rerank_candidates` | `25` | How many top fused results the reranker re-scores. |
| `related_per_hit` | `3` | Linked entries (one hop) attached as `related` to each recall result, at most 8 per query. `0` turns related expansion off. |
| `shared_store` | `off` | `on` adds a memory scope shared by every profile on the machine. |
| `shared_store_path` | blank | Shared store directory (blank = `<hermes root>/nachos-shared`). |
| `redact_secrets` | `on` | Redact secrets in everything written to memory. |
| `checkpoint_max_age_days` | `30` | Delete pre-compression transcript checkpoints older than this; `0` keeps them. |
| `capture` | `review` | `review` suggests memories from finished sessions into a review inbox; `off` disables. |

## Linked memory

Entries can reference each other. Write `[[key]]` (or `[[Entry Title]]`, or `[[key|label]]`) in a body; links inside code spans and fenced blocks are ignored.

- **Stored in the sidecar** (`links` table, forward edges, backlinks by indexed lookup) and kept in step on put, edit, remove, revert and supersede, and reconciled against hand edits. A link to a superseded entry resolves to its active successor; a link to nothing is *broken*, tolerated, and reported by `put` (`broken_links`) and exact-key recall.
- **Exact-key recall** (`nachos_memory_recall key=...`) also returns `links`, `backlinks` and `broken_links`.
- **Query recall** may attach `related` to a result: up to `related_per_hit` (default 3) one-hop neighbours, at most 8 per query, never an entry already in the result list, outgoing links in body order then backlinks by key. Links do not change ranking: a graph boost was measured and rejected (`docs/benchmark.md`).
- **Entities without an LLM.** Identifiers (`db-prim-02`, `dbt_hourly`), machine hostnames, paths, `@handles`, CamelCase and capitalised multiword names, mid-sentence capitalised words and project names used as titles are extracted with regexes into an entity to entries index. `nachos_memory_put` returns up to 3 `suggested_links`, each with a short `reason` ("shares `db-prim-02.internal`", "similar meaning and wording"). Two signals: a shared identifier, host, path, handle or project name (topical entities need two; entities found in many entries are ignored as generic), and similarity that stands out from the rest of memory (IDF-weighted word overlap, plus embedding cosine when embeddings are on, each judged as a z-score against that entry's own similarities, so it needs no per-model threshold; moderate similarity also needs a shared entity). Lexical-only mode uses the word-overlap and entity rules and works with no extra dependency. A project name must be a title's first word, never seen in lowercase, and used by another entry. `nachos_memory_recall entity=...` filters a query to entries that mention an entity, or lists them when there is no query.
- **Manifest.** Entries with links get a `[n links]` suffix only if the annotated manifest still fits `manifest_char_budget` without shortening anything more than the plain one needs.
- **Maintenance.** `nachos-maintain links` reports broken links, orphan hubs (entries named in several others that nothing links to) and entries that strongly share entities but are not linked. Report-only.

## Embeddings

Semantic recall is optional; without it recall is lexical. A failing backend drops to lexical for the rest of the process, and changing the model re-embeds on the next recall.

- `auto` picks the first available: `nachos-embeddings` on PATH, `sentence-transformers`, then an OpenAI-compatible URL (needs `embedding_base_url` and `embedding_model`).
- The `nachos` backend keeps one `nachos-embeddings serve` worker alive, which needs nachos-embeddings 0.6.0 or newer (`npm i -g @nacho-labs/nachos-embeddings`); older versions fall back automatically to one process per call.

```bash
npm i -g @nacho-labs/nachos-embeddings   # local model, found on PATH by `auto`
```

```json
{ "embeddings": "openai",
  "embedding_base_url": "http://localhost:11434/v1",
  "embedding_model": "nomic-embed-text" }
```

Any OpenAI-compatible `/embeddings` endpoint works (Ollama above); for a hosted one also set `embedding_api_key_env`. `{ "embeddings": "off" }` disables semantic recall.

## Tools

| Name | Purpose |
| --- | --- |
| `nachos_memory_recall` | Fetch an entry by `key` (returns a `superseded_by` note if superseded), or search with `query` (`limit`, optional `include_superseded`). |
| `nachos_memory_put` | Add or update an entry (`title`, `body`, optional `summary`, `category`, `supersedes`); rejects low-signal text, lists `possible_duplicates` and `suggested_supersedes`. |
| `nachos_memory_remove` | Delete an entry by `key` or `title`. |
| `nachos_history_search` | Recover details compression removed from the current conversation, including subagent results (`query`, `scope`: `conversation` or `all`, `limit`). For past sessions use Hermes' `session_search`. |

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

With `capture: review`, a finished session (3+ user messages, enough text) queues a trimmed transcript in `$HERMES_HOME/nachos/inbox/pending/`; no LLM runs at exit. A background worker later asks an auxiliary LLM for up to 5 durable facts, applies the quality gate and redaction, and stores them as suggestions. Nothing enters memory until you run `/nachos-memory-inbox approve`; the agent is told how many await review but cannot approve them.

The model comes from `auxiliary.nachos_capture.model` in `config.yaml`. **Cost warning:** if it is unset, Hermes may fall back to your main model, so pin a cheap one. If the call fails, jobs stay queued (dropped after 5 consecutive failures, or one retry on malformed output).

## Maintenance

The wheel/pip package includes `nachos-maintain`, an **offline, report-only**
command with explicit profile selection:

```bash
/path/to/venv/bin/nachos-maintain --profile-home /absolute/path/to/profile --json all
# Independently schedulable: inbox | summaries | duplicates | unused | superseded | links | all
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
