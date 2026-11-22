# Nachos Architecture

`nachos_core` holds the harness-agnostic logic (stores, index, embeddings, recall, history). `nachos_hermes` adapts it to Hermes' provider interface, tools and slash commands.

```
Hermes ── nachos_hermes (provider, tools, slash commands, capture worker)
              │
          nachos_core.MemoryService ── scope: profile [+ shared]
              ├─ Store (sqlite | flatfile)       source of truth
              ├─ Index memory.index.db           vectors, usage, change log, inbox
              └─ $HERMES_HOME/nachos/snapshots/  checkpoints; inbox/ capture jobs
```

## Where Nachos sits

Hermes has three memory layers; Nachos replaces only the last.

| Layer | Lives in | Holds | Managed by |
| --- | --- | --- | --- |
| Soul | `SOUL.md` | Persona, tone | You |
| User | `USER.md` (built-in `memory` tool, `target: "user"`) | Facts about the user | Hermes (stays on) |
| Memory | Nachos store (was `MEMORY.md`) | Projects, environment, decisions, procedures, history | Nachos |

The injected contract tells the agent to route user facts to the user profile and everything else durable to `nachos_memory_put`. Session capture reads `USER.md` through an injected parameter (nachos_core stays Hermes-free) and tells the extraction model not to propose user or persona facts.

## Components

- **MemoryService** is the API the adapter calls. It owns one or two **scopes**: `profile` (always) and an optional `shared` scope used by every Hermes profile on the machine.
- **Store**: SQLite (default) or a flat file. It is the source of truth and holds title, summary, category, body, plus per-entry `created_at`, `updated_at`, and optional `superseded_by` metadata (stored as columns in SQLite and as a backward-compatible `<!-- created: ... | updated: ... | superseded_by: ... -->` comment line under `## Title` in flatfile markdown; legacy entries backfill timestamps from the change log, `usage.first_seen`, or migration time and are marked approximate in status).
- **Sidecar index**: `memory.index.db` next to the store, holding embedding vectors and usage counts. It is derived data; deleting it loses nothing but counts.
- **Manifest**: a `title - summary` table of contents in the system prompt. It never omits an active entry; over budget, summaries shorten and titles stay. Superseded entries are hidden by default and summarised as a count footer (`N superseded entries hidden.`).

Why a sidecar: flat-file memory stays human-editable, and a hand edit can never leave the store inconsistent.

## Recall

The agent calls `nachos_memory_recall` when the manifest shows something relevant. Recall is hybrid: a lexical pass weights query terms by IDF (title and summary count more than body), and with an embedder a vector pass ranks by cosine similarity. The lists are fused with reciprocal-rank fusion, which needs no score calibration between a term weight and a similarity. Without an embedder, or after an embedder error, recall is lexical only. The `nachos` backend's fast `serve` worker needs nachos-embeddings 0.6.0 or newer (`npm i -g @nacho-labs/nachos-embeddings`); older versions fall back automatically to one process per call.

Each entry gets a head vector (`chunk_no = 0`: title, summary and body head). Entries whose body exceeds one chunk (`CHUNK_SIZE = 1200` chars) also get overlapping body-chunk vectors (`~1200`-char chunks with `~200`-char overlap snapped to paragraph/line/sentence boundaries, each prefixed with the title), capped at `MAX_CHUNKS_PER_ENTRY = 16` vectors per entry (1 head + up to 15 body chunks). Vectors are stored under `(key, chunk_no)` with their model id and content hash and reconciled lazily on the next query (existing pre-chunking `vectors` tables migrate in place automatically and rebuild on sync), so hand edits and model changes need no rebuild step or background process. Recall scores each entry by the maximum cosine similarity across its vectors, while duplicate detection (`find_similar`, `duplicate_pairs`) compares head vectors (`chunk_no = 0`) only.

Ranking, in order, on every recall:

1. **Chunk penalty.** An entry's vector score is its head score, or its best body-chunk score minus `0.05` when that is higher (`MemoryService.chunk_penalty`). Chunks are narrower evidence than the head, and without the penalty a chunk matching on one sentence outranks a head that matches the whole topic (paraphrase queries lost hit@1 to this).
2. **Calibrated floor (optional, `similarity_floor: auto`).** Cosine ranges differ between models (random pairs sit near 0.1 for MiniLM and 0.5 for bge), so no absolute cutoff is portable. The index measures each model's own spread, a fixed-seed sample of random head-vector pairs, caches it in the sidecar `calibration` table keyed by `model_id`, and resamples when the head count drifts by more than 20%. Semantic hits below the 75th percentile of that spread are dropped. Stores with fewer than 30 entries are not calibrated and get no floor.
3. **Fusion.** Lexical hits (only those covering at least 40% of the IDF-weighted query terms when embeddings are on) and semantic hits are merged with reciprocal-rank fusion.
4. **Rerank (optional, `rerank: nachos`).** The top `rerank_candidates` fused results go to a cross-encoder served by the same `nachos-embeddings serve` worker (`{"op": "rerank"}`; the ready line announces `"rerank": {"model": ...}`). Each candidate is read as title, summary and its best-matching body chunk (the head's body start for lexical-only hits). The final order blends the rerank rank with the fused rank. If the CLI predates reranking, the worker is started without the flags and the setting is ignored with one logged warning; a rerank error disables it for the session and keeps the fused order.

`duplicate_threshold` is calibrated the same way: with `duplicate_calibration: auto` the effective threshold is `max(duplicate_threshold, model p99 of random-pair head similarity)`, so a model whose unrelated entries already score 0.7 does not flag half of the store. A threshold other than the 0.65 default is never overridden.

There is no speculative prefetch: it adds a fixed per-turn cost for a weak title-only signal, and a better summary fixes a missed recall.

## Writes

Every write (title, summary, body) goes through an injected `redact` function; the adapter supplies Hermes' own redactor, so patterns stay in step. A quality gate (`nachos_core/quality.py`) rejects empty, placeholder or too-short summaries and bodies. Duplicate flagging compares embedding similarity against `duplicate_threshold`, or token overlap without an embedder, and returns `possible_duplicates`. When a duplicate looks like an outdated version (explicit update words such as "now"/"moved to"/"replaced", or changed numbers/versions/dates/paths), the response also returns `suggested_supersedes` without applying it automatically. Passing `supersedes` (key or list of keys) marks the old entries with `superseded_by` without deleting them; superseded entries leave the manifest and default search (still fetchable by exact key with a pointer note, or via `include_superseded=true`), and near-equal fused recall scores break ties in favour of the newer `updated_at`. Both the new write and each `supersede` transition are logged in `changes` so `revert` undoes them. It flags instead of blocking because the agent knows whether two similar entries are the same fact.

Nothing is copied from other providers automatically, so memory never changes without the user asking; `tools/import_memories.py` imports explicitly through this same `put()` path (session id `import:<source>`).

Each mutation first records the entry's previous title, summary, category and body in the index's `changes` table (with the session id), so an overwrite or removal can be undone by `MemoryService.revert`; reverts are logged too. The log is not touched by `delete()` or `retain()`, so removed entries stay revertible, and it is trimmed to the newest 1000 rows. Like the rest of the index it is telemetry-grade: deleting the index file loses undo history, never memories.

## Checkpoints and history

Before each Hermes compression pass the provider saves the full transcript as a gzipped checkpoint (`on_pre_compress`). `nachos_history_search` finds messages containing all query words, de-duplicates across overlapping checkpoints and returns excerpts. Checkpoints stay out of memory entries so generated text never reaches the manifest. They are raw transcripts, so they can contain anything said in the conversation.

The provider tracks a session lineage (`on_session_switch`): compression, `/resume` and `/branch` append the new session id, `/new` and `/reset` start over. `nachos_history_search` defaults to `scope="conversation"` (the lineage), the gap Hermes' own `session_search` leaves, since that tool covers past sessions but not the current one; `scope="all"` also searches older Nachos checkpoints. `on_pre_compress` returns a note naming the checkpoint id and pointing the summary at `nachos_history_search`. `on_delegation` saves each subagent task and result as a `delegation` snapshot, so they are searchable too. `backup_paths` reports the shared-store directory when it lies outside the Hermes home.

Retention is by age (`checkpoint_max_age_days`), not count: the oldest checkpoints hold what later compression dropped, so "keep the newest N" would delete the history worth searching.

## Session capture

`on_session_end` only writes a gzipped, trimmed transcript (user/assistant text, newest ~60k chars) to `inbox/pending/`, because the process is often exiting and no LLM call belongs there. `nachos_core/capture.py` later claims each job by atomic rename into `inbox/processing/` (so two Hermes processes never take the same job), asks an injected `llm(messages) -> str` for at most 5 candidates, and parses defensively: a malformed answer gets one retry, then the job is dropped. Candidates pass the quality gate and redaction, are annotated with duplicates (or marked as updates when the key exists) and land in the profile index's `inbox` table. Candidates wait for review rather than auto-writing because an extractor LLM will invent or over-generalise facts, and only a person should decide what becomes permanent. Approval goes through `MemoryService.put`, so the change log records it; decided rows are trimmed after 30 days. The Hermes adapter supplies the callable via `call_llm(task="nachos_capture")`.

## Usage telemetry

The index counts first-seen time, exact-key recalls and query hits per entry. Nothing reads it on the hot path; it feeds `/nachos-memory-audit` and `tools/memory_audit.py` to surface never-recalled entries and near-duplicates. Deleting is left to a person or the agent.

To keep recall latency low, `memory.index.db` opens in SQLite WAL mode with `PRAGMA synchronous=NORMAL` for usage telemetry (avoiding a per-query `fsync`) and checkpoints on normal close. On an abrupt crash or power loss before a WAL checkpoint, the last few seconds of usage counts may be lost, while `changes` undo-log writes stay committed immediately under `PRAGMA synchronous=FULL`.
