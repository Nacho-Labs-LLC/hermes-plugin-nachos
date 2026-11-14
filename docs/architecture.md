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
- **Store**: SQLite (default) or a flat file. It is the source of truth and holds title, summary, category and body only.
- **Sidecar index**: `memory.index.db` next to the store, holding embedding vectors and usage counts. It is derived data; deleting it loses nothing but counts.
- **Manifest**: a `title - summary` table of contents in the system prompt. It never omits an entry; over budget, summaries shorten and titles stay.

Why a sidecar: flat-file memory stays human-editable, and a hand edit can never leave the store inconsistent.

## Recall

The agent calls `nachos_memory_recall` when the manifest shows something relevant. Recall is hybrid: a lexical pass weights query terms by IDF (title and summary count more than body), and with an embedder a vector pass ranks by cosine similarity. The lists are fused with reciprocal-rank fusion, which needs no score calibration between a term weight and a similarity. Without an embedder, or after an embedder error, recall is lexical only. The `nachos` backend's fast `serve` worker needs nachos-embeddings 0.6.0 or newer (`npm i -g @nacho-labs/nachos-embeddings`); older versions fall back automatically to one process per call.

Vectors are stored with their model id and content hash and reconciled lazily on the next query, so hand edits and model changes need no rebuild step or background process.

There is no speculative prefetch: it adds a fixed per-turn cost for a weak title-only signal, and a better summary fixes a missed recall.

## Writes

Every write (title, summary, body) goes through an injected `redact` function; the adapter supplies Hermes' own redactor, so patterns stay in step. A quality gate (`nachos_core/quality.py`) rejects empty, placeholder or too-short summaries and bodies. Duplicate flagging compares embedding similarity against `duplicate_threshold`, or token overlap without an embedder, and returns `possible_duplicates`. It flags instead of blocking because the agent knows whether two similar entries are the same fact.

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
