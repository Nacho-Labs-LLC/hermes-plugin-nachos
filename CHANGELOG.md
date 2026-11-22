# Changelog

## 0.7.0

### Added
- **Model settings and safe switching.** Set any embedding model explicitly (`embedding_model`; blank = backend default) with `embedding_pooling`, `embedding_max_length`, `embedding_query_prefix` and `embedding_document_prefix` overrides (`none` disables a built-in prefix). Vectors are stored per model, so changing a model keeps the old vectors, re-embeds in time-bounded steps (`reindex_budget_ms`) while recall stays lexical, then recalibrates. `/nachos-memory-status` shows progress. New `nachos-models` CLI (`list`, `status`, `reindex`).
- **Supersede tracking.** Entries have `created_at` / `updated_at`; `nachos_memory_put` accepts `supersedes`. Superseded entries leave the manifest and default search, search results redirect to the current successor (`include_superseded` opts out), and ties favour the newer entry. `put` returns `possible_updates` candidates (alias `suggested_supersedes`) with a reason.
- **Chunked body vectors.** Entries over 1200 characters are embedded in overlapping chunks (up to 16 vectors) so answers deep in a body are found.
- **Confidence and abstention.** Search results carry `confident`, `confidence`, `confidence_score` and `signals`, plus `no_confident_match` when nothing fits. `abstention`: `hint` (default), `on` (adds retry guidance for the agent), `off`. Per-model calibration priors make this and `duplicate_calibration: auto` work on small stores; `similarity_floor: auto` is opt-in.
- **Multi-query recall and reranking.** `nachos_memory_recall` accepts `queries` (up to 4 rephrasings, one embedding call, fused). Optional cross-encoder reranking through `nachos-embeddings`: `rerank: nachos` (every recall) or `borderline` (only uncertain queries), with `rerank_model`, `rerank_dtype`, `rerank_max_length`, `rerank_candidates`.
- **Linked memory.** `[[key]]` / `[[Title]]` links between entries, `related` neighbours on recall (`related_per_hit`), `links` / `backlinks` / `broken_links` on exact reads, regex entity extraction with `suggested_links` on `put`, and `nachos_memory_recall entity=...`. The manifest shows `[n links]` when it fits.
- **History search and provenance.** `nachos_history_search` ranks by IDF-weighted match, returns neighbouring turns (`history_context_before`, `history_context_after`, `history_max_chars`) and stable `ref` handles. Entries record their origin (session, time; for captured entries a verified supporting quote), shown as `source` on recall. Unverified capture suggestions are marked `[ungrounded]` and skipped by `approve all`.
- **Offline maintenance.** `nachos-maintain`: report-only checks `inbox`, `summaries`, `duplicates`, `unused`, `superseded`, `suspicious_supersedes`, `provenance`, `links`, `reconciliation`, `all`, over SQLite and flatfile stores without modifying them. See `docs/memory-maintenance.md`.
- **Reconciliation.** `nachos-reconcile` CLI, an offline review viewer, the `nachos_memory_reconcile` tool and the `nachos-reconcile` skill turn audit findings into proposals (merge, supersede, unlink, rewrite, delete, choose one) that apply only after approval, through the normal write path. See `docs/reconciliation.md`.
- **Benchmarks.** A held-out adversarial set, link and history evaluations, and flags to benchmark your own model. See `docs/benchmark.md`.

### Changed
- Faster repeat recall: per-scope lexical and metadata caches, batched usage writes, and a WAL-mode index.
- `docs/` reorganised: new `docs/models.md`; `docs/benchmark.md` rewritten around method and current results.

### Migration
- Stores and indexes migrate in place on first open: SQLite `memory.db` gains `created_at`, `updated_at`, `superseded_by` and `prev_superseded_by` columns; flatfile `memory.md` entries stay readable and new or updated entries get a `<!-- created: ... | updated: ... | superseded_by: ... -->` comment; `memory.index.db` switches to WAL, re-keys `vectors` by `(key, chunk_no, model)` and adds `calibration`, `links` and `provenance` tables.
- Use `nachos-embeddings >= 0.6.0` for input types, reranking and model overrides. Older CLIs fall back to basic embeddings; lexical-only mode still needs no extra dependency.
- New defaults: `abstention: hint`, `duplicate_calibration: auto`, `similarity_floor: off`, `rerank: off`, `rerank_candidates: 25`, `reindex_budget_ms: 250`, `related_per_hit: 3`, `history_context_before: 1`, `history_context_after: 1`, `history_max_chars: 6000`.
- `possible_updates` replaces `suggested_supersedes` in `nachos_memory_put` results; the old name remains as an alias for this release.
- Downgrading to 0.6.x against a migrated `memory.index.db` is not supported. Back up `$HERMES_HOME/nachos/` before upgrading. To roll back, restore the backup, or keep `memory.db` / `memory.md` and rename `memory.index.db` (and any `-wal` / `-shm` files) so 0.6.x builds a fresh index. A fresh index loses the change log, usage counts, links and provenance, and 0.6.x ignores `superseded_by`.

### Removed
- Nothing is removed. `embedding_normalize` is intentionally not exposed: the index always L2-normalizes vectors.

## 0.6.0

### Added
- Hybrid recall: IDF-weighted lexical search fused with semantic embeddings (`nachos-embeddings`, OpenAI-compatible endpoints or sentence-transformers); persistent `nachos-embeddings serve` worker with one-shot fallback.
- Duplicate flagging on `nachos_memory_put` (`possible_duplicates`), write-time quality gate, Hermes `agent.redact` on all writes.
- Optional `shared` scope across profiles; `backup_paths` exposes the shared store to `hermes backup`.
- Change log with revert: `/nachos-memory-history`, `/nachos-memory-revert`, `memory_audit.py history|revert`.
- Session-end capture into a review inbox (`/nachos-memory-inbox`, `auxiliary.nachos_capture.model`).
- Pre-compression checkpoints with `nachos_history_search`, session lineage across compression, `/resume` and `/branch`, and delegation capture.
- Usage telemetry, `/nachos-memory-audit`, `tools/memory_audit.py`.
- `tools/import_memories.py` for explicit import from Hermes `builtin`, `holographic` and `byterover` memory.
- New config keys (existing: `store`, `manifest_char_budget`): `embeddings`, `embedding_model`, `embedding_dtype`, `embedding_base_url`, `embedding_api_key_env`, `duplicate_threshold`, `shared_store`, `shared_store_path`, `redact_secrets`, `checkpoint_max_age_days`, `capture`.
- Embedding dimension check: the dimension from the `nachos-embeddings` 0.6 `serve` ready line is exposed as `Embedder.dimension` and every returned vector is validated against it (older CLIs without the field still work).
- `embedding_dtype` (e.g. `q8`, about 3x faster) passed to the `nachos` backend as `--dtype`; the model id includes it so a change re-embeds.
- Code-quality gate (`scripts/check.sh`: ruff, vulture, xenon, pylint duplicate-code, pytest).

### Changed
- Refreshed `uv.lock` for the dev tool pins (ruff, mypy, vulture, radon, xenon, pylint).
- Nachos is now a memory provider only and replaces Hermes built-in general memory, MEMORY.md (`memory.memory_enabled: false`). The user profile (USER.md) and SOUL.md are untouched.
- Respects Hermes' soul/user/memory layers: the injected contract routes user facts to the built-in user profile and other durable knowledge to Nachos; session capture is told what the user profile already holds and skips user and persona facts; `import_memories.py builtin` imports only MEMORY.md unless `--include-user-profile`.
- Code split into `nachos_core` (lexical, index, embeddings, capture, history, memory) and `nachos_hermes` (memory_provider, commands, capture_adapter, config, schemas).
- Checkpoints expire by age (`checkpoint_max_age_days`).
- README and architecture rewritten.

### Removed
- **Breaking:** the context engine and the `nachos-context` entry point.
- **Breaking:** the `nachos-policy` plugin.
- **Breaking:** the compaction-archive category and its tools.
- **Breaking:** `on_memory_write` mirroring and the automatic copy of Hermes built-in memory (use `import_memories.py`).
- **Breaking:** the prefetch tier and its scorer config.
