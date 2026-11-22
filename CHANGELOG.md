# Changelog

## Unreleased

### Added
- Entry timestamps (`created_at`, `updated_at`) and `supersedes` support in both SQLite (automatic in-place migration, read-only safe on unmigrated DBs) and flatfile markdown (backward-compatible `<!-- created: ... | updated: ... | superseded_by: ... -->` metadata comments), with backfill from change log / `usage.first_seen` / migration time marked approximate in status.
- `nachos_memory_put(..., supersedes=...)` archives replaced entries without deleting them (logged as `supersede` in the change log for full `revert` support), hides them from the manifest (with a hidden-count footer), redirects superseded search hits to their active successor across `superseded_by` chains (cycle-safe, deduplicated, running before reranking; `include_superseded=true` opts in to raw archived hits), adds a `superseded_by` pointer note on exact-key fetch, surfaces up to 3 `possible_updates` candidates (ranked by head-vector or cached IDF-weighted lexical similarity, gated by the model's calibrated `p99` floor when available, with short machine-made `reason` strings including facet-qualifier differences; `suggested_supersedes` kept as a one-release alias), breaks near-equal recall ties by `updated_at` when entries are at least 60 s apart, and reports long-superseded and suspicious facet-mismatch supersedes in `/nachos-memory-audit` and `nachos-maintain`.
- Ranking calibration and optional cross-encoder reranking (`nachos_core/ranking.py`): body-chunk score penalty (`chunk_penalty = 0.05`), per-model random head-pair cosine calibration cached in a new `calibration` index table powering `duplicate_calibration: auto` (`p99`) and opt-in `similarity_floor: auto` (`p75`, default `off`), and optional `rerank: nachos` (`rerank_model`, `rerank_dtype`, `rerank_candidates`, `sum:1` blend, default `off`).
- Adversarial evaluation fixture (`bench/data/adversarial/`), `--data-dir`, `--suggest-eval` (`recall@1`, `recall@3`, distinct `fp@1`), and 30 `negative` queries in `bench/data/queries.json`.
- Chunked body vectors (`chunk_body`, `embed_chunks`): long entries (>1200-char bodies) store a head vector (`chunk_no = 0`) plus overlapping title-prefixed ~1200-char / ~200-char-overlap body chunks (capped at 16 vectors per entry) under `(key, chunk_no)` with automatic in-place schema migration, max-over-chunks recall scoring, and head-only duplicate detection.
- `nachos-maintain` CLI (`python -m nachos_hermes.maintenance`): offline, report-only memory audit (`inbox`, `summaries`, `duplicates`, `unused`, `superseded`, `suspicious_supersedes`, `all`) with required absolute `--profile-home`, `--json` output, and no writes to store or index files.
- Read-only SQLite mode (`read_only=True`) for `SqliteStore` and `MemoryIndex` using `mode=ro` + `PRAGMA query_only = ON`, with private temporary snapshot support for WAL-mode databases so original `.db`, `-wal`, and `-shm` files are never created, checkpointed, or modified.
- `MemoryService.duplicate_pairs(sync=False)` for offline lexical duplicate audits without mutating index vectors or usage telemetry.
- Shared `nachos_core.quality.flag_reason` reused by `nachos-maintain` and `tools/correct_summaries.py`.
- `docs/memory-maintenance.md` covering profile selection, the read-only SQLite boundary, JSON contract, cron recipes, and human review workflow.

- `embedding_preset` (`small` = MiniLM q8, `balanced` = arctic-embed-s q8, `quality` = gte-modernbert q8) for the `nachos` backend, chosen from the recall benchmark; `embedding_model` / `embedding_dtype` override it. New `embedding_pooling`, `embedding_max_length`, `rerank_max_length` settings, and `embedding_query_prefix` / `embedding_document_prefix` now also override the nachos model's built-in prefixes (blank keeps the model's, `none` disables). Overrides are passed to the `nachos-embeddings` CLI only when set and are part of the model id, so changing one re-embeds. They need a CLI release with the matching flags.
- Safe model switching (`nachos_core/reindex.py`): the vector index keeps one row per entry and model (automatic in-place migration), so a switch keeps the old vectors until the new model has embedded every entry; more than 24 stale entries are re-embedded in time-bounded steps (`reindex_budget_ms`, default 250) while recall, duplicate checks and update hints stay lexical; on completion the old vectors are dropped and similarity thresholds are recalibrated. `/nachos-memory-status` shows progress.
- `python -m nachos_hermes.models` (also `nachos-models`): `list` (presets, effective model settings, and the `auxiliary.nachos_capture` / `auxiliary.nachos_summary` LLM models), read-only `status` (coverage per model, `needs_reindex`) and `reindex` (finish a switch now).
- `python -m bench --model <id>` / `--preset <name>` now run the model beside the lexical baseline by default, with `--pooling`, `--max-length` and `--rerank-max-length` passthroughs. Docs: "Choosing a model" matrix in the README and docs/benchmark.md.

### Changed
- Per-scope in-memory lexical cache (tokenised/stemmed title, summary and body term sets, postings and document frequencies) validated via cheap store fingerprints (`stat` for flatfile, `PRAGMA data_version` + write counter for SQLite) and updated/invalidated on all write paths, plus batched `record_hits` commits during search hydration.
- SQLite `MemoryIndex` opens in WAL mode with `PRAGMA synchronous=NORMAL` for usage telemetry (checkpointed on close) while keeping `PRAGMA synchronous=FULL` for `changes` undo-log commits.
- `MemoryService.find_similar` uses the cached `LexicalIndex` label postings to prune lexical duplicate candidates instead of scanning every entry, and `MDStore` caches parsed entries by fingerprint to avoid re-reading the file per hydrated search result.

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
