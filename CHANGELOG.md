# Changelog

## 0.7.0 (TBD)

### Added
- **Models & safe switching**: explicit `embedding_model` (blank = backend default) plus `embedding_pooling`, `embedding_max_length`, `rerank_max_length`, and `embedding_query_prefix` / `embedding_document_prefix` overrides (including for the `nachos` backend; `none` disables a built-in prefix). Safe model switching (`nachos_core/reindex.py`) keys vectors by `(key, chunk_no, model)` with automatic in-place migration, keeps old vectors until re-embedding completes in time-bounded steps (`reindex_budget_ms`, default 250) while recall stays lexical, then purges old models and recalibrates thresholds; `/nachos-memory-status` shows progress. New `nachos-models` CLI (`python -m nachos_hermes.models` `list` | `status` | `reindex`).
- **Minimum `nachos-embeddings >= 0.6.0` CLI version check**: `CliEmbedder` detects CLI semver via `--version` and/or the `serve` ready line (`version`), logging a one-time warning and degrading gracefully to basic embeddings (disabling input types, reranking, and 0.6+ model overrides) on older CLIs instead of failing.
- **History search & provenance**: `nachos_history_search` now uses BM25/IDF ranking (40% query IDF coverage floor, recency tie-break), returns neighbouring context turns (`history_context_before`/`after`, default 1/1; `history_max_chars`, default 6000; tool calls and results kept together, overlapping windows merged), and emits stable `ref` (`<session>@<checkpoint>#<index>`) handles to reopen spots (`bench/history.py`, `bench/data/history.json`). Memory entries record origin in a new `provenance` index table (session/time on manual saves; verified redacted supporting quote + checkpoint/message ref on approved capture suggestions; ungrounded suggestions flagged `[ungrounded]` and skipped by `approve all`), surfaced as `source` (plus `evidence` on exact reads) on recall and preserved across supersedes and reconciliation merges.
- **Per-query confidence, abstention & cold-start calibration priors**: `nachos_core/calibration.py` ships offline random-pair cosine priors for MiniLM and all model families in `nachos-embeddings` (`e5`, `bge-en-v1.5`, `snowflake-arctic-embed-v1`/`v2.0`, `gte-modernbert`, `nomic-embed-text-v1.5`), used below 30 entries and blended linearly into empirical vectors from 30–100 entries for `possible_updates`, `duplicate_calibration: auto` (`p99`), opt-in `similarity_floor: auto` (`p75`, default `off`), and search confidence. `MemoryService.search_with_confidence` combines calibrated semantic z-score and OOV-aware lexical+agreement coverage via decisive-OR with separate thresholds (`0.48` semantic, `0.38` lexical) to attach `confident` / `confidence` / `confidence_score` / `signals` and `no_confident_match: true` while keeping low-confidence candidates listed (`abstention`: `hint` default without auto-retry guidance, `on` opt-in with retry guidance, `off` to omit).
- **Multi-query recall & cross-encoder reranking**: `nachos_memory_recall` and `MemoryService.search` / `search_with_confidence` accept `queries` (up to 4 variants, embedded in one batched call and fused with RRF keeping best per-hit similarity and confidence). Optional cross-encoder reranking (`rerank`: `off` default, `nachos` full `sum:1` blend over `rerank_candidates`, or `borderline` verifying only top-5 results on borderline-confidence queries within `±0.12` of threshold; multi-query reranking caps to top 5 candidates taking max score across variants).
- **Linked memory & entity filter**: `[[key]]`/`[[Title]]` wikilinks (code skipped) stored in a sidecar `links` table kept in sync on put/edit/remove/revert/supersede/reconcile-merge and reconciled against hand edits; links follow `superseded_by` to active successors, broken links are tolerated and reported, exact-key recall returns `links`/`backlinks`/`broken_links`, query recall attaches one-hop `related` neighbours (`related_per_hit`, default 3, max 8), and regex entity extraction + z-scored lexical/embedding similarity powers `suggested_links` (max 3 with `reason`) and `nachos_memory_recall(entity=...)`. Manifest appends `[n links]` when it fits `manifest_char_budget`.
- **Entry validity & supersedes**: `created_at`/`updated_at` timestamps and `supersedes` in SQLite (in-place migration, read-only safe) and flatfile markdown (`<!-- created: ... | updated: ... | superseded_by: ... -->`), hiding superseded entries from the manifest, redirecting search hits across `superseded_by` chains before reranking (`include_superseded=true` opts in), recency tie-breaking (60 s gap), `possible_updates` candidates (alias `suggested_supersedes`) with facet-qualifier reasons, and `MemoryService.link_supersede` / `unlink_supersede` + `put(key=...)`.
- **Chunked body vectors**: long entries (>1200 chars) store head (`chunk_no = 0`) plus overlapping title-prefixed ~1200-char body chunks (max 16 vectors) with `chunk_penalty = 0.05` max-over-chunks recall scoring and head-only duplicate detection.
- **Memory reconciliation**: `nachos-reconcile` CLI (`inventory`, `packet [--audit]`, `session`, `export`, `validate`, `apply`, `viewer`, `skill`, `cron-evidence`), offline review viewer, sealed v1 JSON artifacts, deterministic findings generator (`merge`, `supersede`, `unlink`, `rewrite`, `delete`, `choose-one` including broken-link fixes and provenance in evidence), gated `apply` through `MemoryService` (preserving provenance and retargeting incoming wikilinks on merge), `nachos_memory_reconcile` agent tool, `nachos-reconcile` companion skill, `docs/reconciliation.md`, and `scripts/browser_gate.sh`.
- **Offline maintenance (`nachos-maintain`)**: report-only CLI (`python -m nachos_hermes.maintenance`, `--profile-home`, `--json`) covering `inbox`, `summaries`, `duplicates`, `unused`, `superseded`, `suspicious_supersedes`, `provenance`, `links`, `reconciliation`, and `all` over read-only / WAL-snapshot SQLite and flatfile stores (`MemoryService.duplicate_pairs(sync=False)`, shared `flag_reason`, `docs/memory-maintenance.md`).
- **Benchmarks**: held-out adversarial set (`bench/data/adversarial/`), held-out link set (`bench/data/links-heldout/`), expanded near-miss negatives (+20 main, +15 adversarial), `--data-dir`, `--suggest-eval`, `--link-eval` (`--calibrate-duplicates`), `--multi-query`, `--rerank-borderline`, `--pooling`/`--max-length`/`--rerank-max-length`, and `python -m bench.history`.

### Changed
- Per-scope in-memory lexical and metadata caches (validated via store fingerprints, updated/invalidated on all write paths, helpers in `nachos_core/support.py`), batched `record_hits` commits during search hydration, and WAL-mode `MemoryIndex` (`synchronous=NORMAL` for telemetry, `synchronous=FULL` for undo-log commits).
- `MemoryService.find_similar` uses cached `LexicalIndex` label postings to prune lexical duplicate candidates, and `MDStore` caches parsed entries by fingerprint.

### Upgrade / Migration Notes
- **Automatic in-place migrations**:
  - SQLite store (`memory.db`): automatically adds `created_at`, `updated_at`, `superseded_by`, and `prev_superseded_by` columns on first write/open (read-only operations tolerate pre-migration schemas).
  - Flatfile store (`memory.md`): existing entries remain valid without changes; new/updated entries include `<!-- created: ... | updated: ... | superseded_by: ... -->` HTML metadata comments.
  - Sidecar index (`memory.index.db`): switches to WAL mode, migrates `vectors` to chunked per-model primary keys `(key, chunk_no, model)` (`chunk_no = 0` for head vectors, up to 15 body chunks for entries >1200 chars), and adds the `calibration`, `links`, and `provenance` tables. The entity index is rebuilt in memory from the store.
- **Embeddings CLI requirement**: `nachos-embeddings >= 0.6.0` is required for input types, cross-encoder reranking, and 0.6+ model override flags. Older CLIs log a one-time warning and fall back gracefully to basic embeddings; lexical-only mode still needs zero extra dependencies.
- **New config defaults**: `abstention: hint`, `duplicate_calibration: auto`, `similarity_floor: off`, `rerank: off` (`rerank_candidates: 25`, `rerank_max_length: 0`), `reindex_budget_ms: 250`, `related_per_hit: 3`, `history_context_before: 1`, `history_context_after: 1`, `history_max_chars: 6000`.
- **New tools and CLIs**: agent tool `nachos_memory_reconcile` (with companion `nachos-reconcile` skill) and console scripts `nachos-maintain`, `nachos-models`, and `nachos-reconcile`.
- **Renamed / omitted options**: `possible_updates` replaces `suggested_supersedes` in `nachos_memory_put` responses (`suggested_supersedes` is retained as a one-release alias in 0.7.0). `embedding_normalize` is not exposed (the index always L2-normalizes vectors).
- **Downgrade guidance**: Direct downgrade to 0.6.x on a migrated `memory.index.db` is unsupported because of the `(key, chunk_no, model)` `vectors` primary key and new tables. Back up `$HERMES_HOME/nachos/` before upgrading. To roll back to 0.6.x, restore the backup, or keep `memory.db`/`memory.md` and remove/rename `memory.index.db` (plus any `-wal`/`-shm` sidecars) so 0.6.x rebuilds a fresh index (rebuilding discards change-log, usage telemetry, links, and provenance stored in the 0.7.0 index; 0.6.x ignores `superseded_by` metadata in the store).

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
