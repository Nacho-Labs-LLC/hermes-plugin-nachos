# Changelog

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
