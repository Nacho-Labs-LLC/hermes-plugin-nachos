# Embedding models and switching

Semantic recall is optional. Without it, recall is lexical (IDF-weighted keyword search). This page covers the embedding backends, how to pick and override a model, and what happens when you change one. Setting names are in the [README configuration reference](../README.md#configuration).

## Backends

`embeddings` selects the backend:

| Value | What it uses |
| --- | --- |
| `auto` (default) | The first available of: `nachos-embeddings` on `PATH`, `sentence-transformers`, then an OpenAI-compatible URL (needs `embedding_base_url` and `embedding_model`). |
| `nachos` | The `nachos-embeddings` CLI. Nachos keeps one `serve` worker alive. |
| `openai` | Any OpenAI-compatible `/embeddings` endpoint. |
| `sentence-transformers` | The Python package, in the Hermes environment. |
| `off` | Lexical recall only. |

A backend that fails at runtime drops recall to lexical for the rest of the process.

```bash
npm i -g @nacho-labs/nachos-embeddings   # local model, found on PATH by `auto`
```

```json
{ "embeddings": "openai",
  "embedding_base_url": "http://localhost:11434/v1",
  "embedding_model": "nomic-embed-text" }
```

For a hosted endpoint also set `embedding_api_key_env` to the name of the environment variable that holds the key (the key itself is never stored).

### `nachos-embeddings` version

Use `nachos-embeddings >= 0.6.0`. Earlier releases lack input types (query vs document), cross-encoder reranking and the model-override flags. Nachos reads what the worker supports from the `serve` ready line, so an older CLI is not an error: Nachos falls back to basic embeddings, without input types, reranking or the 0.6 overrides (and uses one process per call if `serve` is unavailable). When a CLI does report a version below 0.6.0, Nachos also logs one warning.

## Choosing a model

Name the model you want. A blank `embedding_model` means the backend default (MiniLM for `nachos`). For the `nachos` backend `embedding_model` accepts any Hugging Face ONNX encoder or a local directory (`./models/my-model`, fully offline). `nachos-embeddings models` lists the models it ships profiles for, and `python -m nachos_hermes.models list` shows the effective Nachos settings.

The measured options, with the benchmark method, are in the [README](../README.md#choosing-a-model) and [benchmark.md](benchmark.md). To measure another model on your own machine:

```bash
python -m bench --model <id> --dtype q8
```

### Overrides

Built-in per-model profiles (pooling, query/document prefixes) are defaults; explicit settings win. For the `nachos` backend:

- `embedding_pooling` (`mean` or `cls`), `embedding_max_length`, `embedding_query_prefix`, `embedding_document_prefix`. For a prefix, blank keeps the model's own and `none` disables it.
- `rerank_model`, `rerank_dtype` and `rerank_max_length` configure the cross-encoder the same way.
- `embedding_dtype` (for example `q8`) selects the weights precision.

For `openai` and `sentence-transformers` there are no built-in prefixes: set `embedding_query_prefix` and `embedding_document_prefix` yourself (for example `query: ` and `passage: ` for E5).

Everything above except the query prefix and the reranker settings changes the stored vectors, so changing it re-embeds (next section). Vectors are always L2-normalized by the index, so the CLI's `--no-normalize` is not exposed.

## Reranking

`rerank: nachos` re-scores the top `rerank_candidates` fused results with a cross-encoder served by the same `nachos-embeddings serve` worker. It needs the `nachos` backend. `rerank: borderline` re-scores only queries whose confidence is near the abstention threshold, so median latency barely changes. The default cross-encoder is `jinaai/jina-reranker-v1-tiny-en` (about 35 MB at q8). Reranking never re-embeds. Measured effect: [benchmark.md](benchmark.md#current-results).

## Switching models

Changing any embedding setting (model, dtype, pooling, max length, document prefix) puts the profile's vectors in a different space. Nachos switches safely:

- **Old vectors are kept** until the new model has embedded every entry, so a crash or a second change loses nothing and switching back is instant. Then the old vectors are dropped and the thresholds for the new model (`duplicate_calibration`, `similarity_floor`) are recalibrated.
- **Recall never waits for the whole index.** A few edited entries are embedded inline. More than 24 stale entries (a model switch, or a first run on an existing store) are re-embedded in small batches of about `reindex_budget_ms` (default 250 ms) per recall or write, plus the entry in flight (a long entry is up to 16 chunks). On the 208-entry benchmark store, switching MiniLM to gte-modernbert used to block the first recall for 17 s; recalls during the switch now take a median 314 ms and the switch completes after about 43 recalls.
- **Lexical meanwhile.** Until a scope is fully re-embedded, recall, duplicate checks and update hints use lexical matching only, because a partly embedded ranking would mix two vector spaces.
- **Status.** `/nachos-memory-status` shows `re-embedding: ...` while it runs. From a shell:

```bash
python -m nachos_hermes.models status  --profile-home ~/.hermes/profiles/default   # coverage per model; read-only, loads no model
python -m nachos_hermes.models reindex --profile-home ~/.hermes/profiles/default   # finish now (loads the model)
python -m nachos_hermes.models list    [--profile-home H]                          # effective settings and auxiliary models
```

All three accept `--store {sqlite,flatfile}` and `--json`; `--profile-home` is required for `status` and `reindex`.

## Chat models Nachos can call

These are chat models, not embedding models, and use Hermes' own `auxiliary` settings in `config.yaml`. Both can cost money if left unset (Hermes may fall back to your main model), so pin a cheap one.

| Setting | Used by | Notes |
| --- | --- | --- |
| `auxiliary.nachos_capture.model` | Session capture (`capture: review`) | Suggests entries into the review inbox. |
| `auxiliary.nachos_summary.model` | `tools/correct_summaries.py` | Writes replacement summaries; a dry run still calls the model. |

`python -m nachos_hermes.models list` prints both next to the embedding settings when Hermes' config is readable.
