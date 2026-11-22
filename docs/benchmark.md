# Recall benchmark

A reproducible, stdlib-only benchmark for memory recall quality and latency, so changes such as a better embedding model, query/document prefixes, chunked body vectors or a reranker can be judged by numbers. It measures only; it never changes recall behaviour.

## What it does

`bench/recall.py` builds a temporary `MemoryService` (sqlite store in a temp dir), loads the fixture through the normal `put()` path, runs every labelled query through `MemoryService.search(limit=10)` and reports:

- **hit@1 / hit@5**: share of queries whose answer is in the top 1 / top 5.
- **MRR@10**: mean reciprocal rank of the answer within the top 10 (0 if absent).
- **p50 / p95 ms**: search latency per query (median over `--repeat` runs).
- **build s**: wall time to `put()` all entries, which includes embedding them when a backend is on.
- **model load s**: time for a first `embed()` call (model download/load), excluded from build and search timings.

Quality numbers are deterministic for a given backend and model; latency depends on the machine.

## Run it

```bash
python -m bench                                   # lexical only (--embeddings off)
python -m bench --embeddings off,sentence-transformers
python -m bench --embeddings nachos --dtype q8
python -m bench --embeddings openai --base-url http://localhost:11434/v1 --model nomic-embed-text
python -m bench --category paraphrase --misses    # one category, list queries missing the top 5
python -m bench --repeat 5 --json results.json    # full report, per-query ranks included
```

Run from the repository root with the dev environment (`uv sync --group dev`). No extra packages are needed for lexical runs.

| Flag | Meaning |
| --- | --- |
| `--embeddings` | Comma-separated backends: `off`, `nachos`, `sentence-transformers`, `openai`. Each is run with the same options below. |
| `--model`, `--dtype`, `--base-url`, `--api-key-env`, `--query-prefix`, `--document-prefix` | Passed through like the `embedding_*` keys in `nachos_hermes/config.py`. |
| `--no-types` | `nachos` backend: never send the query/document type (previous behaviour), to measure what the type adds. |
| `--label` | Name for the run in the report, so several runs of one backend can be told apart. |
| `--command` | `nachos-embeddings` executable name or path. |
| `--category` | Only run this query category (repeatable). |
| `--repeat N` | Searches per query for latency; the median is used. Default 1. |
| `--json PATH` | Write the full report (`-` prints JSON only). |
| `--misses` | List queries whose answer is not in the top 5. |

A backend that is not available is skipped with a reason, not an error: a missing `nachos-embeddings` on `PATH`, an uninstalled `sentence-transformers`, `openai` without `--base-url`/`--model`, or an embedder that fails its warm-up call. The exit status is non-zero only when no backend ran.

`pytest` does not run the benchmark. `tests/test_bench_recall.py` covers the metric math, fixture integrity and a small lexical smoke run.

## Fixture

`bench/data/memories.json` holds 208 synthetic entries (fictional projects, hosts and people; no real data) shaped like real agent memory: projects, environment, decisions, procedures and history. `bench/data/queries.json` holds 393 queries, each with an `expect` list of memory keys (the slug of the title), a `category`, and `match` (`any` by default; `all` for multi-hop). Edit the JSON directly; `pytest tests/test_bench_recall.py` checks that every expected key exists, keys are unique, every entry passes the write gates, and the paraphrase and long-body rules below hold.

| Category | n | What it tests |
| --- | --- | --- |
| `keyword` | 139 | Short keyword queries, as an agent might type (`harbor invoices`). |
| `natural` | 90 | A full question that shares words with the entry. |
| `paraphrase` | 60 | Same meaning, no shared words (checked by test): needs real semantic matching, so lexical-only scores 0 by construction. |
| `long-body` | 30 | The answer sits only in the body after about 2000 characters, past the text the embedder sees (`embed_text` is cut at 2000). Tests chunked or full-body vectors. |
| `outdated` | 30 | One entry of a superseded/current pair; the expected key is the **current** entry. Current code has no notion of recency, so failures here are expected and are the point. |
| `near-dup` | 24 | A sibling entry differs by one detail (staging vs production, EU vs US); the right one must rank above its twin. |
| `multi-hop` | 20 | Needs two entries (`match: all`): the rank is that of the last expected key to appear, so hit@1 is 0 by construction and hit@5 / MRR are the meaningful numbers. |

## Baseline

Commit 41800d1379 (Nachos 0.6.0), single run, `--repeat 1`. Environment: Linux 6.12 x86_64, Intel Core i7-1360P (16 logical CPUs), Python 3.12.3, sentence-transformers 6.3.0 with torch 2.15.0 (CPU), model `all-MiniLM-L6-v2` (the `sentence-transformers` backend default). The first run also had to download the model, which is not in the timings.

| Backend | hit@1 | hit@5 | MRR@10 | p50 ms | p95 ms | build s | model load s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| off (lexical) | 0.674 | 0.812 | 0.737 | 29.9 | 54.4 | 2.14 | 0.00 |
| sentence-transformers | 0.690 | 0.893 | 0.781 | 52.4 | 79.5 | 10.06 | 7.55 |

Per category (hit@1 / hit@5 / MRR@10):

| Category | n | off | sentence-transformers |
| --- | --- | --- | --- |
| keyword | 139 | 0.950 / 1.000 / 0.972 | 0.978 / 1.000 / 0.989 |
| natural | 90 | 0.711 / 0.889 / 0.796 | 0.767 / 0.978 / 0.862 |
| paraphrase | 60 | 0.000 / 0.000 / 0.000 | 0.233 / 0.400 / 0.305 |
| long-body | 30 | 0.933 / 1.000 / 0.967 | 0.467 / 1.000 / 0.694 |
| outdated | 30 | 0.600 / 1.000 / 0.779 | 0.467 / 0.967 / 0.717 |
| near-dup | 24 | 0.958 / 1.000 / 0.979 | 1.000 / 1.000 / 1.000 |
| multi-hop | 20 | 0.000 / 0.800 / 0.353 | 0.000 / 0.850 / 0.367 |

Readings:

- Hybrid recall helps paraphrases (0.00 to 0.23 hit@1) but they remain the weakest category.
- Long-body hit@1 falls from 0.93 to 0.47 with embeddings on: semantic hits for the truncated text outrank the lexical match on the body tail. Chunked body vectors should target this.
- Outdated hit@1 also falls (0.60 to 0.47): nothing prefers the current entry over the superseded one.
- Lexical search takes about 30 ms for 208 entries because every query reads all bodies; expect it to grow with the store.

## Query/document type and model matrix (before the fusion fix)

These numbers were measured before the fusion change described in the next section. Keep them as the "before" picture; the later tables supersede them.

Same fixture, run with the `nachos` backend against the `nachos-embeddings` branch `naterchrdsn/emb-prefixes` (per-model prefixes and pooling, optional per-request `"type": "query"|"document"`). Nachos now tells the CLI whether it is embedding a search query or a stored entry or duplicate check; the CLI applies the model's own prefix and pooling. Environment as above (i7-1360P, 16 logical CPUs, Python 3.12.3, CLI under Node), one run each (`--repeat 1`), models cached in a directory outside both repos.

How it was run: a shim named `nachos-embeddings` that execs `node <worktree>/bin/cli.js "$@"` was passed with `--command`.

```bash
export NACHOS_EMBEDDINGS_CACHE=/path/outside/both/repos
python -m bench --command ./shim/nachos-embeddings --label minilm-fp32 --embeddings nachos --model Xenova/all-MiniLM-L6-v2
python -m bench ... --label minilm-q8 --model Xenova/all-MiniLM-L6-v2 --dtype q8
python -m bench ... --label bge-small-q8 --model Xenova/bge-small-en-v1.5 --dtype q8
python -m bench ... --label bge-small-q8-notype --model Xenova/bge-small-en-v1.5 --dtype q8 --no-types
python -m bench ... --label arctic-s-q8 --model Snowflake/snowflake-arctic-embed-s --dtype q8   # and --no-types
python -m bench ... --label gte-modernbert-q8 --model Alibaba-NLP/gte-modernbert-base --dtype q8
```

`--no-types` makes the nachos backend omit the type, so queries are embedded like documents (what Nachos did before this change). For `openai` and `sentence-transformers` there is no type; use `--query-prefix` / `--document-prefix` (config keys `embedding_query_prefix` / `embedding_document_prefix`).

| Run | hit@1 | hit@5 | MRR@10 | p50 ms | p95 ms | build s | model load s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lexical (off) | 0.674 | 0.812 | 0.737 | 29.7 | 55.0 | 3.77 | 0.00 |
| MiniLM-L6 fp32 | 0.687 | 0.891 | 0.779 | 42.7 | 63.8 | 8.27 | 0.59 |
| MiniLM-L6 q8 | 0.684 | 0.891 | 0.778 | 39.0 | 65.7 | 6.15 | 0.81 |
| bge-small q8, typed | 0.649 | 0.845 | 0.742 | 45.3 | 74.5 | 7.01 | 1.05 |
| bge-small q8, no type | 0.651 | 0.845 | 0.744 | 41.3 | 65.9 | 7.12 | 0.46 |
| arctic-embed-s q8, typed | 0.656 | 0.840 | 0.742 | 43.4 | 65.5 | 6.92 | 0.99 |
| arctic-embed-s q8, no type | 0.603 | 0.812 | 0.699 | 85.6 | 121.4 | 7.06 | 0.47 |
| gte-modernbert q8 | 0.659 | 0.837 | 0.745 | 57.0 | 84.4 | 18.86 | 1.08 |

Per category, hit@1 / hit@5 / MRR@10:

| Category | n | lexical (off) | MiniLM-L6 fp32 | MiniLM-L6 q8 | gte-modernbert q8 |
|---|---|---|---|---|---|
| keyword | 139 | 0.950 / 1.000 / 0.972 | 0.978 / 1.000 / 0.989 | 0.978 / 1.000 / 0.989 | 0.971 / 1.000 / 0.984 |
| natural | 90 | 0.711 / 0.889 / 0.796 | 0.778 / 0.978 / 0.867 | 0.778 / 0.967 / 0.870 | 0.811 / 0.956 / 0.878 |
| paraphrase | 60 | 0.000 / 0.000 / 0.000 | 0.217 / 0.383 / 0.292 | 0.217 / 0.400 / 0.284 | 0.033 / 0.183 / 0.113 |
| long-body | 30 | 0.933 / 1.000 / 0.967 | 0.433 / 1.000 / 0.673 | 0.433 / 0.967 / 0.678 | 0.300 / 0.667 / 0.492 |
| outdated | 30 | 0.600 / 1.000 / 0.779 | 0.467 / 0.967 / 0.717 | 0.433 / 1.000 / 0.697 | 0.533 / 1.000 / 0.761 |
| near-dup | 24 | 0.958 / 1.000 / 0.979 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| multi-hop | 20 | 0.000 / 0.800 / 0.353 | 0.000 / 0.850 / 0.367 | 0.000 / 0.850 / 0.372 | 0.000 / 0.950 / 0.422 |

| Category | n | bge-small q8, typed | bge-small q8, no type | arctic-embed-s q8, typed | arctic-embed-s q8, no type |
|---|---|---|---|---|---|
| keyword | 139 | 0.942 / 1.000 / 0.969 | 0.935 / 1.000 / 0.965 | 0.964 / 1.000 / 0.981 | 0.942 / 1.000 / 0.970 |
| natural | 90 | 0.744 / 0.956 / 0.848 | 0.744 / 0.956 / 0.848 | 0.744 / 0.967 / 0.850 | 0.667 / 0.956 / 0.795 |
| paraphrase | 60 | 0.017 / 0.117 / 0.070 | 0.017 / 0.133 / 0.086 | 0.033 / 0.117 / 0.081 | 0.000 / 0.100 / 0.048 |
| long-body | 30 | 0.633 / 1.000 / 0.806 | 0.667 / 1.000 / 0.802 | 0.500 / 0.900 / 0.670 | 0.133 / 0.733 / 0.411 |
| outdated | 30 | 0.433 / 1.000 / 0.711 | 0.467 / 1.000 / 0.728 | 0.533 / 1.000 / 0.761 | 0.600 / 1.000 / 0.792 |
| near-dup | 24 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| multi-hop | 20 | 0.000 / 0.800 / 0.356 | 0.000 / 0.750 / 0.336 | 0.000 / 0.800 / 0.345 | 0.000 / 0.600 / 0.269 |

What the query type adds (typed minus untyped, same model):

| Model | hit@1 | hit@5 | MRR@10 | Notable categories |
| --- | --- | --- | --- | --- |
| arctic-embed-s q8 | +0.053 | +0.028 | +0.043 | long-body hit@1 0.133 to 0.500, natural 0.667 to 0.744, multi-hop hit@5 0.600 to 0.800 |
| bge-small q8 | -0.003 | 0.000 | -0.002 | within noise (its instruction is optional for short-query retrieval) |

Notes and caveats:

- arctic-embed-s is the clear case for sending the type: without the query instruction it is the worst embedding run here; with it, it matches bge.
- An extra bge-small fp32 run (hit@1 0.659, hit@5 0.845, MRR 0.746, paraphrase hit@1 0.017) shows its low paraphrase score is not caused by q8.
- bge-small, arctic-embed-s and gte-modernbert score far below MiniLM on `paraphrase` (0.02 to 0.03 hit@1 against 0.22). I did not investigate whether that is the models, the fixture's deliberately word-free paraphrases, or the fusion in `MemoryService.search`; treat it as an open question.
- The first gte-modernbert run failed with `init failed: fetch failed` during the download and was skipped; it succeeded on a retry once the file was cached. Its build is slow (18.9 s) and it is the weakest on long-body.
- Latency is a single run on a shared machine and some runs overlapped with other work (the untyped arctic-embed-s p50 of 85.6 ms is almost certainly noise), so compare builds and p50s only roughly. Build time includes embedding every entry through `put()`.
- The fixture is synthetic and its paraphrase queries are extreme by construction; differences of a few points on 20 to 60 queries are not significant.

### Recommended default (superseded)

On this fixture, keep `Xenova/all-MiniLM-L6-v2` as the default model. It has the best overall hit@5 (0.891) and MRR (0.779) of the semantic runs and by far the best paraphrase recall, and with `dtype: q8` it is essentially identical to fp32 (hit@5 0.891 both, MRR 0.778 vs 0.779) while building about a quarter faster, so q8 is the sensible setting for speed. None of the retrieval-tuned models (bge-small, arctic-embed-s, gte-modernbert) beat it overall here; gte-modernbert is best on natural questions and multi-hop but costs about 2.5x the build time and 768-dimensional vectors, and loses on long-body and paraphrase. If a prefixed model is chosen, send the query type (the default now): it is worth about 5 points of hit@1 for arctic-embed-s. Re-check on real memory data before changing the default; long-body and outdated are weak for every embedding setup, so chunked body vectors and recency handling matter more than the model choice.

## Why paraphrase collapsed, and the fix

The earlier result (bge, arctic, gte at 0.02 to 0.03 paraphrase hit@1 against MiniLM at 0.22) was an artefact of fusion, not of the models. Diagnosis with `python -m bench.diagnose` (cosine spreads and vector-only ranking, no fusion) on the 60 paraphrase queries, q8, typed:

| Model | cosine (query, expected) p10/p50/p90 | cosine (query, random) p10/p50/p90 | random p99 | vector-only hit@1 / hit@5 / MRR@10 | queries with expected cosine < 0.30 |
| --- | --- | --- | --- | --- | --- |
| MiniLM-L6 | 0.19 / 0.33 / 0.49 | -0.03 / 0.08 / 0.22 | 0.35 | 0.367 / 0.633 / 0.478 | 40% |
| bge-small | 0.55 / 0.62 / 0.68 | 0.44 / 0.52 / 0.60 | 0.66 | 0.317 / 0.567 / 0.411 | 0% |
| arctic-embed-s | 0.55 / 0.60 / 0.65 | 0.44 / 0.51 / 0.58 | 0.62 | 0.300 / 0.550 / 0.422 | 0% |
| gte-modernbert | 0.57 / 0.63 / 0.70 | 0.44 / 0.51 / 0.58 | 0.64 | 0.467 / 0.800 / 0.606 | 0% |

- **Model quality is fine.** On vector-only ranking all three are comparable to or better than MiniLM (gte-modernbert is clearly best at 0.467 hit@1, 0.800 hit@5). The models' similarity ranges are compressed and shifted up (random pairs sit around 0.5, not 0.1), which is exactly why an absolute cosine cutoff cannot be shared between models.
- **The 0.30 floor (`_MIN_VECTOR_SCORE`) was not the cause for these models.** It removed nothing for them (every expected cosine is above 0.30), and disabling it changed their fused numbers by zero. It did matter for MiniLM: without it MiniLM paraphrase fused hit@1 falls from 0.23 to 0.03, because the floor was quietly acting as a precision filter on the semantic list.
- **The cause is the interaction of RRF with weak lexical hits.** A paraphrase query shares no words with its answer, but it still shares a generic word or two with other entries, so the lexical list returns up to 30 noise entries. In RRF those sit at ranks 1 to 30 next to the semantic list, and any entry that is in both lists gets a double contribution. With MiniLM the floor made the semantic list short, so overlap with the noise was rare; with a model whose similarities are all above 0.30 the semantic list always had 30 entries and the noise overlapped it constantly, pushing true answers down (fused paraphrase hit@1 0.33 vector-only to 0.02 fused for bge). The fusion also hurt MiniLM (0.37 vector-only, 0.23 fused).

The offline replay used to compare variants (cached vectors, same lexical lists and the same RRF code path; it reproduced the harness numbers to two decimals) gave, for fused paraphrase / overall hit@1 / hit@5:

| Variant | MiniLM | bge-small | arctic-s | gte-modernbert |
| --- | --- | --- | --- | --- |
| before: floor 0.30, all lexical hits | 0.23 / 0.68 / 0.89 | 0.02 / 0.65 / 0.84 | 0.02 / 0.65 / 0.83 | 0.02 / 0.65 / 0.85 |
| floor disabled, all lexical hits | 0.03 / 0.63 / 0.84 | 0.02 / 0.65 / 0.84 | 0.02 / 0.65 / 0.83 | 0.02 / 0.65 / 0.85 |
| per-query relative floor (mean + 2 sd), all lexical hits | 0.12 / 0.64 / 0.90 | 0.10 / 0.66 / 0.88 | 0.12 / 0.65 / 0.90 | 0.10 / 0.66 / 0.93 |
| floor disabled, lexical hits need IDF-weighted query-term coverage >= 0.4 | 0.18 / 0.68 / 0.91 | 0.15 / 0.69 / 0.90 | 0.15 / 0.68 / 0.90 | 0.32 / 0.72 / 0.94 |

**What changed in `MemoryService`:** the absolute cosine floor is removed, and when a query vector exists a lexical hit only joins the fusion if it covers at least 0.4 of the IDF-weighted query terms (`_MIN_LEXICAL_COVERAGE`; lexical-only mode is unchanged). Both are model-independent. 0.35 to 0.45 is a plateau on this fixture; 0.5 helps paraphrase further but costs multi-hop hit@5 (queries that combine words from two entries cover each only about half), so 0.4 is a compromise. This constant was tuned on a synthetic fixture whose paraphrase queries are word-free by construction, so re-check it on real data. A consequence of dropping the floor: a query with nothing relevant in memory now returns the nearest entries instead of possibly none. A calibrated floor (a percentile of random-pair similarity computed lazily from the stored vectors) is the proposal if that matters; it is not implemented.

### Duplicate threshold (`duplicate_threshold`, 0.65)

Same issue, measured on entry/entry cosine (12 near-duplicate sibling pairs from the `near-dup` fixture against every other pair):

| Model | sibling pairs p10 / p50 / p90 | siblings flagged at 0.65 | other pairs p50 / p90 / p99 | other pairs flagged at 0.65 |
| --- | --- | --- | --- | --- |
| MiniLM-L6 | 0.44 / 0.68 / 0.77 | 58% | 0.12 / 0.25 / 0.47 | 0.6% |
| bge-small | 0.70 / 0.77 / 0.88 | 92% | 0.55 / 0.62 / 0.72 | 4.3% |
| arctic-embed-s | 0.78 / 0.82 / 0.92 | 100% | 0.65 / 0.71 / 0.78 | **48%** |
| gte-modernbert | 0.70 / 0.78 / 0.86 | 100% | 0.53 / 0.60 / 0.70 | 2.3% |

With arctic-embed-s the 0.65 threshold would flag about half of all entry pairs as likely duplicates, which makes the flag useless; bge-small is borderline; MiniLM, the model 0.65 was tuned for, misses about 4 in 10 near-duplicates. Proposal, not applied: derive the threshold per model from the stored vectors (for example the 99th percentile of pairwise similarity, clamped to a sane range) instead of one number in config. `duplicate_threshold` stays 0.65.

### Stale vectors after a CLI upgrade

The `nachos` `model_id` was a fixed `nachos:<model>[:<dtype>]`, so an index built by a 0.6.0 CLI (no prefixes, mean pooling) was reused unchanged after upgrading to a type-aware CLI, even though that CLI embeds documents differently for models such as bge (cls pooling) or e5/nomic (document prefix). `CliEmbedder` now learns pooling and the document prefix from the ready line (`warm_up()` is called before the index is compared) and appends `:pooling=<p>` and `:dp=<prefix>` to `model_id` only when they differ from mean pooling and no prefix. Upgrading re-embeds the affected models and leaves MiniLM (whose vectors are unchanged) alone. A not-yet-started CLI reports the base id until first use.

### Results after the fix

Same harness and fixture, typed queries, q8, `--repeat 3`. Machine load average was about 12 on 16 logical CPUs while measuring, so latencies are rough; each row is the quieter of two passes (quality metrics are identical between passes).

| Run | hit@1 | hit@5 | MRR@10 | p50 ms | p95 ms | build s |
| --- | --- | --- | --- | --- | --- | --- |
| lexical | 0.674 | 0.812 | 0.737 | 28.9 | 39.5 | 2.02 |
| MiniLM-L6 q8 | 0.692 | 0.919 | 0.795 | 36.5 | 55.8 | 9.32 |
| bge-small q8 | 0.684 | 0.911 | 0.784 | 39.9 | 56.0 | 11.00 |
| arctic-embed-s q8 | 0.692 | 0.906 | 0.790 | 39.6 | 58.0 | 6.11 |
| gte-modernbert q8 | 0.713 | 0.939 | 0.818 | 56.6 | 94.0 | 13.17 |

Per category, hit@1 / hit@5 / MRR@10:

| Category | n | lexical | MiniLM-L6 q8 | bge-small q8 | arctic-embed-s q8 | gte-modernbert q8 |
|---|---|---|---|---|---|---|
| keyword | 139 | 0.950 / 1.000 / 0.972 | 0.978 / 1.000 / 0.989 | 0.942 / 1.000 / 0.969 | 0.964 / 1.000 / 0.981 | 0.971 / 1.000 / 0.984 |
| natural | 90 | 0.711 / 0.889 / 0.796 | 0.767 / 0.978 / 0.862 | 0.744 / 0.978 / 0.850 | 0.756 / 0.967 / 0.856 | 0.811 / 0.978 / 0.892 |
| paraphrase | 60 | 0.000 / 0.000 / 0.000 | 0.183 / 0.600 / 0.366 | 0.133 / 0.533 / 0.287 | 0.167 / 0.533 / 0.332 | 0.267 / 0.683 / 0.465 |
| long-body | 30 | 0.933 / 1.000 / 0.967 | 0.600 / 0.967 / 0.772 | 0.867 / 0.967 / 0.921 | 0.667 / 0.967 / 0.803 | 0.533 / 0.967 / 0.734 |
| outdated | 30 | 0.600 / 1.000 / 0.779 | 0.467 / 1.000 / 0.722 | 0.433 / 1.000 / 0.711 | 0.533 / 1.000 / 0.761 | 0.533 / 1.000 / 0.761 |
| near-dup | 24 | 0.958 / 1.000 / 0.979 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| multi-hop | 20 | 0.000 / 0.800 / 0.353 | 0.000 / 0.750 / 0.332 | 0.000 / 0.800 / 0.342 | 0.000 / 0.750 / 0.315 | 0.000 / 0.900 / 0.386 |

Change from before the fix (overall hit@1 / hit@5 / MRR@10): MiniLM 0.684 / 0.891 / 0.778 to 0.692 / 0.919 / 0.795; bge-small 0.649 / 0.845 / 0.742 to 0.684 / 0.911 / 0.784; arctic-embed-s 0.656 / 0.840 / 0.742 to 0.692 / 0.906 / 0.790; gte-modernbert 0.659 / 0.837 / 0.745 to 0.713 / 0.939 / 0.818. Paraphrase hit@1 moves from 0.22 / 0.02 / 0.03 / 0.03 to 0.18 / 0.13 / 0.17 / 0.27.

### Recommended default

The earlier recommendation was based on a measurement artefact. With fusion fixed, the ranking changes: `Alibaba-NLP/gte-modernbert-base` (q8) is best overall (hit@1 0.713, hit@5 0.939, MRR 0.818) and clearly best on paraphrase (0.267 hit@1, 0.683 hit@5), natural questions and multi-hop; the cost is a roughly 2x slower build (13 to 19 s against 6 to 9 s for 208 entries), 768-dimensional vectors (about twice the index size and search work) and a 150 MB q8 download, and it is weakest on long-body (0.533 hit@1). The small 384-dimensional models are within about 0.02 of each other and of MiniLM on overall hit@1; MiniLM is still the best on paraphrase among them at hit@5 (0.600) but bge-small is much better on long-body (0.867 against 0.600). Suggested default: keep MiniLM q8 where build time or footprint matter (it is the smallest download and the fastest), and recommend gte-modernbert q8 as the quality choice; on this fixture the gap to the small models is about 0.02 hit@1 and 0.02 to 0.03 hit@5, which is within what a 393-query synthetic set can resolve, so confirm on real memories before changing the default. The default in code is unchanged. Whatever the model, long-body (chunked body vectors) and outdated (recency) remain the weakest categories.

## Chunked body vectors

On `naterchrdsn/memory-next` (integrating lexical cache, WAL telemetry, query/document types and fusion coverage gating), long entries (>1200-char bodies) now store extra overlapping `~1200`-char / `~200`-char-overlap body chunks prefixed with the title (`(key, chunk_no)`, capped at 16 vectors per entry). Recall scores an entry by the max cosine across its vectors; duplicate detection continues to compare head vectors (`chunk_no = 0`) only.

Before vs after chunking on the 208-entry / 393-query fixture (`nachos` backend, `q8`, typed, `--repeat 1`; 15 entries exceed 1200 chars, adding 30 body chunks = 238 total vectors):

| Model | Stage | overall hit@1 / hit@5 / MRR@10 | long-body hit@1 / hit@5 / MRR@10 | p50 / p95 ms | build s | vectors | index KB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniLM-L6 q8 | before | 0.692 / 0.919 / 0.795 | 0.600 / 0.967 / 0.772 | 7.0 / 17.7 | 3.45 | 208 | 500.0 |
| MiniLM-L6 q8 | after | 0.713 / 0.919 / 0.805 | 0.967 / 1.000 / 0.983 | 9.1 / 19.7 | 6.27 | 238 | 560.0 |
| gte-modernbert q8 | before | 0.713 / 0.939 / 0.818 | 0.533 / 0.967 / 0.734 | 26.2 / 63.2 | 15.67 | 208 | 916.0 |
| gte-modernbert q8 | after | 0.735 / 0.936 / 0.828 | 1.000 / 1.000 / 1.000 | 28.0 / 78.8 | 28.88 | 238 | 1036.0 |

Readings:

- **Long-body recall jumps sharply**: MiniLM-L6 q8 rises from `0.600` to `0.967` hit@1 (`+0.367`, MRR `0.772 -> 0.983`) and gte-modernbert q8 reaches `1.000 / 1.000 / 1.000` (`+0.467` hit@1, MRR `0.734 -> 1.000`), lifting overall hit@1 by `+0.021` to `+0.022` and overall MRR@10 by `+0.010` on both models.
- **Small index footprint**: +30 chunks adds +60 KB (+12%) for MiniLM-L6 and +120 KB (+13%) for gte-modernbert. Build time rises when long entries embed multiple chunks synchronously on `put()`, while search latency increases modestly (`+2 ms` p50).
- **Slight paraphrase dip**: Taking the max cosine over 3 vectors for long entries gives them slightly more chances to outrank 1-vector entries on word-free paraphrases (MiniLM paraphrase hit@1 `0.183 -> 0.150`, gte-modernbert `0.267 -> 0.200`), a minor multiple-testing trade-off offset by the large long-body gain.

### Not measured

No OpenAI-compatible server (Ollama etc.) was reachable, so the `openai` backend has no numbers; the `sentence-transformers` row above predates the query type and used no prefixes.
