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

`bench/data/memories.json` holds 208 synthetic entries (fictional projects, hosts and people; no real data) shaped like real agent memory: projects, environment, decisions, procedures and history. `bench/data/queries.json` holds 423 queries (393 answerable plus 30 `negative`), each with an `expect` list of memory keys (the slug of the title), a `category`, and `match` (`any` by default; `all` for multi-hop). Edit the JSON directly; `pytest tests/test_bench_recall.py` checks that every expected key exists, keys are unique, every entry passes the write gates, and the paraphrase and long-body rules below hold.

| Category | n | What it tests |
| --- | --- | --- |
| `keyword` | 139 | Short keyword queries, as an agent might type (`harbor invoices`). |
| `natural` | 90 | A full question that shares words with the entry. |
| `paraphrase` | 60 | Same meaning, no shared words (checked by test): needs real semantic matching, so lexical-only scores 0 by construction. |
| `long-body` | 30 | The answer sits only in the body after about 2000 characters, past the text the embedder sees (`embed_text` is cut at 2000). Tests chunked or full-body vectors. |
| `outdated` | 30 | One entry of a superseded/current pair; the expected key is the **current** entry. Current code has no notion of recency, so failures here are expected and are the point. |
| `near-dup` | 24 | A sibling entry differs by one detail (staging vs production, EU vs US); the right one must rank above its twin. |
| `negative` | 30 | No correct answer exists (off-domain topics and topics adjacent to real ones). Reported apart as false-positive rates: `fp@1` = share of queries returning anything, `fp@5` = mean filled share of the top five slots; lower is better. |
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

## Ranking round: calibration, chunk penalty, reranker

Measured on `naterchrdsn/ranking` (on top of chunked body vectors), `nachos` backend, q8, typed queries, `--repeat 3`. The fixture now also has 30 `negative` queries (no correct answer; listed apart from the hit metrics, which still cover the 393 answerable queries). "Before" below is the branch without this round (chunk penalty 0, no floor, no rerank). Machine load average was 20 to 26 on 16 logical CPUs throughout, so latencies are rough and p95s especially. The CLI for reranking was built from nachos-embeddings commit `f29d2e1` (branch `naterchrdsn/rerank`) in a temp copy; embeddings and rerank models were cached outside both repos.

```bash
export NACHOS_EMBEDDINGS_CACHE=/path/outside/both/repos
python -m bench --command ./shim --embeddings nachos --model Xenova/all-MiniLM-L6-v2 --dtype q8 --chunk-penalty 0   # before
python -m bench ... --dtype q8                                  # chunk penalty 0.05 (the new default)
python -m bench ... --dtype q8 --floor auto                     # + calibrated floor (p75)
python -m bench ... --dtype q8 --rerank --rerank-dtype q8       # + reranker (jina-reranker-v1-tiny-en)
python -m bench.diagnose --command ./shim --model <id> --dtype q8   # duplicate-threshold behaviour
```

### Chunk penalty

`MemoryService.chunk_penalty` is subtracted from a body chunk's cosine: an entry scores `max(head, best chunk - penalty)`. Sweep (overall hit@1 / hit@5 / MRR@10, paraphrase hit@1 / hit@5, long-body hit@1):

| Penalty | MiniLM q8 overall | MiniLM paraphrase | MiniLM long-body | gte q8 overall | gte paraphrase | gte long-body |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (before) | 0.713 / 0.919 / 0.805 | 0.150 / 0.583 | 0.967 | 0.735 / 0.936 / 0.828 | 0.200 / 0.650 | 1.000 |
| 0.02 | 0.713 / 0.921 / 0.807 | 0.150 / 0.600 | 0.967 | 0.735 / 0.944 / 0.830 | 0.200 / 0.700 | 1.000 |
| **0.05** | **0.718 / 0.921 / 0.810** | **0.183 / 0.600** | **0.967** | **0.738 / 0.944 / 0.832** | **0.217 / 0.700** | **1.000** |
| 0.08 | 0.710 / 0.921 / 0.806 | 0.167 / 0.600 | 0.900 | 0.730 / 0.944 / 0.828 | 0.217 / 0.700 | 0.900 |
| 0.12 | 0.713 / 0.921 / 0.807 | 0.167 / 0.600 | 0.900 | 0.723 / 0.942 / 0.823 | 0.233 / 0.700 | 0.767 |

0.05 is the default: it recovers part of the paraphrase hit@1 that chunking cost (MiniLM 0.150 to 0.183, gte 0.200 to 0.217) and lifts hit@5 on both, while long-body stays at 0.967 / 1.000. From 0.08 the penalty starts to hurt long-body (the answer lives in a chunk). It did not get back to the pre-chunking 0.183 / 0.267 paraphrase hit@1 in full. A multiplicative penalty was not tried.

### Calibrated similarity floor (`similarity_floor: auto`, default off)

The index measures a model's own random-pair cosine spread from the stored head vectors (fixed-seed sample of 4000 pairs; `calibration` table in the sidecar index keyed by `model_id`; resampled when the head count drifts by more than 20%; not computed below 30 entries) and drops semantic hits below a percentile of it. Profile of the two models (p50 / p75 / p90 / p95 / p99 of random head pairs): MiniLM 0.12 / 0.18 / 0.25 / 0.29 / 0.43, gte-modernbert 0.53 / 0.56 / 0.59 / 0.62 / 0.69, which is why no absolute cutoff is shared.

Floor percentile sweep with the 0.05 chunk penalty (overall hit@1 / hit@5; negative fp@1 / fp@5; fp@1 is the share of the 30 negative queries that return anything, fp@5 the mean filled share of the top five slots):

| Floor | MiniLM q8 | MiniLM negative | gte q8 | gte negative |
| --- | --- | --- | --- | --- |
| off | 0.718 / 0.921 | 1.00 / 1.00 | 0.738 / 0.944 | 1.00 / 1.00 |
| p75 | 0.723 / 0.921 | 0.90 / 0.83 | 0.753 / 0.942 | 0.80 / 0.73 |
| p80 | 0.720 / 0.921 | 0.83 / 0.77 | 0.753 / 0.942 | 0.80 / 0.65 |
| p85 | 0.725 / 0.921 | 0.80 / 0.73 | 0.753 / 0.942 | 0.80 / 0.64 |
| p90 | 0.733 / 0.921 | 0.80 / 0.62 | 0.756 / 0.934 | 0.80 / 0.60 |
| p95* | 0.735 / 0.908 | 0.77 / 0.51 | 0.761 / 0.916 | 0.73 / 0.51 |
| p99* | 0.720 / 0.827 | 0.73 / 0.36 | 0.728 / 0.830 | 0.67 / 0.35 |

(*measured without the chunk penalty.) The floor raises hit@1 (it removes weak semantic hits that double-count against lexical noise) but cannot make irrelevant queries return nothing: even at p95, 70 to 77% of negative queries still return an entry, because the negatives are deliberately adjacent to real topics (and a lexical hit covering 40% of the terms still passes). Percentiles above p90 cost hit@5 quickly. At the mildest useful setting (p75) MiniLM loses nothing but gte loses one paraphrase query at hit@5 (0.944 to 0.942), so, by the rule that the floor may only ship if it costs no hit@5 anywhere, **it stays off by default**; `similarity_floor: auto` (p75) is available and is a small net win on hit@1 and negatives.

### Duplicate threshold (`duplicate_calibration: auto`)

With `duplicate_calibration: auto` the effective threshold is `max(duplicate_threshold, model p99 of random head-pair cosine)`; it only ever raises the threshold, and a `duplicate_threshold` other than the 0.65 default is used exactly as given (so anyone who set a value keeps it; `duplicate_calibration: off` pins 0.65 for everyone). Measured by `python -m bench.diagnose` on the 12 near-duplicate sibling pairs of the fixture and every other entry pair (siblings flagged = recall; others flagged = false-flag rate):

| Model | Threshold before to after | Sibling recall before to after | Other pairs flagged before to after |
| --- | --- | --- | --- |
| MiniLM-L6 q8 | 0.650 to 0.650 | 58% to 58% | 0.6% to 0.6% |
| bge-small q8 | 0.650 to 0.709 | 92% to 83% | 4.3% to 1.1% |
| arctic-embed-s q8 | 0.650 to 0.774 | 100% to 92% | 48% to 1.1% |
| gte-modernbert q8 | 0.650 to 0.691 | 100% to 92% | 2.3% to 1.1% |

The calibration removes the arctic-embed-s failure (half of all pairs flagged) and roughly equalises the false-flag rate at about 1% for the models whose spread is wide, at the price of one sibling pair out of twelve. It cannot lower the threshold, so MiniLM, which misses 42% of these siblings at 0.65, is unchanged. Twelve sibling pairs is a small sample.

### Reranker

`rerank: nachos` (config keys `rerank_model`, `rerank_dtype`, `rerank_candidates` = 25) re-scores the top fused results with `jinaai/jina-reranker-v1-tiny-en` (q8, 33.4 MB model + 2.0 MB tokenizer) served by the same `nachos-embeddings serve` worker; passages are title, summary and the best-matching chunk. Final order options, on the cached reranker scores of the 423 queries (0.05 penalty; overall hit@1 / hit@5 / MRR@10):

| Final order | MiniLM q8 | gte-modernbert q8 |
| --- | --- | --- |
| no rerank | 0.718 / 0.921 / 0.810 | 0.738 / 0.944 / 0.832 |
| pure rerank order | 0.761 / 0.870 / 0.810 | 0.746 / 0.875 / 0.800 |
| RRF blend of rerank rank and fused rank (k = 10) | 0.725 / 0.921 / 0.812 | 0.738 / 0.929 / 0.825 |
| pure rerank only if the top rerank score is at least 0.5 | 0.768 / 0.926 / 0.836 | 0.771 / 0.942 / 0.845 |
| fused rank term + 0.5 x rerank score | 0.761 / 0.926 / 0.833 | 0.776 / 0.944 / 0.850 |
| **fused rank term + 1.0 x rerank score (`sum:1`)** | **0.763 / 0.926 / 0.833** | **0.773 / 0.942 / 0.848** |
| fused rank term + 2.0 x rerank score | 0.763 / 0.921 / 0.831 | 0.758 / 0.939 / 0.835 |

Pure rerank order gains hit@1 but loses about 0.05 hit@5, because this small English cross-encoder scores word-free paraphrases poorly (paraphrase hit@5 0.60 to 0.25 for MiniLM). Rank-only RRF blending is too timid to help. The default is the score blend `sum:1`: the fused rank contributes `1 / (1 + rank / 60)` and the cross-encoder its 0 to 1 score, so a confident reranker overturns the fused order and a lukewarm one only breaks near-ties. The threshold-gated variant is about as good but has a cliff; `sum:0.5` to `sum:1` are within noise. (`rerank_mode` is a `MemoryService` argument, not a config key.)

Final results, rerank off vs on (floor off), hit@1 / hit@5 / MRR@10 per category; the last row of each overall table adds the floor:

**MiniLM-L6 q8**

| Run | hit@1 | hit@5 | MRR@10 | negative fp@1 / fp@5 | p50 ms | p95 ms | build s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rerank off | 0.718 | 0.921 | 0.810 | 1.00 / 1.00 | 7.1 | 9.0 | 3.6 |
| rerank on | 0.763 | 0.926 | 0.833 | 1.00 / 1.00 | 184.1 | 506.9 | 3.3 |
| floor, rerank off | 0.723 | 0.921 | 0.813 | 0.90 / 0.83 | 7.2 | 9.3 | 3.1 |
| floor + rerank | 0.766 | 0.924 | 0.835 | 0.90 / 0.83 | 157.3 | 482.4 | 3.4 |

| Category | n | rerank off | rerank on | floor, rerank off | floor + rerank |
|---|---|---|---|---|---|
| keyword | 139 | 0.978 / 1.000 / 0.989 | 0.993 / 1.000 / 0.996 | 0.978 / 1.000 / 0.989 | 0.993 / 1.000 / 0.996 |
| natural | 90 | 0.756 / 0.978 / 0.857 | 0.844 / 0.989 / 0.902 | 0.756 / 0.978 / 0.857 | 0.844 / 0.989 / 0.902 |
| paraphrase | 60 | 0.183 / 0.600 / 0.364 | 0.167 / 0.600 / 0.348 | 0.217 / 0.600 / 0.382 | 0.183 / 0.583 / 0.358 |
| long-body | 30 | 0.967 / 1.000 / 0.983 | 1.000 / 1.000 / 1.000 | 0.967 / 1.000 / 0.983 | 1.000 / 1.000 / 1.000 |
| outdated | 30 | 0.467 / 1.000 / 0.722 | 0.733 / 1.000 / 0.861 | 0.467 / 1.000 / 0.722 | 0.733 / 1.000 / 0.861 |
| near-dup | 24 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| multi-hop | 20 | 0.000 / 0.750 / 0.332 | 0.000 / 0.800 / 0.350 | 0.000 / 0.750 / 0.332 | 0.000 / 0.800 / 0.350 |

**gte-modernbert q8**

| Run | hit@1 | hit@5 | MRR@10 | negative fp@1 / fp@5 | p50 ms | p95 ms | build s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rerank off | 0.738 | 0.944 | 0.832 | 1.00 / 1.00 | 16.6 | 20.8 | 12.3 |
| rerank on | 0.773 | 0.942 | 0.848 | 1.00 / 1.00 | 327.3 | 568.7 | 12.9 |
| floor, rerank off | 0.753 | 0.942 | 0.839 | 0.80 / 0.73 | 16.9 | 22.6 | 12.8 |
| floor + rerank | 0.779 | 0.939 | 0.849 | 0.80 / 0.73 | 180.2 | 500.3 | 12.4 |

| Category | n | rerank off | rerank on | floor, rerank off | floor + rerank |
|---|---|---|---|---|---|
| keyword | 139 | 0.971 / 1.000 / 0.984 | 0.993 / 1.000 / 0.996 | 0.971 / 1.000 / 0.984 | 0.993 / 1.000 / 0.996 |
| natural | 90 | 0.800 / 0.978 / 0.886 | 0.867 / 0.978 / 0.918 | 0.833 / 0.978 / 0.906 | 0.878 / 0.978 / 0.924 |
| paraphrase | 60 | 0.217 / 0.700 / 0.429 | 0.217 / 0.700 / 0.416 | 0.283 / 0.683 / 0.456 | 0.233 / 0.683 / 0.419 |
| long-body | 30 | 1.000 / 1.000 / 1.000 | 0.967 / 1.000 / 0.983 | 0.967 / 1.000 / 0.983 | 0.967 / 1.000 / 0.983 |
| outdated | 30 | 0.533 / 1.000 / 0.761 | 0.767 / 1.000 / 0.878 | 0.533 / 1.000 / 0.761 | 0.767 / 1.000 / 0.878 |
| near-dup | 24 | 1.000 / 1.000 / 1.000 | 0.958 / 1.000 / 0.979 | 1.000 / 1.000 / 1.000 | 0.958 / 1.000 / 0.979 |
| multi-hop | 20 | 0.000 / 0.900 / 0.382 | 0.000 / 0.850 / 0.383 | 0.000 / 0.900 / 0.382 | 0.000 / 0.850 / 0.382 |

Reading the tables: the reranker's gains are concentrated where the question shares words with the answer's wording: `outdated` hit@1 jumps (MiniLM 0.467 to 0.733, gte 0.533 to 0.767) because the current entry's text matches a "current X" question better than the superseded one, `natural` hit@1 +0.09 and +0.07, `keyword` to 0.993. It does nothing for paraphrase (0.183 to 0.167 MiniLM, flat for gte), multi-hop hit@5 moves by at most 0.05 either way, and gte near-dup slips from 1.000 to 0.958 hit@1 (one query). It does not reduce false positives on negative queries (still 1.00): the reranker's top score for a negative query (median 0.29) overlaps the scores of true paraphrase hits (median 0.24). Cost: one more model (35 MB) held in the serve worker and 0.2 to 0.3 s per recall (p50 184 ms MiniLM, 327 ms gte at this machine load; p95 about 0.5 to 0.6 s; the CLI README quotes about 0.3 s for 25 passages).

### Final recommended configuration

| | Config | Download | Build (208 entries) | Search p50 / p95 | hit@1 / hit@5 / MRR@10 |
| --- | --- | --- | --- | --- | --- |
| **Recommended** | `gte-modernbert-base`, `q8`, rerank off | 150 MB + 3.6 MB tokenizer | 12.3 s | 17 / 21 ms | 0.738 / 0.944 / 0.832 |
| Quality, if 0.3 s per recall is acceptable | same + `rerank: nachos` (jina tiny, q8) | + 35 MB | 12.9 s | 327 / 569 ms | 0.773 / 0.942 / 0.848 |
| Lightweight | `all-MiniLM-L6-v2`, `q8`, rerank off | 23 MB + 0.7 MB | 3.6 s | 7 / 9 ms | 0.718 / 0.921 / 0.810 |
| Lightweight + rerank | same + `rerank: nachos` | + 35 MB | 3.3 s | 184 / 507 ms | 0.763 / 0.926 / 0.833 |
| Lexical only | `embeddings: off` | none | 0.8 s | 0.1 / 0.2 ms | 0.674 / 0.812 / 0.737 |

The model recommendation from the previous section stands with the new numbers: gte-modernbert q8 is the quality choice (best hit@5 and MRR with rerank off, and cheaper per recall than MiniLM plus reranker), MiniLM q8 the lightweight one. Rerank stays **off by default**: it buys +0.035 to +0.045 hit@1 (mostly natural-language and "current X" questions) for 0.2 to 0.3 s on every recall and a second model, and gives no hit@5 gain on gte. Turn it on when hit@1 matters more than latency. Defaults in code: chunk penalty 0.05 (on), duplicate calibration auto, similarity floor off, rerank off. Caveats: the fixture is synthetic, builds and latencies were measured under heavy machine load (build seconds scale with entries and machine, p95s are noisy), and no real-memory run was done.


### Not measured

No OpenAI-compatible server (Ollama etc.) was reachable, so the `openai` backend has no numbers; the `sentence-transformers` row above predates the query type and used no prefixes.

## Adversarial set

`bench/data/adversarial/` is an independent eval set for `suggested_supersedes` and for recall over a store with superseded entries. It was written before the suggestion heuristic (`nachos_core/quality.py`, `validity.py`) or the main fixture were read, so it was not shaped by what the heuristic does, and the heuristic was not tuned against it afterwards: the numbers below are a one-shot report.

Run it with `--data-dir`:

```bash
python -m bench --data-dir bench/data/adversarial --suggest-eval          # suggestion precision/recall by case
python -m bench --data-dir bench/data/adversarial                         # recall, supersedes applied by label
python -m bench --data-dir bench/data/adversarial --no-supersedes         # same, supersedes links not applied
```

Files (all fictional):

- `memories.json`: 63 base entries (same format as the main fixture).
- `updates.json`: 65 follow-up puts applied in order after the base entries. Each has `update_of: <key>` (37, including 2 chains where the target is itself an update) or `update_of: null` with a `reason` (28 distinct entries), plus a `case` tag. The loader appends them to the memories, turning `update_of` into `supersedes`, so `--suggest-eval` and `--no-supersedes` work unchanged. Every update has a new title (a same-title put would just overwrite), so titles differ from the old one by a little (`... v3`, `... (Postgres 17)`) or a lot (`Thumbnail service` for `Image resizer lambda`).
- `queries.json`: 96 queries over the final state: 37 `current`, 24 `outdated` (worded from the superseded entry only; expected key is the current one), 20 `sibling` (must pick the right twin or additive note) and 15 `no-answer` (`expect: []`).

`--suggest-eval` adds a per-`case` table when entries carry a `case`. No-answer queries are left out of hit@k and counted separately ("returned results").

| Update cases (label: update) | n | Distinct cases (label: distinct) | n |
| --- | --- | --- | --- |
| update-nocue (no cue words) | 7 | distinct-sibling (near-identical wording, other facet) | 10 |
| update-rename (only a name or URL changes) | 5 | distinct-additive-numbers (notes with numbers) | 6 |
| update-negation ("no longer uses X") | 4 | distinct-different-subject (shared vocabulary) | 4 |
| update-question / update-log | 2 / 3 | distinct-negation (additive "has no X") | 3 |
| update-multifact (one fact of several changed) | 4 | distinct-log (incident or retro) | 2 |
| update-title-differs | 4 | distinct-unrelated | 2 |
| update-i18n (German/French) | 3 | distinct-related | 1 |
| update-markdown (code blocks) | 3 | | |
| update-chain (update of an update) | 2 | | |

### Suggestion results

Machine as above; MiniLM-L6 q8 through the `nachos-embeddings` CLI (branch `naterchrdsn/emb-prefixes`, typed, `NACHOS_EMBEDDINGS_CACHE` outside both repos). 37 labelled updates, 28 distinct entries. Suggestions are checked against the labels as each entry is put, with no supersedes applied during the replay.

| Backend | precision | recall | F1 | TP/FP/FN |
| --- | --- | --- | --- | --- |
| lexical (off) | 0.500 | 0.054 | 0.098 | 2/2/35 |
| MiniLM-L6 q8 | 0.429 | 0.081 | 0.136 | 3/4/34 |

By case (TP/n for updates; FP for distinct entries; lexical / MiniLM):

| Case | n | lexical | MiniLM q8 |
| --- | --- | --- | --- |
| update-nocue | 7 | 0/7 | 0/7 |
| update-rename | 5 | 0/5 | 0/5 |
| update-negation | 4 | 0/4 | 0/4 |
| update-question | 2 | 0/2 | 0/2 |
| update-log | 3 | 0/3 | 0/3 |
| update-multifact | 4 | 1/4 | 1/4 |
| update-title-differs | 4 | 0/4 | 0/4 |
| update-i18n | 3 | 0/3 | 0/3 |
| update-markdown | 3 | 0/3 | 0/3 |
| update-chain | 2 | 1/2 | 2/2 |
| distinct-sibling | 10 | 2 FP | 4 FP |
| other distinct cases (18 entries) | 18 | 0 FP | 0 FP |

On this set the suggester is near-silent rather than noisy: it finds the update when the title is the old title plus a version or qualifier (`Plans and prices` to `Plans and prices v3`, `Nimbus billing API` to `... endpoints`) and almost nothing else.

### Why it fails

Read after the run, from `nachos_core/quality.py` and `validity.py` (no change made):

1. **The title gate rejects most real updates.** `_titles_compatible_for_supersede` needs two shared core title tokens (or at least half of them) and no one-sided qualifiers on both titles. 33 of the 37 updates fail it, because an update that is worded as a new entry usually retitles it (`Postgres primary host` to `Primary database server`, `Team standup` to `Daily sync meeting`, `Auth service` to `Why do we get logged out after 5 minutes?`). Applying only the body check (`looks_like_update` without titles) would fire on 26 of the 37, so the title gate is the dominant recall limit. Examples: `Nightly backup job`, `Plans and prices` (nocue), `Supportzeiten`, `Rechnungsnummern` (i18n: the German titles share no token with the old ones), `Thumbnail service` (title-differs).
2. **Candidate retrieval depends on the same titles.** Candidates are the duplicate-similarity hits plus entries whose core title tokens are exactly equal. A retitled update has neither unless the embedding duplicate threshold happens to catch it, so it is never compared. `Feature flags` vs `Feature flag service` passes the title gate and still misses (the body only changes a hostname: no digits or paths the fact regex sees).
3. **The body test only sees numbers, paths, URLs and a few cadence words.** A change of a name, host without a path, tool or person is invisible: `update-rename` (`Grafana dashboards` is a bare hostname change; `On-call roster` swaps two first names; `Transactional email sender` changes one address) never differs in a fact token. Negations (`We no longer use Sentry`, `WireGuard is not used anymore`) and prose changes (`Opsgenie` for `PagerDuty`, `Snowflake` for `BigQuery`, `Adyen` for `Stripe`) have no number change, and the cue list does not contain "no longer", "not used anymore", "is out" or "stopped using".
4. **Facet qualifiers only block when both titles have one.** `Postgres read replica for analytics`, `CI runner pool for macOS`, `Sprint cadence for the design team` and `Support response targets for enterprise` add a qualifier to an existing title; the old title has no qualifier of its own, so `old_only` is empty and the pair passes as "compatible". Their bodies differ in a number or hostname, so the body test fires: 4 false positives for MiniLM (2 for lexical, which only retrieved the macOS and enterprise ones as candidates). A sibling fact whose twin is not itself labelled stays one-sided and is mistaken for a version.
5. **What did not fire falsely:** none of the 18 additive, negation, log, different-subject or unrelated entries produced a suggestion, because they differ in title and the title gate holds. The same gate that protects precision here is what costs recall in item 1.
6. **Chains work only with a close title.** `Primary database server (Postgres 17)` was missed by lexical (core title not equal because of the extra `postgres` token, and no embedding hit) and caught by MiniLM.

### Recall with and without supersedes

96 queries, supersedes applied by label (`--no-supersedes` leaves every entry active). No-answer queries are excluded from the hit metrics (see below). The first row of each pair is lexical, the second MiniLM-L6 q8.

| Run | hit@1 | hit@5 | MRR@10 |
| --- | --- | --- | --- |
| lexical, supersedes applied | 0.889 | 0.951 | 0.914 |
| lexical, no supersedes | 0.617 | 0.926 | 0.768 |
| MiniLM q8, supersedes applied | 0.951 | 1.000 | 0.972 |
| MiniLM q8, no supersedes | 0.679 | 0.988 | 0.826 |

Per category, hit@1 / hit@5 / MRR@10:

| Category | n | lexical, applied | lexical, none | MiniLM, applied | MiniLM, none |
| --- | --- | --- | --- | --- | --- |
| current | 37 | 0.919 / 1.000 / 0.955 | 0.730 / 1.000 / 0.849 | 1.000 / 1.000 / 1.000 | 0.811 / 1.000 / 0.901 |
| outdated | 24 | 0.750 / 0.833 / 0.779 | 0.125 / 0.750 / 0.450 | 0.875 / 1.000 / 0.927 | 0.250 / 0.958 / 0.587 |
| sibling | 20 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 0.950 / 1.000 / 0.975 | 0.950 / 1.000 / 0.975 |

- Applying the labels is worth +0.27 (lexical) and +0.27 (MiniLM) overall hit@1 and +0.63 on `outdated`. This is the ceiling a perfect suggester could reach; today the suggester delivers almost none of it.
- With links applied, lexical `outdated` still misses 4 of 24 at hit@5 (`db-prim-01 host`, `standup 09:45 Google Meet`, `redis-sess.internal 6379 allkeys-lru`, `lyon-app-02 deploiement jeudi`): the old wording is only in the now-hidden entry, so a keyword search has nothing to match. Embeddings recover them. Archiving an old entry makes its wording unfindable unless its text is folded into the successor.
- Without links, `outdated` hit@1 is 0.125 (lexical) and 0.250 (MiniLM): the old entry usually outranks the new one because queries use its words.
- No-answer: 13/15 (lexical) and 15/15 (MiniLM) of the no-answer queries still return results, since recall has no relevance floor. A caller cannot tell "nothing known" from "a weak match" from the result list alone.

## Consolidated results (`memory-next` HEAD)

Measured on `naterchrdsn/memory-next` (commit `9c662836bb`, integrating lexical + metadata caching, chunked body vectors with 0.05 chunk penalty, calibrated duplicate/update floors, `possible_updates` candidates with facet-qualifier reasons, superseded search redirect before reranking, and optional cross-encoder reranking). CLI built from `nachos-embeddings` archive `a0df123` (`rerank` branch), `q8` models (`Xenova/all-MiniLM-L6-v2`, `Alibaba-NLP/gte-modernbert-base`, `jinaai/jina-reranker-v1-tiny-en`), `--repeat 1`, supersedes applied on both fixtures.

### Recall matrix (original + held-out adversarial, supersedes applied)

| Backend | Rerank | Original hit@1 / hit@5 / MRR@10 | Original outdated hit@1 | Original p50 / p95 ms | Original build s | Adversarial hit@1 / hit@5 / MRR@10 | Adversarial outdated hit@1 | Adversarial p50 / p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical (`off`) | off | 0.695 / 0.812 / 0.748 | 0.867 | 0.2 / 0.3 | 1.14 | 0.975 / 1.000 / 0.986 | 1.000 | 0.1 / 0.2 |
| MiniLM-L6 q8 | off | 0.758 / 0.924 / 0.832 | 1.000 | 7.3 / 8.4 | 4.72 | 0.988 / 1.000 / 0.994 | 1.000 | 4.5 / 5.0 |
| MiniLM-L6 q8 | on (`sum:1`) | 0.784 / 0.926 / 0.845 | 0.967 | 181.8 / 490.6 | 4.49 | 0.951 / 1.000 / 0.975 | 0.875 | 122.5 / 149.0 |
| **gte-modernbert q8** | **off** | **0.773 / 0.949 / 0.850** | **1.000** | **17.0 / 26.4** | **13.82** | **1.000 / 1.000 / 1.000** | **1.000** | **12.3 / 14.4** |
| gte-modernbert q8 | on (`sum:1`) | 0.791 / 0.949 / 0.858 | 0.967 | 320.1 / 518.4 | 14.30 | 0.951 / 1.000 / 0.975 | 0.875 | 128.9 / 211.2 |

Notes:
- Original fixture has 208 memories, 393 answerable queries + 30 `negative` queries (`fp@1 / fp@5`: lexical `0.767 / 0.707`, semantic runs `1.000 / 1.000` with `similarity_floor: off`). Adversarial set has 128 memories, 81 answerable queries + 15 no-answer queries (`13/15` lexical, `15/15` semantic still returned results).
- Search redirect (`superseded_by` chain substitution before reranking) eliminates the old-wording lexical gap: adversarial lexical `outdated` hit@1 jumps from `0.750` (before redirect) to `1.000` (`0.125` without supersedes), and both MiniLM q8 and gte-modernbert q8 reach `1.000` on `outdated` across both fixtures with rerank off.
- Rerank trade-off on redirected queries: enabling the cross-encoder raises original `natural` (+0.08–0.10) and `keyword` (to `0.993`) hit@1, lifting overall original hit@1 by `+0.018` to `+0.026`, but drops adversarial `outdated` hit@1 (`1.000 -> 0.875`, 3/24 queries) because a redirected successor that no longer contains the query's old terms receives a low cross-encoder score.

### `possible_updates` candidate evaluation (`--suggest-eval`)

Replayed in order without `supersedes` (original tune set: 15 updates / 193 distinct puts; held-out adversarial set: 37 updates / 91 distinct puts). Calibrated floor reuses the index's cached `p99` random head-pair cosine when `>= 30` vectors are stored, falling back to `max(0.45, duplicate_threshold - 0.17)` (`0.48`) below 30 entries.

| Backend | Floor | Original R@1 / R@3 | Original distinct fp@1 (non-dup) | Adversarial R@1 / R@3 | Adversarial distinct fp@1 |
| --- | --- | --- | --- | --- | --- |
| lexical (`off`) | IDF cosine `>= 0.25` | 0.933 / 1.000 | 0.233 (0.160) | 0.973 / 0.973 | 0.253 |
| MiniLM-L6 q8 | uncalibrated (`0.48`) | 1.000 / 1.000 | 0.399 (0.331) | 1.000 / 1.000 | 0.275 |
| MiniLM-L6 q8 | **calibrated (`p99`)** | **1.000 / 1.000** | **0.394 (0.331)** | **1.000 / 1.000** | **0.275** |
| gte-modernbert q8 | uncalibrated (`0.48`) | 1.000 / 1.000 | 0.995 (0.994) | 1.000 / 1.000 | 0.989 |
| gte-modernbert q8 | **calibrated (`p99`)** | **1.000 / 1.000** | **0.705 (0.692)** | **1.000 / 1.000** | **0.648** |

Readings:
- Replacing the brittle title/cue gate with similarity-ranked `possible_updates` candidates lifts held-out adversarial recall from `0.054 / 0.081` to **`0.973` (lexical, 36/37)** and **`1.000` (MiniLM q8 & gte-modernbert q8, 37/37 at rank 1)**.
- Calibrating the semantic floor to `p99` is essential for shifted-range models like `gte-modernbert`: it cuts distinct `fp@1` by `29–34` percentage points (`0.995 -> 0.705` original, `0.989 -> 0.648` adversarial) with zero recall loss (`1.000` R@1 on both sets). Remaining `gte` distinct `fp@1` comes mainly from the cold-start window (`< 30` entries before calibration activates, where `0.48` sits below `gte`'s `p50` of `0.53`) and parallel siblings (which now carry an explicit `'different qualifier: ...'` warning in `reason`).

### Recommended configuration

| Profile | Settings | Download | Build (208) | Search p50 / p95 (orig) | Original hit@1 / hit@5 / MRR@10 | Adversarial hit@1 / hit@5 |
| --- | --- | --- | --- | --- | --- | --- |
| **Recommended (quality)** | `embeddings: nachos`, `embedding_model: Alibaba-NLP/gte-modernbert-base`, `embedding_dtype: q8`, `rerank: off`, `duplicate_calibration: auto`, `similarity_floor: off` | ~154 MB | 13.8 s | 17 / 26 ms | **0.773 / 0.949 / 0.850** | **1.000 / 1.000** |
| **Lightweight (default)** | `embeddings: nachos`, `embedding_model: Xenova/all-MiniLM-L6-v2`, `embedding_dtype: q8`, `rerank: off`, `duplicate_calibration: auto`, `similarity_floor: off` | ~24 MB | 4.7 s | 7 / 8 ms | 0.758 / 0.924 / 0.832 | 0.988 / 1.000 |
| Max original hit@1 (slower) | `gte-modernbert-base` q8 + `rerank: nachos`, `rerank_dtype: q8` | +35 MB | 14.3 s | 320 / 518 ms | 0.791 / 0.949 / 0.858 | 0.951 / 1.000 |
| Zero-dependency | `embeddings: off` | none | 1.1 s | 0.2 / 0.3 ms | 0.695 / 0.812 / 0.748 | 0.975 / 1.000 |

- **Keep `rerank: off` by default**: with supersedes + redirect active, `gte-modernbert-base` q8 (`rerank: off`) already hits `1.000` on `outdated`, `long-body`, `near-dup`, and every adversarial query at `12–17 ms` p50, while `rerank: on` costs `120–320 ms` per recall and slightly degrades redirected old-wording queries.
- **Defaults in code**: `chunk_penalty = 0.05`, `duplicate_calibration = "auto"`, `calibrate_updates = True` (`p99` floor when calibrated), `similarity_floor = "off"`, `rerank = "off"`.
