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

## Query/document type and model matrix

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

### Recommended default

On this fixture, keep `Xenova/all-MiniLM-L6-v2` as the default model. It has the best overall hit@5 (0.891) and MRR (0.779) of the semantic runs and by far the best paraphrase recall, and with `dtype: q8` it is essentially identical to fp32 (hit@5 0.891 both, MRR 0.778 vs 0.779) while building about a quarter faster, so q8 is the sensible setting for speed. None of the retrieval-tuned models (bge-small, arctic-embed-s, gte-modernbert) beat it overall here; gte-modernbert is best on natural questions and multi-hop but costs about 2.5x the build time and 768-dimensional vectors, and loses on long-body and paraphrase. If a prefixed model is chosen, send the query type (the default now): it is worth about 5 points of hit@1 for arctic-embed-s. Re-check on real memory data before changing the default; long-body and outdated are weak for every embedding setup, so chunked body vectors and recency handling matter more than the model choice.

### Not measured

No OpenAI-compatible server (Ollama etc.) was reachable, so the `openai` backend has no numbers; the `sentence-transformers` row above predates the query type and used no prefixes.
