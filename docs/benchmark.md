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
| `--model`, `--dtype`, `--base-url`, `--api-key-env` | Passed through like the `embedding_*` keys in `nachos_hermes/config.py`. |
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

### Not measured

`nachos-embeddings` is not installed in the baseline environment (not on `PATH`, and no package of that name on the npm registry), so the `nachos` backend (default Transformers.js model, `--dtype q8`) has no numbers yet, and no OpenAI-compatible server (Ollama etc.) was reachable for the `openai` backend. Run the commands above on a machine that has them and add rows here.
