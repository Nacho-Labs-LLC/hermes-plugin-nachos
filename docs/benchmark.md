# Benchmarks

Reproducible, stdlib-only benchmarks for memory recall quality and latency, so a change such as a different embedding model, prefix setting or reranker can be judged by numbers. They only measure; they never change recall behaviour. Use them to compare models for your own machine and to check that a code change did not move ranking.

Contents: [Method](#method) · [Running it](#running-it) · [Benchmarking your own model](#benchmarking-your-own-model) · [Current results](#current-results) · [Known limits](#known-limits)

## Method

### Data sets

All data is synthetic (fictional projects, hosts and people) and shaped like real agent memory. Each set is a directory of JSON under `bench/data/`, loaded through the normal `put()` path into a temporary store.

| Set | Directory | Entries | Queries | Purpose |
| --- | --- | --- | --- | --- |
| Main | `bench/data` | 208 | 443 (393 answerable, 50 `negative`) | Recall quality across query styles. Some settings were tuned on it. |
| Adversarial | `bench/data/adversarial` | 63 base + 65 follow-up updates = 128 | 111 (81 answerable, 30 `no-answer`) | Held-out check for supersede handling and abstention. Written before the update heuristic was read and not tuned on afterwards. |
| Links (held out) | `bench/data/links-heldout` | 76 | 6 | Link-suggestion precision and recall (`--link-eval`). |
| History | `bench/data/history.json` | 6 synthetic sessions | 29 (22 answerable, 7 not) | `nachos_history_search` ranking and context windows (`python -m bench.history`). |

Main-set query categories (`category` in `queries.json`; each query has `expect`, a list of memory keys, and `match`: `any`, or `all` for multi-hop):

| Category | n | What it tests |
| --- | --- | --- |
| `keyword` | 139 | Short keyword queries, as an agent might type. |
| `natural` | 90 | A full question that shares words with the entry. |
| `paraphrase` | 60 | Same meaning, no shared words (enforced by a test), so lexical-only scores 0 by construction. |
| `long-body` | 30 | The answer sits only in the body after about 2000 characters. Tests chunked body vectors. |
| `outdated` | 30 | Worded from the superseded entry of an old/new pair; the expected key is the current entry. |
| `near-dup` | 24 | A sibling entry differs by one detail (staging vs production); the right one must outrank its twin. |
| `multi-hop` | 20 | Needs two entries. The rank is that of the last expected key, so hit@1 is 0 by construction; read hit@5 and MRR. |
| `negative` | 50 | No correct answer exists (30 off-domain, 20 near-miss on a stored topic). Reported as false-positive and abstention rates. |

The adversarial set has `current` (37), `outdated` (24, worded from the superseded entry), `sibling` (20) and `no-answer` (30) queries, and 65 updates applied in order after the base entries. Updates carry an `update_of` label that the loader turns into `supersedes`; `--no-supersedes` leaves them out.

### Metrics

- **hit@1 / hit@5**: share of answerable queries whose expected entry is in the top 1 / top 5 results.
- **MRR@10**: mean reciprocal rank of the expected entry within the top 10 (0 if absent).
- **p50 / p95 ms**: median and 95th-percentile search latency (median over `--repeat` runs per query). Machine-dependent; quality numbers are deterministic for a given backend and model, apart from small run-to-run noise in some model runs (about 0.002 hit@1).
- **build s**: time to `put()` every entry, including embedding. **model load s**: first `embed()` call, excluded from the other timings.
- **fp@1 / fp@5** (negative queries): share returning anything, and mean filled share of the top five slots. Lower is better. Without a relevance floor recall always returns something, which is why abstention exists.
- **Abstention P / R / F1**: a query is "abstained" when `search_with_confidence` reports `no_confident_match`. Recall is the share of unanswerable queries abstained on; precision is the share of abstained queries that really were unanswerable (answerable ones abstained on count against it); F1 combines them. The default `abstention: hint` only reports this; it does not change results.
- **Update suggestion R@1 / R@3** (`--suggest-eval`): replaying the puts in order without `supersedes`, how often `possible_updates` ranks the true older entry first / in the top three. **distinct fp@1** is how often a put of a genuinely new fact still gets a suggestion at rank 1 (lower is better; non-blocking, since the agent decides).
- **Link P / R** (`--link-eval`): precision and recall of `suggested_links` against hand-labelled entry pairs. `hard FP` counts labelled non-pairs that were suggested.
- **Multi-query** (`--multi-query`): hit@1 / hit@5 for one query versus the grouped rephrasings of the same question fused with RRF.
- **History**: `m@k` = the matched message holds the answer within the top k; `ctx@5` = the answer is anywhere in a returned result including neighbouring turns; `FP` = share of unanswerable queries that returned something.

## Running it

From the repository root with the dev environment (`uv sync --group dev`); lexical runs need nothing else.

```bash
python -m bench                                   # lexical only (--embeddings off)
python -m bench --embeddings nachos --model Xenova/all-MiniLM-L6-v2 --dtype q8
python -m bench --embeddings openai --base-url http://localhost:11434/v1 --model nomic-embed-text
python -m bench --data-dir bench/data/adversarial # held-out set
python -m bench --category paraphrase --misses    # one category; list queries missing the top 5
python -m bench --repeat 5 --json results.json    # full report with per-query ranks
python -m bench.history [--noise N] [--repeat R] [--json out.json]
```

| Flag | Meaning |
| --- | --- |
| `--embeddings` | Comma-separated backends to run: `off`, `nachos`, `sentence-transformers`, `openai`. With `--model` and no `--embeddings` it runs `off,nachos` so the result sits next to the lexical baseline. |
| `--model`, `--dtype`, `--pooling`, `--max-length`, `--base-url`, `--api-key-env`, `--query-prefix`, `--document-prefix` | Passed through like the `embedding_*` config keys. |
| `--no-types` | `nachos` backend: do not send the query/document type, to measure what it adds. |
| `--rerank`, `--rerank-borderline`, `--rerank-model`, `--rerank-dtype`, `--rerank-max-length`, `--rerank-candidates` | Measure the cross-encoder (full or borderline-only). |
| `--suggest-eval`, `--link-eval` (with `--calibrate-duplicates`), `--multi-query` | Run the suggestion, link and multi-query evaluations. |
| `--no-supersedes` | Load the fixture without applying supersede links. |
| `--data-dir DIR` | Use another set (`memories.json`, `queries.json`, optional `updates.json`, `links.json`). |
| `--category C` | Only this query category (repeatable). `--misses` lists failures. |
| `--repeat N`, `--label`, `--command`, `--synthetic N`, `--json PATH` | Latency repeats; run name; `nachos-embeddings` executable; pad the corpus with synthetic distractors to N entries; write the report. |

A backend that is not available is skipped with a reason, not an error (no `nachos-embeddings` on `PATH`, `sentence-transformers` not installed, `openai` without `--base-url` and `--model`, an embedder that fails warm-up). The exit status is non-zero only when no backend ran. `pytest` does not run the benchmarks; `tests/test_bench_recall.py` covers the metric math, fixture integrity and a small lexical smoke run.

## Benchmarking your own model

Any Hugging Face ONNX encoder (or a local directory) works with the `nachos` backend:

```bash
python -m bench --model org/some-model --dtype q8                       # main set, next to lexical
python -m bench --model org/some-model --dtype q8 --data-dir bench/data/adversarial
python -m bench --model org/some-model --dtype q8 --suggest-eval --multi-query
```

If the model needs a prefix or a pooling the CLI does not know, pass `--query-prefix`, `--document-prefix`, `--pooling`. Compare the same `--dtype` across models, run on an idle machine, and use `--repeat 3` or more for latency. Treat differences under about 0.01 hit@1 on the main set (4 queries) as noise. If a model wins here, set it in `embedding_model` and `embedding_dtype`; Nachos re-embeds in the background ([docs/models.md](models.md#switching-models)).

## Current results

Measured on the 0.7.0 code with the `nachos-embeddings` CLI, `q8` models (`Xenova/all-MiniLM-L6-v2`, `Alibaba-NLP/gte-modernbert-base`, reranker `jinaai/jina-reranker-v1-tiny-en`), typed queries, supersedes applied, one 16-thread CPU. Latencies are indicative.

### Recall and abstention

| Backend | Rerank | Main hit@1 / hit@5 / MRR | Main outdated hit@1 | Main p50 / p95 ms | Main abstention P / R / F1 | Adv. hit@1 / hit@5 / MRR | Adv. outdated hit@1 | Adv. p50 / p95 ms | Adv. abstention P / R / F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lexical (`off`) | off | 0.695 / 0.812 / 0.748 | 0.867 | 0.5 / 0.7 | 0.312 / 0.680 / 0.428 | 0.975 / 1.000 / 0.986 | 1.000 | 0.5 / 0.7 | 1.000 / 0.667 / 0.800 |
| MiniLM-L6 | off | 0.758 / 0.924 / 0.832 | 1.000 | 9.2 / 19.1 | 0.448 / 0.520 / 0.481 | 0.988 / 1.000 / 0.994 | 1.000 | 5.9 / 6.9 | 1.000 / 0.333 / 0.500 |
| gte-modernbert | off | 0.773 / 0.949 / 0.850 | 1.000 | 22.0 / 43.5 | 0.367 / 0.580 / 0.450 | 1.000 / 1.000 / 1.000 | 1.000 | 16.9 / 32.8 | 1.000 / 0.467 / 0.636 |
| gte-modernbert | `borderline` | 0.773 / 0.949 / 0.850 | 1.000 | 32.4 / 200.1 | 0.326 / 0.560 / 0.412 | 1.000 / 1.000 / 1.000 | 1.000 | 15.8 / 24.9 | 1.000 / 0.567 / 0.723 |
| gte-modernbert | `nachos` | 0.791 / 0.949 / 0.858 | 0.967 | 441.4 / 730.8 | 0.330 / 0.580 / 0.420 | 0.951 / 1.000 / 0.975 | 0.875 | 204.9 / 274.2 | 1.000 / 0.533 / 0.696 |

Build time for the main set: about 1 s lexical, 5 s MiniLM, 14 s gte-modernbert.

- **Embeddings help most on paraphrases.** Lexical hit@1 is 0.000 there by construction; with embeddings and one query it is about 0.2, and with grouped rephrasings (below) 0.97.
- **Supersede redirect fixes outdated queries.** Search follows `superseded_by` to the current entry before ranking. On the adversarial set lexical `outdated` hit@1 is 1.000 with supersedes applied and 0.125 without.
- **Full reranking (`rerank: nachos`) raises main hit@1 by about 0.02** but costs 120-440 ms per recall and lowers adversarial `outdated` hit@1 (1.000 to 0.875): a redirected successor that no longer contains the query's old wording scores low. `borderline` re-scores about 10-13% of queries, keeps latency near the median, and raises adversarial abstention recall (MiniLM 0.333 to 0.567, gte 0.467 to 0.567, precision 1.000). Neither is on by default.
- **Abstention is advisory.** On the main set precision stays under about 0.5 at recall near 0.5 because the 60 paraphrase queries share no words with their answers and look like non-matches. On the adversarial set no answerable query is abstained on (0 of 81). This is why the default is `hint`.
- Negative queries: lexical `fp@1 / fp@5` 0.767 / 0.707; semantic runs 1.000 / 1.000 (`similarity_floor: off`).

### Update suggestions (`--suggest-eval`)

Main set: 15 updates among 193 distinct puts. Adversarial: 37 updates among 91 distinct puts. Calibrated floor is the model's 99th-percentile random-pair similarity.

| Backend | Main R@1 / R@3 | Main distinct fp@1 | Adv. R@1 / R@3 | Adv. distinct fp@1 |
| --- | --- | --- | --- | --- |
| lexical | 0.933 / 1.000 | 0.233 | 0.973 / 0.973 | 0.253 |
| MiniLM-L6 | 1.000 / 1.000 | 0.394 | 1.000 / 1.000 | 0.275 |
| gte-modernbert | 1.000 / 1.000 | 0.539 | 1.000 / 1.000 | 0.352 |

Calibration matters for models with a shifted similarity range: without it gte-modernbert's distinct fp@1 is 0.995 / 0.989 instead of the values above.

### Multi-query recall (`--multi-query`)

Grouped rephrasings of one question (paraphrase + natural + keyword), fused with RRF in one embedding call.

| Backend | 3-query groups (n = 29): 1 query, hit@1 | 3 queries, hit@1 / hit@5 | p50 ms (3 queries) | All groups (n = 151): 1 query, hit@1 | multi-query, hit@1 / hit@5 |
| --- | --- | --- | --- | --- | --- |
| lexical | 0.000 | 0.483 / 0.931 | 0.8 | 0.450 | 0.623 / 0.940 |
| MiniLM-L6 | 0.241 | 0.966 / 1.000 | 33.8 | 0.576 | 0.947 / 1.000 |
| gte-modernbert | 0.172 | 0.966 / 1.000 | 114.9 | 0.596 | 0.974 / 1.000 |

Cost is about 1.6-2.6x the single-query latency. With `rerank: nachos`, multi-query caps reranking at the top 5 candidates and takes the best score across variants, reaching hit@1 1.000 on the triples at lower latency than single-query full reranking.

### Links (`--link-eval`)

Suggestions are deliberately precision-first and recall-low; most labelled pairs are semantic and share no identifier. Adversarial set (61 labelled pairs) and held-out set (78 pairs, 15 hard negatives):

| Backend | Adv. precision / recall | Held-out precision / recall | Held-out hard FP |
| --- | --- | --- | --- |
| lexical | 0.660 / 0.459 | 0.480 / 0.154 | 0 of 15 |
| MiniLM-L6 | 0.631 / 0.557 | 0.550 / 0.282 | 0 of 15 |
| gte-modernbert | 0.646 / 0.557 | 0.538 / 0.269 | 2 of 15 |

Links do not change recall ranking: with no `[[links]]` in the fixtures, hit@1 / hit@5 / MRR are unchanged to three decimals on both sets. A graph boost (adding a share of a strong hit's score to its linked entries) was measured and rejected: it lowered lexical adversarial hit@1 from 0.975 to 0.914 at a small boost, because linked entries are related rather than the answer to the same query.

### History search (`python -m bench.history`)

| Variant | m@1 | m@5 | ctx@5 | FP | p50 ms (base / +200 checkpoints) |
| --- | --- | --- | --- | --- | --- |
| every query word must appear (old rule) | 0.318 | 0.318 | 0.318 | 0.143 | 0.6 / 21.4 |
| IDF ranking, no context | 0.864 | 0.864 | 0.955 | 0.000 | 0.8 / 27.6 |
| IDF ranking + context windows (current) | 0.864 | 0.864 | 1.000 | 0.000 | 0.8 / 33.3 |

### Chunked body vectors, chunk penalty and calibration

- Long entries (over 1200 characters) get overlapping body-chunk vectors (up to 16 per entry). Without them, `long-body` queries failed once embeddings were on, because the embedder only saw the start of the body.
- A chunk's score is reduced by 0.05 (`chunk_penalty`) so a chunk matching one sentence cannot outrank a head vector that matches the whole topic. Removing it cost paraphrase hit@1.
- Cosine ranges differ a lot between models (random pairs sit near 0.1 for MiniLM and 0.5 for bge), so the duplicate threshold and optional similarity floor are calibrated per model from random-pair statistics: offline priors below 30 entries, blended into the store's own distribution from 30 to 100.

### Switching cost

MiniLM q8 to gte-modernbert q8 on the 208-entry store (`reindex_budget_ms` 250): the first recall used to block for 17.2 s. Now recalls during the switch are lexical, take a median 314 ms (max 1.8 s) and the switch completes after 43 recalls.

## Known limits

- **Synthetic, small, partly tuned.** The main set (208 entries) informed several defaults; the adversarial recall/update set and the links-held-out set were not tuned on. Real stores differ in vocabulary, length and noise. Differences under 0.01 hit@1 are not significant.
- **Latency is one machine.** Numbers are medians on one idle 16-thread CPU; some rows of the README model table (notably nomic) were measured on a busier machine.
- **Paraphrase remains the weakest category** for single queries; multi-query rephrasing is the main mitigation and depends on the agent supplying good variants.
- **Abstention cannot separate paraphrases from non-matches** on the main set, so it is advisory. Near-miss negatives (a stored topic, an unstored fact) are the hardest to abstain on.
- **Update and link suggestions are heuristics.** Update suggestions flag some genuinely new facts (distinct fp@1 0.28-0.54 with embeddings); they are non-blocking. Link suggestions miss relations that share no identifier, host, path or name.
- **English only.** The fixtures and the default models are English.
- **Not measured:** stores much larger than a few hundred entries beyond latency padding (`--synthetic`), non-English text, and reranker models other than `jina-reranker-v1-tiny-en`.
