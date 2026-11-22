"""Recall-quality and latency benchmark for Nachos memory (stdlib only).

Builds a temporary ``MemoryService`` (sqlite store in a temp dir), loads the
synthetic fixture through the normal ``put()`` path, runs every labelled query
through ``MemoryService.search`` and reports hit@1, hit@5, MRR@10, search
latency and build time, overall and per query category.

    python -m bench.recall                                  # lexical only
    python -m bench.recall --embeddings off,nachos --json out.json

Measurement only: nothing here changes recall behaviour. See docs/benchmark.md.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nachos_core.embeddings import EMBEDDING_BACKENDS, Embedder, make_embedder  # noqa: E402
from nachos_core.index import MemoryIndex  # noqa: E402
from nachos_core.memory import PROFILE, MemoryService, Scope  # noqa: E402
from nachos_core.store import get_store  # noqa: E402
from nachos_core.store.md_store import slugify  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
SEARCH_LIMIT = 10  # MRR is computed @10, so the service is asked for ten results
BACKENDS = tuple(b for b in EMBEDDING_BACKENDS if b != "auto")
CATEGORIES = (
    "keyword",
    "natural",
    "paraphrase",
    "long-body",
    "outdated",
    "near-dup",
    "multi-hop",
)


# -- dataset ---------------------------------------------------------------


def load_dataset(
    data_dir: Path = DATA_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(memories, queries) from ``memories.json`` and ``queries.json``."""
    memories = json.loads((data_dir / "memories.json").read_text(encoding="utf-8"))
    queries = json.loads((data_dir / "queries.json").read_text(encoding="utf-8"))
    return memories, queries


def expand_synthetic_memories(
    base: list[dict[str, Any]], target: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Deterministically expand ``base`` to ``target`` entries using in-memory distractors."""
    if target <= len(base):
        return list(base)
    rng = random.Random(seed)  # noqa: S311 - deterministic synthetic benchmark fixture
    vocab = sorted(
        {w for m in base for w in f"{m['title']} {m['summary']} {m['body']}".split() if len(w) >= 3}
    )
    cats = ("projects", "infra", "decisions", "procedures", "history", "tools")
    out = list(base)
    idx = 0
    while len(out) < target:
        idx += 1
        words = rng.choices(vocab, k=rng.randint(60, 140))
        out.append(
            {
                "title": f"Synthetic entry {idx:04d} {words[0]} {words[1]}",
                "summary": f"Notes on {words[2]} {words[3]} {words[4]} for {words[5]} subsystem.",
                "category": cats[idx % len(cats)],
                "body": " ".join(words),
            }
        )
    return out


# -- metrics (pure) --------------------------------------------------------


def rank_of(expect: list[str], results: list[str], match: str = "any") -> int | None:
    """1-based rank of the answer in ``results``, or None when it is not there.

    ``any``: rank of the best-placed expected key. ``all``: rank at which every
    expected key has appeared (multi-hop), so the worst-placed one counts.
    """
    positions = {key: i + 1 for i, key in reversed(list(enumerate(results)))}
    found = [positions[k] for k in expect if k in positions]
    if not found or (match == "all" and len(found) < len(expect)):
        return None
    return max(found) if match == "all" else min(found)


def hit_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr_at(ranks: list[int | None], k: int = 10) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None and r <= k) / len(ranks)


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (``pct`` in 0-100); 0.0 for no data."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def summarize(ranks: list[int | None], latencies_ms: list[float]) -> dict[str, Any]:
    return {
        "n": len(ranks),
        "hit@1": round(hit_at(ranks, 1), 4),
        "hit@5": round(hit_at(ranks, 5), 4),
        "mrr@10": round(mrr_at(ranks, 10), 4),
        "p50_ms": round(percentile(latencies_ms, 50), 2),
        "p95_ms": round(percentile(latencies_ms, 95), 2),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Overall and per-category summaries of per-query ``rows`` (rank, ms, category)."""
    out: dict[str, Any] = {
        "overall": summarize([r["rank"] for r in rows], [r["ms"] for r in rows]),
        "by_category": {},
    }
    present = {r["category"] for r in rows}
    for cat in [c for c in CATEGORIES if c in present] + sorted(present - set(CATEGORIES)):
        sel = [r for r in rows if r["category"] == cat]
        out["by_category"][cat] = summarize([r["rank"] for r in sel], [r["ms"] for r in sel])
    return out


# -- running ---------------------------------------------------------------


def build_service(directory: Path, embedder: Embedder | None) -> MemoryService:
    """A profile-only MemoryService over a sqlite store inside ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    scope = Scope(
        PROFILE,
        get_store("sqlite", directory / "memory.db"),
        MemoryIndex(directory / "memory.index.db"),
    )
    return MemoryService(OrderedDict([(PROFILE, scope)]), embedder=embedder)


def load_memories(
    svc: MemoryService, memories: list[dict[str, Any]], *, direct: bool = False
) -> None:
    if direct:
        store = svc.profile.store
        for mem in memories:
            store.put(
                slugify(mem["title"]),
                title=mem["title"],
                summary=mem["summary"],
                category=mem["category"],
                body=mem["body"],
            )
        return
    for mem in memories:
        result = svc.put(
            title=mem["title"],
            summary=mem["summary"],
            category=mem["category"],
            body=mem["body"],
        )
        if "error" in result:
            raise ValueError(f"fixture entry {mem['title']!r} rejected: {result['error']}")
        if result["stored"] != slugify(mem["title"]):
            raise ValueError(f"fixture entry {mem['title']!r} stored under {result['stored']!r}")


def run_queries(
    svc: MemoryService, queries: list[dict[str, Any]], repeat: int = 1
) -> list[dict[str, Any]]:
    """Search every query; latency is the median over ``repeat`` runs, ranks from the first."""
    rows = []
    for q in queries:
        timings, keys = [], []
        for i in range(max(1, repeat)):
            start = time.perf_counter()
            results = svc.search(q["query"], limit=SEARCH_LIMIT)
            timings.append((time.perf_counter() - start) * 1000.0)
            if i == 0:
                keys = [r["key"] for r in results]
        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "expect": q["expect"],
                "rank": rank_of(q["expect"], keys, q.get("match", "any")),
                "ms": percentile(timings, 50),
                "top": keys[:3],
            }
        )
    return rows


def resolve_embedder(backend: str, opts: argparse.Namespace) -> tuple[Embedder | None, str | None]:
    """(embedder, None) when usable, (None, None) for lexical-only, (None, reason) to skip."""
    if backend == "off":
        return None, None
    if backend == "nachos" and shutil.which(opts.command) is None:
        return None, f"`{opts.command}` not found on PATH (install nachos-embeddings)"
    if backend == "sentence-transformers" and (
        importlib.util.find_spec("sentence_transformers") is None
    ):
        return None, "sentence-transformers is not installed"
    if backend == "openai" and not (opts.base_url and opts.model):
        return None, "openai needs --base-url and --model"
    embedder = make_embedder(
        backend,
        model=opts.model,
        base_url=opts.base_url,
        api_key_env=opts.api_key_env,
        command=opts.command,
        dtype=opts.dtype,
    )
    return embedder, None


def probe(embedder: Embedder) -> tuple[float, str | None]:
    """Embed one string (loads the model); returns (seconds, error or None)."""
    start = time.perf_counter()
    try:
        embedder.embed(["warm-up"])
    except Exception as exc:  # any backend failure means "skip this backend"
        return time.perf_counter() - start, f"{type(exc).__name__}: {exc}"[:300]
    return time.perf_counter() - start, None


def run_backend(
    backend: str,
    opts: argparse.Namespace,
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    run: dict[str, Any] = {"backend": backend, "status": "ok"}
    embedder, reason = resolve_embedder(backend, opts)
    if reason:
        return {**run, "status": "skipped", "reason": reason}
    if embedder is not None:
        run["model_id"] = embedder.model_id
        run["model_load_s"], error = probe(embedder)
        run["model_load_s"] = round(run["model_load_s"], 3)
        if error:
            embedder.close()
            return {**run, "status": "skipped", "reason": f"embedder unavailable: {error}"}
        run["dimension"] = embedder.dimension
    with tempfile.TemporaryDirectory(prefix="nachos-bench-") as tmp:
        svc = build_service(Path(tmp), embedder)
        try:
            start = time.perf_counter()
            load_memories(svc, memories, direct=bool(getattr(opts, "synthetic", 0)))
            run["build_s"] = round(time.perf_counter() - start, 3)
            start = time.perf_counter()
            svc.sync_index(svc.profile)  # reconcile pass; ~0 when put() already embedded everything
            run["resync_s"] = round(time.perf_counter() - start, 3)
            if embedder is not None and svc.embedder_error:
                return {**run, "status": "skipped", "reason": svc.embedder_error}
            svc.search(queries[0]["query"], limit=SEARCH_LIMIT)  # warm caches, untimed
            rows = run_queries(svc, queries, opts.repeat)
            if embedder is not None and svc.embedder_error:
                return {**run, "status": "skipped", "reason": svc.embedder_error}
        finally:
            svc.close()
    run.update(aggregate(rows))
    run["queries"] = rows
    return run


# -- output ----------------------------------------------------------------


def environment(memories: int, queries: int, opts: argparse.Namespace) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "memories": memories,
        "queries": queries,
        "repeat": opts.repeat,
    }
    with contextlib.suppress(importlib_metadata.PackageNotFoundError):
        env["nachos_version"] = importlib_metadata.version("hermes-plugin-nachos")
    git = shutil.which("git")
    if git:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [git, "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            env["git_commit"] = proc.stdout.strip()
    return env


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([line, rule, *body])


def _metric_cells(m: dict[str, Any]) -> list[str]:
    return [
        f"{m['hit@1']:.3f}",
        f"{m['hit@5']:.3f}",
        f"{m['mrr@10']:.3f}",
        f"{m['p50_ms']:.1f}",
        f"{m['p95_ms']:.1f}",
    ]


def format_report(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"Nachos recall benchmark: {env['memories']} memories, {env['queries']} queries "
        f"(python {env['python']}, {env['machine']}, {env['cpus']} cpus)",
        "",
    ]
    head = ["backend", "hit@1", "hit@5", "MRR@10", "p50 ms", "p95 ms", "build s", "model load s"]
    rows: list[list[str]] = []
    skipped: list[str] = []
    for run in report["runs"]:
        if run["status"] != "ok":
            skipped.append(f"{run['backend']}: skipped, {run['reason']}")
            continue
        extra = [f"{run['build_s']:.2f}", f"{run.get('model_load_s', 0.0):.2f}"]
        rows.append([run["backend"], *_metric_cells(run["overall"]), *extra])
    lines += [_table(head, rows), *skipped, ""] if rows else [*skipped, ""]
    for run in report["runs"]:
        if run["status"] != "ok":
            continue
        cat_rows = [[cat, str(m["n"]), *_metric_cells(m)] for cat, m in run["by_category"].items()]
        lines += [
            f"{run['backend']} by category"
            + (f" ({run['model_id']})" if "model_id" in run else ""),
            _table(["category", "n", "hit@1", "hit@5", "MRR@10", "p50 ms", "p95 ms"], cat_rows),
            "",
        ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="bench.recall", description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--embeddings",
        default="off",
        help=f"comma-separated backends to run, from {', '.join(BACKENDS)} (default: off)",
    )
    ap.add_argument("--model", default="", help="embedding_model passthrough")
    ap.add_argument("--dtype", default="", help="embedding_dtype passthrough (nachos backend)")
    ap.add_argument("--base-url", default="", help="embedding_base_url passthrough (openai)")
    ap.add_argument("--api-key-env", default="", help="embedding_api_key_env passthrough")
    ap.add_argument("--command", default="nachos-embeddings", help="nachos-embeddings executable")
    ap.add_argument("--category", action="append", help="only run this query category (repeatable)")
    ap.add_argument("--repeat", type=int, default=1, help="searches per query for latency (median)")
    ap.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="expand corpus in memory with synthetic distractors to N entries (0 = base fixture)",
    )
    ap.add_argument("--json", metavar="PATH", help="write the full report as JSON ('-' = stdout)")
    ap.add_argument("--misses", action="store_true", help="list queries whose answer missed top 5")
    opts = ap.parse_args(argv)
    opts.backends = [b.strip() for b in opts.embeddings.split(",") if b.strip()]
    unknown = [b for b in opts.backends if b not in BACKENDS]
    if unknown or not opts.backends:
        ap.error(f"--embeddings: unknown backend {unknown or ''}; choose from {BACKENDS}")
    return opts


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    memories, queries = load_dataset()
    if opts.synthetic:
        memories = expand_synthetic_memories(memories, opts.synthetic)
    if opts.category:
        queries = [q for q in queries if q["category"] in opts.category]
    if not queries:
        print("no queries selected", file=sys.stderr)
        return 2
    runs = [run_backend(b, opts, memories, queries) for b in opts.backends]
    report = {"environment": environment(len(memories), len(queries), opts), "runs": runs}
    if opts.json == "-":
        print(json.dumps(report, indent=1))
        return 0
    print(format_report(report))
    if opts.misses:
        for run in runs:
            for row in run.get("queries", []):
                if row["rank"] is None or row["rank"] > 5:
                    print(f"[{run['backend']}] {row['id']} {row['category']}: {row['query']}")
    if opts.json:
        Path(opts.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if any(r["status"] == "ok" for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
