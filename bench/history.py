"""History-search benchmark: hit rates and latency for ``nachos_core.history.search_history``.

Builds throwaway checkpoint directories from ``bench/data/history.json`` (synthetic sessions
with answerable and unanswerable queries), runs every query through the current ranking and
through the previous "every query term must appear" rule, and reports per-category hit@1,
hit@5, false-positive rate on unanswerable queries and p50/p95 latency.

    python -m bench.history                 # fixture only
    python -m bench.history --noise 200     # plus 200 filler checkpoints (latency at scale)
    python -m bench.history --json out.json

Stdlib only. Measurement only: nothing here changes search behaviour.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nachos_core.history import (  # noqa: E402
    _snapshots_newest_first,
    _terms,
    message_text,
    search_history,
)
from nachos_core.snapshots import SnapshotStore  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "history.json"
Search = Callable[[Path, str], list[dict[str, Any]]]


def build(root: Path, data: dict[str, Any], noise: int = 0) -> None:
    """Write one checkpoint per fixture session, plus ``noise`` filler checkpoints."""
    now = time.time()
    for session in data["sessions"]:
        store = SnapshotStore(root, session["id"])
        meta = store.save(messages=session["messages"], reason="pre-compress-checkpoint")
        _age(store, meta.id, now - session["age_days"] * 86400)
    rng = random.Random(7)  # noqa: S311  (seeded fixture filler, not security)
    vocab = [f"w{n}{rng.choice('abcdefgh')}" for n in range(400)]
    common = ["the", "service", "config", "deploy", "error", "update", "staging", "build", "run"]
    for n in range(noise):
        store = SnapshotStore(root, f"noise-{n}")
        msgs = [
            {
                "role": rng.choice(["user", "assistant"]),
                "content": " ".join(rng.choices(vocab + common * 6, k=rng.randint(15, 60))),
            }
            for _ in range(rng.randint(6, 14))
        ]
        meta = store.save(messages=msgs, reason="pre-compress-checkpoint")
        _age(store, meta.id, now - rng.uniform(0, 30) * 86400)


def _age(store: SnapshotStore, checkpoint_id: str, created_at: float) -> None:
    rows = [json.loads(line) for line in store.index_file.read_text().splitlines()]
    for row in rows:
        if row["id"] == checkpoint_id:
            row["created_at"] = created_at
    store.index_file.write_text("".join(json.dumps(r) + "\n" for r in rows))


def legacy_search(root: Path, query: str, limit: int = 10, max_snapshots: int = 200) -> list:
    """The previous rule: a message must contain every query term; ranked by term count."""
    terms = _terms(query)
    seen: set[str] = set()
    hits: list[tuple[int, float, dict[str, Any]]] = []
    if not terms:
        return []
    for session_dir, row in _snapshots_newest_first(root, None)[:max_snapshots]:
        with gzip.open(session_dir / row["blob"], "rb") as f:
            messages = json.loads(f.read().decode("utf-8")).get("messages", [])
        for message in messages:
            text = message_text(message)
            lower = text.lower()
            if not text or not all(t in lower for t in terms):
                continue
            digest = hashlib.sha1(
                f"{message.get('role')}\0{text}".encode(), usedforsecurity=False
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            context = [{"index": 0, "match": True, "text": text}]
            hits.append(
                (
                    sum(lower.count(t) for t in terms),
                    float(row["created_at"]),
                    {"session_id": row["session_id"], "context": context, "excerpt": text},
                )
            )
    hits.sort(key=lambda h: (-h[0], -h[1]))
    return [h[2] for h in hits[:limit]]


def _hit(result: dict[str, Any], query: dict[str, Any], *, context: bool) -> bool:
    if result["session_id"] != query["session"]:
        return False
    parts = result["context"] if context else [p for p in result["context"] if p["match"]]
    return any(query["contains"].lower() in p["text"].lower() for p in parts)


def evaluate(root: Path, queries: list[dict[str, Any]], search: Search, repeat: int) -> dict:
    cats: dict[str, dict[str, Any]] = {}
    times: list[float] = []
    for q in queries:
        runs = []
        for _ in range(repeat):
            start = time.perf_counter()
            results = search(root, q["q"])
            runs.append((time.perf_counter() - start) * 1000)
        times.append(statistics.median(runs))
        c = cats.setdefault(
            q["category"], {"n": 0, "match@1": 0, "match@5": 0, "ctx@5": 0, "fp": 0}
        )
        c["n"] += 1
        if q["session"] is None:
            c["fp"] += bool(results)
            continue
        c["match@1"] += any(_hit(r, q, context=False) for r in results[:1])
        c["match@5"] += any(_hit(r, q, context=False) for r in results[:5])
        c["ctx@5"] += any(_hit(r, q, context=True) for r in results[:5])
    answerable = [q for q in queries if q["session"] is not None]
    n = len(answerable)
    total = {k: sum(c[k] for c in cats.values()) / n for k in ("match@1", "match@5", "ctx@5")}
    neg = sum(c["fp"] for c in cats.values())
    n_neg = len(queries) - n
    times.sort()
    return {
        "answerable": n,
        "unanswerable": n_neg,
        **{k: round(v, 3) for k, v in total.items()},
        "false_positive_rate": round(neg / n_neg, 3) if n_neg else 0.0,
        "p50_ms": round(statistics.median(times), 2),
        "p95_ms": round(times[min(len(times) - 1, int(len(times) * 0.95))], 2),
        "by_category": cats,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noise", type=int, default=0, help="extra filler checkpoints")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args(argv)
    data = json.loads(DATA.read_text(encoding="utf-8"))
    report: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build(root, data, args.noise)
        runs: dict[str, Search] = {
            "all-terms (before)": legacy_search,
            "idf ranking, no context": lambda r, q: search_history(r, q, before=0, after=0),
            "idf ranking + context": search_history,
        }
        for name, fn in runs.items():
            report[name] = evaluate(root, data["queries"], fn, args.repeat)
    print(f"{'variant':26} {'m@1':>6} {'m@5':>6} {'ctx@5':>6} {'FP':>6} {'p50ms':>7} {'p95ms':>7}")
    for name, r in report.items():
        print(
            f"{name:26} {r['match@1']:6.3f} {r['match@5']:6.3f} {r['ctx@5']:6.3f} "
            f"{r['false_positive_rate']:6.3f} {r['p50_ms']:7.2f} {r['p95_ms']:7.2f}"
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
