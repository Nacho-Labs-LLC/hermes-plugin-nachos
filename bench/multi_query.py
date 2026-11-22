"""``--multi-query``: grouped sub-queries fused by the service versus issued one by one."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any

from bench.recall import (
    SEARCH_LIMIT,
    _metric_cells,
    _table,
    build_service,
    load_memories,
    percentile,
    probe,
    rank_of,
    ranking_options,
    resolve_embedder,
    summarize,
)
from nachos_core.memory import MemoryService

_MULTI_QUERY_PRIORITY = {
    "paraphrase": 0,
    "outdated": 1,
    "long-body": 2,
    "natural": 3,
    "keyword": 4,
    "current": 5,
}


def build_multi_query_groups(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group answerable fixture queries sharing an expected key (hardest phrasing first)."""
    grouped: dict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = {}
    for q in queries:
        expect = q.get("expect") or []
        if q["category"] == "negative" or not expect:
            continue
        key = (tuple(sorted(expect)), str(q.get("match", "any")))
        grouped.setdefault(key, []).append(q)
    out: list[dict[str, Any]] = []
    for (expect_tuple, match), items in grouped.items():
        ordered = sorted(
            items,
            key=lambda q: (_MULTI_QUERY_PRIORITY.get(q["category"], 6), q["id"]),
        )
        variants: list[str] = []
        cats: list[str] = []
        seen: set[str] = set()
        for item in ordered:
            text = item["query"].strip()
            low = text.lower()
            if text and low not in seen:
                seen.add(low)
                variants.append(text)
                cats.append(item["category"])
            if len(variants) == 3:
                break
        if len(variants) >= 2:
            out.append(
                {
                    "expect": list(expect_tuple),
                    "match": match,
                    "primary_category": cats[0],
                    "categories": cats,
                    "queries": variants,
                }
            )
    return out


def _timed_multi_search(
    svc: MemoryService,
    query_list: list[str],
    expect: list[str],
    match: str,
    repeat: int,
) -> tuple[int | None, bool, float]:
    timings: list[float] = []
    rank: int | None = None
    confident = True
    for i in range(max(1, repeat)):
        start = time.perf_counter()
        payload = svc.search_with_confidence(queries=query_list, limit=SEARCH_LIMIT)
        timings.append((time.perf_counter() - start) * 1000.0)
        if i == 0:
            keys = [r["key"] for r in payload["results"]]
            rank = rank_of(expect, keys, match)
            confident = bool(payload["confident"])
    return rank, confident, percentile(timings, 50)


def run_multi_query_groups(
    svc: MemoryService, groups: list[dict[str, Any]], repeat: int = 1
) -> dict[str, Any]:
    """Compare 1-query, 2-query, 3-query, and retry-on-low-confidence recall."""
    all_single: list[tuple[int | None, float]] = []
    all_multi: list[tuple[int | None, float]] = []
    all_retry: list[tuple[int | None, float]] = []
    all_retries = 0
    tri_single: list[tuple[int | None, float]] = []
    tri_two: list[tuple[int | None, float]] = []
    tri_three: list[tuple[int | None, float]] = []
    tri_retry: list[tuple[int | None, float]] = []
    tri_retries = 0

    for g in groups:
        qs, exp, match = g["queries"], g["expect"], g["match"]
        r1, conf1, ms1 = _timed_multi_search(svc, [qs[0]], exp, match, repeat)
        rm, _confm, msm = _timed_multi_search(svc, qs, exp, match, repeat)
        retried = not conf1
        rr = rm if retried else r1
        msr = (ms1 + msm) if retried else ms1
        all_single.append((r1, ms1))
        all_multi.append((rm, msm))
        all_retry.append((rr, msr))
        all_retries += int(retried)
        if len(qs) >= 3:
            r2, _conf2, ms2 = _timed_multi_search(svc, qs[:2], exp, match, repeat)
            tri_single.append((r1, ms1))
            tri_two.append((r2, ms2))
            tri_three.append((rm, msm))
            tri_retry.append((rr, msr))
            tri_retries += int(retried)

    def _pack(pairs: list[tuple[int | None, float]]) -> dict[str, Any]:
        return summarize([r for r, _ in pairs], [ms for _, ms in pairs])

    out: dict[str, Any] = {
        "all_groups": {
            "n": len(groups),
            "mean_queries": round(sum(len(g["queries"]) for g in groups) / len(groups), 2)
            if groups
            else 0.0,
            "1-query": _pack(all_single),
            "multi-query": _pack(all_multi),
            "retry-on-low-conf": {
                **_pack(all_retry),
                "retried": all_retries,
                "retry_rate": round(all_retries / len(groups), 4) if groups else 0.0,
            },
        }
    }
    if tri_single:
        out["triples"] = {
            "n": len(tri_single),
            "1-query": _pack(tri_single),
            "2-query": _pack(tri_two),
            "3-query": _pack(tri_three),
            "retry-on-low-conf": {
                **_pack(tri_retry),
                "retried": tri_retries,
                "retry_rate": round(tri_retries / len(tri_single), 4),
            },
        }
    return out


def eval_multi_query_backend(
    backend: str,
    opts: argparse.Namespace,
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one backend and benchmark 1 vs multi-query recall on grouped phrasings."""
    run: dict[str, Any] = {"backend": opts.label or backend, "status": "ok"}
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
    groups = build_multi_query_groups(queries)
    with tempfile.TemporaryDirectory(prefix="nachos-mq-") as tmp:
        svc = build_service(Path(tmp), embedder, **ranking_options(opts))
        try:
            load_memories(
                svc,
                memories,
                direct=bool(getattr(opts, "synthetic", 0)),
                apply_supersedes=not bool(getattr(opts, "no_supersedes", False)),
            )
            svc.sync_index(svc.profile)
            if groups:
                svc.search(groups[0]["queries"][0], limit=SEARCH_LIMIT)
            run.update(run_multi_query_groups(svc, groups, opts.repeat))
        finally:
            svc.close()
    return run


def format_multi_query_report(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"Nachos multi-query recall benchmark: {env['memories']} memories "
        f"(python {env['python']}, {env['machine']}, {env['cpus']} cpus)",
        "",
    ]
    head = ["backend", "mode", "n", "hit@1", "hit@5", "MRR@10", "p50 ms", "p95 ms", "notes"]
    for section, title, modes in (
        (
            "triples",
            "Exact 3-query groups (paraphrase + natural + keyword)",
            ("1-query", "2-query", "3-query", "retry-on-low-conf"),
        ),
        (
            "all_groups",
            "All multi-variant groups (2-3 phrasings per target)",
            ("1-query", "multi-query", "retry-on-low-conf"),
        ),
    ):
        rows: list[list[str]] = []
        for run in report["runs"]:
            if run["status"] != "ok" or section not in run:
                continue
            sec = run[section]
            for mode in modes:
                m = sec[mode]
                note = (
                    f"retried {m['retried']}/{m['n']} ({m['retry_rate']:.0%})"
                    if "retried" in m
                    else (f"avg {sec['mean_queries']} q" if mode == "multi-query" else "")
                )
                rows.append([run["backend"], mode, str(m["n"]), *_metric_cells(m), note])
        if rows:
            lines += [title, _table(head, rows), ""]
    for run in report["runs"]:
        if run["status"] != "ok":
            lines.append(f"{run['backend']}: skipped, {run['reason']}")
    return "\n".join(lines)
