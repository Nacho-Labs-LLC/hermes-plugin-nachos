"""Suggestion and link evals: ``--suggest-eval`` (possible-update hints) and ``--link-eval``
(suggested_links), with their reports."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from bench.recall import (
    _run_name,
    _table,
    build_service,
    probe,
    ranking_options,
    resolve_embedder,
)
from nachos_core.memory import MemoryService
from nachos_core.store.md_store import slugify


def _case_table(by_case: dict[str, dict[str, int]]) -> str:
    rows = []
    for case, c in sorted(by_case.items()):
        upd, dist = c["updates"], c["distinct"]
        r1 = f"{c['hit@1'] / upd:.3f}" if upd else "-"
        r3 = f"{c['hit@3'] / upd:.3f}" if upd else "-"
        fp1 = f"{c['fp@1'] / dist:.3f}" if dist else "-"
        rows.append([case, str(c["n"]), r1, r3, fp1])
    return _table(["case", "n", "recall@1", "recall@3", "fp@1"], rows)


def _score_suggestions(
    svc: MemoryService,
    memories: list[dict[str, Any]],
    gt: dict[str, set[str]],
    near_dup_keys: set[str],
) -> dict[str, Any]:
    counts = dict.fromkeys(
        (
            "updates",
            "hit1",
            "hit3",
            "distinct",
            "distinct_fp1",
            "near_dup",
            "near_dup_fp1",
            "other_distinct",
            "other_distinct_fp1",
        ),
        0,
    )
    misses: list[dict[str, Any]] = []
    by_case: dict[str, dict[str, int]] = {}
    for mem in memories:
        key = slugify(mem["title"])
        res = svc.put(
            title=mem["title"],
            summary=mem["summary"],
            category=mem["category"],
            body=mem["body"],
            supersedes=None,
        )
        cands = [u["key"] for u in (res.get("possible_updates") or [])]
        expected = gt.get(key, set())
        case = (
            by_case.setdefault(
                mem["case"],
                dict.fromkeys(("n", "updates", "hit@1", "hit@3", "distinct", "fp@1"), 0),
            )
            if mem.get("case")
            else None
        )
        if case is not None:
            case["n"] += 1
        if expected:
            counts["updates"] += 1
            hit1 = int(bool(cands[:1] and cands[0] in expected))
            hit3 = int(any(c in expected for c in cands[:3]))
            counts["hit1"] += hit1
            counts["hit3"] += hit3
            if case is not None:
                case["updates"] += 1
                case["hit@1"] += hit1
                case["hit@3"] += hit3
            if not hit3:
                misses.append({"key": key, "expected": sorted(expected), "top3": cands[:3]})
        else:
            counts["distinct"] += 1
            fp1 = int(bool(cands))
            counts["distinct_fp1"] += fp1
            bucket = "near_dup" if key in near_dup_keys else "other_distinct"
            counts[bucket] += 1
            counts[f"{bucket}_fp1"] += fp1
            if case is not None:
                case["distinct"] += 1
                case["fp@1"] += fp1
    upd = counts["updates"]
    dist = counts["distinct"]
    non_nd = counts["other_distinct"]
    return {
        "updates": upd,
        "hit@1": counts["hit1"],
        "hit@3": counts["hit3"],
        "recall@1": round(counts["hit1"] / upd, 4) if upd else 1.0,
        "recall@3": round(counts["hit3"] / upd, 4) if upd else 1.0,
        "distinct_puts": dist,
        "distinct_fp1_count": counts["distinct_fp1"],
        "distinct_fp@1": round(counts["distinct_fp1"] / dist, 4) if dist else 0.0,
        "near_dup_puts": counts["near_dup"],
        "near_dup_fp@1": (
            round(counts["near_dup_fp1"] / counts["near_dup"], 4) if counts["near_dup"] else 0.0
        ),
        "other_distinct_fp@1": (round(counts["other_distinct_fp1"] / non_nd, 4) if non_nd else 0.0),
        "misses": misses,
        **({"by_case": by_case} if by_case else {}),
    }


def eval_suggest_backend(
    backend: str,
    opts: argparse.Namespace,
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay ``memories`` through put() without supersedes and evaluate possible_updates."""
    run: dict[str, Any] = {"backend": _run_name(backend, opts), "status": "ok"}
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
    gt = {slugify(m["title"]): set(m["supersedes"]) for m in memories if m.get("supersedes")}
    near_dup_keys = {k for q in queries if q["category"] == "near-dup" for k in q["expect"]}
    with tempfile.TemporaryDirectory(prefix="nachos-suggest-") as tmp:
        svc = build_service(Path(tmp), embedder, **ranking_options(opts))
        try:
            scored = _score_suggestions(svc, memories, gt, near_dup_keys)
        finally:
            svc.close()
    run.update(scored)
    return run


def _link_roots(memories: list[dict[str, Any]]) -> dict[str, str]:
    """key -> root of its supersedes chain, so a newer version inherits its predecessor's labels."""
    root: dict[str, str] = {}
    for mem in memories:
        key = slugify(mem["title"])
        old = (mem.get("supersedes") or [None])[0]
        root[key] = root.get(old, old) if old else key
    return root


def _score_links(
    svc: MemoryService,
    memories: list[dict[str, Any]],
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = _link_roots(memories)
    hard = {
        frozenset((root.get(n["a"], n["a"]), root.get(n["b"], n["b"]))) for n in negatives or []
    }
    hard_fp = 0
    wanted = {frozenset((root.get(p["a"], p["a"]), root.get(p["b"], p["b"]))) for p in positives}
    found: set[frozenset[str]] = set()
    tp = fp = neutral = 0
    false_positives: list[dict[str, Any]] = []
    for mem in memories:
        res = svc.put(
            title=mem["title"],
            summary=mem["summary"],
            category=mem["category"],
            body=mem["body"],
            supersedes=mem.get("supersedes"),
        )
        key = slugify(mem["title"])
        for sug in res.get("suggested_links") or []:
            pair = frozenset((root[key], root.get(sug["key"], sug["key"])))
            if len(pair) == 1:
                neutral += 1  # two versions of one fact: an update, not a link
            elif pair in wanted:
                tp += 1
                found.add(pair)
            else:
                fp += 1
                hard_fp += pair in hard
                false_positives.append(
                    {"key": key, "suggested": sug["key"], "reason": sug.get("reason", "")}
                )
    return {
        "labelled_pairs": len(wanted),
        "suggestions": tp + fp,
        "tp": tp,
        "fp": fp,
        "neutral": neutral,
        "hard_fp": hard_fp,
        "hard_negatives": len(hard),
        "precision": round(tp / (tp + fp), 4) if tp + fp else 1.0,
        "recall": round(len(found) / len(wanted), 4) if wanted else 1.0,
        "false_positives": false_positives,
    }


def eval_link_backend(
    backend: str, opts: argparse.Namespace, memories: list[dict[str, Any]]
) -> dict[str, Any]:
    """Replay ``memories`` through put() and score ``suggested_links`` against ``links.json``.

    Labelled updates are applied as ``supersedes`` (as an agent following ``possible_updates``
    would), so an old version is hidden and never competes with its replacement.
    """
    run: dict[str, Any] = {"backend": opts.label or backend, "status": "ok"}
    path = opts.data_dir / "links.json"
    if not path.exists():
        return {**run, "status": "skipped", "reason": f"no links.json in {opts.data_dir}"}
    embedder, reason = resolve_embedder(backend, opts)
    if reason:
        return {**run, "status": "skipped", "reason": reason}
    labels = json.loads(path.read_text(encoding="utf-8"))
    positives, negatives = labels["positives"], labels.get("negatives", [])
    with tempfile.TemporaryDirectory(prefix="nachos-links-") as tmp:
        svc = build_service(Path(tmp), embedder, **ranking_options(opts))
        try:
            run.update(_score_links(svc, memories, positives, negatives))
        finally:
            svc.close()
    return run


def format_link_report(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [f"Nachos suggested_links eval: {env['memories']} memories", ""]
    rows = [
        [
            r["backend"],
            str(r["labelled_pairs"]),
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['tp']}/{r['fp']}",
            f"{r['hard_fp']}/{r['hard_negatives']}",
        ]
        for r in report["runs"]
        if r["status"] == "ok"
    ]
    lines += (
        [_table(["backend", "pairs", "precision", "recall", "TP/FP", "hard-neg FP"], rows)]
        if rows
        else []
    )
    lines += [
        f"{r['backend']}: skipped, {r['reason']}" for r in report["runs"] if r["status"] != "ok"
    ]
    for r in report["runs"]:
        for fp in r.get("false_positives", [])[:40]:
            lines.append(f"  FP {fp['key']} -> {fp['suggested']}  ({fp['reason']})")
    return "\n".join(lines) + "\n"


def format_suggest_report(report: dict[str, Any]) -> str:
    env = report["environment"]
    lines = [
        f"Nachos possible_updates eval: {env['memories']} memories "
        f"(python {env['python']}, {env['machine']}, {env['cpus']} cpus)",
        "",
    ]
    head = [
        "backend",
        "updates",
        "recall@1",
        "recall@3",
        "distinct",
        "distinct fp@1",
        "non-dup fp@1",
    ]
    rows: list[list[str]] = []
    skipped: list[str] = []
    for run in report["runs"]:
        if run["status"] != "ok":
            skipped.append(f"{run['backend']}: skipped, {run['reason']}")
            continue
        rows.append(
            [
                run["backend"],
                f"{run['hit@1']}/{run['hit@3']}/{run['updates']}",
                f"{run['recall@1']:.3f}",
                f"{run['recall@3']:.3f}",
                str(run["distinct_puts"]),
                f"{run['distinct_fp@1']:.3f}",
                f"{run['other_distinct_fp@1']:.3f}",
            ]
        )
    lines += [_table(head, rows), *skipped, ""] if rows else [*skipped, ""]
    for run in report["runs"]:
        if run["status"] == "ok" and run.get("by_case"):
            lines += [f"{run['backend']} by case", _case_table(run["by_case"]), ""]
    return "\n".join(lines)
