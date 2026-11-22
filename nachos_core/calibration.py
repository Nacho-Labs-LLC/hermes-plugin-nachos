"""Per-model calibration priors, cold-start blending, and per-query search confidence.

Priors were measured offline with ``nachos-embeddings`` (``q8``, ``--type document``) over an
80-topic generic English corpus (3,160 distinct document pairs, zero overlap with the recall
benchmarks). They bridge cold start (< 30 entries) and blend linearly into the store's own
empirical random-pair distribution from 30 to 100 entries.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

#: Minimum entries before empirical store calibration is computed (must match index.py).
CALIBRATION_MIN_HEADS = 30
#: Head count at which the store's own empirical distribution gets 100% weight.
CALIBRATION_FULL_HEADS = 100

#: Per-backend confidence score cutoffs (0..1).
LEXICAL_ABSTAIN_THRESHOLD = 0.38
SEMANTIC_ABSTAIN_THRESHOLD = 0.48
ABSTAIN_THRESHOLD = LEXICAL_ABSTAIN_THRESHOLD
HIGH_CONFIDENCE_THRESHOLD = 0.65
#: Half-width around the abstain threshold that triggers optional cross-encoder verification.
BORDERLINE_BAND = 0.12

NO_MATCH_NOTE = "No matching memory found."
LOW_CONFIDENCE_NOTE = (
    "No confident match in memory. Results below are low-confidence candidates only "
    "— verify before relying on them or treat the topic as not stored."
)
HINT_LOW_CONFIDENCE_NOTE = (
    "Low-confidence match; verify the returned candidates before relying on them."
)

# Order matches nachos-embeddings/src/providers/model-profiles.ts plus MiniLM (default).
_PRIOR_PROFILES: tuple[tuple[re.Pattern[str], str, dict[str, float]], ...] = (
    (
        re.compile(r"(?:all-minilm-l6-v2|^nachos:default(?::|$))", re.IGNORECASE),
        "all-minilm-l6-v2",
        {
            "p50": 0.0389,
            "p75": 0.0944,
            "p80": 0.1106,
            "p85": 0.1292,
            "p90": 0.1608,
            "p95": 0.2120,
            "p99": 0.3195,
            "p99.5": 0.3730,
            "mean": 0.0481,
            "std": 0.0898,
        },
    ),
    (
        re.compile(r"(?:^|[/_:-])e5-", re.IGNORECASE),
        "e5",
        {
            "p50": 0.7745,
            "p75": 0.7972,
            "p80": 0.8026,
            "p85": 0.8095,
            "p90": 0.8192,
            "p95": 0.8327,
            "p99": 0.8536,
            "p99.5": 0.8620,
            "mean": 0.7765,
            "std": 0.0315,
        },
    ),
    (
        re.compile(r"bge-.*-en-v1\.5", re.IGNORECASE),
        "bge-en-v1.5",
        {
            "p50": 0.4699,
            "p75": 0.5156,
            "p80": 0.5270,
            "p85": 0.5429,
            "p90": 0.5624,
            "p95": 0.5949,
            "p99": 0.6505,
            "p99.5": 0.6730,
            "mean": 0.4760,
            "std": 0.0655,
        },
    ),
    (
        re.compile(r"snowflake-arctic-embed-[a-z]+-v2\.0", re.IGNORECASE),
        "snowflake-arctic-embed-v2.0",
        {
            "p50": 0.1309,
            "p75": 0.1789,
            "p80": 0.1922,
            "p85": 0.2065,
            "p90": 0.2259,
            "p95": 0.2542,
            "p99": 0.3317,
            "p99.5": 0.3713,
            "mean": 0.1355,
            "std": 0.0716,
        },
    ),
    (
        re.compile(r"snowflake-arctic-embed-", re.IGNORECASE),
        "snowflake-arctic-embed-v1",
        {
            "p50": 0.6713,
            "p75": 0.7021,
            "p80": 0.7108,
            "p85": 0.7203,
            "p90": 0.7315,
            "p95": 0.7476,
            "p99": 0.7790,
            "p99.5": 0.7897,
            "mean": 0.6724,
            "std": 0.0451,
        },
    ),
    (
        re.compile(r"gte-modernbert", re.IGNORECASE),
        "gte-modernbert",
        {
            "p50": 0.4985,
            "p75": 0.5375,
            "p80": 0.5470,
            "p85": 0.5614,
            "p90": 0.5774,
            "p95": 0.6071,
            "p99": 0.6667,
            "p99.5": 0.6827,
            "mean": 0.5052,
            "std": 0.0561,
        },
    ),
    (
        re.compile(r"nomic-embed-text-v1(?:\.5)?(?![\d.])", re.IGNORECASE),
        "nomic-embed-text-v1.5",
        {
            "p50": 0.5666,
            "p75": 0.5973,
            "p80": 0.6051,
            "p85": 0.6155,
            "p90": 0.6289,
            "p95": 0.6484,
            "p99": 0.6838,
            "p99.5": 0.7058,
            "mean": 0.5694,
            "std": 0.0454,
        },
    ),
)


def prior_profile(model_id: str) -> dict[str, float] | None:
    """Return a copy of the offline calibration prior matching ``model_id``, or None."""
    if not model_id:
        return None
    for pattern, _family, stats in _PRIOR_PROFILES:
        if pattern.search(model_id):
            return dict(stats)
    return None


def blend_profiles(
    prior: Mapping[str, float] | None,
    empirical: Mapping[str, float] | None,
    head_count: int,
    *,
    full_heads: int = CALIBRATION_FULL_HEADS,
) -> dict[str, float] | None:
    """Blend ``prior`` into ``empirical`` linearly as ``head_count`` grows to ``full_heads``."""
    if prior is None and empirical is None:
        return None
    if empirical is None:
        return dict(prior or {})
    if prior is None:
        return dict(empirical)
    alpha = min(1.0, max(0.0, head_count / float(full_heads)))
    keys = set(prior) | set(empirical)
    out: dict[str, float] = {}
    for k in keys:
        if k in prior and k in empirical:
            out[k] = (1.0 - alpha) * prior[k] + alpha * empirical[k]
        elif k in empirical:
            out[k] = empirical[k]
        else:
            out[k] = prior[k]
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _profile_center_scale(profile: Mapping[str, float] | None) -> tuple[float, float]:
    if profile is None:
        return 0.15, 0.10
    center = float(profile.get("mean", profile["p50"]))
    if "std" in profile:
        scale = float(profile["std"])
    else:
        scale = (float(profile["p95"]) - float(profile["p50"])) / 1.645
    return center, max(0.02, scale)


def assess_confidence(
    *,
    has_results: bool,
    lexical_coverage: float,
    lexical_margin: float = 0.0,
    semantic_top1: float | None = None,
    semantic_topk: float | None = None,
    profile: Mapping[str, float] | None = None,
    agreement: bool = False,
    rerank_score: float | None = None,
    abstain_threshold: float | None = None,
) -> dict[str, Any]:
    """Combine calibrated semantic, lexical, agreement, margin and optional rerank signals.

    In semantic mode, semantic z-score and lexical-hybrid paths combine via ``max`` so
    word-free paraphrases with a strong calibrated z-score are not penalised for zero
    lexical overlap, and lexical-strong hits (e.g. long-body) are not dragged down by
    moderate semantic z-scores. Lexical-only and semantic modes use separate calibrated
    thresholds unless ``abstain_threshold`` is explicitly overridden.
    """
    if not has_results:
        return {
            "confident": False,
            "confidence": "none",
            "confidence_score": 0.0,
            "signals": {"lexical_coverage": 0.0},
        }
    s_lex = _clamp01(lexical_coverage)
    if semantic_top1 is not None:
        threshold = (
            SEMANTIC_ABSTAIN_THRESHOLD if abstain_threshold is None else float(abstain_threshold)
        )
        center, scale = _profile_center_scale(profile)
        sem_z = (semantic_top1 - center) / scale
        s_sem = _clamp01((sem_z - 1.1) / 2.7)
        if semantic_topk is not None:
            margin_z = (semantic_top1 - semantic_topk) / scale
            s_mar = _clamp01((margin_z - 0.3) / 2.0)
        else:
            margin_z = 0.0
            s_mar = 0.5
        s_agr = 0.12 if agreement else 0.0
        lex_path = _clamp01(0.80 * s_lex + 0.12 * s_mar + s_agr) if s_lex > 0.0 else 0.0
        score = max(s_sem, lex_path)
        signals: dict[str, Any] = {
            "semantic_z": round(sem_z, 2),
            "margin_z": round(margin_z, 2),
            "lexical_coverage": round(s_lex, 3),
            "agreement": agreement,
        }
        if rerank_score is not None:
            score = 0.55 * score + 0.45 * _clamp01(rerank_score)
            signals["rerank"] = round(rerank_score, 3)
    else:
        threshold = (
            LEXICAL_ABSTAIN_THRESHOLD if abstain_threshold is None else float(abstain_threshold)
        )
        s_mar = _clamp01(lexical_margin)
        score = (0.80 * s_lex + 0.20 * s_mar) if s_lex > 0.0 else 0.0
        signals = {
            "lexical_coverage": round(s_lex, 3),
            "lexical_margin": round(s_mar, 3),
        }
    confident = score >= threshold
    tier = "high" if score >= HIGH_CONFIDENCE_THRESHOLD else ("medium" if confident else "low")
    return {
        "confident": confident,
        "confidence": tier,
        "confidence_score": round(score, 3),
        "signals": signals,
    }


def is_borderline(
    assessment: Mapping[str, Any],
    *,
    semantic: bool,
    band: float = BORDERLINE_BAND,
) -> bool:
    """True when a non-empty search's confidence sits within ``band`` of its abstain threshold."""
    if assessment.get("confidence") == "none":
        return False
    threshold = SEMANTIC_ABSTAIN_THRESHOLD if semantic else LEXICAL_ABSTAIN_THRESHOLD
    return abs(float(assessment.get("confidence_score", 0.0)) - threshold) <= band


def format_search_response(
    query: str,
    results: list[dict[str, Any]],
    assessment: Mapping[str, Any],
    *,
    mode: str = "on",
) -> dict[str, Any]:
    """Assemble the structured recall payload with confidence and explicit abstention fields."""
    payload: dict[str, Any] = {
        "query": query,
        "confident": bool(assessment["confident"]),
        "confidence": str(assessment["confidence"]),
        "confidence_score": float(assessment["confidence_score"]),
        "signals": dict(assessment["signals"]),
        "count": len(results),
        "results": results,
    }
    if not payload["confident"]:
        payload["no_confident_match"] = True
        low_note = HINT_LOW_CONFIDENCE_NOTE if mode == "hint" else LOW_CONFIDENCE_NOTE
        payload["note"] = low_note if results else NO_MATCH_NOTE
    return payload
