"""Summary quality gate — shared by the write path and the audit tool.

The manifest's title+summary line is the ONLY signal the agent has to
decide whether an entry is worth recalling. Low-signal summaries
("nothing", a bare restatement of the title, a stray placeholder) make
the manifest useless without the agent ever finding out — it just never
recalls that entry. This module defines what "low-signal" means in one
place so both the write-time gate (nachos_memory_put) and the periodic
audit (tools/correct_summaries.py) agree.

No LLM, no network, stdlib only — this must stay cheap enough to run
in the write-time hot path.
"""

from __future__ import annotations

import re

from .lexical import STOPWORDS, tokens

_BANNED = {
    "nothing",
    "n/a",
    "na",
    "none",
    "todo",
    "tbd",
    "placeholder",
    "test",
    "testing",
    "foo",
    "bar",
    "lorem ipsum",
    "-",
    "...",
    "tbd.",
}

_MIN_CHARS = 8
_MIN_WORDS = 2
_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list:
    return _WORD_RE.findall((text or "").lower())


def low_signal_reason(text: str) -> str | None:
    """Return a human-readable reason text is low-signal, or None if fine."""
    t = (text or "").strip()
    if not t:
        return "empty"
    if t.lower().strip(".! ") in _BANNED:
        return f"banned placeholder text ({t!r})"
    if len(t) < _MIN_CHARS:
        return f"too short ({len(t)} chars, minimum {_MIN_CHARS})"
    words = _words(t)
    if len(words) < _MIN_WORDS:
        return f"too few words ({len(words)}, minimum {_MIN_WORDS})"
    if len(set(words)) == 1:
        return "repeats a single word"
    return None


def is_redundant_summary(summary: str, title: str) -> bool:
    """True if summary adds nothing beyond restating the title."""
    s = (summary or "").strip().lower().rstrip(".")
    t = (title or "").strip().lower().rstrip(".")
    return bool(s) and s == t


def flag_reason(title: str, summary: str, body: str) -> str | None:
    """Shared audit heuristic; a flag is a review hint, not evidence of falsehood."""
    reason = low_signal_reason(summary)
    if reason:
        return f"low-signal ({reason})"
    if is_redundant_summary(summary, title):
        return "redundant (just restates the title)"
    if not shares_vocabulary(summary, body):
        return "no shared vocabulary with body (likely stale)"
    return None


def shares_vocabulary(summary: str, body: str, *, min_overlap: int = 1) -> bool:
    """True if summary and body share at least `min_overlap` content words.

    Used by the audit pass to flag summaries that look unrelated to the
    entry they describe (e.g. stale after an edit to body but not summary).
    """
    summary_words = set(_words(summary)) - STOPWORDS
    body_words = set(_words(body)) - STOPWORDS
    if not summary_words:
        return False
    return len(summary_words & body_words) >= min_overlap


_STRONG_UPDATE_CUES_RE = re.compile(
    r"\b(?:no longer|moved to|moved from|replaced|migrated to|migrated from|"
    r"switched to|retired|deprecated|cancelled|canceled|end of life|supersedes|"
    r"instead of|formerly|updated to|changed to|decommissioned|sunset|"
    r"dropped|removed)\b",
    re.IGNORECASE,
)
_WEAK_UPDATE_CUE_RE = re.compile(r"\bnow\b", re.IGNORECASE)
_FACT_NUMBER_RE = re.compile(r"\b(?:v?\d+(?:[.:-]\d+)+|\d{4}-\d{2}-\d{2}|\d+)\b")
_FACT_PATH_RE = re.compile(r"(?:https?://[^\s,;)]+|(?:/[\w.-]+){2,}|[a-z0-9_.-]+/[a-z0-9_.-]+)")
_FACT_HOST_RE = re.compile(r"\b[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)+\b")
_FACT_WORD_RE = re.compile(
    r"\b(?:single|double|triple|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"hourly|daily|weekly|biweekly|fortnightly|monthly|quarterly|yearly|annually)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19\d\d|20\d\d)\b")
_OLD_TEMPORAL_WORDS = frozenset(
    {
        "old",
        "legacy",
        "previous",
        "prior",
        "former",
        "early",
        "initial",
        "deprecated",
        "retired",
        "archived",
        "original",
    }
)
_NEW_TEMPORAL_WORDS = frozenset({"new", "current", "latest", "update", "updated", "revised"})


def _has_temporal_progression(old_title: str, new_title: str) -> bool:
    old_words = tokens(old_title)
    new_words = tokens(new_title)
    old_years = [int(y) for y in _YEAR_RE.findall(old_title)]
    new_years = [int(y) for y in _YEAR_RE.findall(new_title)]
    if old_years and new_years and max(new_years) > max(old_years):
        return True
    old_dated = bool(old_years or (old_words & _OLD_TEMPORAL_WORDS))
    new_current = bool(new_words & _NEW_TEMPORAL_WORDS)
    return old_dated and new_current


def _host_tokens(text: str) -> set[str]:
    return {
        m
        for m in _FACT_HOST_RE.findall(text)
        if any(ch.isalpha() for ch in m.rsplit(".", 1)[-1]) and not m.startswith("v")
    }


def _fact_groups(text: str) -> tuple[set[str], set[str], set[str], set[str]]:
    lowered = (text or "").lower()
    paths = set(_FACT_PATH_RE.findall(lowered))
    hosts = {h for h in _host_tokens(lowered) if not any(h in p for p in paths)}
    numbers = set(_FACT_NUMBER_RE.findall(lowered))
    words = set(_FACT_WORD_RE.findall(lowered))
    return paths, hosts, numbers, words


def _number_shape(tok: str) -> str:
    if ":" in tok:
        return "time"
    if "-" in tok:
        return "date"
    if "." in tok:
        return "version"
    return "year" if _YEAR_RE.fullmatch(tok) else "int"


def _pick_changed_pair(removed: list[str], added: list[str], *, by_shape: bool) -> tuple[str, str]:
    if by_shape:
        for old_tok in removed:
            shape = _number_shape(old_tok)
            for new_tok in added:
                if _number_shape(new_tok) == shape:
                    return old_tok, new_tok
    return removed[0], added[0]


def _changed_fact_pair(old_text: str, new_text: str) -> tuple[str, str] | None:
    old_groups = _fact_groups(old_text)
    new_groups = _fact_groups(new_text)
    for idx, (old_set, new_set) in enumerate(zip(old_groups, new_groups)):
        removed = sorted(old_set - new_set)
        added = sorted(new_set - old_set)
        if removed and added:
            return _pick_changed_pair(removed, added, by_shape=(idx == 2))
    return None


def _format_update_reason(
    fact_pair: tuple[str, str] | None,
    strong_cue: re.Match[str] | None,
    temporal: bool,
    weak_cue: re.Match[str] | None,
    shared_terms: tuple[str, ...],
) -> str:
    if fact_pair:
        return f"changed: {fact_pair[0]} -> {fact_pair[1]}"
    if strong_cue:
        return f"new text says {strong_cue.group(0).lower()}"
    if temporal:
        return "newer version of topic"
    if weak_cue:
        return "new text says now"
    if shared_terms:
        return f"shared terms: {', '.join(shared_terms[:3])}"
    return "similar topic"


def update_signal(
    old_text: str,
    new_text: str,
    *,
    old_title: str = "",
    new_title: str = "",
    shared_terms: tuple[str, ...] = (),
) -> tuple[float, str]:
    """Return ``(rank_boost, short_reason)`` describing why ``new_text`` may update ``old_text``.

    Never gates candidates: when no update cue or factual diff fires, returns a zero boost
    and a neutral similarity reason so the calling agent can still inspect the candidate.
    """
    fact_pair = _changed_fact_pair(old_text, new_text)
    strong_cue = _STRONG_UPDATE_CUES_RE.search(new_text or "")
    weak_cue = None if strong_cue else _WEAK_UPDATE_CUE_RE.search(new_text or "")
    temporal = bool(
        old_title
        and new_title
        and _has_temporal_progression(old_title, new_title)
        and (old_text or "").strip() != (new_text or "").strip()
    )
    boost = min(
        (0.05 if fact_pair else 0.0)
        + (0.06 if strong_cue else (0.02 if weak_cue else 0.0))
        + (0.03 if temporal else 0.0),
        0.10,
    )
    reason = _format_update_reason(fact_pair, strong_cue, temporal, weak_cue, shared_terms)
    return boost, reason


def looks_like_update(
    old_text: str,
    new_text: str,
    *,
    old_title: str = "",
    new_title: str = "",
) -> bool:
    """True when ``new_text`` shows an update cue, changed factual token, or temporal progression."""
    boost, _reason = update_signal(old_text, new_text, old_title=old_title, new_title=new_title)
    return boost > 0.0
