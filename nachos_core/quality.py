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


_UPDATE_CUES_RE = re.compile(
    r"\b(?:now|no longer|moved to|moved from|replaced|migrated to|migrated from|"
    r"switched to|retired|deprecated|cancelled|canceled|end of life|supersedes|"
    r"instead of|formerly|updated to|changed to)\b",
    re.IGNORECASE,
)
_FACT_NUMBER_RE = re.compile(r"\b(?:v?\d+(?:[.:-]\d+)+|\d{4}-\d{2}-\d{2}|\d+)\b")
_FACT_PATH_RE = re.compile(r"(?:https?://[^\s,;)]+|(?:/[\w.-]+){2,}|[a-z0-9_.-]+/[a-z0-9_.-]+)")
_FACT_WORD_RE = re.compile(
    r"\b(?:single|double|triple|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"hourly|daily|weekly|biweekly|fortnightly|monthly|quarterly|yearly|annually)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19\d\d|20\d\d)\b")
_YEAR_OR_VER_RE = re.compile(r"^(?:19\d\d|20\d\d|v\d+(?:\.\d+)*)$")
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
_TEMPORAL_TITLE_WORDS = _OLD_TEMPORAL_WORDS | _NEW_TEMPORAL_WORDS | frozenset({"setup"})


def core_title_tokens(title: str) -> frozenset[str]:
    """Stemmed subject tokens of ``title``, stripping years, versions and temporal tags."""
    return frozenset(
        t for t in tokens(title) if t not in _TEMPORAL_TITLE_WORDS and not _YEAR_OR_VER_RE.match(t)
    )


def _titles_compatible_for_supersede(old_title: str, new_title: str) -> bool:
    """True when two titles refer to the same subject rather than distinct facets or topics."""
    old_core = core_title_tokens(old_title)
    new_core = core_title_tokens(new_title)
    if not old_core or not new_core:
        return True
    shared = old_core & new_core
    if not shared:
        return False
    if len(shared) < 2 and len(shared) / len(old_core | new_core) < 0.5:
        return False
    # Distinct non-temporal qualifiers on both sides (e.g. production vs staging, EU vs US)
    # indicate parallel siblings rather than an update of the same entry.
    old_only = old_core - new_core
    new_only = new_core - old_core
    return not (old_only and new_only)


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


def _factual_tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    return (
        set(_FACT_NUMBER_RE.findall(lowered))
        | set(_FACT_PATH_RE.findall(lowered))
        | set(_FACT_WORD_RE.findall(lowered))
    )


def looks_like_update(
    old_text: str,
    new_text: str,
    *,
    old_title: str = "",
    new_title: str = "",
) -> bool:
    """Deterministic heuristic: True when ``new_text`` reads like an update of ``old_text``.

    When titles are supplied, rejects unrelated topics and distinct-facet siblings
    (such as staging vs production or EU vs US). Fires on explicit replacement wording,
    changed numeric/date/version/cadence/path tokens, or temporal title progression.
    """
    if old_title and new_title and not _titles_compatible_for_supersede(old_title, new_title):
        return False
    if _UPDATE_CUES_RE.search(new_text or ""):
        return True
    if (
        old_title
        and new_title
        and _has_temporal_progression(old_title, new_title)
        and (old_text or "").strip() != (new_text or "").strip()
    ):
        return True
    old_facts = _factual_tokens(old_text)
    new_facts = _factual_tokens(new_text)
    return bool(old_facts and new_facts and old_facts != new_facts)
