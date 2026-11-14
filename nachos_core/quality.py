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

from .lexical import STOPWORDS

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
