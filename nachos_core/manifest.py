"""Memory manifest — compact pointer list, recall on demand.

The manifest is the most distinctive Nachos idea: instead of dumping
all memory inline (expensive, blows context budget), surface a
SUMMARY of what's available — preference keys, recent topics, fact
counts — so the LLM can decide whether to call memory_recall to get
the details.

A good manifest is ~200-400 tokens. It tells the model 'I know things
about X, Y, Z. Ask me if you need them.'

Build steps:
  1. Query the memory store for preferences (kind == 'preference')
  2. Optionally pull recent session topics from session metadata
  3. Group remaining facts by kind, count them
  4. Render as a compact text block

The 'memory store' protocol below is intentionally minimal — the
adapter wraps Hermes' built-in memory file, or whatever backend the
host harness has. nachos_core does not care.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol


# ---------------------------------------------------------------------------
# Storage protocol — implemented by the adapter
# ---------------------------------------------------------------------------

class MemoryReadProtocol(Protocol):
    """What the manifest needs from the host's memory store.

    Adapters wrap whatever the harness exposes (Hermes memory file,
    a SQLite table, an external API) into these three calls.
    """

    def list_entries(self, kind: Optional[str] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Return [{content, kind, tags, target, created_at}, ...]."""
        ...

    def list_recent_topics(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Return [{topic, session_age, session_id}, ...] best-effort."""
        ...

    def fact_counts_by_kind(self) -> Dict[str, int]:
        """Return {kind: count} for grouped surfacing."""
        ...


# ---------------------------------------------------------------------------
# Manifest model
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    preferences: List[Dict[str, str]] = field(default_factory=list)  # {key, value}
    recent_topics: List[Dict[str, str]] = field(default_factory=list)  # {topic, age}
    fact_counts: List[Dict[str, Any]] = field(default_factory=list)    # {kind, count}
    total_entries: int = 0
    total_facts: int = 0
    generated_at: str = ""


@dataclass
class ManifestConfig:
    max_tokens: int = 400              # target render size
    include_preferences: bool = True
    recent_topic_count: int = 5
    include_fact_counts: bool = True
    preference_value_max_chars: int = 60


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_manifest(memory: MemoryReadProtocol,
                   config: Optional[ManifestConfig] = None) -> Manifest:
    """Build a manifest by querying the memory store for METADATA only.

    Never loads full entry content into the manifest. That's the whole
    point — the manifest is a pointer list, recall is on demand.
    """
    cfg = config or ManifestConfig()
    manifest = Manifest()

    # 1. Preferences — surface key/short-value pairs
    if cfg.include_preferences:
        try:
            entries = memory.list_entries(kind="preference", limit=20)
            for e in entries:
                manifest.preferences.append({
                    "key": _extract_pref_key(e.get("content", "")),
                    "value": _truncate(
                        e.get("content", ""),
                        cfg.preference_value_max_chars,
                    ),
                })
        except Exception:
            pass

    # 2. Recent topics
    if cfg.recent_topic_count > 0:
        try:
            topics = memory.list_recent_topics(limit=cfg.recent_topic_count)
            for t in topics:
                manifest.recent_topics.append({
                    "topic": _truncate(t.get("topic", ""), 80),
                    "age": t.get("session_age", "recent"),
                })
        except Exception:
            pass

    # 3. Fact counts
    if cfg.include_fact_counts:
        try:
            counts = memory.fact_counts_by_kind()
            for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                if count > 0:
                    manifest.fact_counts.append({"kind": kind, "count": count})
        except Exception:
            pass

    # 4. Totals (best-effort — no failure if backend can't tell us)
    try:
        all_entries = memory.list_entries(limit=1000)
        manifest.total_entries = len(all_entries)
        manifest.total_facts = sum(c["count"] for c in manifest.fact_counts)
    except Exception:
        pass

    manifest.generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return manifest


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_manifest(manifest: Manifest) -> str:
    """Render the manifest as a compact prompt section.

    Stays under ~400 tokens by truncating preference values and capping
    topic count. If the manifest is fully empty, returns empty string
    (caller should drop the section).
    """
    if (not manifest.preferences
            and not manifest.recent_topics
            and not manifest.fact_counts):
        return ""

    lines = ["# Memory Manifest",
             "(Pointers only — call memory_recall for full content.)"]

    if manifest.preferences:
        lines.append("")
        lines.append("Preferences:")
        for p in manifest.preferences:
            key = (p.get("key") or "").strip()
            val = (p.get("value") or "").strip()
            # If the key is just a prefix of (or identical to) the value,
            # don't render "key: key…" — just render the value. The key
            # extraction was always a best-effort summary, never a real
            # column name. When entries are sentence-shaped (the Hermes
            # default), there is no key — so don't pretend.
            if not key or _key_is_redundant(key, val):
                lines.append(f"- {val}")
            else:
                lines.append(f"- {key}: {val}")

    if manifest.recent_topics:
        lines.append("")
        lines.append("Recent topics:")
        for t in manifest.recent_topics:
            lines.append(f"- {t['topic']} ({t['age']})")

    if manifest.fact_counts:
        lines.append("")
        lines.append("Known facts by kind:")
        for c in manifest.fact_counts:
            lines.append(f"- {c['kind']}: {c['count']}")

    if manifest.total_entries:
        lines.append("")
        lines.append(f"Total entries: {manifest.total_entries}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PREF_KEY_RE = re.compile(r"^([A-Z][A-Za-z0-9 _-]{1,40}?):\s")


def _extract_pref_key(content: str) -> str:
    """Best-effort key extraction from a preference entry.

    Real Hermes memory entries are sentence-shaped — they don't follow a
    'Key: value' contract. Only return a key when the entry actually
    starts with one followed by a colon (e.g. 'Editor: vim'). Otherwise
    return empty string and let the renderer fall back to value-only.
    """
    if not content:
        return ""
    m = _PREF_KEY_RE.match(content.strip())
    if m:
        return m.group(1).strip()
    return ""


def _key_is_redundant(key: str, value: str) -> bool:
    """True if the key is a prefix of the value (or equal). Avoids 'X: X…'."""
    if not key:
        return True
    k = key.strip().lower()
    v = (value or "").strip().lower()
    if not v:
        return True
    if v.startswith(k):
        return True
    # Trailing ellipsis on truncated values shouldn't fool the check
    if v.rstrip("…").startswith(k):
        return True
    return False


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
