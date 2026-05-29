"""LLM-driven durable fact extraction.

Reads a conversation transcript, asks an LLM to distill it into typed
(subject, predicate, object, confidence, kind) triples, validates the
JSON output, returns the parsed facts. Pure logic — the LLM call
itself is injected by the caller as a callable, so this module never
imports any provider SDK.

Pipeline:

    transcript -> EXTRACTION_SYSTEM_PROMPT + user message ->
    LLM call (caller-supplied) -> JSON array of facts ->
    validate + map to MemoryFact -> dedup against existing -> store

This file owns only the first half (transcript → MemoryFact list).
Dedup lives in dedup.py. Storage is the plugin adapter's concern.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

from .types import MemoryFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction assistant. Your job is to read a conversation and extract facts that are worth remembering long-term about the user, their projects, preferences, decisions, environment, and relationships.

RULES:
- Extract only facts with lasting value. Skip transient task details, temporary errors, one-off debugging steps, or anything tied to the current session's specific work.
- Focus on: who the user is, what they prefer, what tools/languages/frameworks they use, what they have decided, durable project conventions, environment facts, and relationships between people or systems.
- Each fact must be a clear, standalone triple: subject, predicate, object.
- Be concise — objects should be short phrases, not sentences.
- Assign a confidence score (0.0-1.0). Use 0.9+ for explicit statements ("I prefer X"), 0.6-0.8 for inferred facts.
- Use only these kinds: preference, environment, project, skill, relationship, decision, general.

OUTPUT FORMAT — strict JSON array, no other text, no markdown fences:
[
  {"subject": "user", "predicate": "prefers", "object": "TypeScript over JavaScript", "kind": "preference", "confidence": 0.95},
  {"subject": "project deposco", "predicate": "uses", "object": "Angular 21.2", "kind": "project", "confidence": 0.9}
]

If no durable facts are present, return: []

DO NOT include:
- Markdown code fences
- Facts about the conversation itself ("user asked about X")
- Temporary or context-specific information
- Secrets, API keys, passwords, tokens, or credit-card numbers"""


# ---------------------------------------------------------------------------
# Inputs / config
# ---------------------------------------------------------------------------

class Message(Protocol):
    """Minimal duck-type for a transcript message."""
    role: str
    content: Any


@dataclass
class ExtractionConfig:
    max_conversation_chars: int = 50_000
    max_response_tokens: int = 2048
    min_confidence: float = 0.6           # facts below this are dropped
    default_kind: str = "general"
    default_source_session: Optional[str] = None


@dataclass
class ExtractionResult:
    facts: List[MemoryFact] = field(default_factory=list)
    raw_count: int = 0
    parse_success: bool = False
    error: Optional[str] = None

    @property
    def kept(self) -> int:
        return len(self.facts)


# ---------------------------------------------------------------------------
# LLMCall protocol — host injects this
# ---------------------------------------------------------------------------

LLMCall = Callable[[str, str, int], str]
# Signature: llm_call(system_prompt, user_message, max_tokens) -> response_text
# The caller (Hermes plugin) wraps whatever provider is configured.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_user_message(messages: Iterable[Dict[str, Any]],
                       max_chars: int = 50_000) -> str:
    """Render a transcript as a compact user message for the extractor.

    - Filters to user/assistant turns
    - Coerces non-string content to its string repr
    - Truncates from the BEGINNING if total exceeds max_chars (keep recent)
    """
    rendered_lines: List[str] = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        text = _content_to_text(content).strip()
        if not text:
            continue
        label = "User" if role == "user" else "Assistant"
        rendered_lines.append(f"{label}: {text}")

    if not rendered_lines:
        return "The conversation was empty. Return: []"

    transcript = "\n\n".join(rendered_lines)
    if len(transcript) > max_chars:
        transcript = "[…earlier turns truncated…]\n\n" + transcript[-max_chars:]

    return (
        "Extract durable facts from this conversation:\n\n"
        f"---\n{transcript}\n---\n\n"
        "Return a JSON array of facts (or [] if none). "
        "Strict JSON only — no markdown fences."
    )


def extract_facts(
    messages: Iterable[Dict[str, Any]],
    llm_call: LLMCall,
    config: Optional[ExtractionConfig] = None,
) -> ExtractionResult:
    """End-to-end: transcript → validated MemoryFact list.

    Never raises on LLM/parse errors — returns ExtractionResult with
    parse_success=False and an error string. Callers can decide whether
    to retry. This keeps extraction non-blocking on the host side.
    """
    cfg = config or ExtractionConfig()
    user_msg = build_user_message(messages, cfg.max_conversation_chars)

    try:
        raw = llm_call(EXTRACTION_SYSTEM_PROMPT, user_msg, cfg.max_response_tokens)
    except Exception as e:
        logger.warning("Extraction LLM call failed: %s", e)
        return ExtractionResult(error=f"llm_call failed: {e}")

    try:
        raw_facts = _parse_response(raw)
    except ValueError as e:
        logger.warning("Extraction response parse failed: %s | raw=%r",
                       e, raw[:300] if raw else "")
        return ExtractionResult(error=str(e))

    facts: List[MemoryFact] = []
    for raw_fact in raw_facts:
        fact = _validate_and_map(raw_fact, cfg)
        if fact is not None:
            facts.append(fact)

    return ExtractionResult(
        facts=facts,
        raw_count=len(raw_facts),
        parse_success=True,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_VALID_KINDS = {"preference", "environment", "project", "skill",
                "relationship", "decision", "general"}

# Strip optional ```json … ``` fences if the model leaks them despite the prompt.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_response(raw: str) -> List[Dict[str, Any]]:
    """Parse the LLM response into a list of dicts. Lenient about whitespace
    and accidental fences. Strict about types beyond that.
    """
    if not raw or not raw.strip():
        return []
    text = _FENCE_RE.sub("", raw).strip()
    # Try to find the first '[' so models that prefix a sentence still parse
    if not text.startswith("["):
        bracket = text.find("[")
        if bracket >= 0:
            text = text[bracket:]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array, got {type(parsed).__name__}")
    return [item for item in parsed if isinstance(item, dict)]


def _validate_and_map(raw_fact: Dict[str, Any],
                      cfg: ExtractionConfig) -> Optional[MemoryFact]:
    """Validate one raw extraction dict and map it to MemoryFact.

    Drops facts that are missing fields, fall below confidence threshold,
    or look like sensitive content (heuristic block).
    """
    subject = _str_or_empty(raw_fact.get("subject")).strip()
    predicate = _str_or_empty(raw_fact.get("predicate")).strip()
    obj = _str_or_empty(raw_fact.get("object")).strip()
    if not subject or not predicate or not obj:
        return None

    # Confidence
    raw_conf = raw_fact.get("confidence")
    try:
        confidence = float(raw_conf) if raw_conf is not None else 0.7
    except (TypeError, ValueError):
        confidence = 0.7
    confidence = max(0.0, min(1.0, confidence))
    if confidence < cfg.min_confidence:
        return None

    # Kind — accept either 'kind' or legacy 'type'
    kind = (_str_or_empty(raw_fact.get("kind"))
            or _str_or_empty(raw_fact.get("type"))
            or cfg.default_kind).lower().strip()
    if kind not in _VALID_KINDS:
        kind = cfg.default_kind

    # Cheap secret-shape filter
    if _looks_like_secret(obj) or _looks_like_secret(subject):
        return None

    return MemoryFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=confidence,
        kind=kind,
        source_session=cfg.default_source_session,
    )


def _str_or_empty(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _content_to_text(content: Any) -> str:
    """Coerce assistant message content (string or content-block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Anthropic-style: [{"type": "text", "text": "..."}]
        parts: List[str] = []
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


_SECRET_HINTS = (
    "sk-", "ghp_", "gho_", "ghu_", "ghs_", "xox", "Bearer ",
    "AKIA", "ASIA", "AIza",  # AWS keys, Google API keys
    "-----BEGIN",            # PEM
)


def _looks_like_secret(text: str) -> bool:
    if not text:
        return False
    if any(h in text for h in _SECRET_HINTS):
        return True
    # 32+ run of base64/hex chars without spaces — strong secret indicator
    if re.search(r"[A-Za-z0-9_\-+/=]{32,}", text) and " " not in text:
        return True
    return False
