"""Hermes adapter for the Nachos extractor.

Provides:
  - HermesLLMCall: wraps Hermes' active model into the LLMCall protocol
    so the extractor can run end-of-session fact extraction without
    knowing which provider is active.
  - JsonlFactStore: persists extracted MemoryFacts to a jsonl file under
    ~/.hermes/<profile>/nachos/facts.jsonl. Uses dedup-then-rewrite so
    the file always reflects the merged truth.

These are intentionally tiny — extraction policy is in nachos_core.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional

from nachos_core.types import MemoryFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM call wrapper — Hermes' auxiliary client
# ---------------------------------------------------------------------------

def make_hermes_llm_call(
    model: Optional[str] = None,
    timeout: float = 60.0,
) -> Callable[[str, str, int], str]:
    """Build an LLMCall closure that runs against Hermes' configured model.

    Uses agent.auxiliary_client.get_text_auxiliary_client(task) — the
    same path Hermes uses for compression, web extraction, and other
    background tasks. Returns a string body. Raises if Hermes isn't
    importable (caller catches).

    The auxiliary client returns a (client, default_model_slug) tuple
    every call. We resolve once at construction and cache; if the
    user switches providers mid-session the closure stays on the
    original — fine for v0.2.
    """
    from agent.auxiliary_client import get_text_auxiliary_client  # type: ignore

    client, default_model = get_text_auxiliary_client(task="extraction")
    if client is None:
        raise RuntimeError("Hermes auxiliary client unavailable (no provider configured)")

    resolved_model = model or default_model
    if not resolved_model:
        raise RuntimeError("Hermes auxiliary client has no default model")

    # Warn loudly if an expensive model slipped through — extraction is
    # a high-volume background task. Opus should never be the extraction
    # model. This is a safeguard against 'provider: auto' falling back
    # to whatever the main model is.
    _EXPENSIVE_MODEL_HINTS = ("opus", "gpt-4o", "gemini-ultra")
    if any(h in resolved_model.lower() for h in _EXPENSIVE_MODEL_HINTS):
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Nachos extraction resolved to expensive model %r — "
            "set auxiliary.extraction.model in config.yaml to a cheaper model "
            "(e.g. claude-sonnet-4-6) to control cost.",
            resolved_model,
        )

    def _call(system_prompt: str, user_message: str, max_tokens: int) -> str:
        # OpenAI-compatible chat-completions surface across all aux providers.
        kwargs = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,  # extraction wants determinism
        }

        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            # Some providers reject 'temperature' or unrecognized kwargs.
            kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**kwargs)

        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        if msg is None:
            return ""
        content = getattr(msg, "content", "") or ""
        return content if isinstance(content, str) else str(content)

    return _call


# ---------------------------------------------------------------------------
# Fact store — jsonl with dedup-on-write
# ---------------------------------------------------------------------------

class JsonlFactStore:
    """Stores MemoryFacts as one JSON object per line.

    Reads are O(file). For v0.2 fact volumes (dozens to low-hundreds per
    user across all time) that's fine. SQLite migration is a v0.3 problem
    if the file ever gets big enough to matter.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list_all(self) -> List[MemoryFact]:
        if not self.path.exists():
            return []
        out: List[MemoryFact] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        out.append(MemoryFact(
                            subject=d["subject"],
                            predicate=d["predicate"],
                            object=d["object"],
                            confidence=float(d.get("confidence", 1.0)),
                            kind=d.get("kind", "general"),
                            source_session=d.get("source_session"),
                            extracted_at=d.get("extracted_at", ""),
                        ))
                    except (KeyError, ValueError, TypeError) as e:
                        logger.debug("Skipping malformed fact line %d: %s",
                                     line_num, e)
        except Exception as e:
            logger.warning("Could not read fact store %s: %s", self.path, e)
        return out

    def replace_all(self, facts: List[MemoryFact]) -> None:
        """Atomic-ish rewrite. Used by upsert flow."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for fact in facts:
                    f.write(json.dumps(_fact_to_dict(fact)) + "\n")
            tmp.replace(self.path)
        except Exception as e:
            logger.warning("Could not write fact store %s: %s", self.path, e)
            try:
                tmp.unlink()
            except Exception:
                pass

    def upsert(self, incoming: List[MemoryFact]) -> tuple[int, int]:
        """Dedup incoming against existing, write merged set, return counts.

        Returns (inserted, updated).
        """
        from nachos_core.dedup import deduplicate_facts

        existing = self.list_all()
        result = deduplicate_facts(incoming, existing)

        # Build new full list: existing minus replaced + merged + inserted
        replaced_keys = {id(old) for old, _ in result.to_update}
        kept = [f for f in existing if id(f) not in replaced_keys]
        merged_replacements = [merged for _, merged in result.to_update]
        all_facts = kept + merged_replacements + result.to_insert
        self.replace_all(all_facts)
        return len(result.to_insert), len(result.to_update)


def _fact_to_dict(fact: MemoryFact) -> dict:
    return {
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "confidence": fact.confidence,
        "kind": fact.kind,
        "source_session": fact.source_session,
        "extracted_at": fact.extracted_at,
    }
