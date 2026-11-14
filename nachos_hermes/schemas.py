"""Tool schemas the Nachos provider exposes to the Hermes agent, plus small arg helpers."""

from __future__ import annotations

import json
from typing import Any

from nachos_core.memory import PROFILE, SHARED


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def with_scope(schema: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(schema))
    out["parameters"]["properties"]["scope"] = {
        "type": "string",
        "enum": [PROFILE, SHARED],
        "description": (
            "'profile' (default for writes) is private to this Hermes "
            "profile; 'shared' is visible to every profile. For recall "
            "and remove, omit to search both (profile first)."
        ),
    }
    return out


RECALL_SCHEMA = {
    "name": "nachos_memory_recall",
    "description": (
        "Pull full memory entry content on demand. Provide 'key' for an exact "
        "entry from the manifest, or 'query' to search: matches by keywords and, "
        "when embeddings are enabled, by meaning, so a paraphrase still finds the "
        "entry. Use when the manifest shows a relevant entry or you suspect one "
        "exists."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Exact entry key (slug of title)."},
            "query": {"type": "string", "description": "Natural-language or keyword search."},
            "limit": {"type": "integer", "description": "Max results for a query (default 10)."},
        },
    },
}

PUT_SCHEMA = {
    "name": "nachos_memory_put",
    "description": (
        "Add or update a memory entry. The key is derived from the title (slug); "
        "putting an existing title updates that entry. If summary is omitted it is "
        "taken from the first line of body, so write body summary-first. category "
        "groups the entry in the manifest. Low-signal body/summary (empty, "
        "placeholder, too short) is rejected. The response lists "
        "possible_duplicates when similar entries exist: merge rather than keep "
        "near-copies. Secrets are redacted before storage."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short label (becomes the key)."},
            "summary": {
                "type": "string",
                "description": "One keyword-dense line for the manifest. Optional; derived from body[0] if absent.",
            },
            "category": {
                "type": "string",
                "description": "Manifest group (e.g. 'project-x', 'user'). Default 'general'.",
            },
            "body": {"type": "string", "description": "Full entry content."},
        },
        "required": ["title", "body"],
    },
}

REMOVE_SCHEMA = {
    "name": "nachos_memory_remove",
    "description": "Delete a memory entry by 'key' or 'title'.",
    "parameters": {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Entry key."},
            "title": {"type": "string", "description": "Entry title (slugified to key)."},
        },
    },
}

HISTORY_SCHEMA = {
    "name": "nachos_history_search",
    "description": (
        "Recover exact details that context compression removed from the CURRENT "
        "conversation (commands, paths, error text, numbers, decisions, subagent "
        "results); Hermes' own session_search cannot reach these. Every compression "
        "checkpoints the full transcript first; this finds messages containing all "
        "of the query's words and returns excerpts. For PAST sessions prefer "
        'Hermes\' session_search; scope="all" additionally searches older Nachos '
        "checkpoints."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Words that must all appear in the message.",
            },
            "scope": {
                "type": "string",
                "enum": ["conversation", "all"],
                "description": "'conversation' (default) searches the current conversation, including sessions it continued from; 'all' also searches older Nachos checkpoints.",
            },
            "limit": {"type": "integer", "description": "Max results (default 8)."},
        },
        "required": ["query"],
    },
}
