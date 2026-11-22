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


_RECALL_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "key": {"type": "string", "description": "Exact entry key (slug of title)."},
        "query": {"type": "string", "description": "Natural-language or keyword search."},
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional 1-4 query rephrasings (fused with RRF; embeddings batched in one "
                "call). Use with or instead of 'query'."
            ),
        },
        "entity": {
            "type": "string",
            "description": "Only entries that mention this identifier, host, path, @handle "
            "or name; with no query, lists them.",
        },
        "limit": {"type": "integer", "description": "Max results for a query (default 10)."},
        "include_superseded": {
            "type": "boolean",
            "description": "Include superseded entries in query results (default false).",
        },
    },
}

_RECALL_BASE_DESC = (
    "Pull full memory entry content on demand. Provide 'key' for an exact "
    "entry from the manifest (superseded entries still return their content "
    "plus a superseded_by pointer, links, backlinks and broken_links), or "
    "'query' / 'queries' (up to 4 variants, fused with RRF in one batched "
    "embedding call) / 'entity' to search: matches by keywords and, when "
    "embeddings are enabled, by meaning, redirecting superseded hits to their "
    "current successor unless include_superseded is true; query results may "
    "carry `related` (linked entries one hop away). Entries saved with "
    "provenance carry a compact 'source' (session, date, ref); pass its ref to "
    "nachos_history_search to reopen the original conversation. "
)

RECALL_SCHEMA_HINT: dict[str, Any] = {
    "name": "nachos_memory_recall",
    "description": (
        _RECALL_BASE_DESC
        + "Query responses include confident/confidence and no_confident_match as "
        "informational hints. Use when the manifest shows a relevant entry or you "
        "suspect one exists."
    ),
    "parameters": _RECALL_PARAMS,
}

RECALL_SCHEMA: dict[str, Any] = {
    "name": "nachos_memory_recall",
    "description": (
        _RECALL_BASE_DESC
        + "Query responses include confident/confidence; when confidence is low or "
        "no_confident_match is true, treat listed results as low-confidence "
        "candidates and retry with 1-3 rephrasings in 'queries', including one "
        "written as the stored fact would be (e.g. 'Decision: migrations run through "
        "Alembic...'). Use when the manifest shows a relevant entry or you suspect "
        "one exists."
    ),
    "parameters": _RECALL_PARAMS,
}

PUT_SCHEMA = {
    "name": "nachos_memory_put",
    "description": (
        "Add or update a memory entry. The key is derived from the title (slug); "
        "putting an existing title updates that entry. If summary is omitted it is "
        "taken from the first line of body, so write body summary-first. category "
        "groups the entry in the manifest. Low-signal body/summary (empty, "
        "placeholder, too short) is rejected. Pass supersedes (key or list of keys) "
        "up front when replacing an outdated entry so it stays archived out of "
        "default recall and the manifest. The response lists possible_duplicates "
        "and possible_updates; if one of possible_updates is an older version of "
        "the same fact, call put again with supersedes=[key]. Link related entries by "
        "writing [[key]] (or [[Entry Title]]) in the body; suggested_links lists entries "
        "that share identifiers, hosts or names with the new one, and broken_links any "
        "[[...]] that points nowhere. Secrets are "
        "redacted before storage."
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
            "supersedes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional key or list of keys this entry replaces; marks old entries "
                    "superseded without deleting them."
                ),
            },
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
        "checkpoints the full transcript first; this ranks messages by how well they "
        "match the query's words (rarer words count more; no need for every word) and "
        "returns each hit with its neighbouring turns, labelled by role and position "
        "(a tool call and its result always come together). Each result has a stable "
        "'ref'; pass a ref (or a memory entry's source.ref) back to reopen that "
        "spot. For PAST sessions prefer Hermes' session_search; "
        'scope="all" additionally searches older Nachos checkpoints.'
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Words to look for; the more distinctive, the better the ranking.",
            },
            "ref": {
                "type": "string",
                "description": "A result 'ref' or a memory entry's source.ref ('<session>@<checkpoint>#<index>' opens that message with context; a bare '<session>' limits the query to that session).",
            },
            "scope": {
                "type": "string",
                "enum": ["conversation", "all"],
                "description": "'conversation' (default) searches the current conversation, including sessions it continued from; 'all' also searches older Nachos checkpoints.",
            },
            "limit": {"type": "integer", "description": "Max results (default 8)."},
            "before": {
                "type": "integer",
                "description": "Turns of context before each match (default from config, 1).",
            },
            "after": {
                "type": "integer",
                "description": "Turns of context after each match (default from config, 1).",
            },
        },
    },
}
