"""Bounded strict JSON IO. No path interpretation, file IO or provider initialization."""

import json
from typing import Any

from .hashing import MAX_BYTES, canonical_json
from .models import validate_document


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _forbidden(_value: str) -> Any:
    raise ValueError("unsupported JSON number")


def loads(raw: Any, *, expected_profile: Any = None) -> dict[str, Any]:
    """Read a JSON string or UTF-8 bytes, with strict v1 validation."""
    if type(raw) not in (str, bytes):
        raise ValueError("expected JSON text or bytes")
    try:
        size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
        if size > MAX_BYTES:
            raise ValueError("JSON byte limit exceeded")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = json.loads(
            raw, object_pairs_hook=_pairs, parse_float=_forbidden, parse_constant=_forbidden
        )
        return validate_document(value, expected_profile=expected_profile)
    except (UnicodeError, RecursionError, TypeError, KeyError) as exc:
        raise ValueError("invalid reconciliation JSON") from exc


def dumps(value: Any) -> bytes:
    """Validate and serialize a complete artifact to canonical UTF-8 bytes."""
    return canonical_json(validate_document(value))
