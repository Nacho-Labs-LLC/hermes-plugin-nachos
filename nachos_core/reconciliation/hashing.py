"""Versioned integrity hashes; these are NOT signatures or proof of authenticity."""

import hashlib
import json
from typing import Any

MAX_BYTES = 8 * 1024 * 1024
MAX_DEPTH = 32
MAX_NODES = 100_000
MAX_ITEMS = 10_000
MEMORY_FIELDS = frozenset(("key", "title", "summary", "category", "body"))


def _check_json(value: Any, depth: int, remaining: list[int]) -> None:
    remaining[0] -= 1
    if depth > MAX_DEPTH or remaining[0] < 0:
        raise ValueError("JSON depth/node limit exceeded")
    if value is None or type(value) in (bool, int, str):
        if type(value) is int and abs(value) > 2**53 - 1:
            raise ValueError("integer out of range")
        return
    if type(value) not in (dict, list) or len(value) > MAX_ITEMS:
        raise ValueError("unsupported JSON type or collection limit exceeded")
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        values = value.values()
    else:
        values = value
    for item in values:
        _check_json(item, depth + 1, remaining)


def canonical_json(value: Any) -> bytes:
    """Exact strings, sorted keys, compact UTF-8; no floats or implicit coercions."""
    _check_json(value, 0, [MAX_NODES])
    try:
        raw = json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("invalid canonical JSON") from exc
    if len(raw) > MAX_BYTES:
        raise ValueError("JSON byte limit exceeded")
    return raw


def file_sha256(raw: bytes) -> str:
    """SHA-256 of raw bytes, with no decoding or newline normalization."""
    if type(raw) is not bytes:
        raise ValueError("raw file content must be bytes")
    return hashlib.sha256(raw).hexdigest()


def content_digest(domain: str, value: Any) -> str:
    """Domain-separated SHA-256 of canonical JSON (full lowercase hex)."""
    if domain not in ("memory-record", "proposal", "packet", "session", "plan", "inventory"):
        raise ValueError("unknown hash domain")
    return file_sha256(
        b"nachos.reconciliation/" + domain.encode("ascii") + b"/v1\0" + canonical_json(value)
    )


def memory_sha256(record: dict[str, Any]) -> str:
    """Bind all five logical-record fields, including the exact record key."""
    if type(record) is not dict or set(record) != MEMORY_FIELDS:
        raise ValueError("memory record requires key/title/summary/category/body")
    if any(type(value) is not str for value in record.values()):
        raise ValueError("memory fields must be strings")
    return content_digest("memory-record", record)
