"""Sealed inventory of current sources: discovery and hashing only, no review claims.

Usage, enablement and provenance are not observable from these sources, so they
stay unknown (enabled=None, no provenance, usage status "unknown"); unknown is
never reported as unused. Coverage is staged honestly: content search, section
review and body review are all false. ``discovered`` means only that the explicit
labels the caller supplied were found; the profile itself is never enumerated.
"""

from __future__ import annotations

from typing import Any

from .acquire import CurrentSource, SourceSpec, read_current_sources
from .boundary import AcquisitionError, SourceBoundary
from .hashing import file_sha256
from .models import SCHEMAS, seal_document

_LIMITATIONS = [
    "Sources are only the explicit bounded labels supplied (memory store plus named "
    "skill/reference files); this is not a discovery of the whole profile.",
    "Skills, references and other profile files outside those labels were not examined.",
    "'discovered' means the listed explicit sources were found and hashed, nothing more.",
    "No content search, section review or body review has been performed.",
    "Enablement, provenance and usage are unknown, not absent.",
]


def _coverage() -> dict[str, Any]:
    return {
        "discovered": True,
        "hashed": True,
        "content_searched": False,
        "section_reviewed": False,
        "body_reviewed": False,
        "limitations": list(_LIMITATIONS),
    }


def source_id(item: CurrentSource) -> str:
    """Deterministic opaque id derived from kind/identity/locator."""
    key = "\0".join((item.kind, item.identity, item.locator)).encode("utf-8")
    return "src-" + file_sha256(key)[:16]


def _source(item: CurrentSource) -> dict[str, Any]:
    return {
        "id": source_id(item),
        "kind": item.kind,
        "role": item.role,
        "identity": item.identity,
        "locator": item.locator,
        "sha256": item.sha256,
        "before": item.before if isinstance(item.before, str) else dict(item.before),
        "coverage": _coverage(),
        "metadata": {
            "byte_size": item.byte_size,
            "char_size": item.char_size,
            "enabled": None,
            "provenance": [],
            "usage": {"status": "unknown", "references": []},
        },
    }


def _check_profile(boundary: SourceBoundary, profile: dict[str, str]) -> dict[str, str]:
    if (
        type(profile) is not dict
        or set(profile) != {"id", "root"}
        or any(type(value) is not str for value in profile.values())
        or str(boundary.root) != profile["root"]
    ):
        raise AcquisitionError("profile_mismatch")
    return {"id": profile["id"], "root": profile["root"]}


def build_inventory(
    boundary: SourceBoundary,
    profile: dict[str, str],
    spec: SourceSpec,
) -> dict[str, Any]:
    """Read the explicit sources and return a sealed, validated inventory."""
    trusted = _check_profile(boundary, profile)
    current = read_current_sources(boundary, spec)
    sources = sorted(
        (_source(item) for item in current.values()),
        key=lambda s: (s["kind"], s["locator"], s["identity"]),
    )
    return seal_document(
        {
            "schema": SCHEMAS["inventory"],
            "profile": trusted,
            "sources": sources,
            "coverage": _coverage(),
        }
    )
