"""Strict v1 JSON shapes. Paths are inert labels, never filesystem capabilities.

Every artifact has schema/profile/digest. All objects reject extra keys. Public
validators return detached JSON dictionaries, not mutable references to inputs.

V1 wire shapes (all fields required; [] allowed unless validated otherwise)::

    profile = {id: str, root: str}  # Explicit inert profile identity, not access.
    coverage = {discovered, hashed, content_searched, section_reviewed,
                body_reviewed: bool, limitations: [str]}  # independent flags
    source = {id, kind, role, identity, locator, sha256, before, coverage,
              metadata}
    # kind: memory_record | memory_file | skill_file | reference_file
    # role: primary | reference
    # metadata = {byte_size: int, char_size: int, enabled: bool | null,
    #             provenance: [str], usage: {status, ...}}; sizes must match the
    # snapshot. usage.status: available | unavailable | unknown; only
    # "available" may list references. Unknown is never "unused".
    # before: exact UTF-8 file text OR {key,title,summary,category,body: str}
    finding = {id, type, source_ids: [str], excerpts: [{source_id,text}],
               rationale, evidence: [observation], limitations: [str]}
    # type: stale | duplicate | conflict | unsupported | clarification
    # observation = {kind, status, source_ids, claim, observation, observed_at}
    # kind: schedule | execution | delivery | manual | source_content | usage
    # status: verified | unverified | missing | conflicting
    # Every changed source needs its own exact excerpt in a cited finding.
    proposal = {id, revision: int, finding_ids: [str], depends_on: [str],
                changes: [{source_id,source_sha256,operation,before,after}],
                assessments: [{finding_id, revision: int, evidence: [observation],
                               preservation_checklist: [{item,preserved: bool,
                               explanation}]}]}
    # Assessments bind the exact proposal revision; an edit or revision request
    # drops stale ones, and acceptance/export need every cited finding assessed
    # with all preserved == true (approval_safe).
    # operation: edit (typed after) | disable (null after, proposal-only)
    inventory = {schema,profile,digest,sources: [source],coverage}
    packet = {schema,profile,digest,inventory,findings: [finding],
              proposals: [proposal]}  # Initial proposal revisions are exactly 1.
    session = {schema,profile,digest,packet,reviews: [review]}
    review = {proposal_id,revisions: [proposal],history: [event],
              current_revision: int,decision,accepted_digest: str | null}
    event = {revision: int,action,digest,note}
    plan = {schema,profile,digest,packet_digest,session_digest,approved: [approval],
            ledger: [{proposal_id,revision,decision,status,reason}],
            execution: "not_performed",source_validation: "not_performed"}
    approval = {proposal,findings: [finding],sources: [source],
                inventory_coverage: coverage,review_digest}

Unannotated leaf fields above are strings, except before/after and nested objects.
SHA256 fields are full lowercase hex. Evidence times require ISO8601 timezone.
IDs are opaque exact strings; locators are relative POSIX labels without dot
segments, backslashes or colons. Coverage never upgrades discovery to body review.
Schema strings are exactly ``nachos.reconciliation.<kind>.v1``. See hashing.py
for byte-level canonicalization/domain separation and resource limits. Recompute
an artifact digest using seal_document; this validates, never signs or approves.
"""

from copy import deepcopy
from datetime import datetime
from typing import Any

from .hashing import canonical_json, content_digest, file_sha256, memory_sha256

SCHEMAS = {
    kind: f"nachos.reconciliation.{kind}.v1" for kind in ("inventory", "packet", "session", "plan")
}


def fields(value: Any, names: str) -> None:
    """Require precisely the named object fields."""
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError(f"expected object fields: {names}")


def text(value: Any, *, empty: bool = False) -> None:
    if type(value) is not str or (not empty and not value):
        raise ValueError("expected string")


def sequence(value: Any, *, nonempty: bool = False) -> None:
    if type(value) is not list or (nonempty and not value):
        raise ValueError("expected list")


def strings(value: Any, *, nonempty: bool = False) -> None:
    sequence(value, nonempty=nonempty)
    for item in value:
        text(item)
    if len(set(value)) != len(value):
        raise ValueError("duplicate string/ID")


def digest_text(value: Any) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected full lowercase SHA256")


def indexed(values: Any) -> dict[str, Any]:
    sequence(values)
    result: dict[str, Any] = {}
    for value in values:
        if type(value) is not dict or "id" not in value:
            raise ValueError("expected identified object")
        text(value["id"])
        if value["id"] in result:
            raise ValueError("duplicate ID")
        result[value["id"]] = value
    return result


def profile(value: Any) -> None:
    fields(value, "id root")
    text(value["id"])
    text(value["root"])
    if "\x00" in value["root"]:
        raise ValueError("profile root contains NUL")


def coverage(value: Any) -> None:
    fields(value, "discovered hashed content_searched section_reviewed body_reviewed limitations")
    for key in ("discovered", "hashed", "content_searched", "section_reviewed", "body_reviewed"):
        if type(value[key]) is not bool:
            raise ValueError("coverage flags must be booleans")
    strings(value["limitations"])


def _locator(locator: str) -> None:
    if "\x00" in locator:
        raise ValueError("source locator contains NUL")
    if "\\" in locator or ":" in locator or any(p in ("", ".", "..") for p in locator.split("/")):
        raise ValueError("locator must be a relative POSIX label without traversal")


def _metadata(value: dict[str, Any]) -> None:
    metadata = value["metadata"]
    fields(metadata, "byte_size char_size enabled provenance usage")
    snapshot = (
        canonical_json(value["before"]).decode("utf-8")
        if value["kind"] == "memory_record"
        else value["before"]
    )
    for key, size in (("byte_size", len(snapshot.encode("utf-8"))), ("char_size", len(snapshot))):
        if type(metadata[key]) is not int or metadata[key] != size:
            raise ValueError("snapshot size mismatch")
    if metadata["enabled"] is not None and type(metadata["enabled"]) is not bool:
        raise ValueError("enabled must be boolean or null (unknown)")
    strings(metadata["provenance"])
    fields(metadata["usage"], "status references")
    if metadata["usage"]["status"] not in ("available", "unavailable", "unknown"):
        raise ValueError("unsupported usage availability")
    strings(metadata["usage"]["references"])
    if metadata["usage"]["status"] != "available" and metadata["usage"]["references"]:
        raise ValueError("unavailable usage cannot claim references")


def source(value: Any) -> None:
    fields(value, "id kind role identity locator sha256 before coverage metadata")
    for key in ("id", "identity", "locator"):
        text(value[key])
    _locator(value["locator"])
    if value["kind"] not in ("memory_record", "memory_file", "skill_file", "reference_file"):
        raise ValueError("unsupported source kind")
    if value["role"] not in ("primary", "reference"):
        raise ValueError("unsupported source role")
    digest_text(value["sha256"])
    if value["kind"] == "memory_record":
        actual = memory_sha256(value["before"])
    else:
        text(value["before"], empty=True)
        actual = file_sha256(value["before"].encode("utf-8"))
    if actual != value["sha256"]:
        raise ValueError("source snapshot hash mismatch")
    coverage(value["coverage"])
    if not value["coverage"]["hashed"]:
        raise ValueError("snapshot must be hashed")
    _metadata(value)


def _inventory(value: dict[str, Any]) -> None:
    fields(value, "schema profile digest sources coverage")
    indexed(value["sources"])
    identities = set()
    for item in value["sources"]:
        source(item)
        identity = (item["kind"], item["identity"], item["locator"])
        if identity in identities:
            raise ValueError("duplicate source identity")
        identities.add(identity)
    coverage(value["coverage"])


def finding(value: Any, sources: dict[str, Any]) -> None:
    fields(value, "id type source_ids excerpts rationale evidence limitations")
    text(value["id"])
    if value["type"] not in ("stale", "duplicate", "conflict", "unsupported", "clarification"):
        raise ValueError("unsupported finding type")
    strings(value["source_ids"], nonempty=True)
    if not set(value["source_ids"]) <= sources.keys():
        raise ValueError("unknown finding source")
    text(value["rationale"])
    strings(value["limitations"])
    sequence(value["excerpts"], nonempty=True)
    for excerpt in value["excerpts"]:
        fields(excerpt, "source_id text")
        text(excerpt["text"])
        if excerpt["source_id"] not in value["source_ids"]:
            raise ValueError("excerpt source mismatch")
        before = sources[excerpt["source_id"]]["before"]
        bodies = before.values() if isinstance(before, dict) else [before]
        if not any(excerpt["text"] in body for body in bodies):
            raise ValueError("excerpt is not exact source text")
    observations(value["evidence"], value["source_ids"])


def timestamp(value: Any) -> None:
    """Require an ISO8601 human observation/review time with an explicit zone."""
    text(value)
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid timestamp") from exc
    if observed.tzinfo is None:
        raise ValueError("timestamp must include timezone")


def observations(value: Any, source_ids: list[str]) -> None:
    """Validate typed observations without promoting schedules to execution evidence."""
    sequence(value, nonempty=True)
    for evidence in value:
        fields(evidence, "kind status source_ids claim observation observed_at")
        if evidence["kind"] not in (
            "schedule",
            "execution",
            "delivery",
            "manual",
            "source_content",
            "usage",
        ):
            raise ValueError("unsupported evidence kind")
        if evidence["status"] not in ("verified", "unverified", "missing", "conflicting"):
            raise ValueError("unsupported evidence status")
        strings(evidence["source_ids"], nonempty=True)
        if not set(evidence["source_ids"]) <= set(source_ids):
            raise ValueError("unknown evidence source")
        text(evidence["claim"])
        text(evidence["observation"])
        timestamp(evidence["observed_at"])


def preservation(value: Any) -> None:
    """False assessments are valid review material, never export authority."""
    sequence(value, nonempty=True)
    for check in value:
        fields(check, "item preserved explanation")
        text(check["item"])
        text(check["explanation"])
        if type(check["preserved"]) is not bool:
            raise ValueError("preserved must be boolean")


def _assessments(value: dict[str, Any], findings: dict[str, Any]) -> None:
    sequence(value["assessments"])
    assessed = set()
    for assessment in value["assessments"]:
        fields(assessment, "finding_id revision evidence preservation_checklist")
        fid = assessment["finding_id"]
        text(fid)
        if fid not in value["finding_ids"] or fid in assessed:
            raise ValueError("unknown or duplicate assessment finding")
        assessed.add(fid)
        if type(assessment["revision"]) is not int or assessment["revision"] != value["revision"]:
            raise ValueError("preservation assessment must bind exact revision")
        observations(assessment["evidence"], findings[fid]["source_ids"])
        preservation(assessment["preservation_checklist"])


def _change(change: Any, src: dict[str, Any]) -> None:
    fields(change, "source_id source_sha256 operation before after")
    if change["source_sha256"] != src["sha256"] or change["before"] != src["before"]:
        raise ValueError("source precondition mismatch")
    if change["operation"] == "disable":
        if change["after"] is not None:
            raise ValueError("disable is proposal-only and requires null after")
    elif change["operation"] == "edit":
        if src["kind"] == "memory_record":
            memory_sha256(change["after"])
            if change["after"]["key"] != change["before"]["key"]:
                raise ValueError("memory key identity cannot change")
        else:
            text(change["after"], empty=True)
    else:
        raise ValueError("unsupported operation")


def proposal(value: Any, packet: dict[str, Any]) -> None:
    fields(value, "id revision finding_ids depends_on changes assessments")
    text(value["id"])
    if type(value["revision"]) is not int or value["revision"] < 1:
        raise ValueError("revision must be positive integer")
    strings(value["finding_ids"], nonempty=True)
    strings(value["depends_on"])
    findings = indexed(packet["findings"])
    if not set(value["finding_ids"]) <= findings.keys():
        raise ValueError("unknown proposal finding")
    _assessments(value, findings)
    sources = indexed(packet["inventory"]["sources"])
    covered = {sid for fid in value["finding_ids"] for sid in findings[fid]["source_ids"]}
    excerpted = {
        excerpt["source_id"]
        for fid in value["finding_ids"]
        for excerpt in findings[fid]["excerpts"]
    }
    sequence(value["changes"], nonempty=True)
    changed = set()
    for change in value["changes"]:
        if type(change) is not dict:
            raise ValueError("change must be an object")
        text(change.get("source_id"))
        sid = change["source_id"]
        if sid not in covered or sid in changed:
            raise ValueError("unknown, uncovered or duplicate change source")
        changed.add(sid)
        if sid not in excerpted:
            raise ValueError("each changed source requires an exact supported excerpt")
        _change(change, sources[sid])


def approval_safe(item: dict[str, Any]) -> bool:
    """Require fresh complete proposal-specific preservation confirmation."""
    return {a["finding_id"] for a in item["assessments"]} == set(item["finding_ids"]) and all(
        check["preserved"]
        for assessment in item["assessments"]
        for check in assessment["preservation_checklist"]
    )


def _packet(value: dict[str, Any]) -> None:
    fields(value, "schema profile digest inventory findings proposals")
    validate_document(value["inventory"], expected_profile=value["profile"])
    if value["inventory"]["schema"] != SCHEMAS["inventory"]:
        raise ValueError("packet requires inventory schema")
    sources = indexed(value["inventory"]["sources"])
    for item in indexed(value["findings"]).values():
        finding(item, sources)
    for item in indexed(value["proposals"]).values():
        proposal(item, value)
        if item["revision"] != 1:
            raise ValueError("packet proposals must start at revision 1")


def validate_document(value: Any, *, expected_profile: Any = None) -> dict[str, Any]:
    """Validate structure, relationships and integrity, then return a detached copy.

    All malformed input failures use ValueError, including nested wrong types.
    """
    try:
        return _validate_document(value, expected_profile=expected_profile)
    except (KeyError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("malformed reconciliation document") from exc


def _validate_document(value: Any, *, expected_profile: Any = None) -> dict[str, Any]:
    canonical_json(value)
    if type(value) is not dict or value.get("schema") not in SCHEMAS.values():
        raise ValueError("unsupported schema/version")
    if "profile" not in value or "digest" not in value:
        raise ValueError("missing profile/digest")
    profile(value["profile"])
    if expected_profile is not None and value["profile"] != expected_profile:
        raise ValueError("profile binding mismatch")
    kind = next(kind for kind, schema in SCHEMAS.items() if schema == value["schema"])
    digest_text(value["digest"])
    payload = {key: item for key, item in value.items() if key != "digest"}
    if value["digest"] != content_digest(kind, payload):
        raise ValueError("artifact digest mismatch")
    if kind == "inventory":
        _inventory(value)
    elif kind == "packet":
        _packet(value)
    elif kind == "session":
        from .review import validate_session  # noqa: PLC0415 -- Mutually dependent validators.

        validate_session(value)
    else:
        from .export import validate_plan  # noqa: PLC0415 -- Mutually dependent validators.

        validate_plan(value)
    return deepcopy(value)


def seal_document(value: dict[str, Any]) -> dict[str, Any]:
    """Compute an integrity digest, then validate. This does not confer trust."""
    canonical_json(value)
    result = deepcopy(value)
    if result.get("schema") not in SCHEMAS.values():
        raise ValueError("unsupported schema/version")
    kind = next(kind for kind, schema in SCHEMAS.items() if schema == result["schema"])
    result.pop("digest", None)
    result["digest"] = content_digest(kind, result)
    return validate_document(result)
