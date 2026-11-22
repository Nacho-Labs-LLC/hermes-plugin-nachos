"""Synthetic-only packet builder tests."""

import copy
import json

import pytest

from nachos_core.reconciliation import packets
from nachos_core.reconciliation.acquire import SourceSpec
from nachos_core.reconciliation.boundary import SourceBoundary
from nachos_core.reconciliation.inventory import build_inventory
from nachos_core.reconciliation.models import seal_document, validate_document
from nachos_core.reconciliation.packets import (
    bounded_views,
    build_packet,
    heuristic_findings,
    load_findings_json,
)
from nachos_core.reconciliation.review import new_session

AT = "2026-11-16T13:00:00+00:00"
MD_TEXT = (
    "# ops\n\n## Deploy\ntbd\nthe deploy runbook steps live in the wiki\n\n"
    "## Plain Note\nplain note about plain things\nplain details\n"
)


@pytest.fixture
def inventory(tmp_path):
    (tmp_path / "nachos").mkdir()
    (tmp_path / "nachos/memory.md").write_text(MD_TEXT, encoding="utf-8")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills/SKILL.md").write_text("pre\n# One\naaa\n## Two\nbbb\n", encoding="utf-8")
    boundary = SourceBoundary(tmp_path, ("nachos/memory.md", "skills/SKILL.md"))
    profile = {"id": "p", "root": str(tmp_path)}
    return build_inventory(boundary, profile, SourceSpec("flatfile", ("skills/SKILL.md",)))


def _source(inv, identity):
    return next(s for s in inv["sources"] if s["identity"] == identity)


def _finding(inv, identity="skill:skills/SKILL.md", text="aaa"):
    sid = _source(inv, identity)["id"]
    return {
        "id": "f1",
        "type": "stale",
        "source_ids": [sid],
        "excerpts": [{"source_id": sid, "text": text}],
        "rationale": "r",
        "evidence": [
            {
                "kind": "manual",
                "status": "unverified",
                "source_ids": [sid],
                "claim": "c",
                "observation": "o",
                "observed_at": "2026-01-01T00:00:00Z",
            }
        ],
        "limitations": [],
    }


def test_views_split_on_headings_and_are_deterministic(inventory):
    views = bounded_views(inventory)
    assert views == bounded_views(inventory)
    skill = next(
        v for v in views if v["source_id"] == _source(inventory, "skill:skills/SKILL.md")["id"]
    )
    assert [s["heading"] for s in skill["sections"]] == ["", "One", "Two"]
    assert [s["excerpt"] for s in skill["sections"]] == ["pre", "aaa", "bbb"]
    assert skill["omitted_sections"] == 0


def test_views_truncate_and_flag(inventory):
    views = bounded_views(inventory, max_section_chars=2)
    flat = [s for v in views for s in v["sections"]]
    assert any(s["truncated"] for s in flat)
    assert all(len(s["excerpt"]) <= 2 for s in flat)
    limited = bounded_views(inventory, max_sections_per_source=1)
    assert any(v["omitted_sections"] > 0 for v in limited)
    total = bounded_views(inventory, max_total_chars=3)
    assert sum(len(s["excerpt"]) for v in total for s in v["sections"]) <= 3


def test_views_do_not_mutate_inventory(inventory):
    snapshot = copy.deepcopy(inventory)
    bounded_views(inventory)
    assert inventory == snapshot


def test_heuristics_are_clarifications_with_exact_excerpts(inventory):
    found = heuristic_findings(inventory, observed_at=AT)
    assert found == heuristic_findings(inventory, observed_at=AT)
    assert found
    assert {f["type"] for f in found} == {"clarification"}
    for item in found:
        assert item["evidence"][0]["kind"] == "source_content"
        assert item["evidence"][0]["status"] == "unverified"
    packet = build_packet(inventory, findings=found)
    assert packet["proposals"] == []


def test_load_findings_strict():
    assert load_findings_json(b"[]") == []
    assert load_findings_json(b'[{"a": 1}]') == [{"a": 1}]
    for bad in (
        b'[{"a":1,"a":2}]',
        b'[{"a":1.5}]',
        b"[NaN]",
        b"{}",
        b'["x"]',
        b"\xff",
        b"[",
        b"[" * 100000,
        b"[" + b",".join([b"{}"] * (packets.MAX_FINDINGS + 1)) + b"]",
    ):
        with pytest.raises(ValueError, match=r"findings|JSON|duplicate|number|finding"):
            load_findings_json(bad)
    with pytest.raises(ValueError, match=r"findings|JSON|duplicate|number|finding"):
        load_findings_json("[]")  # type: ignore[arg-type]


def test_packet_validates_and_session_opens(inventory):
    packet = build_packet(inventory, findings=[_finding(inventory)])
    validate_document(packet)
    session = new_session(packet)
    validate_document(session)
    assert packet["inventory"] == inventory
    assert packet["proposals"] == []
    assert build_packet(inventory, findings=[_finding(inventory)])["digest"] == packet["digest"]


def test_packet_rejects_inexact_excerpt(inventory):
    with pytest.raises(ValueError, match="exact source text"):
        build_packet(inventory, findings=[_finding(inventory, text="not in source")])


def test_packet_rejects_tampered_inventory(inventory):
    bad = copy.deepcopy(inventory)
    bad["sources"][0]["metadata"]["usage"]["status"] = "available"
    with pytest.raises(ValueError, match="digest mismatch"):
        build_packet(bad)


def test_packet_rejects_non_inventory(inventory):
    packet = build_packet(inventory)
    with pytest.raises(ValueError, match="expected an inventory"):
        build_packet(packet)


def test_finder_only_when_injected(inventory):
    calls = []

    def finder(views):
        calls.append(views)
        return [_finding(inventory)]

    build_packet(inventory, findings=[_finding(inventory)])
    assert not calls
    packet = build_packet(inventory, finder=finder)
    assert len(calls) == 1
    assert calls[0] == bounded_views(inventory)
    assert [f["id"] for f in packet["findings"]] == ["f1"]


def test_finder_output_is_validated(inventory):
    with pytest.raises(ValueError, match="expected object fields"):
        build_packet(inventory, finder=lambda _v: [{"id": "x"}])


def test_proposals_only_from_explicit_input_and_validated(inventory):
    finding = _finding(inventory)
    source = _source(inventory, "skill:skills/SKILL.md")
    proposal = {
        "id": "p1",
        "revision": 1,
        "finding_ids": ["f1"],
        "depends_on": [],
        "changes": [
            {
                "source_id": source["id"],
                "source_sha256": source["sha256"],
                "operation": "edit",
                "before": source["before"],
                "after": "changed",
            }
        ],
        "assessments": [],
    }
    packet = build_packet(inventory, findings=[finding], proposals=[proposal])
    assert [p["id"] for p in packet["proposals"]] == ["p1"]
    stale = copy.deepcopy(proposal)
    stale["changes"][0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="precondition mismatch"):
        build_packet(inventory, findings=[finding], proposals=[stale])


def test_building_does_not_upgrade_coverage(inventory):
    packet = build_packet(inventory, findings=heuristic_findings(inventory, observed_at=AT))
    cov = packet["inventory"]["coverage"]
    assert cov["content_searched"] is False
    assert cov["section_reviewed"] is False
    assert cov["body_reviewed"] is False
    for source in packet["inventory"]["sources"]:
        assert source["coverage"]["body_reviewed"] is False


def test_findings_json_round_trip(inventory):
    raw = json.dumps([_finding(inventory)]).encode()
    packet = build_packet(inventory, findings=load_findings_json(raw))
    assert seal_document(packet)["digest"] == packet["digest"]


def test_heuristics_require_explicit_aware_observed_at(inventory):
    with pytest.raises(TypeError):
        heuristic_findings(inventory)  # type: ignore[call-arg]
    for bad in ("2026-11-16T13:00:00", "yesterday", ""):
        with pytest.raises(ValueError, match=r"timestamp|string"):
            heuristic_findings(inventory, observed_at=bad)
    found = heuristic_findings(inventory, observed_at=AT)
    assert {f["evidence"][0]["observed_at"] for f in found} == {AT}
    assert not any("placeholder" in lim for f in found for lim in f["limitations"])
    assert not hasattr(packets, "PLACEHOLDER_TIME")


def _skill_inventory(tmp_path, text):
    (tmp_path / "nachos").mkdir()
    (tmp_path / "nachos/memory.md").write_text("# ops\n\n## A\nalpha\n", encoding="utf-8")
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills/SKILL.md").write_bytes(text.encode("utf-8"))
    boundary = SourceBoundary(tmp_path, ("nachos/memory.md", "skills/SKILL.md"))
    profile = {"id": "p", "root": str(tmp_path)}
    return build_inventory(boundary, profile, SourceSpec("flatfile", ("skills/SKILL.md",)))


@pytest.mark.parametrize("sep", ["\r\n", "\r", "\u2028", "\x0c", "\x0b", "\x1c", "\x85", "\u2029"])
def test_view_excerpts_are_exact_substrings_for_exotic_line_breaks(tmp_path, sep):
    text = f"pre{sep}more{sep}# One{sep}aaa{sep}bbb\n## Two{sep}ccc{sep}\n\nlast{sep}"
    inv = _skill_inventory(tmp_path, text)
    source = _source(inv, "skill:skills/SKILL.md")
    view = next(v for v in bounded_views(inv) if v["source_id"] == source["id"])
    assert view["sections"]
    for section in view["sections"]:
        assert section["excerpt"] in source["before"]
        assert section["heading"] in source["before"] or section["heading"] == ""
    # A finding built straight from a view excerpt validates inside a sealed packet.
    excerpt = next(
        s["excerpt"] for s in view["sections"] if "ccc" in s["excerpt"] or sep in s["excerpt"]
    )
    finding = _finding(inv, text=excerpt)
    packet = build_packet(inv, findings=[finding])
    validate_document(packet)


def test_crlf_headings_split_only_on_newline(tmp_path):
    inv = _skill_inventory(tmp_path, "pre\r\n# One\r\naaa\r\nbbb\r\n")
    source = _source(inv, "skill:skills/SKILL.md")
    view = next(v for v in bounded_views(inv) if v["source_id"] == source["id"])
    assert [s["heading"] for s in view["sections"]] == ["", "One"]
    assert view["sections"][1]["excerpt"] == "aaa\r\nbbb\r"
    assert view["sections"][1]["excerpt"] in source["before"]
