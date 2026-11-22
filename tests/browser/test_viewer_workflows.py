"""Visible-UI workflows: request-revision, filters, reverify controls, dirty-edit prompts."""

from __future__ import annotations

from viewer_ui import (
    DETAIL,
    accept_disabled,
    click,
    decision_of,
    error_text,
    export_plan,
    fill_verification,
    finding_panel,
    item_row,
    load_packet,
    open_viewer,
    queue_ids,
    reverify,
    save_session,
    select,
    set_preserved,
    wait_decision,
    wait_revision,
)

SKILL = "p-skill-section"
FINDING = "f-skill-section"


def _expected_ids(packet, kind=None, finding=None):
    kinds = {s["id"]: s["kind"] for s in packet["inventory"]["sources"]}
    types = {f["id"]: f["type"] for f in packet["findings"]}
    out = []
    for proposal in packet["proposals"]:
        if kind and kind not in {kinds[c["source_id"]] for c in proposal["changes"]}:
            continue
        if finding and not any(types.get(f) == finding for f in proposal["finding_ids"]):
            continue
        out.append(proposal["id"])
    return sorted(out)


def test_request_revision_is_visible_saved_and_not_exported(viewer_page, viewer_html_path):
    vp = viewer_page
    open_viewer(vp, viewer_html_path)
    select(vp, SKILL)
    click(vp, "accept")
    wait_decision(vp, SKILL, "accept")
    select(vp, "p-memory")
    vp.page.locator('[data-testid="note"]').fill("please clarify the second item")
    click(vp, "request-revision")
    wait_decision(vp, "p-memory", "request_revision")
    history = vp.page.locator('[data-testid="history"]').inner_text()
    assert "request_revision" in history
    assert "please clarify the second item" in history
    assert decision_of(vp, SKILL) == "accept"  # another proposal's acceptance is untouched
    session = save_session(vp)
    decisions = {r["proposal_id"]: r["decision"] for r in session["reviews"]}
    assert decisions["p-memory"] == "request_revision"
    assert decisions[SKILL] == "accept"
    _, plan = export_plan(vp)
    assert [a["proposal"]["id"] for a in plan["approved"]] == [SKILL]
    vp.assert_isolated()


def test_filters_reduce_combine_and_reset(viewer_page, viewer_html_path):
    vp = viewer_page
    packet = load_packet()
    everything = sorted(p["id"] for p in packet["proposals"])
    open_viewer(vp, viewer_html_path)
    assert sorted(queue_ids(vp)) == everything
    page = vp.page

    page.locator('[data-testid="filter-kind"]').select_option("skill_file")
    assert sorted(queue_ids(vp)) == _expected_ids(packet, kind="skill_file")
    assert _expected_ids(packet, kind="skill_file") != everything

    page.locator('[data-testid="reset-filters"]').click()
    assert sorted(queue_ids(vp)) == everything

    page.locator('[data-testid="filter-finding"]').select_option("id:f-duplicate")
    assert queue_ids(vp) == ["p-multi"]
    page.locator('[data-testid="reset-filters"]').click()

    page.locator('[data-testid="filter-finding"]').select_option("type:clarification")
    clar = _expected_ids(packet, finding="clarification")
    assert sorted(queue_ids(vp)) == clar
    page.locator('[data-testid="filter-kind"]').select_option("memory_file")
    both = sorted(set(clar) & set(_expected_ids(packet, kind="memory_file")))
    assert both == ["p-dep-child"]
    assert queue_ids(vp) == both

    page.locator('[data-testid="reset-filters"]').click()
    select(vp, "p-multi")
    click(vp, "reject")
    wait_decision(vp, "p-multi", "reject")
    page.locator('[data-testid="filter-decision"]').select_option("reject")
    assert queue_ids(vp) == ["p-multi"]
    page.locator('[data-testid="filter-decision"]').select_option("accept")
    assert page.locator('[data-testid="no-results"]').count() == 1
    page.locator('[data-testid="reset-filters"]').click()
    assert sorted(queue_ids(vp)) == everything
    assert page.locator('[data-testid="counts"]').inner_text() == (
        f"Total {len(everything)} | Filtered {len(everything)}"
    )
    vp.assert_isolated()


def test_reverify_controls_gate_acceptance(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    open_viewer(vp, viewer_html_path)
    select(vp, SKILL)
    assert page.locator('[data-testid="verify-panel"]').count() == 0
    assert page.locator(f'{DETAIL} [data-action="reverify"]').is_hidden()
    assert page.locator('[data-testid="edit-assessments"]').count() == 0

    click(vp, "edit")
    assert page.locator(f'{DETAIL} [data-action="reverify"]').is_visible()
    assert item_row(vp, FINDING, 0).locator('[data-testid="verify-preserved"]').is_checked()
    set_preserved(vp, FINDING, 0, False)
    reverify(vp)
    wait_revision(vp, SKILL, 2)
    wait_decision(vp, SKILL, "unreviewed")
    assert accept_disabled(vp)
    assert "not all preserved" in page.locator('[data-testid="accept-reason"]').inner_text()

    click(vp, "edit")
    assert not item_row(vp, FINDING, 0).locator('[data-testid="verify-preserved"]').is_checked()
    set_preserved(vp, FINDING, 0, True)
    reverify(vp)
    wait_revision(vp, SKILL, 3)
    wait_decision(vp, SKILL, "unreviewed")  # reverify never auto-accepts
    assert not accept_disabled(vp)
    click(vp, "accept")
    wait_decision(vp, SKILL, "accept")
    reviews = {r["proposal_id"]: r for r in save_session(vp)["reviews"]}
    assert len(reviews[SKILL]["revisions"]) == 3
    assert reviews[SKILL]["revisions"][-1]["assessments"][0]["revision"] == 3
    vp.assert_isolated()


def test_reverify_validation_error_keeps_state_and_session(viewer_page, viewer_html_path):
    vp = viewer_page
    open_viewer(vp, viewer_html_path)
    select(vp, SKILL)
    before = save_session(vp)
    click(vp, "edit")
    fill_verification(vp, FINDING)
    blank = item_row(vp, FINDING, 0).locator('[data-testid="verify-explanation"]')
    blank.fill("")
    reverify(vp)
    vp.page.wait_for_function(
        "document.querySelector('[data-testid=error]').textContent.length > 0"
    )
    assert "explanation" in error_text(vp)
    assert finding_panel(vp, FINDING).count() == 1  # still editing, buffers kept
    assert blank.input_value() == ""
    assert item_row(vp, FINDING, 0).locator('[data-testid="verify-item-text"]').input_value()
    wait_revision(vp, SKILL, 1)
    assert save_session(vp) == before
    vp.assert_isolated()


def test_dirty_edit_prompts_before_switching(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    open_viewer(vp, viewer_html_path)
    select(vp, SKILL)
    click(vp, "edit")
    assert page.locator('[data-testid="dirty-warning"]').inner_text() == ""
    page.locator('[data-testid="edit-0-text"]').fill("dirty text")
    assert "Unsaved edits" in page.locator('[data-testid="dirty-warning"]').inner_text()
    for action in ("accept", "reject", "defer", "request-revision"):
        assert page.locator(f'{DETAIL} [data-action="{action}"]').is_disabled()

    page.locator('[data-testid="queue-item"][data-proposal="p-memory"]').click()
    assert "Discard unsaved edits?" in page.locator("#modal-title").inner_text()
    page.locator('[data-testid="modal-cancel"]').click()
    wait_revision(vp, SKILL, 1)
    assert page.locator('[data-testid="edit-0-text"]').input_value() == "dirty text"

    page.locator('[data-testid="queue-item"][data-proposal="p-memory"]').click()
    page.locator('[data-testid="modal-ok"]').click()
    wait_revision(vp, "p-memory", 1)
    assert page.locator('[data-testid="verify-panel"]').count() == 0

    click(vp, "edit")  # toggling a verify control also counts as unsaved
    set_preserved(vp, "f-clarification", 0, False)
    assert "Unsaved edits" in page.locator('[data-testid="dirty-warning"]').inner_text()
    vp.assert_isolated()
