"""Regressions: verify-panel focus, unlisted evidence sources, modal inertness, discarded results."""

from __future__ import annotations

from viewer_ui import (
    PACKET,
    click,
    finding_panel,
    item_row,
    open_document,
    open_viewer,
    select,
    wait_decision,
)

MEMORY = "p-memory"
FINDING = "f-clarification"
ACTIVE = "document.activeElement"


def _patch_core(vp, body):
    """Before load, swap the frozen NachosCore for a mutable copy that ``body(c)`` may wrap."""
    vp.page.add_init_script(
        "(() => { let real; Object.defineProperty(window, 'NachosCore', { configurable: true,"
        " get() { return real; }, set(v) { real = { ...v }; (" + body + ")(real); } }); })()"
    )


def _enter_edit(vp, pid=MEMORY):
    select(vp, pid)
    click(vp, "edit")
    finding_panel(vp, FINDING).wait_for()


def _focus_testid(vp) -> str:
    return vp.page.evaluate(f"{ACTIVE}?.dataset?.testid || ''")


def test_focus_after_adding_and_removing_verify_rows(viewer_page, viewer_html_path):
    vp = viewer_page
    open_viewer(vp, viewer_html_path)
    _enter_edit(vp)
    panel = finding_panel(vp, FINDING)
    panel.locator('[data-testid="verify-add-item"]').click()
    assert panel.locator('[data-testid="verify-item"]').count() == 3
    assert _focus_testid(vp) == "verify-item-text"
    assert vp.page.evaluate(f"{ACTIVE}.closest('[data-testid=verify-item]').dataset.index") == "2"

    item_row(vp, FINDING, 1).locator('[data-testid="verify-remove-item"]').click()
    assert _focus_testid(vp) == "verify-remove-item"  # stays in the list, not on body
    assert panel.locator('[data-testid="verify-item"]').count() == 2

    panel.locator('[data-testid="verify-add-evidence"]').click()
    rows = panel.locator('[data-testid="verify-evidence"]')
    assert rows.count() == 2
    assert _focus_testid(vp) == "verify-claim"
    assert (
        vp.page.evaluate(f"{ACTIVE}.closest('[data-testid=verify-evidence]').dataset.index") == "1"
    )
    rows.nth(1).locator('[data-testid="verify-remove-evidence"]').click()
    assert rows.count() == 1
    assert _focus_testid(vp) == "verify-add-evidence"  # last row's remove is disabled


def test_unlisted_evidence_sources_are_visible_checkboxes(viewer_page, viewer_html_path):
    vp = viewer_page
    # Core validation rejects such evidence, so inject it after validation, in memory only.
    _patch_core(
        vp,
        "c => { const orig = c.newSession; c.newSession = async (...a) => {"
        " const s = await orig(...a);"
        " const rec = s.reviews.find(r => r.proposal_id === 'p-memory');"
        " const last = rec.revisions[rec.revisions.length - 1];"
        " last.assessments[0].evidence[0].source_ids.push('s-extra-1'); return s; }; }",
    )
    vp.goto(viewer_html_path)
    open_document(vp, PACKET, "Loaded packet-mixed.json")
    _enter_edit(vp)
    boxes = finding_panel(vp, FINDING).locator('[data-testid="verify-source"]')
    values = boxes.evaluate_all("els => els.map(e => [e.value, e.checked])")
    assert values == [["s-mem-1", True], ["s-extra-1", True]]
    for index in range(boxes.count()):
        assert boxes.nth(index).is_visible()
    boxes.nth(1).uncheck()
    boxes.nth(1).check()
    assert boxes.evaluate_all("els => els.map(e => e.checked)") == [True, True]


def test_unlisted_evidence_sources_round_trip_into_the_saved_revision(
    viewer_page, viewer_html_path
):
    vp = viewer_page
    page = vp.page
    _patch_core(
        vp,
        "c => { const ns = c.newSession; c.newSession = async (...a) => {"
        " const s = await ns(...a);"
        " const rec = s.reviews.find(r => r.proposal_id === 'p-memory');"
        " rec.revisions[rec.revisions.length - 1].assessments[0].evidence[0]"
        ".source_ids.push('s-extra-1'); return s; };"
        " const rp = c.reviewProposal; c.reviewProposal = async (...a) => {"
        " window.__sent = a[3] && a[3].editedProposal; return rp(...a); }; }",
    )
    vp.goto(viewer_html_path)
    open_document(vp, PACKET, "Loaded packet-mixed.json")
    _enter_edit(vp)
    click(vp, "reverify")
    page.wait_for_function("() => !!window.__sent")
    sent = page.evaluate("window.__sent.assessments[0].evidence[0].source_ids")
    assert sent == ["s-mem-1", "s-extra-1"]


def _open_profile_dialog(vp, html_path):
    vp.goto(html_path)
    vp.import_file('[data-testid="file-input"]', PACKET)
    vp.page.locator('[data-testid="modal-cancel"]').wait_for(state="visible")


def _app_state(vp):
    return vp.page.evaluate(
        "(() => { const a = document.getElementById('app');"
        " return [a.inert, a.getAttribute('aria-hidden'),"
        " document.getElementById('modal').hidden]; })()"
    )


def test_modal_makes_background_inert_and_traps_tab(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    _open_profile_dialog(vp, viewer_html_path)
    assert _app_state(vp) == [True, "true", False]
    assert page.evaluate(f"{ACTIVE}.id") == "modal-cancel"
    page.keyboard.press("Tab")
    assert page.evaluate(f"{ACTIVE}.id") == "modal-ok"
    page.keyboard.press("Tab")  # wraps forward
    assert page.evaluate(f"{ACTIVE}.id") == "modal-cancel"
    page.keyboard.press("Shift+Tab")
    assert page.evaluate(f"{ACTIVE}.id") == "modal-ok"
    page.keyboard.press("Shift+Tab")  # wraps backward
    assert page.evaluate(f"{ACTIVE}.id") == "modal-cancel"
    page.keyboard.press("Escape")
    page.locator("#modal").wait_for(state="hidden")
    assert _app_state(vp) == [False, None, True]
    assert page.locator('[data-testid="queue-item"]').count() == 0  # cancelled: nothing loaded


def test_escape_restores_focus_to_the_opener(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    open_viewer(vp, viewer_html_path)
    select(vp, MEMORY)
    click(vp, "edit")
    finding_panel(vp, FINDING).wait_for()
    item_row(vp, FINDING, 0).locator('[data-testid="verify-item-text"]').fill("changed")
    other = page.locator('[data-testid="queue-item"][data-proposal="p-skill-section"]')
    other.focus()
    page.keyboard.press("Enter")  # dirty edit -> discard prompt
    page.locator("#modal").wait_for(state="visible")
    assert _app_state(vp)[:2] == [True, "true"]
    page.keyboard.press("Escape")
    page.locator("#modal").wait_for(state="hidden")
    assert _app_state(vp) == [False, None, True]
    assert page.evaluate(f"{ACTIVE}.dataset.proposal") == "p-skill-section"
    page.get_by_role("heading", name=f"Proposal {MEMORY} (revision", exact=False).wait_for()


def test_discarded_action_result_is_announced(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    _patch_core(
        vp,
        "c => { const orig = c.reviewProposal; c.reviewProposal = async (...a) => {"
        " const r = await orig(...a);"
        " await new Promise(res => { window.__release = res; }); return r; }; }",
    )
    open_viewer(vp, viewer_html_path)
    select(vp, MEMORY)
    click(vp, "reject")
    page.wait_for_function("() => !!window.__release")
    page.locator('[data-testid="queue-item"][data-proposal="p-skill-section"]').click()
    page.get_by_role("heading", name="Proposal p-skill-section (revision", exact=False).wait_for()
    page.evaluate("window.__release()")
    status = page.locator('[data-testid="status"]')
    page.wait_for_function(
        "() => document.querySelector('[data-testid=status]').textContent.includes('discarded')"
    )
    text = status.inner_text()
    assert MEMORY in text
    assert "selection changed" in text
    page.get_by_role("heading", name="Proposal p-skill-section (revision", exact=False).wait_for()
    assert (
        page.locator('[data-testid="queue-item"][data-proposal="p-memory"] .tag')
        .nth(1)
        .inner_text()
        == "unreviewed"
    )


def test_failed_action_after_selection_change_is_announced(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    _patch_core(
        vp,
        "c => { c.reviewProposal = async () => {"
        " await new Promise(res => { window.__release = res; });"
        " throw new Error('boom'); }; }",
    )
    open_viewer(vp, viewer_html_path)
    select(vp, MEMORY)
    click(vp, "reject")
    page.wait_for_function("() => !!window.__release")
    page.locator('[data-testid="queue-item"][data-proposal="p-skill-section"]').click()
    page.get_by_role("heading", name="Proposal p-skill-section (revision", exact=False).wait_for()
    page.evaluate("window.__release()")
    page.wait_for_function(
        "() => document.querySelector('[data-testid=status]').textContent.includes('context changed')"
    )
    assert "boom" in page.locator('[data-testid="status"]').inner_text()


def test_escape_still_cancels_after_a_backdrop_click(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    _open_profile_dialog(vp, viewer_html_path)
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("Escape")
    page.locator("#modal").wait_for(state="hidden")
    assert _app_state(vp) == [False, None, True]


def _active_desc(vp) -> str:
    return vp.page.evaluate(
        f"(() => {{ const a = {ACTIVE}; return a === document.body ? 'BODY' :"
        " (a.dataset.testid || a.id || a.tagName) + (a.dataset.proposal ? ':' + a.dataset.proposal : ''); })()"
    )


def test_focus_after_keyboard_selecting_a_proposal(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    open_viewer(vp, viewer_html_path)
    other = page.locator('[data-testid="queue-item"][data-proposal="p-skill-section"]')
    other.focus()
    page.keyboard.press("Enter")
    page.get_by_role("heading", name="Proposal p-skill-section (revision", exact=False).wait_for()
    assert _active_desc(vp) == "queue-item:p-skill-section"
    assert page.evaluate(f"{ACTIVE}.getAttribute('aria-current')") == "true"


def test_focus_after_edit_and_cancel_edit(viewer_page, viewer_html_path):
    vp = viewer_page
    open_viewer(vp, viewer_html_path)
    select(vp, MEMORY)
    click(vp, "edit")
    finding_panel(vp, FINDING).wait_for()
    assert vp.page.evaluate(f"!!{ACTIVE}.closest('[data-testid=verify-panel]')")
    assert _active_desc(vp) != "BODY"
    click(vp, "cancel-edit")
    assert _active_desc(vp) == "edit"


def test_focus_after_review_action_stays_in_actions(viewer_page, viewer_html_path):
    vp = viewer_page
    open_viewer(vp, viewer_html_path)
    select(vp, MEMORY)
    click(vp, "defer")
    wait_decision(vp, MEMORY, "defer")
    assert _active_desc(vp) == "defer"
    click(vp, "reject")
    wait_decision(vp, MEMORY, "reject")
    assert _active_desc(vp) == "reject"


def test_tab_cycle_with_three_dialog_controls(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    _open_profile_dialog(vp, viewer_html_path)
    # TEST-ONLY fixture: a third focusable control placed in the dialog after it opens.
    page.evaluate(
        "(() => { const b = document.createElement('button'); b.id = 'modal-extra';"
        " b.textContent = 'extra'; document.getElementById('modal').appendChild(b); })()"
    )
    order = page.evaluate("Array.from(document.querySelectorAll('#modal button')).map(b => b.id)")
    assert len(order) >= 3
    page.evaluate(f"document.getElementById('{order[0]}').focus()")
    for expected in [*order[1:], order[0]]:  # forward, wrapping last -> first
        page.keyboard.press("Tab")
        assert page.evaluate(f"{ACTIVE}.id") == expected
    for expected in [order[-1], *order[-2::-1]]:  # reverse, wrapping first -> last
        page.keyboard.press("Shift+Tab")
        assert page.evaluate(f"{ACTIVE}.id") == expected
    page.keyboard.press("Escape")
    page.locator("#modal").wait_for(state="hidden")


def test_second_dialog_unreachable_while_modal_open(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    # Invariant proved: while the modal is open #app is inert and aria-hidden, so no input can start
    # a second dialog; the production `!m.hidden` guard in confirmDialog is defensive only (not tested).
    _open_profile_dialog(vp, viewer_html_path)
    assert _app_state(vp) == [True, "true", False]
    page.evaluate(
        "document.activeElement && document.activeElement.blur()"
    )  # Enter must not hit Cancel
    box = page.locator("#app").bounding_box()
    page.mouse.click(box["x"] + 5, box["y"] + 5)  # lands on the modal backdrop / inert app
    page.keyboard.press("Enter")
    assert page.locator("#modal").is_visible()
    assert page.locator("#modal-title").count() == 1
    assert _app_state(vp) == [True, "true", False]
    assert page.locator('[data-testid="queue-item"]').count() == 0
    page.keyboard.press("Escape")
    page.locator("#modal").wait_for(state="hidden")
