"""Keyboard-only operation and the 375px narrow viewport."""

from __future__ import annotations

import pytest
from viewer_ui import (
    DETAIL,
    PACKET,
    accept_disabled,
    decision_of,
    item_row,
    open_viewer,
    queue_item,
    select,
    wait_decision,
    wait_revision,
)

SKILL = "p-skill-section"
FINDING = "f-skill-section"
MAX_TABS = 200
OVERFLOW = "document.documentElement.scrollWidth - document.documentElement.clientWidth"


def tab_to(vp, selector):
    """Press Tab until the focused element matches ``selector``; assert focus is visible."""
    page = vp.page
    for _ in range(MAX_TABS):
        page.keyboard.press("Tab")
        if page.evaluate("s => !!document.activeElement?.matches(s)", selector):
            outline = page.evaluate(
                "(() => { const s = getComputedStyle(document.activeElement);"
                " return [s.outlineStyle, parseFloat(s.outlineWidth)]; })()"
            )
            assert outline[0] != "none", f"no visible focus ring on {selector}"
            assert outline[1] >= 2
            return
    pytest.fail(f"{selector} not reachable by Tab within {MAX_TABS} presses")


def _action(name):
    return f'{DETAIL} [data-action="{name}"]'


def test_review_is_operable_without_a_mouse(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    vp.goto(viewer_html_path)
    vp.import_file('[data-testid="file-input"]', PACKET)
    page.locator('[data-testid="modal-cancel"]').wait_for(state="visible")
    assert page.evaluate("document.activeElement.id") == "modal-cancel"
    page.keyboard.press("Tab")  # the dialog traps focus and OK is reachable by keyboard
    assert page.evaluate("document.activeElement.id") == "modal-ok"
    page.keyboard.press("Enter")
    page.wait_for_function(
        "document.querySelector('[data-testid=status]').textContent.includes('Loaded')"
    )

    tab_to(vp, f'[data-testid="queue-item"][data-proposal="{SKILL}"]')
    page.keyboard.press("Enter")
    wait_revision(vp, SKILL, 1)
    tab_to(vp, _action("accept"))
    page.keyboard.press("Space")
    wait_decision(vp, SKILL, "accept")

    tab_to(vp, _action("edit"))
    page.keyboard.press("Enter")
    tab_to(vp, f'[data-finding="{FINDING}"] [data-testid="verify-preserved"]')
    page.keyboard.press("Space")
    assert not item_row(vp, FINDING, 0).locator('[data-testid="verify-preserved"]').is_checked()
    tab_to(vp, _action("reverify"))
    page.keyboard.press("Enter")
    wait_revision(vp, SKILL, 2)
    wait_decision(vp, SKILL, "unreviewed")
    assert accept_disabled(vp)

    tab_to(vp, _action("edit"))
    page.keyboard.press("Enter")
    tab_to(vp, f'[data-finding="{FINDING}"] [data-testid="verify-preserved"]')
    page.keyboard.press("Space")
    tab_to(vp, _action("reverify"))
    page.keyboard.press("Enter")
    wait_revision(vp, SKILL, 3)
    assert not accept_disabled(vp)
    tab_to(vp, _action("accept"))
    page.keyboard.press("Space")
    wait_decision(vp, SKILL, "accept")
    assert decision_of(vp, SKILL) == "accept"
    vp.assert_isolated()


def _fits(vp, selector):
    box = vp.page.locator(selector).first.bounding_box()
    assert box is not None, selector
    width = vp.page.evaluate("document.documentElement.clientWidth")
    assert box["x"] >= -1, (selector, box)
    assert box["x"] + box["width"] <= width + 1, (selector, box, width)


def test_narrow_viewport_has_no_horizontal_overflow(narrow_viewer_page, viewer_html_path):
    vp = narrow_viewer_page
    page = vp.page
    assert page.viewport_size["width"] == 375
    vp.goto(viewer_html_path)
    assert page.evaluate(OVERFLOW) <= 1
    open_viewer(vp, viewer_html_path)
    assert page.evaluate(OVERFLOW) <= 1
    queue_item(vp, SKILL).scroll_into_view_if_needed()
    select(vp, SKILL)
    assert page.evaluate(OVERFLOW) <= 1
    for action in ("accept", "edit", "reject", "defer", "request-revision"):
        assert page.locator(_action(action)).is_visible(), action
        _fits(vp, _action(action))
    page.locator(_action("edit")).click()
    assert page.evaluate(OVERFLOW) <= 1
    for testid in ("verify-panel", "verify-item-text", "verify-claim", "verify-evidence-status"):
        _fits(vp, f'[data-testid="{testid}"]')
    page.locator('[data-testid="verify-preserved"]').first.scroll_into_view_if_needed()
    page.locator('[data-testid="verify-preserved"]').first.uncheck()
    page.locator(_action("reverify")).scroll_into_view_if_needed()
    page.locator(_action("reverify")).click()
    wait_revision(vp, SKILL, 2)
    assert page.evaluate(OVERFLOW) <= 1
    assert accept_disabled(vp)
    for hostile in ("p-unassessed", "p-multi"):
        select(vp, hostile)
        assert page.evaluate(OVERFLOW) <= 1, hostile
    vp.assert_isolated()
