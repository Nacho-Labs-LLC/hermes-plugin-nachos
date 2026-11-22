"""Full review lifecycle through the real viewer using only visible UI actions."""

from __future__ import annotations

from dataclasses import dataclass

from viewer_ui import (
    DETAIL,
    accept_disabled,
    click,
    export_plan,
    fill_verification,
    finding_panel,
    load_packet,
    open_document,
    open_viewer,
    reverify,
    select,
    wait_decision,
    wait_revision,
)

from nachos_core.reconciliation import codec, export

SKILL = "p-skill-section"
FINDING = "f-skill-section"


@dataclass
class Lifecycle:
    vp: object
    sec: dict

    @property
    def page(self):
        return self.vp.page

    def text(self, testid):
        return self.page.locator(f'[data-testid="{testid}"]').inner_text()


def _accept_then_edit_text(lc):
    vp = lc.vp
    select(vp, SKILL)
    click(vp, "accept")
    wait_decision(vp, SKILL, "accept")
    assert "(none)" not in lc.page.locator(f"{DETAIL} dd").nth(1).inner_text()  # digest set
    click(vp, "edit")
    lc.page.locator('[data-testid="edit-0-text"]').fill(lc.sec["changes"][0]["after"] + "\nEDITED")
    click(vp, "save-edit")
    wait_revision(vp, SKILL, 2)
    wait_decision(vp, SKILL, "unreviewed")
    assert "No assessments on this revision" in lc.text("assessments")
    assert accept_disabled(vp)
    assert "Accept disabled" in lc.text("accept-reason")
    assert "edit" in lc.text("history")


def _reverify_and_reaccept(lc):
    vp = lc.vp
    click(vp, "edit")
    assert finding_panel(vp, FINDING).count() == 1
    fill_verification(vp, FINDING)
    reverify(vp)
    wait_revision(vp, SKILL, 3)
    wait_decision(vp, SKILL, "unreviewed")  # never auto-accepted
    assert "Finding" in lc.text("assessments")
    click(vp, "accept")
    wait_decision(vp, SKILL, "accept")


def _defer_and_reject_others(lc):
    select(lc.vp, "p-memory")
    click(lc.vp, "defer")
    wait_decision(lc.vp, "p-memory", "defer")
    select(lc.vp, "p-multi")
    click(lc.vp, "reject")
    wait_decision(lc.vp, "p-multi", "reject")


def _save_reload_reopen(lc):
    vp = lc.vp
    saved = vp.capture_download(lambda: lc.page.locator('[data-testid="save-session"]').click())
    assert saved.name == "reconciliation-session.json"
    session = codec.loads(saved.read_bytes())
    assert {r["proposal_id"]: r["decision"] for r in session["reviews"]}[SKILL] == "accept"
    lc.page.reload()
    open_document(vp, saved, "Loaded reconciliation-session.json")
    wait_decision(vp, SKILL, "accept")
    wait_decision(vp, "p-memory", "defer")
    wait_decision(vp, "p-multi", "reject")
    select(vp, SKILL)
    assert lc.page.get_by_role("heading", name=f"Proposal {SKILL} (revision 3)").is_visible()
    assert "EDITED" in lc.page.locator(DETAIL).inner_text()


def _export_and_validate(lc):
    plan_path, plan = export_plan(lc.vp)
    assert plan_path.name == "reconciliation-plan.json"
    export.validate_plan(plan)
    assert [a["proposal"]["id"] for a in plan["approved"]] == [SKILL]
    assert plan["execution"] == "not_performed"


def test_review_lifecycle(viewer_page, viewer_html_path):
    packet = load_packet()
    sec = next(p for p in packet["proposals"] if p["id"] == SKILL)
    open_viewer(viewer_page, viewer_html_path)
    assert viewer_page.page.locator('[data-testid="queue-item"]').count() == len(
        packet["proposals"]
    )
    lc = Lifecycle(viewer_page, sec)
    _accept_then_edit_text(lc)
    _reverify_and_reaccept(lc)
    _defer_and_reject_others(lc)
    _save_reload_reopen(lc)
    _export_and_validate(lc)
    viewer_page.assert_isolated()
