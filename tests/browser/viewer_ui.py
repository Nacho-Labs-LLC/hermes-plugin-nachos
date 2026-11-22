"""Shared visible-UI helpers for the viewer browser tests (selectors follow viewer.html/ui.js)."""

from __future__ import annotations

import json
from pathlib import Path

from nachos_core.reconciliation import codec

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "viewer"
PACKET = FIXTURES / "packet-mixed.json"
TAG = ".tag"
DETAIL = '[data-testid="detail"]'


def load_packet(path: Path = PACKET) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def queue_item(vp, pid):
    return vp.page.locator(f'[data-testid="queue-item"][data-proposal="{pid}"]')


def queue_ids(vp) -> list[str]:
    return vp.page.locator('[data-testid="queue-item"]').evaluate_all(
        "els => els.map(e => e.dataset.proposal)"
    )


def open_document(vp, path, name_hint):
    """Import a packet/session through the file input and confirm the profile dialog."""
    vp.import_file('[data-testid="file-input"]', path)
    vp.page.locator('[data-testid="modal-ok"]').click()
    vp.page.wait_for_function(
        "t => document.querySelector('[data-testid=status]').textContent.includes(t)", arg=name_hint
    )


def open_viewer(vp, html_path, path=PACKET):
    vp.goto(html_path)
    open_document(vp, path, f"Loaded {Path(path).name}")


def select(vp, pid):
    queue_item(vp, pid).click()
    vp.page.get_by_role("heading", name=f"Proposal {pid} (revision", exact=False).wait_for()


def click(vp, action):
    vp.page.locator(f'{DETAIL} [data-action="{action}"]').click()


def decision_of(vp, pid) -> str:
    return queue_item(vp, pid).locator(TAG).nth(1).inner_text()


def wait_decision(vp, pid, decision):
    vp.page.wait_for_function(
        "([p, d]) => [...document.querySelectorAll('[data-testid=queue-item]')]"
        ".find(b => b.dataset.proposal === p)?.querySelectorAll('.tag')[1]?.textContent === d",
        arg=[pid, decision],
    )


def wait_revision(vp, pid, revision):
    vp.page.get_by_role("heading", name=f"Proposal {pid} (revision {revision})").wait_for()


def error_text(vp) -> str:
    return vp.page.locator('[data-testid="error"]').inner_text()


def save_session(vp) -> dict:
    saved = vp.capture_download(lambda: vp.page.locator('[data-testid="save-session"]').click())
    return codec.loads(saved.read_bytes())


def export_plan(vp) -> tuple[Path, dict]:
    path = vp.capture_download(lambda: vp.page.locator('[data-testid="export-plan"]').click())
    return path, codec.loads(path.read_bytes())


# -- reverify controls -------------------------------------------------------------------------
def finding_panel(vp, fid):
    return vp.page.locator(f'[data-testid="verify-finding"][data-finding="{fid}"]')


def item_row(vp, fid, index=0):
    return finding_panel(vp, fid).locator(f'[data-testid="verify-item"][data-index="{index}"]')


def set_preserved(vp, fid, index, checked):
    box = item_row(vp, fid, index).locator('[data-testid="verify-preserved"]')
    box.set_checked(checked)


def reverify(vp):
    click(vp, "reverify")


def accept_disabled(vp) -> bool:
    return vp.page.locator(f'{DETAIL} [data-action="accept"]').is_disabled()


def fill_verification(vp, fid, *, text="intent kept", explanation="unchanged purpose"):
    """Complete every checklist/evidence row of a finding so reverify can validate."""
    panel = finding_panel(vp, fid)
    for row in panel.locator('[data-testid="verify-item"]').all():
        row.locator('[data-testid="verify-item-text"]').fill(text)
        row.locator('[data-testid="verify-explanation"]').fill(explanation)
    for row in panel.locator('[data-testid="verify-evidence"]').all():
        row.locator('[data-testid="verify-claim"]').fill("claim checked")
        row.locator('[data-testid="verify-observation"]').fill("observed in source")
        sources = row.locator('[data-testid="verify-source"]')
        if sources.count() and not sources.first.is_checked():
            sources.first.check()
