"""Bounded rendering of large packets and inert handling of hostile imported text."""

from __future__ import annotations

import sys

from viewer_ui import (
    DETAIL,
    FIXTURES,
    load_packet,
    open_viewer,
    queue_ids,
    select,
)

LARGE = FIXTURES / "packet-large.json"
MAX_ROWS = 500  # ui.js MAX_ROWS
SPARSE = "p-big-sparse"
DENSE = "p-big-dense"


def _oversized_packet(path, count=MAX_ROWS + 20):
    """A packet with more proposals than the queue renders at once (built with the fixture API)."""
    sys.path.insert(0, str(FIXTURES))
    try:
        import build_viewer_fixtures as bvf  # noqa: PLC0415 -- fixture builder
    finally:
        sys.path.remove(str(FIXTURES))
    bvf.PLAIN.append(True)
    try:
        sources, findings, proposals = [], [], []
        for n in range(count):
            key = f"many-{n:04d}"
            src = bvf.make_source(
                f"s-{key}", "memory_record", bvf.DB, bvf.memory(key, n, plain=True),
                identity=f"memory:{key}",
            )  # fmt: skip
            fnd = bvf.make_finding(
                f"f-{key}", "stale", [(src["id"], f"Decision: keep {key} offline.")],
                [bvf.obs("manual", "verified", [src["id"]], "bulk claim")],
            )  # fmt: skip
            after = bvf._edit_memory(src, summary=f"many summary {n}")
            proposals.append(bvf.make_proposal(f"p-{key}", [fnd], [(src, after, "edit")]))
            sources.append(src)
            findings.append(fnd)
        path.write_bytes(bvf.dumps(bvf.packet_of(sources, findings, proposals)))
    finally:
        bvf.PLAIN.clear()
    return path


def test_large_packet_renders_and_falls_back_for_huge_diffs(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    open_viewer(vp, viewer_html_path, LARGE)
    total = len(load_packet(LARGE)["proposals"])
    assert total > 300
    assert len(queue_ids(vp)) == min(total, MAX_ROWS)
    assert (
        f"Total {total} | Filtered {total}" in page.locator('[data-testid="counts"]').inner_text()
    )

    select(vp, "p-000")  # small diff: real comparison
    assert page.locator('[data-testid="diff-unified"]').count() >= 1
    assert page.locator('[data-testid="diff-fallback"]').count() == 0
    for pid in (SPARSE, DENSE):  # > 2000 lines: no comparison, full text shown instead
        select(vp, pid)
        fallback = page.locator('[data-testid="diff-fallback"]')
        fallback.wait_for()
        assert "Diff too large" in fallback.inner_text()
        assert page.locator('[data-testid="diff-unified"]').count() == 0
        assert "L02000" in fallback.inner_text()
        # bounded DOM: the fallback shows text, it does not build per-line diff rows
        assert page.locator(f"{DETAIL} *").count() < 5000
    select(vp, "p-001")  # still responsive afterwards
    assert page.locator(f"{DETAIL} h2").inner_text().startswith("Proposal p-001 ")
    assert page.locator("*").count() < 8000  # queue capped at MAX_ROWS rows
    assert vp.recorder.page_errors == []
    assert [m for m in vp.recorder.console if m[0] == "error"] == []
    vp.assert_isolated()


def test_queue_is_bounded_with_show_all(viewer_page, viewer_html_path, tmp_path):
    vp = viewer_page
    count = MAX_ROWS + 20
    big = _oversized_packet(tmp_path / "packet-many.json", count)
    open_viewer(vp, viewer_html_path, big)
    assert len(queue_ids(vp)) == MAX_ROWS
    counts = vp.page.locator('[data-testid="counts"]').inner_text()
    assert counts == f"Total {count} | Filtered {count}"
    show_all = vp.page.get_by_role("button", name=f"Show all {count}")
    assert show_all.is_visible()
    show_all.click()
    assert len(queue_ids(vp)) == count
    assert vp.page.get_by_role("button", name=f"Show all {count}").count() == 0
    assert (
        vp.page.evaluate("document.activeElement.dataset.testid") == "queue-item"
    )  # focus not dropped to body
    select(vp, f"p-many-{count - 1:04d}")  # a row that was beyond the bound is usable
    vp.assert_isolated()


def test_hostile_text_is_inert_and_literal(viewer_page, viewer_html_path):
    vp = viewer_page
    page = vp.page
    packet = load_packet()
    open_viewer(vp, viewer_html_path)
    scripts_before = page.locator("script").count()
    requests_before = list(vp.recorder.requests)
    shown = {}
    for proposal in packet["proposals"]:
        pid = proposal["id"]
        select(vp, pid)
        heading = page.locator(f"{DETAIL} h2").inner_text()
        assert heading == f"Proposal {pid} (revision {proposal['revision']})"  # literal id
        shown[pid] = page.locator(DETAIL).text_content()
        assert not page.evaluate("typeof window.__pwned !== 'undefined'"), pid
        for tag in ("img", "svg", "iframe", "object", "embed", "link", "style"):
            assert page.locator(f"{DETAIL} {tag}").count() == 0, (pid, tag)
        assert page.locator("a[href^='javascript:']").count() == 0
        assert page.locator("[onerror], [onload]").count() == 0
        assert page.locator("script").count() == scripts_before
    assert "<script>" in shown["p-unassessed"]  # hostile finding id rendered as text
    assert "<b>1</b>" in shown["p-unassessed"]
    everything = "".join(shown.values())
    assert '<img src="https://invalid.example/probe-img"' in everything
    assert "</script><script>window.__pwned=2</script>" in everything
    assert "javascript:window.__pwned=3" in everything
    assert chr(0x202E) in everything  # bidi override is data, not interpreted away
    assert vp.recorder.requests == requests_before
    assert vp.recorder.page_errors == []
    assert [m for m in vp.recorder.console if m[0] == "error"] == []
    vp.assert_isolated()


def test_hostile_proposal_edit_mode_stays_inert(viewer_page, viewer_html_path):
    pid = "p-unassessed"
    vp = viewer_page
    open_viewer(vp, viewer_html_path)
    select(vp, pid)
    vp.page.locator(f'{DETAIL} [data-action="edit"]').click()
    panel = vp.page.locator('[data-testid="verify-finding"]')
    panel.first.wait_for()
    assert "f-hostile-<script>" in panel.first.inner_text()
    assert (
        vp.page.locator("[data-testid=verify-panel] img, [data-testid=verify-panel] script").count()
        == 0
    )
    vp.assert_isolated()
    assert vp.recorder.page_errors == []
