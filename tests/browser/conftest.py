"""Browser-gate fixtures. See browser_harness for the environment variables.

Gate semantics: with NACHOS_REQUIRE_BROWSER=1 a missing toolchain or viewer FAILS every test
here. Without it tests are skipped with a loud 'BROWSER GATE NOT RUN' reason, never passed.
Tests that need no browser (the wheel loop) carry ``@pytest.mark.no_browser_needed``.
"""

from __future__ import annotations

import functools
import importlib
from pathlib import Path

import pytest
from browser_harness import (
    VIEWPORTS,
    Recorder,
    ViewerPage,
    gate_required,
    toolchain_problem,
    unavailable_reason,
)

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures" / "viewer"
WS_BLOCK = """(() => {
  const blocked = function WebSocket(url) {
    window.__nachosWsAttempt(String(url));
    throw new DOMException('network denied by test harness', 'SecurityError');
  };
  Object.defineProperty(window, 'WebSocket', {value: blocked, configurable: false});
})();"""
_state = {"skipped": 0, "ran": 0}


@functools.lru_cache(maxsize=1)
def _problem() -> str | None:
    return toolchain_problem()


def _in_browser_dir(item) -> bool:
    return HERE in Path(str(item.fspath)).resolve().parents


def _needs_browser(item) -> bool:
    return _in_browser_dir(item) and item.get_closest_marker("no_browser_needed") is None


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    mine = [item for item in items if _needs_browser(item)]
    problem = _problem() if mine else None
    if problem and not gate_required():
        skip = pytest.mark.skip(reason=unavailable_reason(problem))
        for item in mine:
            item.add_marker(skip)
        _state["skipped"] = len(mine)


def pytest_terminal_summary(terminalreporter):
    if _state["skipped"]:
        terminalreporter.section("BROWSER GATE NOT RUN", sep="!")
        terminalreporter.write_line(
            f"{_state['skipped']} browser test(s) were SKIPPED, not passed: {_problem()}"
        )
        terminalreporter.write_line("Set NACHOS_REQUIRE_BROWSER=1 to require the gate.")


@pytest.fixture(autouse=True)
def _browser_gate(request):
    if request.node.get_closest_marker("no_browser_needed") is not None:
        return
    problem = _problem()
    if problem is not None and gate_required():
        pytest.fail(f"BROWSER GATE FAILED (required): {problem}", pytrace=False)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def browser_toolchain():
    """Fails (gate required) or skips (not required) before anything imports playwright."""
    problem = _problem()
    if problem is None:
        return
    if gate_required():
        pytest.fail(f"BROWSER GATE FAILED (required): {problem}", pytrace=False)
    pytest.skip(unavailable_reason(problem))


@pytest.fixture(scope="session")
def playwright_driver(browser_toolchain):  # noqa: ARG001 -- ordering only
    from playwright.sync_api import sync_playwright  # noqa: PLC0415 -- optional dev tool

    with sync_playwright() as driver:
        yield driver


@pytest.fixture(scope="session")
def chromium_browser(playwright_driver):
    import os  # noqa: PLC0415

    args = ["--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost"]
    args += ["--no-sandbox"] if os.environ.get("NACHOS_BROWSER_NO_SANDBOX") == "1" else []
    browser = playwright_driver.chromium.launch(headless=True, args=args)
    yield browser
    browser.close()


@pytest.fixture
def downloads_dir(tmp_path) -> Path:
    path = tmp_path / "downloads"
    path.mkdir()
    return path


@pytest.fixture
def open_viewer_page(chromium_browser, downloads_dir):
    """Factory ``open(viewport='desktop'|'narrow')`` -> ViewerPage; all contexts closed after."""
    made = []

    def open_page(viewport: str = "desktop") -> ViewerPage:
        recorder = Recorder()
        context = chromium_browser.new_context(
            accept_downloads=True,
            service_workers="block",
            viewport=VIEWPORTS[viewport],
            offline=False,
        )

        def route(rt):
            url = rt.request.url
            (rt.continue_ if url.lower().startswith(("file:", "data:", "blob:")) else rt.abort)()

        context.route("**/*", route)

        # WebSocket: the page-side constructor is replaced (records + throws). Playwright's own
        # route_web_socket is avoided: its mock supersedes this script and ws.close() inside a
        # sync handler deadlocks the dispatcher. Chromium's DNS is also dead (launch args).
        def _ws_attempt(url):
            recorder.websockets.append(url)

        context.expose_function("__nachosWsAttempt", _ws_attempt)
        context.add_init_script(WS_BLOCK)
        context.on("request", lambda req: recorder.requests.append(req.url))
        context.on("page", lambda pg: _watch(pg, recorder))
        page = context.new_page()
        made.append(context)
        return ViewerPage(page, context, recorder, downloads_dir, viewport)

    yield open_page
    for context in made:
        context.close()


def _watch(page, recorder: Recorder) -> None:
    page.on("console", lambda msg: recorder.console.append((msg.type, msg.text)))
    page.on("pageerror", lambda err: recorder.page_errors.append(str(err)))


@pytest.fixture
def viewer_page(open_viewer_page) -> ViewerPage:
    return open_viewer_page("desktop")


@pytest.fixture
def narrow_viewer_page(open_viewer_page) -> ViewerPage:
    return open_viewer_page("narrow")


@pytest.fixture(params=sorted(VIEWPORTS))
def any_viewer_page(request, open_viewer_page) -> ViewerPage:
    """Parametrized over the desktop and 375px narrow viewports."""
    return open_viewer_page(request.param)


@pytest.fixture(scope="session")
def viewer_html_path(tmp_path_factory) -> Path:
    """Write ``nachos_core.reconciliation.viewer.viewer_html()`` to a session temp file."""
    try:
        module = importlib.import_module("nachos_core.reconciliation.viewer")
        produce = module.viewer_html
    except (ImportError, AttributeError):
        message = "viewer module not present yet (nachos_core.reconciliation.viewer.viewer_html)"
        if gate_required():
            pytest.fail(f"BROWSER GATE FAILED (required): {message}", pytrace=False)
        pytest.skip(f"{unavailable_reason(message)}")
    html = produce()
    path = tmp_path_factory.mktemp("viewer") / "viewer.html"
    if isinstance(html, Path):
        html = html.read_bytes()
    path.write_bytes(html.encode("utf-8") if isinstance(html, str) else html)
    return path
