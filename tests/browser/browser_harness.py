"""Toolchain detection and an isolated, network-denied Chromium context for viewer tests.

Environment:
  NACHOS_REQUIRE_BROWSER=1   the gate is mandatory: missing tools FAIL, never skip.
  NACHOS_BROWSER_PYTHON      interpreter that has playwright (used by scripts/browser_gate.sh;
                             this module needs playwright importable in the running one).
  PLAYWRIGHT_BROWSERS_PATH   where `playwright install chromium` put the browsers.
  NACHOS_BROWSER_NO_SANDBOX  set to 1 to pass --no-sandbox (containers running as root).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

GATE_NOT_RUN = "BROWSER GATE NOT RUN"
VIEWPORTS = {"desktop": {"width": 1280, "height": 900}, "narrow": {"width": 375, "height": 667}}
LOCAL_SCHEMES = frozenset({"file", "about", "data", "blob"})
DENIED_URL = re.compile(r"^(https?|wss?|ftp)://", re.IGNORECASE)


def gate_required() -> bool:
    return os.environ.get("NACHOS_REQUIRE_BROWSER") == "1"


def toolchain_problem() -> str | None:
    """None when playwright and a chromium build are usable, else a one-line reason."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415 -- optional dev tool
    except ImportError:
        hint = os.environ.get("NACHOS_BROWSER_PYTHON", "<python with playwright>")
        return f"playwright is not importable in {sys.executable}; run pytest with {hint}"
    paths = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if paths and not Path(paths).is_dir():
        return f"PLAYWRIGHT_BROWSERS_PATH={paths} is not a directory"
    try:
        with sync_playwright() as playwright:
            executable = Path(playwright.chromium.executable_path)
    except Exception as exc:
        return f"playwright driver failed to start: {type(exc).__name__}: {exc}"
    if not executable.exists():
        return (
            f"chromium is not installed ({executable}); run `python -m playwright install chromium`"
        )
    return None


def unavailable_reason(problem: str) -> str:
    return f"{GATE_NOT_RUN}: {problem} (set NACHOS_REQUIRE_BROWSER=1 to make this a failure)"


def is_external(url: str) -> bool:
    scheme = urlsplit(url).scheme.lower()
    return bool(DENIED_URL.match(url)) or scheme not in LOCAL_SCHEMES


@dataclass
class Recorder:
    """Everything the page tried to reach, including requests the harness aborted."""

    requests: list[str] = field(default_factory=list)
    websockets: list[str] = field(default_factory=list)
    console: list[tuple[str, str]] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)

    @property
    def external(self) -> list[str]:
        return [u for u in self.requests if is_external(u)] + list(self.websockets)

    @property
    def non_file(self) -> list[str]:
        """Requests other than file:// loads (data:/blob: included); useful for tight asserts."""
        return [u for u in self.requests if not u.lower().startswith("file:")]


class ViewerPage:
    """A page in an isolated context: file:// only, network denied, downloads captured."""

    def __init__(self, page, context, recorder: Recorder, downloads_dir: Path, viewport: str):
        self.page = page
        self.context = context
        self.recorder = recorder
        self.downloads_dir = downloads_dir
        self.viewport = viewport

    def goto(self, path: Path | str):
        url = Path(path).resolve().as_uri()
        self.page.goto(url)
        return self.page

    def import_file(self, selector: str, path: Path | str) -> None:
        self.page.set_input_files(selector, str(path))

    def import_bytes(self, selector: str, name: str, data: bytes) -> None:
        payload = {"name": name, "mimeType": "application/json", "buffer": data}
        self.page.set_input_files(selector, payload)

    def capture_download(self, action, name: str | None = None) -> Path:
        """Run ``action()`` (e.g. a click), save the resulting download, return its path."""
        with self.page.expect_download() as info:
            action()
        download = info.value
        target = self.downloads_dir / (name or download.suggested_filename or "download.bin")
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        download.save_as(str(target))
        return target

    def canary_fired(self):
        """Whether any hostile fixture payload executed (they all set window.__pwned)."""
        return self.page.evaluate("typeof window.__pwned !== 'undefined'")

    def assert_isolated(self) -> None:
        assert self.recorder.external == [], self.recorder.external
        assert not self.canary_fired(), "hostile fixture markup executed"
