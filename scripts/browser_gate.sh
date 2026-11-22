#!/usr/bin/env bash
# Real-browser gate for the reconciliation viewer. Never skips silently: a missing tool, a
# skipped test or an empty run is a hard failure.
#
#   NACHOS_BROWSER_PYTHON=/path/to/python bash scripts/browser_gate.sh
#
# Needs: python with playwright + pytest, a chromium build (`python -m playwright install
# chromium`; set PLAYWRIGHT_BROWSERS_PATH if it lives outside the default cache), and `node`
# when tests/test_viewer_core_node.py exists. For the installed-wheel loop test also set
# NACHOS_WHEEL_BUILD_PYTHON to an interpreter that has pip + setuptools if PY lacks setuptools.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${NACHOS_BROWSER_PYTHON:-python}"
export NACHOS_REQUIRE_BROWSER=1

fail() {
    echo "BROWSER GATE FAILED: $*" >&2
    exit 2
}

command -v "$PY" >/dev/null 2>&1 || fail "interpreter '$PY' not found (set NACHOS_BROWSER_PYTHON)"
"$PY" -c "import playwright, pytest" >/dev/null 2>&1 \
    || fail "'$PY' lacks playwright and/or pytest (pip install --group browser; pip install pytest)"
"$PY" - <<'PYEOF' || fail "chromium is not installed or PLAYWRIGHT_BROWSERS_PATH is wrong (python -m playwright install chromium)"
import pathlib, sys
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    exe = pathlib.Path(p.chromium.executable_path)
    if not exe.exists():
        sys.exit(f"missing {exe}")
    print(f"playwright chromium executable: {exe}")
PYEOF

tests=(tests/browser tests/test_viewer_fixtures.py)
if [ -f tests/test_viewer_core_node.py ]; then
    command -v node >/dev/null 2>&1 || fail "node is required by tests/test_viewer_core_node.py"
    tests+=(tests/test_viewer_core_node.py)
fi

log="$(mktemp "${TMPDIR:-/tmp}/browser-gate.XXXXXX")"
trap 'rm -f "$log"' EXIT
set +e
"$PY" -m pytest -q -rs -p no:cacheprovider "${tests[@]}" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
[ "$status" -eq 0 ] || fail "pytest exited $status"
if grep -Eq '[0-9]+ (skipped|deselected|xfailed)' "$log"; then
    fail "tests were skipped/deselected; the gate requires every test to run"
fi
grep -Eq '[0-9]+ passed' "$log" || fail "no tests passed"
echo "browser gate passed"
