#!/usr/bin/env bash
# Nachos pre-commit hook: runs the shared quality gate (scripts/check.sh):
# ruff check, ruff format --check, vulture, xenon, pylint duplicate-code, pytest.
# Tracked in-repo so it travels across machines. Install with:
#     bash scripts/install-hooks.sh
#
# Skip in a pinch with:  git commit --no-verify
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "── nachos pre-commit ──────────────────────────────────"
bash scripts/check.sh
echo "── pre-commit OK ──────────────────────────────────────"
