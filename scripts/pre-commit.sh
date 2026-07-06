#!/usr/bin/env bash
# Nachos pre-commit hook — Python lint stack (ruff + vulture) + tests.
# Tracked in-repo so it travels across machines. Install with:
#     bash scripts/install-hooks.sh
#
# Skip in a pinch with:  git commit --no-verify
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Prefer the repo venv python if present, else system python3.
if [ -x ".venv/bin/python3" ]; then
    PY=".venv/bin/python3"
else
    PY="python3"
fi

# Only lint staged Python files (fast, relevant).
STAGED_PY=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.py$' || true)

echo "── nachos pre-commit ──────────────────────────────────"

# 1. ruff — lint (code smells, unused imports/vars, simplify, bugbear).
if $PY -m ruff --version >/dev/null 2>&1; then
    echo "ruff check…"
    $PY -m ruff check nachos_core plugins tools
else
    echo "  (ruff not installed in venv — skipping; pip install ruff)"
fi

# 2. vulture — dead-code / slop finder (fallow-equivalent).
if $PY -m vulture --version >/dev/null 2>&1; then
    echo "vulture (dead code)…"
    $PY -m vulture nachos_core plugins/memory/nachos/__init__.py tools \
        .vulture_allowlist.py --min-confidence 70
else
    echo "  (vulture not installed in venv — skipping; pip install vulture)"
fi

# 3. tests — the spine has no external deps, runs in <1s.
echo "pytest…"
$PY -m pytest tests/ -q

echo "── pre-commit OK ──────────────────────────────────────"
