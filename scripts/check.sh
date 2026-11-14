#!/usr/bin/env bash
# One-shot code-quality gate, shared by the pre-commit hook and CI:
#   ruff check -> ruff format --check -> mypy -> vulture -> xenon -> pylint duplicate-code -> pytest
# Config lives in pyproject.toml ([tool.ruff], [tool.mypy], [tool.vulture], [tool.pylint]); xenon's
# thresholds are the flags below (xenon has no config file).
#
#   bash scripts/check.sh            # run every check, stop at the first failure
#
# Install the pinned tools with:  uv sync --group dev   (or: pip install --group dev)
# Uses .venv/bin/python3 when present, else python3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -x ".venv/bin/python3" ]; then
    PY=".venv/bin/python3"
else
    PY="python3"
fi

SRC_DIRS=(nachos_core nachos_hermes plugins tools)
LINT_PATHS=(__init__.py nachos_core nachos_hermes plugins tests tools)

# Missing tools are a hard failure, never a silent skip.
missing=()
for tool in ruff mypy vulture radon xenon pylint pytest; do
    "$PY" -m "$tool" --version >/dev/null 2>&1 || missing+=("$tool")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: missing dev tools: ${missing[*]}" >&2
    echo "  Install the pinned set:  uv sync --group dev" >&2
    echo "  or:                      $PY -m pip install --group dev   (pip >= 25.1)" >&2
    exit 1
fi

step() { echo; echo "── $* ──"; }

step "ruff check (lint)"
"$PY" -m ruff check "${LINT_PATHS[@]}"

step "ruff format --check"
"$PY" -m ruff format --check "${LINT_PATHS[@]}"

step "mypy (type check; config in [tool.mypy], python_version 3.9)"
"$PY" -m mypy

step "vulture (dead code, min_confidence from pyproject)"
"$PY" -m vulture

step "radon (report: blocks rated C or worse, average complexity, modules below B)"
"$PY" -m radon cc -nc "${SRC_DIRS[@]}"
"$PY" -m radon cc -a "${SRC_DIRS[@]}" | tail -n 2
"$PY" -m radon mi -n B "${SRC_DIRS[@]}"

step "xenon (gate: no block worse than C, no module worse than B, average A)"
"$PY" -m xenon --max-absolute C --max-modules B --max-average A "${SRC_DIRS[@]}"

step "pylint duplicate-code (min-similarity-lines from pyproject)"
"$PY" -m pylint --disable=all --enable=duplicate-code "${SRC_DIRS[@]}" tests

step "pytest"
"$PY" -m pytest tests -q

echo
echo "── all checks passed ──"
