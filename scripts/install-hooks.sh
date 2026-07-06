#!/usr/bin/env bash
# Install the tracked nachos git hooks. Run once per checkout / machine.
#     bash scripts/install-hooks.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
ln -sfn "../../scripts/pre-commit.sh" "$ROOT/.git/hooks/pre-commit"
chmod +x "$ROOT/scripts/pre-commit.sh"
echo "Installed pre-commit hook -> scripts/pre-commit.sh"
echo "Lint stack: ruff + vulture + pytest. Bypass with 'git commit --no-verify'."
echo
echo "Ensure the lint tools are in your venv:"
echo "    .venv/bin/python3 -m pip install ruff vulture"
