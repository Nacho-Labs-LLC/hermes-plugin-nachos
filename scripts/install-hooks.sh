#!/usr/bin/env bash
# Install the tracked nachos git hooks. Run once per checkout / machine.
#     bash scripts/install-hooks.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
ln -sfn "../../scripts/pre-commit.sh" "$ROOT/.git/hooks/pre-commit"
chmod +x "$ROOT/scripts/pre-commit.sh"
echo "Installed pre-commit hook -> scripts/pre-commit.sh"
echo "Runs scripts/check.sh: ruff, vulture, xenon, pylint duplicate-code, pytest."
echo "Bypass with 'git commit --no-verify'."
echo
echo "Install the pinned dev tools (missing tools fail the hook, they are not skipped):"
echo "    uv sync --group dev"
