#!/usr/bin/env bash
# Installed-wheel end-to-end loop (offline): wheel from a scratch copy -> fresh --without-pip
# venv -> real `nachos-reconcile` on a disposable synthetic profile:
#   inventory -> packet -> session -> [review] -> export -> validate (exit 0)
#   -> mutate one source -> validate (exit 3), asserting nothing else changed.
#
#   bash scripts/loop_wheel.sh [--keep]
#
# LOOP_VIEWER_STEP="cmd args"  replaces the default review step (a viewer-driving command). It
#   runs with LOOP_PACKET, LOOP_SESSION, LOOP_REVIEWED_SESSION (write here), LOOP_WORKDIR,
#   LOOP_PROFILE_HOME and LOOP_VENV_PYTHON in its environment.
# NACHOS_WHEEL_BUILD_PYTHON    interpreter with pip+setuptools for the offline wheel build.
# NACHOS_BROWSER_PYTHON        interpreter running this script (default: python3).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${NACHOS_BROWSER_PYTHON:-python3}"
exec "$PY" "$ROOT/tests/browser/wheel_loop.py" "$@"
