"""Drop-in compatibility shim for the packaged Nachos memory provider."""

import sys
from pathlib import Path

# A legacy directory install exposes this directory but not the repository
# root. Resolve the symlinked source path so the packaged implementation and
# nachos_core remain importable in that supported compatibility mode.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nachos_hermes.memory_provider import (  # noqa: E402, F401
    NachosMemoryProvider,
    register,
)