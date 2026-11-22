"""Where a saved memory came from: compact views of the index's provenance rows.

A row is written when an entry is saved (``MemoryService.put``): manual saves record the
session and time; approved capture candidates also carry the supporting quote and, when the
quote could be located in a transcript checkpoint, a ref ``nachos_history_search`` can open.
Rows live in the index (``provenance`` table), so both store backends gain provenance
without any change to the stored entries, and entries saved before this existed simply have none.
"""

from __future__ import annotations

import time
from typing import Any

from .history import make_ref

MANUAL = "manual"
CAPTURE = "capture"
MAX_QUOTE_CHARS = 300


def source_view(prov: dict[str, Any] | None, *, evidence: bool = False) -> dict[str, Any] | None:
    """Compact ``{"session", "date", "ref"}`` (plus ``evidence`` quote on exact reads), or None."""
    if not prov:
        return None
    out: dict[str, Any] = {
        "session": prov.get("session_id") or "",
        "date": time.strftime("%Y-%m-%d", time.gmtime(float(prov.get("captured_at") or 0))),
    }
    if out["session"]:
        out["ref"] = make_ref(out["session"], prov.get("checkpoint_id"), prov.get("message_index"))
    if prov.get("source") == CAPTURE:
        out["via"] = "capture"
        if prov.get("grounded") is False:
            out["ungrounded"] = True
    if evidence and prov.get("quote"):
        out["evidence"] = prov["quote"]
    return out
