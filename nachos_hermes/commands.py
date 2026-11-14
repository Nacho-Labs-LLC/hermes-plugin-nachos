"""Slash-command text rendering and registration for the Nachos Hermes provider.

Renderers are plain functions over a ``MemoryService`` (plus whatever small
facts they need), so they are testable without a provider. ``register_commands``
wires the handlers into a Hermes plugin context.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from nachos_core.history import checkpoint_stats
from nachos_core.memory import PROFILE, MemoryService

UNAVAILABLE = "Nachos memory store unavailable."
INBOX_USAGE = "Usage: /nachos-memory-inbox [show <id> | approve <id|all> | reject <id|all>]"


def status_text(
    service: MemoryService | None,
    cfg: dict[str, Any],
    checkpoint_root: Path | None,
    session_id: str,
    pending: int,
) -> str:
    lines = ["Nachos memory", f"  store       : {cfg['store']}"]
    if not service:
        lines.append("  (store unavailable — see logs)")
        return "\n".join(lines)
    status = service.status()
    emb = status["embeddings"]
    if status["embeddings_error"]:
        emb += f" (disabled: {status['embeddings_error']})"
    lines.append(f"  embeddings  : {emb}")
    for name, s in status["scopes"].items():
        indexed = f", {s['indexed']} embedded" if service.embedder else ""
        lines.append(
            f"  {name:<11} : {s['entries']} entries{indexed}, {s['never_used']} never recalled"
        )
    if checkpoint_root:
        cp = checkpoint_stats(checkpoint_root)
        age = f", oldest {cp['oldest_days']}d" if cp["count"] else ""
        lines.append(
            f"  checkpoints : {cp['count']} across {cp['sessions']} sessions{age} "
            f"(expire after {cfg['checkpoint_max_age_days']}d)"
        )
    lines.append(f"  capture     : {cfg['capture']}, {pending} pending in /nachos-memory-inbox")
    lines.append(f"  session     : {session_id or '-'}")
    changes = ", ".join(f"{n} {s['changes']}" for n, s in status["scopes"].items())
    lines.append(f"  change log  : {changes} (/nachos-memory-history, /nachos-memory-revert)")
    return "\n".join(lines)


def changes_text(service: MemoryService | None, key: str = "") -> str:
    if not service:
        return UNAVAILABLE
    rows = service.history(key=key or None, limit=20)
    if not rows:
        return "No memory changes recorded."
    lines = [
        f"Recent memory changes ({len(rows)}), newest first; revert with "
        "/nachos-memory-revert <scope:id>"
    ]
    for r in rows:
        when = time.strftime("%m-%d %H:%M", time.localtime(r["ts"]))
        sess = f" [{r['session_id'][:8]}]" if r["session_id"] else ""
        prev = f" (was: {r['preview']})" if r["preview"] else ""
        lines.append(f"  {r['scope']}:{r['id']} {when} {r['action']:<6} {r['key']}{sess}{prev}")
    return "\n".join(lines)


def revert_text(service: MemoryService | None, handle: str) -> str:
    if not service:
        return UNAVAILABLE
    scope, sep, raw_id = (handle or "").strip().rpartition(":")
    if not raw_id.isdigit():
        return "Usage: /nachos-memory-revert <scope:id>  (see /nachos-memory-history)"
    res = service.revert(int(raw_id), scope if sep else PROFILE)
    if "error" in res:
        return str(res["error"])
    return f"Reverted {res['undid']} of {res['key']} ({res['scope']}): entry {res['result']}."


def audit_text(service: MemoryService | None, older_than_days: float = 30) -> str:
    if not service:
        return UNAVAILABLE
    pairs = service.duplicate_pairs()
    unused = service.unused(older_than_days)
    lines = [f"Likely duplicates: {len(pairs)}"]
    lines.extend(
        f"  {p['similarity']:.2f} {p['match']:<8} {p['a']['scope']}:{p['a']['key']}"
        f"  ~  {p['b']['scope']}:{p['b']['key']}"
        for p in pairs[:15]
    )
    lines.append("")
    lines.append(f"Never recalled, first seen > {older_than_days:g} days ago: {len(unused)}")
    lines.extend(f"  {row['age_days']:>6}d  {row['scope']}:{row['key']}" for row in unused[:15])
    return "\n".join(lines)


# -- review inbox ------------------------------------------------------------


def _inbox_item(index, target: str) -> dict[str, Any] | None:
    ident = target.lstrip("#")
    return index.get_inbox(int(ident)) if ident.isdigit() else None


def _inbox_hint(dups: list[dict[str, Any]]) -> str:
    if not dups:
        return ""
    top = dups[0]
    return (
        f" [updates {top['key']}]" if top.get("match") == "update" else f" [similar: {top['key']}]"
    )


def _inbox_list(index, queued: int) -> str:
    note = (
        f"\nProcessing {queued} captured session(s) in the background -- run again shortly."
        if queued
        else ""
    )
    rows = index.list_inbox("pending")
    if not rows:
        return "No suggested memories awaiting review." + note
    lines = [f"{len(rows)} suggested memories (/nachos-memory-inbox show|approve|reject <id|all>)"]
    lines.extend(
        f"  #{r['id']} {r['title']}{_inbox_hint(r['duplicates'])}\n      {r['summary']}"
        f"\n      why: {r['reason'] or '-'}"
        for r in rows
    )
    return "\n".join(lines) + note


def _inbox_show(index, target: str) -> str:
    row = _inbox_item(index, target)
    if not row:
        return f"No inbox item {target}."
    dups = ", ".join(f"{d['key']} ({d['match']} {d['similarity']})" for d in row["duplicates"])
    return (
        f"#{row['id']} [{row['status']}] {row['title']}\n"
        f"category: {row['category']}\nsummary: {row['summary']}\n"
        f"why: {row['reason'] or '-'}\nsimilar: {dups or '-'}\n\n{row['body']}"
    )


def _inbox_decide(service: MemoryService, action: str, target: str) -> str:
    index = service.profile.index
    if target.lower() == "all":
        rows = index.list_inbox("pending")
    else:
        row = _inbox_item(index, target)
        rows = [row] if row and row["status"] == "pending" else []
        if not rows:
            return f"No pending inbox item {target}."
    if not rows:
        return "No suggested memories awaiting review."
    return "\n".join(_inbox_apply(service, action, r) for r in rows)


def _inbox_apply(service: MemoryService, action: str, r: dict[str, Any]) -> str:
    index = service.profile.index
    if action == "reject":
        index.set_inbox_status(r["id"], "rejected")
        return f"Rejected #{r['id']} {r['title']}"
    res = service.put(
        title=r["title"],
        body=r["body"],
        summary=r["summary"],
        category=r["category"],
        scope=PROFILE,
    )
    if "error" in res:
        return f"Could not approve #{r['id']} {r['title']}: {res['error']}"
    index.set_inbox_status(r["id"], "approved")
    return f"Approved #{r['id']} -> {res['stored']}{' (updated)' if res.get('updated') else ''}"


def inbox_text(
    service: MemoryService | None, raw_args: str = "", queued: Callable[[], int] = lambda: 0
) -> str:
    """Render ``/nachos-memory-inbox``; ``queued()`` reports capture jobs still processing."""
    if not service:
        return UNAVAILABLE
    parts = (raw_args or "").split()
    action = parts[0].lower() if parts else "list"
    target = parts[1] if len(parts) > 1 else ""
    if action == "list":
        return _inbox_list(service.profile.index, queued())
    if action not in ("show", "approve", "reject"):
        return INBOX_USAGE
    if not target:
        return f"Usage: /nachos-memory-inbox {action} <id{'|all' if action != 'show' else ''}>"
    if action == "show":
        return _inbox_show(service.profile.index, target)
    return _inbox_decide(service, action, target)


# -- registration --------------------------------------------------------------


def _audit_days(raw_args: str) -> float:
    try:
        return float((raw_args or "").strip() or 30)
    except ValueError:
        return 30.0


def register_commands(ctx, provider) -> None:
    """Register the ``/nachos-memory-*`` slash commands for ``provider`` on a plugin context."""
    ctx.register_command(
        "nachos-memory-status",
        lambda _args: provider.status_text(),
        "Nachos memory status — scopes, embeddings, usage, checkpoints",
    )
    ctx.register_command(
        "nachos-memory-list",
        lambda _args: provider.system_prompt_block() or "Nachos memory is empty.",
        "Render the Nachos memory manifest (table of contents)",
    )
    ctx.register_command(
        "nachos-memory-audit",
        lambda args: provider.audit_text(_audit_days(args)),
        "List likely duplicate entries and entries never recalled",
        args_hint="[days]",
    )
    ctx.register_command(
        "nachos-memory-history",
        lambda args: provider.changes_text((args or "").strip()),
        "Recent memory changes (create/update/remove) with revert handles",
        args_hint="[key]",
    )
    ctx.register_command(
        "nachos-memory-revert",
        provider.revert_text,
        "Undo a memory change by its scope:id handle",
        args_hint="<scope:id>",
    )
    ctx.register_command(
        "nachos-memory-inbox",
        provider.inbox_text,
        "Review memories suggested from past sessions (approve or reject)",
        args_hint="[show <id> | approve <id|all> | reject <id|all>]",
    )
