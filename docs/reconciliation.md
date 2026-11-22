# Memory reconciliation

Reconciliation turns the audit signals from [`nachos-maintain`](memory-maintenance.md)
(duplicates, outdated entries, suspicious supersede links, low-quality entries, conflicting
facts) into **reviewable proposals**, lets a human accept or reject them, and applies the
accepted ones through `MemoryService`. Nothing changes memory until a human has approved a
specific proposal and `apply` runs with `--run`. Design context: [architecture](architecture.md).

```
inventory -> packet (findings + proposals) -> session (review) -> plan (accepted only)
                                                                      |
                                    validate (drift check) -> apply (dry run, then --run)
```

Every artifact is a sealed, versioned JSON document (`nachos.reconciliation.<kind>.v1`)
with a content digest. Approval binds an exact proposal revision: an edit or a revision
request revokes it. Imported text is untrusted and rendered inertly.

## Findings and proposals

`nachos-reconcile packet --audit` (and the agent tool) generate them deterministically,
with no LLM and no network:

| Finding | Proposal `kind` | Changes | Confidence |
| --- | --- | --- | --- |
| Near-duplicate pair | `merge` | `edit` the newer entry with the union of lines only the older has, `disable` (remove) the older | high at similarity >= 0.9 or equal titles, else medium/low |
| Newer entry states an update or changed fact over an older one | `supersede` | `supersede` the older (`superseded_by` newer) | high with a changed fact, else medium |
| Supersede link whose old/new differ in a facet qualifier (region, team, ...) | `unlink` | `unlink` the old entry | medium |
| Low-signal summary | `rewrite` | `edit` the summary to the first substantive body line, only if that passes the same checks | medium (low for vocabulary mismatch) |
| Placeholder-like body | `delete` | `disable` | medium |
| Same facet, different values, no update cue | `choose-one` | default: newer wins (`supersede` older); the alternative is to reject | always low |

A finding that cannot be turned into a safe proposal (for example a conflict whose age
order is unknown) is reported as a flag only. Proposals never overlap on a source and
start with no preservation assessments, so none can be accepted before a reviewer
verifies it. `confidence` is a ranking aid, not evidence; each finding lists its limits.

Optional LLM drafting of the merged text is only possible through the Python API
(`generate_findings(..., drafter=...)`) with a caller-supplied function; neither the CLI
nor the tool wires one.

## Schema additions (v1)

Change `operation` is one of `edit`, `disable` (null `after`), `supersede` or `unlink`
(entries only; `after` is exactly `{"superseded_by": "<other key in the
inventory>"}`). A proposal may carry optional `kind` and `confidence`
(`low|medium|high`). The Python validators and the viewer's JS core agree on all of this
(`tests/test_reconciliation_extension_parity.py`).

## Command line

```bash
H=/abs/path/to/profile
nachos-reconcile inventory --profile-home $H --out inv.json
nachos-reconcile packet --inventory inv.json --audit --profile-home $H \
    --observed-at 2026-11-22T10:00:00Z --out packet.json     # outputs go outside the profile
nachos-reconcile session --packet packet.json --out session.json
nachos-reconcile viewer --out viewer.html                    # open offline, import session.json
# review in the viewer, export the plan, then:
nachos-reconcile validate --plan plan.json --profile-home $H --out validation.json
nachos-reconcile apply --plan plan.json --profile-home $H --out dry.json            # dry run
nachos-reconcile apply --plan plan.json --profile-home $H --run --allow-remove --out applied.json
```

Exit codes: 0 ok, 2 error, 3 stale or refused, 4 unavailable, 5 partial failure.

### `apply`

* Re-validates the plan against the **current** sources and refuses on any drift
  (`source_status: stale`), then preflights every change; one refusal and nothing is
  written. Each entry is re-checked again right before its write.
* Writes go through `MemoryService`, so Hermes' secret redaction (required unless
  `--allow-unredacted`), the write-time quality gate, the change log and its 1000-record
  window all apply. A text changed by redaction is reported (`redacted: true`).
* `edit` keeps the entry's key (the title may change without moving the key). `supersede`
  and `unlink` refuse self links, existing links and cycles.
* Removal (`disable`, from merges and deletes) needs `--allow-remove`.
* Entries only. Skill and reference file changes stay with the primary session.
* Each action reports a `change_id`; undo with `/nachos-memory-revert` or
  `tools/memory_audit.py revert ID`.

Source reading shares the WAL-safe read-only snapshot used by `nachos-maintain`: the
original `.db`, `-wal` and `-shm` files are never created, checkpointed or modified.

## Agent flow (Hermes)

The provider exposes `nachos_memory_reconcile` with `action` = `review | status | approve |
apply`, and ships the companion skill `nachos-reconcile` (`nachos-reconcile skill --out FILE`).

1. `review` audits the store and writes `packet.json`, `session.json` and `viewer.html`
   into `<profile>/nachos/reconcile/<review_id>/` (mode 0600), returning short proposal
   summaries. Nothing is changed.
2. The agent shows them to the user. The user either approves by naming proposal ids in
   chat, or opens `viewer.html`, imports `session.json`, reviews with full evidence and
   exports a plan.
3. `approve` (chat path) needs `confirm_user_approved=true` and explicit `proposal_ids`; it
   records the approval as manual evidence stating it came from the user's instruction and
   exports `plan.json`. `apply` accepts that plan or a viewer-exported one via `plan_path`.
4. `apply` is a dry run unless `run=true`; removals need `allow_remove=true`.

Everything except `status` is primary-agent only (`skipped: non-primary agent context`
otherwise). The tool cannot prove a human is present: the chat approval path trusts the
agent to relay the user's words, so it is the convenient path and the viewer is the
careful one. Plans are re-validated against the live store at apply time either way, and
every change is undoable.

## Non-goals

No scheduling, no implicit network or LLM use, no multi-profile review, no skill-file
writes, no apply for unaccepted or blocked proposals.
