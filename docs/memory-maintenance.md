# Opt-in memory maintenance

`nachos-maintain` is installed with the pip/wheel package. It also runs as
`python -m nachos_hermes.maintenance`. It is a deterministic **report-only**
CLI, not an autonomous memory curator. Installing or running it creates no
cron jobs, entries, approvals, or model requests. To act on its findings see
[reconciliation](reconciliation.md); for the overall design see
[architecture](architecture.md).

## Select exactly one profile

```sh
/path/to/venv/bin/nachos-maintain --profile-home /home/alice/.hermes/profiles/work --json all
/path/to/venv/bin/nachos-maintain --profile-home /home/alice/.hermes/profiles/work --store flatfile summaries
/path/to/venv/bin/nachos-maintain --profile-home /home/alice/.hermes/profiles/work --json links
```

Replace the example paths with your actual interpreter/install and profile
home. `--profile-home` is required and must be absolute (the default profile
can be selected explicitly as `/home/alice/.hermes`). The CLI ignores ambient
`HERMES_HOME`, sticky profile selection, provider configuration, and embedding
credentials. `--store sqlite` is the default; specify `flatfile` when that is
the profile's configured driver. It never guesses between two existing stores.

Only `<profile-home>/nachos/memory.db` (or `memory.md`), its existing
`memory.index.db`, and capture queue filenames are inspected. Resolved paths
escaping the selected profile are rejected. Missing stores are errors, not
empty reports. Shared stores and arbitrary custom store paths are deliberately
outside this CLI's scope; use existing explicit-path tools for those.

### Read-only SQLite boundary

Store and index connections use `mode=ro` and `PRAGMA query_only = ON`, skip
source schema initialization, and never sync vectors or record usage:

- **Rollback-mode databases** without WAL sidecars are opened directly with
  `mode=ro` and `query_only = ON`.
- **WAL-mode databases** (SQLite header format version 2, or any existing
  `-wal` / `-shm` sidecar) cannot be opened in place without SQLite creating or
  updating `-shm` read marks. Instead, `nachos-maintain` copies the database
  file (and its `-wal` file when present) into a private (`0700`) temporary
  directory, verifies that `(size, mtime_ns)` of both source files stayed stable
  across the copy (retrying up to 3 times on concurrent writes or checkpoints),
  opens the copy with `mode=ro` and `query_only = ON`, and runs
  `PRAGMA quick_check` on the copy before reading. If the sources keep changing
  or `quick_check` is not `ok`, it fails closed with an error. SQLite rebuilds
  a transient `-shm` inside the temporary directory and reads any committed,
  un-checkpointed WAL writes while leaving the original database, `-wal`, and
  `-shm` bytes and modification times untouched. The temporary directory is
  deleted when the connection closes.
- An absent index is represented by an in-memory empty index, **not created on
  disk**. The flatfile uses the existing Markdown parser without invoking write
  operations.

`nachos-maintain` never creates, checkpoints, switches journal mode on, or
modifies files in the profile directory. Note that auditing while another
process is actively writing is not a cross-file atomic snapshot across
`memory.db` and `memory.index.db` (and a concurrent checkpoint during a WAL
copy could cause a transient read error); stop writers for a strictly quiescent
audit, and rerun before making approved edits.

## Independent checks

| Check | Existing mechanism reused | Review action |
| --- | --- | --- |
| `inbox` | Index pending-candidate count and capture queue layout/helpers | Review backlog; investigate stuck capture processing. No transcripts are opened, claimed, retried, or extracted. |
| `summaries` | The same `nachos_core.quality.flag_reason` used by `tools/correct_summaries.py` | Inspect low-signal, title-repeating, or vocabulary-mismatched summaries against the full body. |
| `duplicates` | `MemoryService.duplicate_pairs(sync=False)`, embeddings disabled | Read both entries and decide whether they actually describe the same fact. No orphan-vector or telemetry cleanup. |
| `unused` | `MemoryService.unused`, existing usage telemetry | Review never-recalled/never-query-hit entries first seen at least `--days N` ago (default 30). |
| `superseded` | `MemoryService.superseded` | Entries superseded at least `--days N` ago (default 30): candidates to remove once you are sure the successor covers them. |
| `suspicious_supersedes` | `MemoryService.suspicious_supersedes` | Supersede links where the old and new entries differ by a qualifier such as region or team, so the new entry may not replace the old one. Unlink with reconciliation if it is wrong. |
| `provenance` | Index `provenance` table | Entries with no recorded origin. Information only: entries saved before 0.7.0 never have one. |
| `reconciliation` | The deterministic proposal generator used by `nachos-reconcile` | A compact count of proposals (merge, supersede, unlink, rewrite, delete, choose-one) that a reconciliation review would offer. Nothing is written. |
| `links` | `MemoryService.link_report`, straight from the store (no sidecar, no embeddings) | Fix `[[...]]` links that point nowhere (`broken_link`), link or fold entries that many others name but nothing links to (`orphan_hub`), and review entry pairs that strongly share identifiers, hosts, paths or names but are not linked (`unlinked_related`). Entity matching is regex-based and precision-first, so it misses relations that share no identifier; it never edits anything. |
| `all` | Every check above | Monthly review report, not an action plan executed by the CLI. |

`--days` (used by `unused` and `superseded`) must be finite and nonnegative. First-seen age is **not** creation
age or last-modified age. Untracked entries are returned separately in
`untracked_keys`, not labeled unused. Rebuilding/deleting an index loses usage
history. An unused or old entry is not necessarily false or unimportant.

Lexical duplicate checks can miss paraphrases and flag related but distinct
facts; the all-pairs scan is quadratic, so schedule it less often for large
stores. Vocabulary mismatch can indicate a stale summary but does not prove
the body or summary is false. Flatfile bodies contain their summary line, so
vocabulary overlap is less useful there. No check detects factual truth,
conflicts, expired authorizations, or semantic redundancy reliably.

## JSON and automation contract

With `--json`, stdout contains one JSON object. Success has `schema_version: 1`,
`report_only: true`, resolved `profile_home`/`store_path`, `index_available`, and
`checks` containing only the requested checks. Summary, duplicate, unused,
superseded and link checks have `count` and `items` (no hidden truncation). Unused includes `days`,
`telemetry_available`, and `untracked_keys`; superseded includes `days`. Inbox has `pending_candidates`,
`pending_jobs`, and `processing_jobs`. Missing index means
`pending_candidates: null` and `telemetry_available: false`, not evidence of a
clean inbox. Queue counts are separate from extracted candidate counts.

Exit **0** means the report completed, even when findings exist; consumers must
inspect counts and availability. Store/path/schema/read failures return **2**
and an object with `schema_version`, `report_only`, and `error`, without partial
`checks`. Argparse syntax/help follows the normal CLI convention (usage errors
on stderr, exit 2; not JSON). Without `--json`, reports are human-readable.
Reports contain entry keys, categories, and duplicate titles; even without full bodies
or transcripts they may be sensitive. Keep them private and set retention.

## Separate daily / weekly / monthly cron recipes

These are **examples to opt into manually**, not jobs installed by Nachos.
Use ordinary OS cron to avoid spending an agent turn just to execute a report.
Use absolute executable/profile/output paths and the same OS user that owns
the profile. First run each command by hand against your profile. Create a
private report directory once (for example
`mkdir -m 700 /home/alice/nachos-reports`) and choose a retention policy; each
recipe below replaces one report per check, rather than accumulating
transcripts or daily archives. Protect stderr logs as well.

Example crontab entries, independently removable/pausable:

```cron
# Daily: capture backlog. Does not run extraction or approve anything.
10 7 * * * umask 077; /path/to/venv/bin/nachos-maintain --profile-home /home/alice/.hermes/profiles/work --json inbox > /home/alice/nachos-reports/inbox.json 2> /home/alice/nachos-reports/inbox.err
# Weekly: summary maintenance review queue (offline, no corrections).
20 7 * * 1 umask 077; /path/to/venv/bin/nachos-maintain --profile-home /home/alice/.hermes/profiles/work --json summaries > /home/alice/nachos-reports/summaries.json 2> /home/alice/nachos-reports/summaries.err
# Weekly, separately: potential duplicate review.
30 7 * * 1 umask 077; /path/to/venv/bin/nachos-maintain --profile-home /home/alice/.hermes/profiles/work --json duplicates > /home/alice/nachos-reports/duplicates.json 2> /home/alice/nachos-reports/duplicates.err
# Monthly: full inventory of review signals, including unused >=90d.
40 7 1 * * umask 077; /path/to/venv/bin/nachos-maintain --profile-home /home/alice/.hermes/profiles/work --days 90 --json all > /home/alice/nachos-reports/monthly.json 2> /home/alice/nachos-reports/monthly.err
```

Redirection replaces a previous file even on failure, so inspect exit status
and the JSON `error`, not just file existence. If a separate reader requires
atomic publication, wrap the command with a private same-directory temporary
file and rename only on exit 0. Do not combine findings with an automatic
`remove`, `purge-category`, approval, or consolidation step. A report cannot
establish semantic safety. Monitor failures using your scheduler's existing
notification mechanism; Nachos does not provide delivery, locking, or retention.
For slow large-store scans, add your platform's existing timeout/lock wrapper.

## Finish the lifecycle with a human review

1. **Daily intake:** inspect `/nachos-memory-status` in the selected profile.
   Capture already has a background/on-demand processor, retry logic, and
   pending/processing ownership; do not add a competing extraction mechanism.
   It uses `auxiliary.nachos_capture`, potentially network and money. Inspect
   `/nachos-memory-inbox show <id>` before approving/rejecting individual IDs.
   The report CLI never starts capture, approves, or rejects.
2. **Weekly summary review:** compare each flagged summary with its body in an
   authorized primary session; rewrite only reviewed entries through supported
   memory tools. The offline report does not draft replacement prose.
3. **Optional LLM summary correction:** existing source-checkout tool
   `tools/correct_summaries.py --path /absolute/profile/nachos/memory.db --limit 10`
   generates proposed diffs. **Its default dry-run still calls an LLM**; this is
   not the offline report command. Run it in the intended Hermes profile
   environment with `auxiliary.nachos_summary.model` pinned and credentials
   explicitly configured. Review cost/privacy and output first. `--run` writes
   summaries directly to the store and is a separately approved standalone
   maintenance operation, not an unattended default. It does not route through
   the service change log/redactor, so do not assume those edits are revertible
   there; use your approved backup/recovery process and avoid concurrent writers.
   This legacy corrector and `tools/memory_audit.py` remain source-checkout tools;
   the wheel-installed `nachos-maintain` covers their offline audit needs without
   requiring repo-local imports or Hermes itself.
4. **Monthly consolidation:** read full bodies, scope/provenance and recent
   history. Retain distinct facts and historical decisions; identify an explicit
   survivor and proposed merged body only when justified. Ask the owner to
   approve exact edits/deletions. Apply supported tools in an authorized primary
   session, read back changed keys, then rerun reports. Never treat age, zero
   usage, or a duplicate score as permission to delete. No automatic semantic
   deletion, merge, approval, or correction is provided.
5. **Recovery:** existing `/nachos-memory-history` and
   `/nachos-memory-revert <scope:id>` cover service-tracked edits; the sidecar
   retains at most 1000 change records. Existing source audit commands `history`
   and `revert` are also available with explicit paths. Neither the log nor an
   audit report is a full backup. Checkpoint pruning is already owned by the
   provider's `checkpoint_max_age_days`; the toolkit does not duplicate it.

The primary-agent write guard is unchanged. If a scheduled/delegated session
reports `skipped: non-primary agent context`, stop and hand the audit report to
an authorized primary session. Never impersonate a primary provider, change
context flags, or edit SQLite to get around that refusal.

## Source-checkout tools

These scripts live in the repository (not the wheel) and take explicit paths: `--store {sqlite,flatfile}` and `--path` (default `$HERMES_HOME/nachos/memory.db`).

- `tools/memory_audit.py` (`--embeddings`, `--embedding-model`, `--json`, ...): `duplicates`, `unused`, `purge-category NAME`, `history [--key K] [--limit N]`, `revert ID`. `purge-category` and `revert` are dry runs unless `--run`. Nothing changes content otherwise, but opening it can still initialise store and index schema; use `nachos-maintain` when you need a strictly read-only audit.
- `tools/correct_summaries.py [--limit N] [--run]`: generates replacement summaries with `auxiliary.nachos_summary.model`. See step 3 above for its cost and safety limits.
- `tools/import_memories.py`: one-time import; see the [README](../README.md#migrate).

To turn these findings into reviewed, applied fixes see [reconciliation](reconciliation.md).
