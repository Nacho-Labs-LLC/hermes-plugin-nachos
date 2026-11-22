"""Read-only SQLite access that never creates, checkpoints or modifies the source files."""

from __future__ import annotations

import contextlib
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

_SQLITE_MAGIC = b"SQLite format 3\x00"
_SNAPSHOT_ATTEMPTS = 4  # initial attempt + 3 retries


def _needs_wal_snapshot(path: Path) -> bool:
    """True when opening ``path`` directly with ``mode=ro`` could create/mutate WAL sidecars."""
    if any(os.path.lexists(f"{path}{suffix}") for suffix in ("-wal", "-shm")):
        return True
    with path.open("rb") as fh:
        header = fh.read(20)
    return (
        len(header) == 20 and header[:16] == _SQLITE_MAGIC and (header[18] == 2 or header[19] == 2)
    )


def _stat_sig(path: Path, *, required: bool = False) -> tuple[int, int] | None:
    if not required and not os.path.lexists(path):
        return None
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def _copy_wal_snapshot(resolved: Path) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Copy a WAL database and its ``-wal`` sidecar into a private temp dir consistently.

    Records ``(size, mtime_ns)`` of both files before and after copying and retries if either
    changes mid-copy (for example due to a concurrent write or checkpoint). Fails closed if the
    sources do not stabilise.
    """
    wal_src = Path(f"{resolved}-wal")
    for _ in range(_SNAPSHOT_ATTEMPTS):
        before = (_stat_sig(resolved, required=True), _stat_sig(wal_src))
        temp_dir = tempfile.TemporaryDirectory(prefix="nachos-ro-")
        try:
            target = Path(temp_dir.name) / resolved.name
            shutil.copyfile(resolved, target)
            if before[1] is not None:
                shutil.copyfile(wal_src, Path(f"{target}-wal"))
            after = (_stat_sig(resolved, required=True), _stat_sig(wal_src))
            if after == before:
                return target, temp_dir
        except FileNotFoundError:
            if not os.path.lexists(resolved) or os.path.lexists(wal_src):
                temp_dir.cleanup()
                raise
        except Exception:
            temp_dir.cleanup()
            raise
        temp_dir.cleanup()
    raise OSError(
        f"SQLite WAL snapshot did not stabilise after {_SNAPSHOT_ATTEMPTS} attempts: {resolved}"
    )


def _verify_sqlite_copy(conn: sqlite3.Connection, source: Path) -> None:
    rows = [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]
    if rows != ["ok"]:
        detail = "; ".join(rows) or "empty quick_check result"
        raise sqlite3.DatabaseError(f"SQLite snapshot failed quick_check for {source}: {detail}")


def open_read_only_sqlite(
    path: str | os.PathLike[str],
) -> tuple[sqlite3.Connection, tempfile.TemporaryDirectory[str] | None]:
    """Open a SQLite file read-only without creating, checkpointing, or modifying source files.

    Rollback-mode databases are opened directly with ``mode=ro`` and ``PRAGMA query_only = ON``.
    WAL-mode databases (header version 2 or existing ``-wal``/``-shm`` sidecars) are copied along
    with any ``-wal`` file into a private temporary directory with stability retries and verified
    via ``PRAGMA quick_check`` so SQLite can rebuild its transient ``-shm`` index there without
    touching the original files or directory.
    """
    resolved = Path(path).resolve()
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    target = resolved
    try:
        if _needs_wal_snapshot(resolved):
            target, temp_dir = _copy_wal_snapshot(resolved)
        conn = sqlite3.connect(f"{target.as_uri()}?mode=ro", uri=True, check_same_thread=False)
        try:
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA schema_version").fetchone()
            if temp_dir is not None:
                _verify_sqlite_copy(conn, resolved)
        except Exception:
            conn.close()
            raise
    except Exception:
        if temp_dir is not None:
            temp_dir.cleanup()
        raise
    return conn, temp_dir


def close_sqlite(
    conn: sqlite3.Connection, temp_dir: tempfile.TemporaryDirectory[str] | None
) -> None:
    """Close a SQLite connection and clean up any read-only snapshot temp directory."""
    with contextlib.suppress(Exception):
        conn.close()
    if temp_dir is not None:
        with contextlib.suppress(Exception):
            temp_dir.cleanup()
