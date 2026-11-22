"""Sidecar index schema: table definitions and in-place migrations for older index files.

Everything in the index is derived or telemetry, so a migration may rebuild a table whose
contents the next sync recreates (``vectors``) but must preserve the others.
"""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    key          TEXT NOT NULL,
    chunk_no     INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    vec          BLOB NOT NULL,
    PRIMARY KEY (key, chunk_no, model)
);
CREATE TABLE IF NOT EXISTS usage (
    key          TEXT PRIMARY KEY,
    first_seen   REAL NOT NULL,
    recall_count INTEGER NOT NULL DEFAULT 0,
    hit_count    INTEGER NOT NULL DEFAULT 0,
    last_used    REAL
);
CREATE TABLE IF NOT EXISTS changes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 REAL NOT NULL,
    key                TEXT NOT NULL,
    action             TEXT NOT NULL,
    session_id         TEXT,
    prev_title         TEXT,
    prev_summary       TEXT,
    prev_category      TEXT,
    prev_body          TEXT,
    prev_superseded_by TEXT
);
CREATE INDEX IF NOT EXISTS changes_key ON changes (key);
CREATE TABLE IF NOT EXISTS calibration (
    model      TEXT PRIMARY KEY,
    head_count INTEGER NOT NULL,
    profile    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS links (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    pos INTEGER NOT NULL,
    PRIMARY KEY (src, dst)
);
CREATE INDEX IF NOT EXISTS links_dst ON links (dst);
CREATE TABLE IF NOT EXISTS inbox (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts      REAL NOT NULL,
    session_id      TEXT,
    title           TEXT NOT NULL,
    summary         TEXT NOT NULL,
    category        TEXT NOT NULL,
    body            TEXT NOT NULL,
    reason          TEXT,
    duplicates_json TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS provenance (
    key           TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL DEFAULT '',
    captured_at   REAL NOT NULL,
    source        TEXT NOT NULL DEFAULT 'manual',
    checkpoint_id TEXT,
    message_index INTEGER,
    quote         TEXT,
    grounded      INTEGER
);
"""

INBOX_COLS = (
    "id",
    "created_ts",
    "session_id",
    "title",
    "summary",
    "category",
    "body",
    "reason",
    "duplicates_json",
    "status",
    "quote",
    "grounded",
    "checkpoint_id",
    "message_index",
)
#: Columns added after the first release; ``migrate`` adds them to old files.
INBOX_EXTRA = (
    ("quote", "TEXT"),
    ("grounded", "INTEGER"),
    ("checkpoint_id", "TEXT"),
    ("message_index", "INTEGER"),
)

_VECTORS_DDL = """
CREATE TABLE vectors (
    key          TEXT NOT NULL,
    chunk_no     INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    vec          BLOB NOT NULL,
    PRIMARY KEY (key, chunk_no, model)
);
"""


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def existing_inbox_cols(conn: sqlite3.Connection) -> tuple[str, ...]:
    have = table_columns(conn, "inbox")
    return tuple(c for c in INBOX_COLS if c in have)


def create_and_migrate(conn: sqlite3.Connection) -> None:
    """Create missing tables, then bring older files up to the current layout."""
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate_vectors(conn)
    _migrate_changes(conn)
    _migrate_inbox(conn)


def _migrate_changes(conn: sqlite3.Connection) -> None:
    cols = table_columns(conn, "changes")
    if cols and "prev_superseded_by" not in cols:
        conn.execute("ALTER TABLE changes ADD COLUMN prev_superseded_by TEXT")
        conn.commit()


def _migrate_inbox(conn: sqlite3.Connection) -> None:
    cols = table_columns(conn, "inbox")
    for name, decl in INBOX_EXTRA:
        if name not in cols:
            conn.execute(f"ALTER TABLE inbox ADD COLUMN {name} {decl}")
    conn.commit()


def _migrate_vectors(conn: sqlite3.Connection) -> None:
    cols = table_columns(conn, "vectors")
    if not cols:
        return
    if "chunk_no" in cols:
        _add_model_to_vectors_key(conn)
        return
    # Pre-chunking schema had PRIMARY KEY (key) without chunk_no. Vectors are derived
    # data, so recreating the table lets lazy sync rebuild chunked vectors automatically.
    conn.executescript("DROP TABLE vectors;" + _VECTORS_DDL)
    conn.commit()


def _add_model_to_vectors_key(conn: sqlite3.Connection) -> None:
    """Add ``model`` to the primary key so a model switch keeps the old vectors until done.

    Existing rows are copied over unchanged; the table is rebuilt only for databases that
    still have the ``(key, chunk_no)`` key.
    """
    pk_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(vectors)") if row[5]}
    if "model" in pk_columns:
        return
    conn.executescript(
        "ALTER TABLE vectors RENAME TO vectors_old;"
        + _VECTORS_DDL
        + "INSERT INTO vectors SELECT key, chunk_no, content_hash, model, vec FROM vectors_old;"
        "DROP TABLE vectors_old;"
    )
    conn.commit()
