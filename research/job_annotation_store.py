"""Human annotation storage — for the criterion-validity check (research/
job_annotation_agreement.py).

Deliberately decoupled from any LLM call at annotation time: the person
annotating reads the description and picks signals BLIND, without ever
seeing what the extractor detected — otherwise the comparison would be
worthless (an annotator who sees the LLM's answer first tends to agree
with it, consciously or not). The comparison against the LLM happens
later, at report-generation time, in a separate call.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "job_annotations.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    description        TEXT NOT NULL,
    human_signal_ids   TEXT NOT NULL,   -- JSON list
    annotator_note     TEXT,            -- optional: who/what context, free text
    created_at         TEXT NOT NULL
);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


@contextmanager
def _connect(db_path: Path | None = None):
    resolved = db_path if db_path is not None else DB_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def save_annotation(
    description: str,
    human_signal_ids: list[str],
    annotator_note: str = "",
    db_path: Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO annotations (description, human_signal_ids, annotator_note, created_at) "
            "VALUES (?, ?, ?, ?)",
            (description, json.dumps(human_signal_ids), annotator_note, _utcnow_iso()),
        )
    return cur.lastrowid


def all_annotations(db_path: Path | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM annotations ORDER BY created_at").fetchall()
    return [
        {
            "id": r["id"],
            "description": r["description"],
            "human_signal_ids": json.loads(r["human_signal_ids"]),
            "annotator_note": r["annotator_note"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def delete_annotation(annotation_id: int, db_path: Path | None = None) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
    return cur.rowcount
