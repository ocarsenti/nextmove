"""Job-extraction log — persists every /jobs/extract call so the signal
vocabulary can eventually be calibrated against real usage, not synthetic
examples.

Separate from research/... conceptually (job descriptions are employer-
authored text about a role, not personal data about an identifiable
candidate) — but still logged deliberately and minimally: the description
text itself is kept (needed to re-verify citations and for future
paraphrase-robustness checks), the detected signals, and a timestamp.
Nothing about who submitted it.

This is intentionally a separate, lighter-weight design than
research/... would be for candidate data: no consent flow, no
participant_code, no withdrawal endpoint — because the privacy calculus
is different (job posting text vs. an individual's self-reported
psychological answers). If job descriptions submitted here ever include
identifiable third-party information (a named hiring manager's email in
the text, for instance), that's a real edge case worth revisiting before
this log is used for anything beyond internal calibration.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "job_extraction_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extractions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    description       TEXT NOT NULL,
    detected_signals  TEXT NOT NULL,   -- JSON list of signal_ids
    axis_requirements TEXT NOT NULL,   -- JSON {axis_id: {level, importance}}
    created_at        TEXT NOT NULL
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


def log_extraction(
    description: str,
    detected_signal_ids: list[str],
    axis_requirements: dict[str, dict],
    db_path: Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO extractions (description, detected_signals, axis_requirements, created_at) "
            "VALUES (?, ?, ?, ?)",
            (description, json.dumps(detected_signal_ids), json.dumps(axis_requirements), _utcnow_iso()),
        )
    return cur.lastrowid


def all_extractions(db_path: Path | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM extractions ORDER BY created_at").fetchall()
    return [
        {
            "id": r["id"],
            "description": r["description"],
            "detected_signals": json.loads(r["detected_signals"]),
            "axis_requirements": json.loads(r["axis_requirements"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]
