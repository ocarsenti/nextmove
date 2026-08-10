"""Archetype calibration log — persists raw_affinity from every COMPLETE
profile so DISPLAY_TEMPERATURE/DISPLAY_FLOOR/DISPLAY_MIN_GAP (see
engine/archetype.py) can eventually be picked from a real distribution of
gaps instead of the 5 synthetic personas used so far.

Privacy calculus (same reasoning style as research/job_extraction_log.py,
but a different call since this IS derived from an individual's personal
questionnaire answers, not employer-authored job text):

- What's stored: only the 5 archetype-level raw_affinity floats (already
  a heavy aggregation of the person's answers — you cannot reconstruct
  their individual answers or even their per-axis scores from this),
  plus dominant/secondary archetype names, low_confidence, context_id,
  and a timestamp.
- What's deliberately NOT stored: no user_id, no session identifier, no
  answers, no per-axis scores. Nothing here can be linked back to a
  specific person or to their other passages (including their own
  retest-study passages in engine/retest_store.py, which use a
  participant_code specifically to allow that linkage under explicit
  consent — this store must never gain a joinable key with that one).
- No consent flow, same as job_extraction_log — the data is anonymous
  and in aggregate by construction, not a candidate's identifiable
  psychological profile. If this reasoning changes (e.g. someone finds a
  way to re-identify from raw_affinity + context_id + timestamp
  triangulation at low volume), revisit before relying on this data.
- Logged for every COMPLETE profile only (not partial in-progress
  answers) — see the is_complete check at the call site in api/main.py.
- Recommended: mention this aggregate, anonymous logging in the
  transparency page (methode.html) alongside the existing retest-study
  disclosure, even though it doesn't need the same opt-in mechanics.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "archetype_calibration_log.db"

# Same guidance as DISPLAY_TEMPERATURE's comment in engine/archetype.py:
# not a hard cutoff, just the point below which picking a new constant
# from this data would be no more principled than the synthetic personas
# it replaces. Shared by /study/archetype-calibration/report and
# scripts/check_calibration_and_alert.py — keep both reading this, not a
# copy of the number, so they can't drift apart.
RECOMMENDED_N = 50

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_affinity_passages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_affinity   TEXT NOT NULL,   -- JSON {archetype_name: float}, pre-normalization (see compute_archetype_distribution_with_raw)
    dominant       TEXT,
    secondary      TEXT,
    low_confidence INTEGER NOT NULL,
    context_id     TEXT,
    created_at     TEXT NOT NULL
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


def log_passage(
    raw_affinity: dict[str, float],
    dominant: Optional[str],
    secondary: Optional[str],
    low_confidence: bool,
    context_id: Optional[str],
    db_path: Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO raw_affinity_passages "
            "(raw_affinity, dominant, secondary, low_confidence, context_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                json.dumps(raw_affinity),
                dominant,
                secondary,
                int(low_confidence),
                context_id,
                _utcnow_iso(),
            ),
        )
    return cur.lastrowid


def all_passages(db_path: Path | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM raw_affinity_passages ORDER BY created_at"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "raw_affinity": json.loads(r["raw_affinity"]),
            "dominant": r["dominant"],
            "secondary": r["secondary"],
            "low_confidence": bool(r["low_confidence"]),
            "context_id": r["context_id"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def count_passages(db_path: Path | None = None) -> int:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM raw_affinity_passages").fetchone()
    return row["n"]
