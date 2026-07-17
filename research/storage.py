"""Research persistence layer — for the reliability/validation study, not the
product itself.

Deliberately separate from any future product database: a different SQLite
file, a different consent (participant_code, never an email or account id),
and it stores only what's needed to recompute a Profile later — raw answers
+ the ontology version in effect when they were given — never derived scores.
Recomputing from raw answers means a future scoring-logic change never makes
stored data silently stale or inconsistent with itself (see the "store raw
answers, not derived Profile" principle discussed for this study).

Two tables:
    participants(participant_code, consent_given, consent_date, created_at)
    submissions(id, participant_code, registry_version, context_id,
                answers_json, submitted_at)

participant_code is a random token generated server-side and handed back to
the client to keep locally (e.g. localStorage) — it lets a retest be paired
with a first submission without ever holding a name, email, or account id.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "reliability_study.db"


def _utcnow_iso() -> str:
    """Naive UTC ISO string — deliberately timezone-naive to stay comparable
    with SQLite's own datetime('now') (also naive UTC), used both in tests
    and for any future SQL-side date arithmetic."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    participant_code TEXT PRIMARY KEY,
    consent_given INTEGER NOT NULL DEFAULT 1,
    consent_date TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_code TEXT NOT NULL REFERENCES participants(participant_code),
    registry_version INTEGER NOT NULL,
    context_id TEXT,
    answers_json TEXT NOT NULL,
    submitted_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_submissions_participant ON submissions(participant_code);
"""


@contextmanager
def _connect(db_path: Path | None = None):
    resolved = db_path if db_path is not None else DB_PATH
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


def create_participant(db_path: Path | None = None) -> str:
    """New research participant, distinct from any product account. Returns
    the participant_code the caller must keep (e.g. localStorage) to pair a
    future retest with this one — nothing else identifies this person."""
    code = secrets.token_urlsafe(16)
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO participants (participant_code, consent_given, consent_date, created_at) VALUES (?, 1, ?, ?)",
            (code, now, now),
        )
    return code


def participant_exists(participant_code: str, db_path: Path | None = None) -> bool:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM participants WHERE participant_code = ?", (participant_code,)
        ).fetchone()
    return row is not None


def withdraw_consent(participant_code: str, db_path: Path | None = None) -> int:
    """Right to withdraw: deletes this participant's submissions and record entirely."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM submissions WHERE participant_code = ?", (participant_code,))
        conn.execute("DELETE FROM participants WHERE participant_code = ?", (participant_code,))
    return cur.rowcount


def save_submission(
    participant_code: str,
    registry_version: int,
    answers: dict[str, str],
    context_id: str | None = None,
    db_path: Path | None = None,
) -> int:
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO submissions (participant_code, registry_version, context_id, answers_json, submitted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (participant_code, registry_version, context_id, json.dumps(answers), now),
        )
    return cur.lastrowid


def all_submissions(db_path: Path | None = None) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM submissions ORDER BY submitted_at").fetchall()
    return [
        {
            "id": r["id"],
            "participant_code": r["participant_code"],
            "registry_version": r["registry_version"],
            "context_id": r["context_id"],
            "answers": json.loads(r["answers_json"]),
            "submitted_at": r["submitted_at"],
        }
        for r in rows
    ]


def submissions_for_version(registry_version: int, db_path: Path | None = None) -> list[dict]:
    """Only submissions taken under a given ontology version — alpha/correlation
    are only meaningful within one fixed version, per the versioning discipline
    discussed (an axis split/merge between versions changes what's being measured)."""
    return [s for s in all_submissions(db_path) if s["registry_version"] == registry_version]


def retest_pairs(
    min_gap_hours: float = 12,
    max_gap_days: float = 21,
    db_path: Path | None = None,
) -> list[tuple[dict, dict]]:
    """Pairs of submissions from the same participant, same ontology version,
    spaced within a short-term reproducibility window (default 12h-21 days) —
    close enough that a real preference shift is unlikely, far enough to not
    just be someone re-clicking through with the answers still fresh in mind."""
    subs = all_submissions(db_path)
    by_participant: dict[str, list[dict]] = {}
    for s in subs:
        by_participant.setdefault(s["participant_code"], []).append(s)

    pairs: list[tuple[dict, dict]] = []
    for participant_subs in by_participant.values():
        participant_subs.sort(key=lambda s: s["submitted_at"])
        for i in range(len(participant_subs) - 1):
            a, b = participant_subs[i], participant_subs[i + 1]
            if a["registry_version"] != b["registry_version"]:
                continue
            t_a = datetime.fromisoformat(a["submitted_at"])
            t_b = datetime.fromisoformat(b["submitted_at"])
            gap = t_b - t_a
            if timedelta(hours=min_gap_hours) <= gap <= timedelta(days=max_gap_days):
                pairs.append((a, b))
    return pairs
