"""
Minimal SQLite persistence for shared Decision Cards.
"""
import json
import secrets
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nextmove_v4.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shares (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            current_job_title TEXT NOT NULL,
            opportunity_title TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    return conn


def create_share(current_job_title: str, opportunity_title: str, payload: dict[str, Any]) -> str:
    share_id = secrets.token_urlsafe(6)
    conn = _connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO shares (id, current_job_title, opportunity_title, payload) "
                "VALUES (?, ?, ?, ?)",
                (share_id, current_job_title, opportunity_title, json.dumps(payload)),
            )
    finally:
        conn.close()
    return share_id


def get_share(share_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT current_job_title, opportunity_title, payload FROM shares WHERE id = ?",
            (share_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    current_job_title, opportunity_title, payload = row
    return {
        "current_job_title": current_job_title,
        "opportunity_title": opportunity_title,
        "result": json.loads(payload),
    }
