"""Test-retest study store — SQLite persistence for repeated questionnaire passages.

Links passages from the same real person via a server-generated participant
code (no email/PII collected) so short-term reproducibility can be measured:
see methode.html's "Reproductibilité à court terme" evidence box.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Two common French words, easy to remember and re-type after a few days —
# simpler than an alphanumeric code for a low-volume study (~thousands of
# combinations is plenty of headroom for the expected participant count).
_WORDS = [
    "chat", "chien", "tigre", "lion", "ours", "loup", "renard", "lapin",
    "cheval", "aigle", "requin", "dauphin", "baleine", "singe", "panda",
    "soleil", "lune", "etoile", "nuage", "pluie", "vent", "orage", "neige",
    "ocean", "riviere", "montagne", "foret", "desert", "volcan", "iceberg",
    "table", "chaise", "porte", "fenetre", "lampe", "miroir", "tapis",
    "piano", "guitare", "tambour", "violon", "trompette", "flute",
    "pomme", "orange", "citron", "fraise", "cerise", "raisin", "mangue",
    "rouge", "bleu", "vert", "jaune", "violet", "marron", "rose", "gris",
    "matin", "midi", "soir", "nuit", "printemps", "ete", "automne", "hiver",
]
_WORDS_PER_CODE = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS retest_passages (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    participant_code    TEXT NOT NULL,
    passage_number      INTEGER NOT NULL,
    session_user_id     TEXT NOT NULL,
    context_id          TEXT,
    registry_version    INTEGER NOT NULL,
    answers_json        TEXT NOT NULL,
    axis_scores_json    TEXT NOT NULL,
    weighted_scores_json TEXT NOT NULL,
    archetype_dominant  TEXT,
    archetype_secondary TEXT,
    is_complete         INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    UNIQUE(participant_code, passage_number)
);
CREATE INDEX IF NOT EXISTS idx_retest_participant_code
    ON retest_passages(participant_code);
"""


class UnknownParticipantCode(Exception):
    """Raised when a retest is submitted with a code that was never issued."""


class RetestStore:
    """SQLite-backed store for test-retest study passages."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or Path(__file__).parent.parent / "data" / "retest" / "retest_study.db"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _generate_unused_code(self, conn: sqlite3.Connection) -> str:
        while True:
            code = "-".join(secrets.choice(_WORDS) for _ in range(_WORDS_PER_CODE))
            row = conn.execute(
                "SELECT 1 FROM retest_passages WHERE participant_code = ? LIMIT 1",
                (code,),
            ).fetchone()
            if row is None:
                return code

    def save_passage(
        self,
        *,
        session_user_id: str,
        context_id: Optional[str],
        registry_version: int,
        answers: dict[str, str],
        axis_scores: dict[str, dict],
        weighted_scores: dict[str, float],
        archetype_dominant: Optional[str],
        archetype_secondary: Optional[str],
        is_complete: bool,
        participant_code: Optional[str] = None,
    ) -> tuple[str, int]:
        """Persist one passage. Returns (participant_code, passage_number).

        If participant_code is None, this is a first passage — a fresh code
        is minted. If it's given, it must already exist (a typo'd code fails
        loudly rather than silently starting a new, disconnected series).
        """
        with self._connect() as conn:
            if participant_code is None:
                participant_code = self._generate_unused_code(conn)
                passage_number = 1
            else:
                participant_code = participant_code.strip().lower()
                row = conn.execute(
                    "SELECT MAX(passage_number) AS n FROM retest_passages "
                    "WHERE participant_code = ?",
                    (participant_code,),
                ).fetchone()
                if row is None or row["n"] is None:
                    raise UnknownParticipantCode(participant_code)
                passage_number = row["n"] + 1

            conn.execute(
                """
                INSERT INTO retest_passages (
                    participant_code, passage_number, session_user_id,
                    context_id, registry_version, answers_json,
                    axis_scores_json, weighted_scores_json,
                    archetype_dominant, archetype_secondary,
                    is_complete, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    participant_code,
                    passage_number,
                    session_user_id,
                    context_id,
                    registry_version,
                    json.dumps(answers, ensure_ascii=False),
                    json.dumps(axis_scores, ensure_ascii=False),
                    json.dumps(weighted_scores, ensure_ascii=False),
                    archetype_dominant,
                    archetype_secondary,
                    1 if is_complete else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return participant_code, passage_number

    def passages_for(self, participant_code: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM retest_passages WHERE participant_code = ? "
                "ORDER BY passage_number ASC",
                (participant_code,),
            ).fetchall()
            return [dict(r) for r in rows]
