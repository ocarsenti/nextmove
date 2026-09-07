"""Tests for research/archetype_calibration_log.py — the anonymous
raw_affinity log used to eventually recalibrate DISPLAY_TEMPERATURE/
DISPLAY_FLOOR/DISPLAY_MIN_GAP on real data (see engine/archetype.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from research import archetype_calibration_log as cal_log


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_archetype_calibration.db"


def test_init_db_creates_file(db_path):
    cal_log.init_db(db_path)
    assert db_path.exists()


def test_log_and_read_back_a_passage(db_path):
    cal_log.init_db(db_path)
    raw_affinity = {
        "Builder": 0.21, "Expert": 0.25, "Operator": 0.18,
        "Leader": 0.31, "Connecteur": 0.17,
    }
    row_id = cal_log.log_passage(
        raw_affinity, "Leader", "Expert", False, "startup", db_path=db_path
    )
    assert row_id == 1

    rows = cal_log.all_passages(db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["raw_affinity"] == raw_affinity
    assert row["dominant"] == "Leader"
    assert row["secondary"] == "Expert"
    assert row["low_confidence"] is False
    assert row["context_id"] == "startup"
    assert row["created_at"]


def test_log_passage_with_no_secondary_or_context(db_path):
    cal_log.init_db(db_path)
    raw_affinity = {"Builder": 0.2, "Expert": 0.2, "Operator": 0.2, "Leader": 0.2, "Connecteur": 0.2}
    cal_log.log_passage(raw_affinity, "Builder", None, True, None, db_path=db_path)

    rows = cal_log.all_passages(db_path)
    assert rows[0]["secondary"] is None
    assert rows[0]["context_id"] is None
    assert rows[0]["low_confidence"] is True


def test_no_pii_fields_in_schema(db_path):
    """Structural guarantee matching the module docstring's privacy
    calculus: nothing joinable to a person or to their other passages."""
    cal_log.init_db(db_path)
    cal_log.log_passage(
        {"Builder": 0.2, "Expert": 0.2, "Operator": 0.2, "Leader": 0.2, "Connecteur": 0.2},
        "Builder", None, False, None, db_path=db_path,
    )
    row = cal_log.all_passages(db_path)[0]
    forbidden_keys = {"user_id", "session_user_id", "participant_code", "answers", "axis_scores"}
    assert forbidden_keys.isdisjoint(row.keys())


def test_count_passages(db_path):
    cal_log.init_db(db_path)
    assert cal_log.count_passages(db_path) == 0
    for _ in range(3):
        cal_log.log_passage(
            {"Builder": 0.2, "Expert": 0.2, "Operator": 0.2, "Leader": 0.2, "Connecteur": 0.2},
            "Builder", None, False, None, db_path=db_path,
        )
    assert cal_log.count_passages(db_path) == 3


def test_multiple_passages_ordered_by_created_at(db_path):
    cal_log.init_db(db_path)
    for name in ["Builder", "Expert", "Operator"]:
        cal_log.log_passage(
            {"Builder": 0.2, "Expert": 0.2, "Operator": 0.2, "Leader": 0.2, "Connecteur": 0.2},
            name, None, False, None, db_path=db_path,
        )
    rows = cal_log.all_passages(db_path)
    assert [r["dominant"] for r in rows] == ["Builder", "Expert", "Operator"]
