"""Tests for scripts/check_calibration_and_alert.py — threshold check,
idempotent alerting via the flag file, and that a missing destination
address or SMTP failure surfaces as a non-zero exit rather than being
silently swallowed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest

from research import archetype_calibration_log as cal_log
import check_calibration_and_alert as alert_script


def _log_n_passages(n, db_path):
    for _ in range(n):
        cal_log.log_passage(
            {"Builder": 0.2, "Expert": 0.2, "Operator": 0.2, "Leader": 0.2, "Explorer": 0.2},
            "Builder", None, False, None, db_path=db_path,
        )


@pytest.fixture(autouse=True)
def isolated_calibration_env(monkeypatch, tmp_path):
    """Point both the calibration DB and the alert flag file at tmp_path
    so tests never touch the real data/ files, and lower the threshold to
    keep test fixtures small."""
    db_path = tmp_path / "calib.db"
    flag_path = tmp_path / "alert.flag"
    cal_log.init_db(db_path)
    monkeypatch.setattr(cal_log, "DB_PATH", db_path)
    monkeypatch.setattr(cal_log, "RECOMMENDED_N", 3)
    monkeypatch.setattr(alert_script, "FLAG_PATH", flag_path)
    return db_path, flag_path


def test_below_threshold_no_email_no_flag(monkeypatch, isolated_calibration_env):
    db_path, flag_path = isolated_calibration_env
    _log_n_passages(2, db_path)
    monkeypatch.setenv("ARCHETYPE_CALIBRATION_ALERT_TO", "olivier@example.com")

    sent = []
    monkeypatch.setattr(alert_script, "_send_email", lambda *a, **kw: sent.append(a))

    rc = alert_script.check_and_alert()
    assert rc == 0
    assert sent == []
    assert not flag_path.exists()


def test_threshold_reached_sends_email_and_writes_flag(monkeypatch, isolated_calibration_env):
    db_path, flag_path = isolated_calibration_env
    _log_n_passages(3, db_path)
    monkeypatch.setenv("ARCHETYPE_CALIBRATION_ALERT_TO", "olivier@example.com")

    sent = []
    monkeypatch.setattr(alert_script, "_send_email", lambda *a, **kw: sent.append(a))

    rc = alert_script.check_and_alert()
    assert rc == 0
    assert len(sent) == 1
    assert sent[0][0] == "olivier@example.com"
    assert flag_path.exists()


def test_second_run_does_not_resend(monkeypatch, isolated_calibration_env):
    db_path, flag_path = isolated_calibration_env
    _log_n_passages(3, db_path)
    monkeypatch.setenv("ARCHETYPE_CALIBRATION_ALERT_TO", "olivier@example.com")

    sent = []
    monkeypatch.setattr(alert_script, "_send_email", lambda *a, **kw: sent.append(a))

    alert_script.check_and_alert()
    alert_script.check_and_alert()  # simulates the next day's cron run
    assert len(sent) == 1


def test_reset_allows_resend(monkeypatch, isolated_calibration_env):
    db_path, flag_path = isolated_calibration_env
    _log_n_passages(3, db_path)
    monkeypatch.setenv("ARCHETYPE_CALIBRATION_ALERT_TO", "olivier@example.com")

    sent = []
    monkeypatch.setattr(alert_script, "_send_email", lambda *a, **kw: sent.append(a))

    alert_script.check_and_alert()
    alert_script.check_and_alert(reset=True)
    assert len(sent) == 2


def test_missing_destination_address_is_a_reported_failure(monkeypatch, isolated_calibration_env):
    db_path, flag_path = isolated_calibration_env
    _log_n_passages(3, db_path)
    monkeypatch.delenv("ARCHETYPE_CALIBRATION_ALERT_TO", raising=False)

    rc = alert_script.check_and_alert()
    assert rc == 1
    assert not flag_path.exists()


def test_smtp_failure_does_not_write_flag(monkeypatch, isolated_calibration_env):
    db_path, flag_path = isolated_calibration_env
    _log_n_passages(3, db_path)
    monkeypatch.setenv("ARCHETYPE_CALIBRATION_ALERT_TO", "olivier@example.com")

    def _boom(*a, **kw):
        raise RuntimeError("SMTP connection refused")
    monkeypatch.setattr(alert_script, "_send_email", _boom)

    rc = alert_script.check_and_alert()
    assert rc == 1
    assert not flag_path.exists()  # so the next cron run retries
