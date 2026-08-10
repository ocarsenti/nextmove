"""Check archetype-calibration progress and send ONE email alert the first
time the recommended threshold (see research/archetype_calibration_log.RECOMMENDED_N)
is reached. Meant to run on a schedule (cron) on the VPS — see the crontab
line in this file's module docstring / STAGING.md.

Idempotency: a marker file (data/archetype_calibration_alert_sent.flag) is
written after a successful send, so re-running this script (e.g. daily)
doesn't spam the inbox once the threshold has been crossed. After you
actually recalibrate DISPLAY_TEMPERATURE/DISPLAY_FLOOR/DISPLAY_MIN_GAP on
the accumulated data, delete that flag file (or rerun with --reset) so the
next round of calibration can alert again once it, in turn, reaches the
threshold.

Required environment variables (put these in the VPS .env, never commit
them):
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
    ARCHETYPE_CALIBRATION_ALERT_TO   — destination email address

Suggested crontab (daily at 8am):
    0 8 * * * cd /opt/nextmove && /opt/nextmove/.venv/bin/python3 scripts/check_calibration_and_alert.py >> data/calibration_alert.log 2>&1
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research import archetype_calibration_log as cal_log

FLAG_PATH = Path(__file__).parent.parent / "data" / "archetype_calibration_alert_sent.flag"


def _send_email(to_addr: str, subject: str, body: str) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    from_addr = os.environ.get("SMTP_FROM", user)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(from_addr, [to_addr], msg.as_string())


def check_and_alert(reset: bool = False) -> int:
    """Returns 0 normally; non-zero only on a real error, so cron logs an
    obvious failure — reaching or not reaching the threshold is never an
    error by itself."""
    if reset and FLAG_PATH.exists():
        FLAG_PATH.unlink()
        print("Flag réinitialisé — la prochaine alerte pourra repartir.")

    n = cal_log.count_passages()
    threshold = cal_log.RECOMMENDED_N
    print(f"Passages loggés : {n} / {threshold}")

    if n < threshold:
        return 0

    if FLAG_PATH.exists():
        print("Seuil atteint, alerte déjà envoyée précédemment — rien à faire.")
        return 0

    to_addr = os.environ.get("ARCHETYPE_CALIBRATION_ALERT_TO")
    if not to_addr:
        print(
            "ARCHETYPE_CALIBRATION_ALERT_TO n'est pas défini — impossible d'envoyer "
            "l'alerte email. Seuil pourtant atteint, ne pas ignorer.",
            file=sys.stderr,
        )
        return 1

    subject = f"NextMove — seuil de calibration atteint ({n}/{threshold})"
    body = (
        f"{n} profils complets ont été loggés dans archetype_calibration_log.db "
        f"(seuil recommandé : {threshold}).\n\n"
        "Il est temps de recalibrer DISPLAY_TEMPERATURE / DISPLAY_FLOOR / "
        "DISPLAY_MIN_GAP dans engine/archetype.py sur ces données réelles, "
        "au lieu des personas synthétiques actuels.\n\n"
        "Rapport détaillé : GET /study/archetype-calibration/report\n\n"
        "Une fois la recalibration faite, supprime data/archetype_calibration_"
        "alert_sent.flag (ou relance ce script avec --reset) pour permettre "
        "une future alerte au prochain seuil."
    )

    try:
        _send_email(to_addr, subject, body)
    except Exception as e:
        print(f"Échec de l'envoi de l'alerte email : {e}", file=sys.stderr)
        return 1

    FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FLAG_PATH.write_text(f"alerted at n={n}\n")
    print(f"Alerte envoyée à {to_addr}.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true",
        help="Efface le flag 'déjà alerté' avant de vérifier (à utiliser après une recalibration).",
    )
    args = parser.parse_args()
    sys.exit(check_and_alert(reset=args.reset))
