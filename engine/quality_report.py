"""Axis quality report — internal consistency + inter-axis correlation.

Reuses `SignalComputer` (originally built for the V6 ontology-evolution
rule engine, never wired to a production endpoint) to answer two of
methode.html's still-"pending" validation rows: "Cohérence interne" and
"Validité de construit" — distinct from the test-retest reproducibility
report in `retest_analysis.py`, which answers a third, separate row.

The only real profile data available today is the test-retest study's
opt-in passages (`RetestStore`) — every completed passage counts as one
profile here (no pairing needed, unlike the retest reliability report,
so single-passage participants still contribute). This is a small,
self-selected sample (people who opted into the reproducibility study),
not the full questionnaire-taking population — the report says so
explicitly rather than presenting itself as representative.
"""

from __future__ import annotations

import json

from ontology.models import AxisScore, Profile
from engine.signals import SignalComputer
from engine.retest_store import RetestStore

# Correlation/consistency estimates need a bigger sample than a paired
# ICC does (no natural "2 occasions" structure to lean on) — 30 is the
# usual rule-of-thumb floor for a stable Pearson r estimate.
MIN_PROFILES_FOR_RELIABLE_REPORT = 30

# Two axes correlating at or above this are flagged as possibly redundant
# (candidates for the V6 "merge" rule) — not proof, just a pointer.
HIGH_CORRELATION_THRESHOLD = 0.7

SAMPLE_CAVEAT = (
    "Échantillon = participants opt-in de l'étude de reproductibilité, "
    "pas l'ensemble des utilisateurs du questionnaire."
)


def _row_to_profile(row: dict) -> Profile:
    raw = json.loads(row["axis_scores_json"])
    axis_scores = {
        axis_id: AxisScore.model_validate(data) for axis_id, data in raw.items()
    }
    return Profile(
        user_id=row["session_user_id"],
        axis_scores=axis_scores,
        is_complete=bool(row["is_complete"]),
    )


def compute_quality_report(store: RetestStore) -> dict:
    rows = store.all_complete_passages()
    profiles = [_row_to_profile(r) for r in rows]
    signals = SignalComputer().compute(profiles)

    axes_report: dict[str, dict] = {}
    correlations: list[dict] = []
    seen_pairs: set[frozenset] = set()

    for key, value in signals.items():
        if isinstance(key, tuple):
            pair = frozenset(key)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            ax_a, ax_b = key
            correlations.append(
                {
                    "axis_a": ax_a,
                    "axis_b": ax_b,
                    "correlation": value["correlation"],
                    "flagged_redundant": abs(value["correlation"])
                    >= HIGH_CORRELATION_THRESHOLD,
                }
            )
        else:
            axes_report[key] = value

    correlations.sort(key=lambda c: -abs(c["correlation"]))

    n_profiles = len(profiles)
    return {
        "n_profiles": n_profiles,
        "min_profiles_for_reliable_report": MIN_PROFILES_FOR_RELIABLE_REPORT,
        "sample_sufficient": n_profiles >= MIN_PROFILES_FOR_RELIABLE_REPORT,
        "sample_caveat": SAMPLE_CAVEAT,
        "high_correlation_threshold": HIGH_CORRELATION_THRESHOLD,
        "axes": axes_report,
        "correlations": correlations,
    }
