"""Axis quality report — internal consistency + inter-axis correlation.

Answers two of methode.html's still-"pending" validation rows: "Cohérence
interne" and "Validité de construit" — distinct from the test-retest
reproducibility report in `retest_analysis.py`, which answers a third,
separate row.

Cohérence interne comes from `internal_consistency.py`'s real, item-level
Cronbach's alpha (`cronbachs_alpha` / `band` per axis) — not from
`SignalComputer`'s `internal_consistency` proxy, kept here only under
`internal_consistency_proxy` for comparison (it measures self-consistency
of one person's answers, not whether an axis's questions correlate with
each other, which is what "cohérence interne" actually asks). Validité de
construit still comes from `SignalComputer`'s inter-axis correlation,
which is a legitimate use of that mechanism (it was originally built for
the V6 ontology-evolution rule engine, never wired to a production
endpoint until this and the alpha computation).

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
from ontology.registry import AxisRegistry, QuestionBank
from engine.signals import SignalComputer
from engine.retest_store import RetestStore
from engine.internal_consistency import compute_internal_consistency_report

# Correlation/consistency estimates need a bigger sample than a paired
# ICC does (no natural "2 occasions" structure to lean on) — 30 is the
# usual rule-of-thumb floor for a stable Pearson r estimate.
MIN_PROFILES_FOR_RELIABLE_REPORT = 30

# Two axes correlating at or above this are flagged as possibly redundant
# (candidates for the V6 "merge" rule) — not proof, just a pointer.
HIGH_CORRELATION_THRESHOLD = 0.7

# An axis with a discrimination_index below this barely separates
# respondents from each other (everyone answers about the same) — flagged
# as a possible "split" candidate (the axis may be too coarse, bundling
# together things that should be measured separately), or simply a
# poorly-worded item set. Either way, a pointer for review, not proof —
# same status as HIGH_CORRELATION_THRESHOLD above, and same caveat: this
# never triggers a split by itself (see engine/transformation.py — a split
# still needs a human to define what the new axes actually are).
LOW_DISCRIMINATION_THRESHOLD = 0.3

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


def compute_quality_report(
    store: RetestStore, bank: QuestionBank, registry: AxisRegistry
) -> dict:
    rows = store.all_complete_passages()
    profiles = [_row_to_profile(r) for r in rows]
    signals = SignalComputer().compute(profiles)
    alpha_report = compute_internal_consistency_report(bank, registry, rows)

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
            entry = dict(value)
            entry["internal_consistency_proxy"] = entry.pop("internal_consistency")
            entry["flagged_low_discrimination"] = (
                entry["n_profiles"] >= MIN_PROFILES_FOR_RELIABLE_REPORT
                and entry["discrimination_index"] < LOW_DISCRIMINATION_THRESHOLD
            )
            axes_report[key] = entry

    for axis_id, alpha_data in alpha_report["axes"].items():
        if axis_id not in axes_report and alpha_data["n_respondents"] == 0:
            continue  # nothing to show for this axis from either source yet
        axes_report.setdefault(axis_id, {})
        axes_report[axis_id].setdefault("flagged_low_discrimination", False)
        axes_report[axis_id].update(alpha_data)

    correlations.sort(key=lambda c: -abs(c["correlation"]))

    n_profiles = len(profiles)
    return {
        "n_profiles": n_profiles,
        "min_profiles_for_reliable_report": MIN_PROFILES_FOR_RELIABLE_REPORT,
        "sample_sufficient": n_profiles >= MIN_PROFILES_FOR_RELIABLE_REPORT,
        "sample_caveat": SAMPLE_CAVEAT,
        "high_correlation_threshold": HIGH_CORRELATION_THRESHOLD,
        "low_discrimination_threshold": LOW_DISCRIMINATION_THRESHOLD,
        "min_respondents_for_reliable_alpha": alpha_report["min_respondents_for_reliable_alpha"],
        "axes": axes_report,
        "correlations": correlations,
    }
