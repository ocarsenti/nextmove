"""Reliability analysis on accumulated real submissions — computes what the
methode.html transparency page currently lists as 'pending':

    - internal consistency (Cronbach's alpha) per axis, from the 3 fixed
      base items every respondent answers — not the coarse per-profile-
      variance proxy engine/signals.py uses internally for the split/merge
      rule engine, a proper item-level alpha.
    - inter-axis correlation matrix (reuses engine.signals.SignalComputer,
      which already computes this correctly from reconstructed Profiles).
    - short-term test-retest correlation, from participant pairs returned
      by research.storage.retest_pairs().

Everything here is computed FROM raw stored answers, reconstructed into
Profile objects via the same ProfileScorer used in production — never from
a separately-maintained analysis pipeline that could silently drift from
what the product actually computes.
"""
from __future__ import annotations

import math

from ontology.registry import AxisRegistry, QuestionBank
from engine.scorer import ProfileScorer
from engine.signals import SignalComputer
from research.storage import all_submissions, submissions_for_version, retest_pairs

MIN_RESPONDENTS_FOR_ALPHA = 20  # below this, an alpha estimate is too noisy to report


def _cronbach_alpha(item_matrix: list[list[float]]) -> float | None:
    """item_matrix: one row per respondent, one column per item (fixed item
    count, e.g. the 3 base questions of one axis). Classic formula:

        alpha = (k / (k - 1)) * (1 - sum(item_variances) / variance_of_total_score)

    Returns None if there are too few respondents or fewer than 2 items.
    """
    if not item_matrix or len(item_matrix[0]) < 2:
        return None
    n = len(item_matrix)
    if n < 2:
        return None
    k = len(item_matrix[0])

    def _variance(values: list[float]) -> float:
        m = sum(values) / len(values)
        return sum((v - m) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0

    item_variances = [_variance([row[i] for row in item_matrix]) for i in range(k)]
    total_scores = [sum(row) for row in item_matrix]
    total_variance = _variance(total_scores)

    if total_variance == 0:
        return None  # everyone answered identically — alpha is undefined, not zero
    return round((k / (k - 1)) * (1 - sum(item_variances) / total_variance), 4)


def compute_internal_consistency(registry_version: int | None = None) -> dict[str, dict]:
    """Cronbach's alpha per axis, using only the 3 fixed base items (same
    for every respondent, so the item matrix is well-formed)."""
    registry = AxisRegistry()
    bank = QuestionBank()

    subs = submissions_for_version(registry_version) if registry_version is not None else all_submissions()

    result: dict[str, dict] = {}
    for axis in registry.non_meta_axes():
        base_qs = bank.base_questions_for(axis.id)
        if len(base_qs) < 2:
            continue
        base_qids = [q.id for q in base_qs]

        item_matrix: list[list[float]] = []
        for sub in subs:
            row = []
            complete = True
            for qid in base_qids:
                answer_value = sub["answers"].get(qid)
                if answer_value is None:
                    complete = False
                    break
                q = bank.get(qid)
                score_val = next((opt.score for opt in q.options if opt.value == answer_value), None)
                if score_val is None:
                    complete = False
                    break
                row.append(score_val)
            if complete:
                item_matrix.append(row)

        n = len(item_matrix)
        alpha = _cronbach_alpha(item_matrix) if n >= MIN_RESPONDENTS_FOR_ALPHA else None
        result[axis.id] = {
            "n_respondents": n,
            "alpha": alpha,
            "reportable": n >= MIN_RESPONDENTS_FOR_ALPHA,
        }
    return result


def compute_inter_axis_correlations(registry_version: int | None = None) -> dict:
    """Reconstructs Profiles from stored raw answers and reuses SignalComputer
    — the same statistical machinery engine/v6_rules.py is meant to eventually
    consume for data-driven split/merge (see engine/signals.py)."""
    registry = AxisRegistry()
    bank = QuestionBank()
    scorer = ProfileScorer(registry, bank)

    subs = submissions_for_version(registry_version) if registry_version is not None else all_submissions()
    profiles = [
        scorer.score(f"research_{s['id']}", s["answers"], s.get("context_id"))
        for s in subs
    ]
    return SignalComputer().compute(profiles)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(num / denom, 4) if denom > 1e-9 else None


def compute_test_retest(min_gap_hours: float = 12, max_gap_days: float = 21) -> dict[str, dict]:
    """Short-term reproducibility per axis: correlation between a person's
    first and second pass, for pairs close enough that a genuine preference
    shift is unlikely (see research/storage.py::retest_pairs docstring)."""
    registry = AxisRegistry()
    bank = QuestionBank()
    scorer = ProfileScorer(registry, bank)

    pairs = retest_pairs(min_gap_hours, max_gap_days)
    per_axis_first: dict[str, list[float]] = {}
    per_axis_second: dict[str, list[float]] = {}

    for a, b in pairs:
        if a["registry_version"] != b["registry_version"]:
            continue
        profile_a = scorer.score(f"retest_a_{a['id']}", a["answers"], a.get("context_id"))
        profile_b = scorer.score(f"retest_b_{b['id']}", b["answers"], b.get("context_id"))
        for axis_id in profile_a.axis_scores:
            if axis_id not in profile_b.axis_scores:
                continue
            per_axis_first.setdefault(axis_id, []).append(profile_a.axis_scores[axis_id].raw_value)
            per_axis_second.setdefault(axis_id, []).append(profile_b.axis_scores[axis_id].raw_value)

    result: dict[str, dict] = {}
    for axis_id in per_axis_first:
        xs, ys = per_axis_first[axis_id], per_axis_second[axis_id]
        result[axis_id] = {
            "n_pairs": len(xs),
            "correlation": _pearson(xs, ys) if len(xs) >= 10 else None,
            "reportable": len(xs) >= 10,
        }
    return result


def full_reliability_report(registry_version: int | None = None) -> dict:
    registry = AxisRegistry()
    version = registry_version if registry_version is not None else registry.version
    return {
        "registry_version": version,
        "n_total_submissions": len(submissions_for_version(version)),
        "internal_consistency": compute_internal_consistency(version),
        "inter_axis_correlations": {
            f"{k[0]}__{k[1]}": v["correlation"]
            for k, v in compute_inter_axis_correlations(version).items()
            if isinstance(k, tuple)
        },
        "test_retest": compute_test_retest(),
    }
