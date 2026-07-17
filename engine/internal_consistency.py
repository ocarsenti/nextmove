"""Cronbach's alpha — real, item-level internal consistency per axis.

Distinct from `SignalComputer.compute()`'s `internal_consistency` proxy
(1 - mean per-profile answer variance), which doesn't measure whether an
axis's individual questions correlate with each other — it only measures
how consistent one person's answers were within themselves. This module
computes the textbook statistic instead: how well an axis's questions
covary with each other across many respondents.

Restricted to *base* questions (`Question.trigger is None`) specifically:
those are the one item set every complete passage answers identically.
Adaptive follow-up questions are shown to a variable subset of
respondents (only when confidence is low), which would break the
fixed-item-set assumption the formula relies on — mixing them in would
silently bias the result, not just add noise.
"""

from __future__ import annotations

import json

from ontology.registry import AxisRegistry, QuestionBank

MIN_ITEMS = 2      # alpha is mathematically undefined for a single item
MIN_RESPONDENTS = 2
MIN_RESPONDENTS_FOR_RELIABLE_ALPHA = 30  # below this, alpha is too noisy to act on


def _item_scores(bank: QuestionBank, axis_id: str, answers: dict[str, str]) -> list[float] | None:
    """Numeric scores for axis_id's base questions from one respondent.

    None if any base question is missing from `answers` (listwise
    deletion — keeps every respondent's row the same length/item order).
    """
    scores = []
    for q in bank.base_questions_for(axis_id):
        answer_value = answers.get(q.id)
        if answer_value is None:
            return None
        opt_score = next((o.score for o in q.options if o.value == answer_value), None)
        if opt_score is None:
            return None
        scores.append(opt_score)
    return scores


def _variance(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def cronbachs_alpha(item_matrix: list[list[float]]) -> float | None:
    """item_matrix: one row per respondent, one column per item (fixed order)."""
    n = len(item_matrix)
    if n < MIN_RESPONDENTS:
        return None
    k = len(item_matrix[0])
    if k < MIN_ITEMS:
        return None

    item_variances = [_variance([row[j] for row in item_matrix]) for j in range(k)]
    total_scores = [sum(row) for row in item_matrix]
    total_variance = _variance(total_scores)
    if total_variance == 0:
        return None

    return (k / (k - 1)) * (1 - sum(item_variances) / total_variance)


def _alpha_band(alpha: float | None) -> str:
    """George & Mallery (2003) conventional thresholds."""
    if alpha is None:
        return "n/a"
    if alpha < 0.5:
        return "inacceptable"
    if alpha < 0.6:
        return "faible"
    if alpha < 0.7:
        return "discutable"
    if alpha < 0.8:
        return "acceptable"
    if alpha < 0.9:
        return "bonne"
    return "excellente"


def compute_internal_consistency_report(
    bank: QuestionBank, registry: AxisRegistry, passages: list[dict]
) -> dict:
    """passages: rows from RetestStore.all_complete_passages()."""
    axes_report: dict[str, dict] = {}

    for axis_id in registry.active_ids():
        n_items = len(bank.base_questions_for(axis_id))
        if n_items < MIN_ITEMS:
            continue  # alpha undefined regardless of data — nothing to report

        matrix: list[list[float]] = []
        for row in passages:
            answers = json.loads(row["answers_json"])
            scores = _item_scores(bank, axis_id, answers)
            if scores is not None:
                matrix.append(scores)

        alpha = cronbachs_alpha(matrix)
        axes_report[axis_id] = {
            "n_items": n_items,
            "n_respondents": len(matrix),
            "cronbachs_alpha": alpha,
            "band": _alpha_band(alpha),
            "sample_sufficient": len(matrix) >= MIN_RESPONDENTS_FOR_RELIABLE_ALPHA,
        }

    return {
        "min_respondents_for_reliable_alpha": MIN_RESPONDENTS_FOR_RELIABLE_ALPHA,
        "axes": axes_report,
    }
