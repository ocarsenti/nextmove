"""Tests for real, item-level Cronbach's alpha (engine/internal_consistency.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from ontology.registry import AxisRegistry, QuestionBank
from engine.internal_consistency import (
    cronbachs_alpha,
    _item_scores,
    _alpha_band,
    compute_internal_consistency_report,
    MIN_RESPONDENTS_FOR_RELIABLE_ALPHA,
)

bank = QuestionBank()
registry = AxisRegistry()


# ===================================================================
# CRONBACH'S ALPHA — pure math, hand-verified
# ===================================================================

class TestCronbachsAlpha:
    def test_perfect_agreement_gives_alpha_one(self):
        # Every respondent gives the same value to all 3 items — each
        # item is a perfect proxy for the total score.
        # Hand-verified: item variances = 1/6 each, total variance = 1.5,
        # alpha = (3/2) * (1 - 0.5/1.5) = 1.0.
        matrix = [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        assert cronbachs_alpha(matrix) == pytest.approx(1.0)

    def test_uncorrelated_items_give_low_alpha(self):
        # Each item independently randomized per respondent — no shared
        # signal for a total score to pick up on.
        matrix = [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [1.0, 1.0, 0.0],
        ]
        alpha = cronbachs_alpha(matrix)
        assert alpha is not None
        assert alpha < 0.5

    def test_single_respondent_returns_none(self):
        assert cronbachs_alpha([[1.0, 0.5, 0.0]]) is None

    def test_single_item_returns_none(self):
        assert cronbachs_alpha([[1.0], [0.5], [0.0]]) is None

    def test_zero_variance_total_returns_none(self):
        # Everyone answered identically on every item — no variance to
        # apportion between items and error, alpha is undefined (0/0).
        matrix = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        assert cronbachs_alpha(matrix) is None


class TestAlphaBand:
    @pytest.mark.parametrize(
        "alpha,expected",
        [
            (None, "n/a"),
            (0.3, "inacceptable"),
            (0.55, "faible"),
            (0.65, "discutable"),
            (0.75, "acceptable"),
            (0.85, "bonne"),
            (0.95, "excellente"),
        ],
    )
    def test_bands(self, alpha, expected):
        assert _alpha_band(alpha) == expected


# ===================================================================
# ITEM SCORES — extraction from real question bank
# ===================================================================

class TestItemScores:
    def test_extracts_scores_for_all_base_questions(self):
        qs = bank.base_questions_for("autonomy")
        answers = {q.id: "C" for q in qs}  # "C" = 1.0 on every autonomy question
        scores = _item_scores(bank, "autonomy", answers)
        assert scores == [1.0] * len(qs)

    def test_missing_answer_returns_none(self):
        qs = bank.base_questions_for("autonomy")
        answers = {q.id: "C" for q in qs[:-1]}  # last base question unanswered
        assert _item_scores(bank, "autonomy", answers) is None

    def test_unknown_axis_returns_empty_list(self):
        assert _item_scores(bank, "not_a_real_axis", {}) == []


# ===================================================================
# FULL REPORT — real bank/registry, synthetic passages
# ===================================================================

def _passage_row(answers: dict[str, str]) -> dict:
    import json
    return {"answers_json": json.dumps(answers)}


class TestComputeInternalConsistencyReport:
    def test_empty_passages_returns_zero_respondents_per_axis(self):
        report = compute_internal_consistency_report(bank, registry, [])
        assert report["axes"]["autonomy"]["n_respondents"] == 0
        assert report["axes"]["autonomy"]["cronbachs_alpha"] is None
        assert report["axes"]["autonomy"]["sample_sufficient"] is False

    def test_every_active_axis_with_enough_base_items_is_reported(self):
        report = compute_internal_consistency_report(bank, registry, [])
        for axis_id in registry.active_ids():
            assert axis_id in report["axes"]
            assert report["axes"][axis_id]["n_items"] == len(
                bank.base_questions_for(axis_id)
            )

    def test_consistent_answers_across_respondents_give_high_alpha(self):
        qs = bank.base_questions_for("autonomy")
        passages = [
            _passage_row({q.id: "C" for q in qs}),
            _passage_row({q.id: "A" for q in qs}),
            _passage_row({q.id: "B" for q in qs}),
        ]
        report = compute_internal_consistency_report(bank, registry, passages)
        axis = report["axes"]["autonomy"]
        assert axis["n_respondents"] == 3
        assert axis["cronbachs_alpha"] == pytest.approx(1.0)
        assert axis["band"] == "excellente"

    def test_passage_missing_a_base_answer_is_excluded_via_listwise_deletion(self):
        qs = bank.base_questions_for("autonomy")
        complete = {q.id: "C" for q in qs}
        incomplete = {q.id: "A" for q in qs[:-1]}  # missing one base answer
        report = compute_internal_consistency_report(
            bank, registry, [_passage_row(complete), _passage_row(incomplete)]
        )
        assert report["axes"]["autonomy"]["n_respondents"] == 1

    def test_sample_sufficient_flag(self):
        qs = bank.base_questions_for("autonomy")
        qids = [q.id for q in qs]
        passages = [
            _passage_row(dict(zip(qids, ["A", "B", "C"])))
            for _ in range(MIN_RESPONDENTS_FOR_RELIABLE_ALPHA)
        ]
        report = compute_internal_consistency_report(bank, registry, passages)
        assert report["axes"]["autonomy"]["n_respondents"] == MIN_RESPONDENTS_FOR_RELIABLE_ALPHA
        assert report["axes"]["autonomy"]["sample_sufficient"] is True
