"""Tests for engine/matching.py — candidate Profile vs. JobCard tension engine."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.models import AxisStatus, JobAxisRequirement, JobCard
from ontology.registry import AxisRegistry, QuestionBank
from engine.scorer import ProfileScorer
from engine.matching import compute_tensions, _intensity, _direction
from questionnaire.adaptive import AdaptiveQuestionnaire


def _build_full_answers(bank: QuestionBank, registry: AxisRegistry, score_value: str = "B") -> dict[str, str]:
    questionnaire = AdaptiveQuestionnaire(registry, bank)
    base = questionnaire.initial_sequence()
    return {qid: score_value for qid in base}


class TestIntensityAndDirection(unittest.TestCase):

    def test_low_intensity_below_threshold(self):
        self.assertEqual(_intensity(0.05), "low")

    def test_medium_intensity(self):
        self.assertEqual(_intensity(0.25), "medium")

    def test_high_intensity(self):
        self.assertEqual(_intensity(0.6), "high")

    def test_direction_aligned_within_epsilon(self):
        self.assertEqual(_direction(0.5, 0.53), "aligned")

    def test_direction_job_demands_more(self):
        self.assertEqual(_direction(0.3, 0.9), "job_demands_more")

    def test_direction_job_demands_less(self):
        self.assertEqual(_direction(0.9, 0.3), "job_demands_less")


class TestComputeTensions(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.bank = QuestionBank()
        self.scorer = ProfileScorer(self.registry, self.bank)
        answers = _build_full_answers(self.bank, self.registry, score_value="C")  # high answers
        self.profile = self.scorer.score("candidate_1", answers)
        self.axis_id = "autonomy"
        self.assertIn(self.axis_id, self.profile.axis_scores, "fixture must produce a scored autonomy axis")

    def _job_requiring(self, axis_id: str, level: float, importance: float = 1.0) -> JobCard:
        return JobCard(
            job_id="job_1",
            title="Test role",
            axis_requirements={axis_id: JobAxisRequirement(axis_id=axis_id, level=level, importance=importance)},
        )

    def test_aligned_axis_has_low_distance(self):
        candidate_val = self.profile.axis_scores[self.axis_id].raw_value
        job = self._job_requiring(self.axis_id, level=candidate_val)
        result = compute_tensions(self.profile, job, self.registry)
        self.assertEqual(len(result.tensions), 1)
        self.assertAlmostEqual(result.tensions[0].distance, 0.0, places=6)
        self.assertEqual(result.tensions[0].direction, "aligned")

    def test_opposite_requirement_produces_high_tension(self):
        candidate_val = self.profile.axis_scores[self.axis_id].raw_value
        opposite = 0.0 if candidate_val > 0.5 else 1.0
        job = self._job_requiring(self.axis_id, level=opposite)
        result = compute_tensions(self.profile, job, self.registry)
        t = result.tensions[0]
        self.assertGreater(t.distance, MEDIUM_THRESHOLD if (MEDIUM_THRESHOLD := 0.35) else 0)
        self.assertEqual(t.intensity, "high")
        self.assertIn(t.direction, ("job_demands_more", "job_demands_less"))

    def test_low_confidence_axis_is_flagged_and_excluded(self):
        # Single-answer profile → low confidence on every axis
        thin_registry = AxisRegistry()
        thin_bank = QuestionBank()
        thin_scorer = ProfileScorer(thin_registry, thin_bank)
        base_q = thin_bank.base_questions_for(self.axis_id)[0]
        profile = thin_scorer.score("candidate_thin", {base_q.id: "B"})

        job = JobCard(
            job_id="job_thin",
            title="Test role",
            axis_requirements={self.axis_id: JobAxisRequirement(axis_id=self.axis_id, level=0.9)},
        )
        result = compute_tensions(profile, job, thin_registry)
        self.assertIn(self.axis_id, result.low_confidence_axes)
        self.assertNotIn(self.axis_id, result.top_tensions)

    def test_skipped_axis_when_absent_from_profile(self):
        job = self._job_requiring("nonexistent_axis_xyz", level=0.5)
        result = compute_tensions(self.profile, job, self.registry)
        self.assertIn("nonexistent_axis_xyz", result.skipped_axes)
        self.assertEqual(result.tensions, [])
        self.assertEqual(result.fit_score, 0.0)

    def test_masked_axis_is_skipped_not_scored(self):
        self.registry.mask(self.axis_id, reason="test")
        # Rescore so the mask is reflected in the profile's axis status
        answers = _build_full_answers(self.bank, self.registry, score_value="C")
        profile = self.scorer.score("candidate_2", answers)
        job = self._job_requiring(self.axis_id, level=0.5)
        result = compute_tensions(profile, job, self.registry)
        self.assertIn(self.axis_id, result.skipped_axes)
        self.registry.unmask(self.axis_id)  # restore for other tests in this run

    def test_fit_score_is_one_when_perfectly_aligned(self):
        candidate_val = self.profile.axis_scores[self.axis_id].raw_value
        job = self._job_requiring(self.axis_id, level=candidate_val, importance=1.0)
        result = compute_tensions(self.profile, job, self.registry)
        self.assertAlmostEqual(result.fit_score, 1.0, places=2)

    def test_multi_axis_fit_score_is_weighted_by_importance(self):
        axes = list(self.profile.axis_scores.keys())[:2]
        v0 = self.profile.axis_scores[axes[0]].raw_value
        v1 = self.profile.axis_scores[axes[1]].raw_value
        job = JobCard(
            job_id="job_multi",
            title="Test role",
            axis_requirements={
                axes[0]: JobAxisRequirement(axis_id=axes[0], level=v0, importance=1.0),   # perfectly aligned
                axes[1]: JobAxisRequirement(axis_id=axes[1], level=max(0.0, v1 - 1.0) if v1 >= 1.0 else 1.0 - v1, importance=0.2),  # far off, low importance
            },
        )
        result = compute_tensions(self.profile, job, self.registry)
        # Dominated by the high-importance aligned axis, so fit_score should stay high.
        self.assertGreater(result.fit_score, 0.7)


if __name__ == "__main__":
    unittest.main()
