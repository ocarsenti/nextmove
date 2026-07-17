"""Tests for research/storage.py and research/reliability.py — uses a
temporary SQLite file per test, never the real reliability_study.db."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.registry import AxisRegistry, QuestionBank
from questionnaire.adaptive import AdaptiveQuestionnaire
from research import storage
from research.reliability import (
    _cronbach_alpha, compute_internal_consistency, compute_test_retest,
    full_reliability_report, MIN_RESPONDENTS_FOR_ALPHA,
)


def _full_answers(score_value="B") -> dict[str, str]:
    registry = AxisRegistry()
    bank = QuestionBank()
    q = AdaptiveQuestionnaire(registry, bank)
    return {qid: score_value for qid in q.initial_sequence()}


class TestStorage(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        storage.init_db(self.db_path)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def test_create_participant_returns_unique_codes(self):
        c1 = storage.create_participant(self.db_path)
        c2 = storage.create_participant(self.db_path)
        self.assertNotEqual(c1, c2)
        self.assertTrue(storage.participant_exists(c1, self.db_path))

    def test_unknown_participant_does_not_exist(self):
        self.assertFalse(storage.participant_exists("not-a-real-code", self.db_path))

    def test_save_and_retrieve_submission(self):
        code = storage.create_participant(self.db_path)
        sid = storage.save_submission(code, registry_version=1, answers={"q1": "A"}, db_path=self.db_path)
        subs = storage.all_submissions(self.db_path)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["id"], sid)
        self.assertEqual(subs[0]["answers"], {"q1": "A"})

    def test_withdraw_deletes_participant_and_submissions(self):
        code = storage.create_participant(self.db_path)
        storage.save_submission(code, 1, {"q1": "A"}, db_path=self.db_path)
        storage.save_submission(code, 1, {"q1": "B"}, db_path=self.db_path)
        deleted = storage.withdraw_consent(code, self.db_path)
        self.assertEqual(deleted, 2)
        self.assertFalse(storage.participant_exists(code, self.db_path))
        self.assertEqual(storage.all_submissions(self.db_path), [])

    def test_submissions_for_version_filters_correctly(self):
        code = storage.create_participant(self.db_path)
        storage.save_submission(code, registry_version=1, answers={"q1": "A"}, db_path=self.db_path)
        storage.save_submission(code, registry_version=2, answers={"q1": "B"}, db_path=self.db_path)
        v1 = storage.submissions_for_version(1, self.db_path)
        self.assertEqual(len(v1), 1)
        self.assertEqual(v1[0]["registry_version"], 1)

    def test_retest_pairs_within_window(self):
        import sqlite3
        code = storage.create_participant(self.db_path)
        sid1 = storage.save_submission(code, 1, {"q1": "A"}, db_path=self.db_path)
        sid2 = storage.save_submission(code, 1, {"q1": "B"}, db_path=self.db_path)
        # Backdate sid1 by 3 days to fall inside the default 12h-21d window
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE submissions SET submitted_at = datetime('now', '-3 days') WHERE id = ?",
                (sid1,),
            )
        pairs = storage.retest_pairs(db_path=self.db_path)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0]["id"], sid1)
        self.assertEqual(pairs[0][1]["id"], sid2)

    def test_retest_pairs_excludes_too_close_together(self):
        code = storage.create_participant(self.db_path)
        storage.save_submission(code, 1, {"q1": "A"}, db_path=self.db_path)
        storage.save_submission(code, 1, {"q1": "B"}, db_path=self.db_path)
        # Both submitted "now" — gap near zero, below the 12h minimum
        pairs = storage.retest_pairs(db_path=self.db_path)
        self.assertEqual(pairs, [])

    def test_retest_pairs_excludes_different_registry_versions(self):
        import sqlite3
        code = storage.create_participant(self.db_path)
        sid1 = storage.save_submission(code, registry_version=1, answers={"q1": "A"}, db_path=self.db_path)
        storage.save_submission(code, registry_version=2, answers={"q1": "B"}, db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE submissions SET submitted_at = datetime('now', '-3 days') WHERE id = ?",
                (sid1,),
            )
        pairs = storage.retest_pairs(db_path=self.db_path)
        self.assertEqual(pairs, [])


class TestCronbachAlpha(unittest.TestCase):

    def test_perfect_agreement_gives_high_alpha(self):
        # Every respondent answers all 3 items identically to each other —
        # maximal internal consistency.
        matrix = [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.75, 0.75, 0.75]]
        alpha = _cronbach_alpha(matrix)
        self.assertIsNotNone(alpha)
        self.assertGreater(alpha, 0.9)

    def test_random_noise_gives_low_alpha(self):
        import random
        random.seed(42)
        matrix = [[random.random(), random.random(), random.random()] for _ in range(30)]
        alpha = _cronbach_alpha(matrix)
        self.assertIsNotNone(alpha)
        self.assertLess(alpha, 0.5)

    def test_too_few_items_returns_none(self):
        self.assertIsNone(_cronbach_alpha([[0.5], [0.6]]))

    def test_too_few_respondents_returns_none(self):
        self.assertIsNone(_cronbach_alpha([[0.5, 0.5, 0.5]]))

    def test_zero_variance_returns_none_not_zero(self):
        matrix = [[0.5, 0.5, 0.5] for _ in range(10)]
        self.assertIsNone(_cronbach_alpha(matrix))


class TestComputeInternalConsistency(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        storage.init_db(self.db_path)
        self._orig_db_path = storage.DB_PATH
        storage.DB_PATH = self.db_path  # reliability.py's helpers use module-level default

    def tearDown(self):
        storage.DB_PATH = self._orig_db_path
        self.db_path.unlink(missing_ok=True)

    def test_below_threshold_not_reportable(self):
        code = storage.create_participant(self.db_path)
        storage.save_submission(code, 1, _full_answers(), db_path=self.db_path)
        result = compute_internal_consistency(registry_version=1)
        for axis_id, r in result.items():
            self.assertFalse(r["reportable"])
            self.assertIsNone(r["alpha"])

    def test_above_threshold_is_reportable(self):
        for _ in range(MIN_RESPONDENTS_FOR_ALPHA + 5):
            code = storage.create_participant(self.db_path)
            storage.save_submission(code, 1, _full_answers("B"), db_path=self.db_path)
        result = compute_internal_consistency(registry_version=1)
        reportable = [r for r in result.values() if r["reportable"]]
        self.assertGreater(len(reportable), 0)


class TestFullReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        storage.init_db(self.db_path)
        self._orig_db_path = storage.DB_PATH
        storage.DB_PATH = self.db_path

    def tearDown(self):
        storage.DB_PATH = self._orig_db_path
        self.db_path.unlink(missing_ok=True)

    def test_empty_db_does_not_crash(self):
        report = full_reliability_report(registry_version=1)
        self.assertEqual(report["n_total_submissions"], 0)
        self.assertIn("internal_consistency", report)
        self.assertIn("test_retest", report)

    def test_report_shape_with_data(self):
        code = storage.create_participant(self.db_path)
        storage.save_submission(code, 1, _full_answers(), db_path=self.db_path)
        report = full_reliability_report(registry_version=1)
        self.assertEqual(report["n_total_submissions"], 1)
        self.assertIsInstance(report["inter_axis_correlations"], dict)


if __name__ == "__main__":
    unittest.main()
