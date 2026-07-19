"""Tests for the job-side validation protocol (research/job_extraction_*.py)."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.models import DetectedSignal
from engine.job_extraction import load_signal_library, signals_to_axes
from research import job_extraction_log as log_module
from research.job_extraction_validation import (
    calibrate_signal_vocabulary, compute_extraction_stability,
    compute_paraphrase_stability, _jaccard,
    MIN_EXTRACTIONS_FOR_RELIABLE_CALIBRATION,
)
from research.job_annotation_agreement import score_agreement


class TestJobExtractionLog(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp.name)
        log_module.init_db(self.db_path)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def test_log_and_retrieve(self):
        log_module.log_extraction(
            "Une description de poste.", ["cadre_reglementaire"],
            {"cognitive_structuring": {"level": 0.8, "importance": 0.7}},
            db_path=self.db_path,
        )
        rows = log_module.all_extractions(self.db_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["detected_signals"], ["cadre_reglementaire"])

    def test_multiple_logs_accumulate(self):
        for i in range(5):
            log_module.log_extraction(f"desc {i}", [], {}, db_path=self.db_path)
        self.assertEqual(len(log_module.all_extractions(self.db_path)), 5)


class TestSignalCalibration(unittest.TestCase):

    def setUp(self):
        self.library = load_signal_library()

    def test_below_sample_floor_not_flagged(self):
        rows = [{"detected_signals": ["cadre_reglementaire"]} for _ in range(5)]
        report = calibrate_signal_vocabulary(rows, self.library)
        self.assertFalse(report["sample_sufficient"])
        for sig in report["signals"].values():
            self.assertFalse(sig["flagged_never_fires"])
            self.assertFalse(sig["flagged_always_fires"])

    def test_signal_that_never_fires_is_flagged_above_sample_floor(self):
        n = MIN_EXTRACTIONS_FOR_RELIABLE_CALIBRATION + 5
        rows = [{"detected_signals": ["cadre_reglementaire"]} for _ in range(n)]
        report = calibrate_signal_vocabulary(rows, self.library)
        self.assertTrue(report["sample_sufficient"])
        never_fired_signal = next(s.id for s in self.library if s.id != "cadre_reglementaire")
        self.assertTrue(report["signals"][never_fired_signal]["flagged_never_fires"])
        self.assertFalse(report["signals"]["cadre_reglementaire"]["flagged_never_fires"])

    def test_signal_that_always_fires_is_flagged(self):
        n = MIN_EXTRACTIONS_FOR_RELIABLE_CALIBRATION + 5
        all_ids = [s.id for s in self.library]
        rows = [{"detected_signals": list(all_ids)} for _ in range(n)]
        report = calibrate_signal_vocabulary(rows, self.library)
        for sig_id in all_ids:
            self.assertTrue(report["signals"][sig_id]["flagged_always_fires"])

    def test_empty_log_does_not_crash(self):
        report = calibrate_signal_vocabulary([], self.library)
        self.assertEqual(report["n_extractions"], 0)
        self.assertFalse(report["sample_sufficient"])


class TestJaccard(unittest.TestCase):
    def test_identical_sets(self):
        self.assertEqual(_jaccard({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint_sets(self):
        self.assertEqual(_jaccard({"a"}, {"b"}), 0.0)

    def test_both_empty_returns_none(self):
        self.assertIsNone(_jaccard(set(), set()))

    def test_partial_overlap(self):
        self.assertAlmostEqual(_jaccard({"a", "b"}, {"b", "c"}), 1 / 3)


class TestExtractionStability(unittest.TestCase):

    def setUp(self):
        self.library = load_signal_library()

    def test_perfectly_deterministic_extractor_gives_jaccard_one(self):
        def fake_extract(description, library):
            return [DetectedSignal(signal_id="cadre_reglementaire", source_phrase="x")]

        result = compute_extraction_stability("desc", self.library, n_runs=4, extract_fn=fake_extract)
        self.assertEqual(result.mean_pairwise_jaccard, 1.0)
        for stddev in result.axis_level_stddev.values():
            self.assertEqual(stddev, 0.0)

    def test_inconsistent_extractor_gives_lower_jaccard(self):
        calls = {"n": 0}

        def flaky_extract(description, library):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                return [DetectedSignal(signal_id="cadre_reglementaire", source_phrase="x")]
            return [DetectedSignal(signal_id="innovation_produit", source_phrase="y")]

        result = compute_extraction_stability("desc", self.library, n_runs=4, extract_fn=flaky_extract)
        self.assertLess(result.mean_pairwise_jaccard, 1.0)

    def test_no_signals_detected_gives_none_jaccard_not_crash(self):
        def empty_extract(description, library):
            return []

        result = compute_extraction_stability("desc", self.library, n_runs=3, extract_fn=empty_extract)
        self.assertIsNone(result.mean_pairwise_jaccard)


class TestParaphraseStability(unittest.TestCase):

    def setUp(self):
        self.library = load_signal_library()

    def test_identical_extraction_on_both_gives_perfect_stability(self):
        def fake_extract(description, library):
            return [DetectedSignal(signal_id="cadre_reglementaire", source_phrase="x")]

        result = compute_paraphrase_stability("original", "paraphrase", self.library, extract_fn=fake_extract)
        self.assertEqual(result.signal_jaccard, 1.0)
        self.assertEqual(result.axis_level_mean_abs_diff, 0.0)

    def test_divergent_extraction_flagged_by_low_jaccard(self):
        def diverging_extract(description, library):
            if description == "original":
                return [DetectedSignal(signal_id="cadre_reglementaire", source_phrase="x")]
            return [DetectedSignal(signal_id="innovation_produit", source_phrase="y")]

        result = compute_paraphrase_stability("original", "paraphrase", self.library, extract_fn=diverging_extract)
        self.assertEqual(result.signal_jaccard, 0.0)


class TestAnnotationAgreement(unittest.TestCase):

    def setUp(self):
        self.library = load_signal_library()

    def test_perfect_agreement_gives_precision_recall_one(self):
        def fake_extract(description, library):
            return [DetectedSignal(signal_id="cadre_reglementaire", source_phrase="x")]

        annotations = [
            {"description": "d1", "human_signal_ids": ["cadre_reglementaire"]},
            {"description": "d2", "human_signal_ids": ["cadre_reglementaire"]},
        ]
        report = score_agreement(annotations, self.library, extract_fn=fake_extract)
        self.assertEqual(report["overall_precision"], 1.0)
        self.assertEqual(report["overall_recall"], 1.0)

    def test_llm_over_flags_reduces_precision_not_recall(self):
        def over_flagging_extract(description, library):
            return [
                DetectedSignal(signal_id="cadre_reglementaire", source_phrase="x"),
                DetectedSignal(signal_id="innovation_produit", source_phrase="y"),
            ]

        annotations = [{"description": "d1", "human_signal_ids": ["cadre_reglementaire"]}]
        report = score_agreement(annotations, self.library, extract_fn=over_flagging_extract)
        self.assertEqual(report["overall_recall"], 1.0)
        self.assertLess(report["overall_precision"], 1.0)

    def test_llm_under_flags_reduces_recall_not_precision(self):
        def under_flagging_extract(description, library):
            return []

        annotations = [{"description": "d1", "human_signal_ids": ["cadre_reglementaire"]}]
        report = score_agreement(annotations, self.library, extract_fn=under_flagging_extract)
        self.assertEqual(report["overall_recall"], 0.0)
        self.assertIsNone(report["overall_precision"])  # no positive predictions at all


if __name__ == "__main__":
    unittest.main()
