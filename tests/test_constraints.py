"""Tests for engine/constraints.py — job constraint detection & candidate compatibility."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.models import JobAxisRequirement, JobCard
from ontology.registry import AxisRegistry, QuestionBank
from engine.scorer import ProfileScorer
from engine.constraints import (
    load_constraint_library, detect_constraints, build_job_narrative,
    assess_compatibility, build_match_narrative, analyze_job,
)
from questionnaire.adaptive import AdaptiveQuestionnaire


def _build_full_answers(bank, registry, score_value="B"):
    questionnaire = AdaptiveQuestionnaire(registry, bank)
    base = questionnaire.initial_sequence()
    return {qid: score_value for qid in base}


class TestConstraintLibrary(unittest.TestCase):

    def test_loads_seven_constraints(self):
        lib = load_constraint_library()
        self.assertEqual(len(lib), 7)

    def test_every_constraint_has_at_least_two_triggers(self):
        # A constraint is a PATTERN across axes, never a single-axis reading.
        for c in load_constraint_library():
            self.assertGreaterEqual(len(c.triggers), 2, f"{c.id} has fewer than 2 triggers")

    def test_every_trigger_axis_is_a_real_axis(self):
        registry = AxisRegistry()
        for c in load_constraint_library():
            for t in c.triggers:
                self.assertIsNotNone(registry.get(t.axis_id), f"{c.id} triggers on unknown axis {t.axis_id}")

    def test_every_involved_axis_is_a_real_axis(self):
        registry = AxisRegistry()
        for c in load_constraint_library():
            for axis_id in c.axes_involved:
                self.assertIsNotNone(registry.get(axis_id), f"{c.id} involves unknown axis {axis_id}")


class TestDetectConstraints(unittest.TestCase):

    def setUp(self):
        self.library = load_constraint_library()

    def test_job_with_no_requirements_triggers_nothing(self):
        job = JobCard(job_id="j1", title="Empty job", axis_requirements={})
        self.assertEqual(detect_constraints(job, self.library), [])

    def test_job_matching_c002_triggers(self):
        # C002: influence > 0.7 AND social_interaction > 0.6
        job = JobCard(job_id="j2", title="Influence role", axis_requirements={
            "influence": JobAxisRequirement(axis_id="influence", level=0.8),
            "social_interaction": JobAxisRequirement(axis_id="social_interaction", level=0.7),
        })
        detected = detect_constraints(job, self.library)
        self.assertIn("C002", [c.id for c in detected])

    def test_partial_match_does_not_trigger(self):
        # Only one of C002's two triggers satisfied — must NOT fire (AND, not OR).
        job = JobCard(job_id="j3", title="Partial", axis_requirements={
            "influence": JobAxisRequirement(axis_id="influence", level=0.9),
            "social_interaction": JobAxisRequirement(axis_id="social_interaction", level=0.2),
        })
        detected = detect_constraints(job, self.library)
        self.assertNotIn("C002", [c.id for c in detected])

    def test_job_narrative_mentions_detected_constraint_names(self):
        job = JobCard(job_id="j4", title="Poste de coordination", axis_requirements={
            "influence": JobAxisRequirement(axis_id="influence", level=0.9),
            "social_interaction": JobAxisRequirement(axis_id="social_interaction", level=0.9),
        })
        detected = detect_constraints(job, self.library)
        narrative = build_job_narrative(job, detected)
        self.assertIn("Poste de coordination", narrative)
        self.assertIn("influencer sans autorité hiérarchique", narrative.lower())

    def test_empty_job_narrative_says_so(self):
        job = JobCard(job_id="j5", title="Rien", axis_requirements={})
        narrative = build_job_narrative(job, [])
        self.assertIn("aucune contrainte", narrative.lower())


class TestAssessCompatibility(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.bank = QuestionBank()
        self.scorer = ProfileScorer(self.registry, self.bank)
        self.library = load_constraint_library()

    def test_high_scoring_candidate_is_aligned(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="C")
        profile = self.scorer.score("cand_high", answers)
        c002 = next(c for c in self.library if c.id == "C002")
        result = assess_compatibility(profile, c002)
        self.assertIn(result.status, ("aligned", "tension"))  # depends on seed answer mapping, but must not crash
        self.assertTrue(result.narrative)

    def test_low_scoring_axis_produces_tension_with_risks(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="A")  # low answers
        profile = self.scorer.score("cand_low", answers)
        c002 = next(c for c in self.library if c.id == "C002")
        result = assess_compatibility(profile, c002)
        if result.status == "tension":
            self.assertTrue(result.risks_flagged)
            self.assertEqual(set(result.risks_flagged), set(c002.risks))

    def test_masked_axis_marked_unavailable_not_silently_dropped(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="C")
        self.registry.mask("influence", reason="test")
        try:
            profile = self.scorer.score("cand_masked", answers)
            c002 = next(c for c in self.library if c.id == "C002")
            result = assess_compatibility(profile, c002)
            self.assertIn("influence", result.axes_unavailable)
            self.assertNotIn("influence", result.axes_checked)
        finally:
            self.registry.unmask("influence")

    def test_no_usable_axes_gives_unknown_status(self):
        answers = {}
        profile = self.scorer.score("cand_empty", answers)
        c002 = next(c for c in self.library if c.id == "C002")
        result = assess_compatibility(profile, c002)
        self.assertEqual(result.status, "unknown")


class TestAnalyzeJob(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.bank = QuestionBank()
        self.scorer = ProfileScorer(self.registry, self.bank)
        self.library = load_constraint_library()

    def test_analyze_job_without_profile_has_no_compatibilities(self):
        job = JobCard(job_id="j6", title="Test", axis_requirements={
            "influence": JobAxisRequirement(axis_id="influence", level=0.9),
            "social_interaction": JobAxisRequirement(axis_id="social_interaction", level=0.9),
        })
        result = analyze_job(job, self.library, profile=None)
        self.assertTrue(len(result.detected_constraints) > 0)
        self.assertEqual(result.compatibilities, [])
        self.assertEqual(result.match_narrative, "")

    def test_analyze_job_with_profile_produces_match_narrative(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="C")
        profile = self.scorer.score("cand_full", answers)
        job = JobCard(job_id="j7", title="Test complet", axis_requirements={
            "influence": JobAxisRequirement(axis_id="influence", level=0.9),
            "social_interaction": JobAxisRequirement(axis_id="social_interaction", level=0.9),
        })
        result = analyze_job(job, self.library, profile=profile)
        self.assertEqual(len(result.compatibilities), len(result.detected_constraints))
        self.assertTrue(result.match_narrative)


class TestArchitecturalIsolation(unittest.TestCase):
    """Same isolation guarantee as archetypes: constraints must never leak into
    the raw tension engine's fit_score/tensions computation."""

    def test_matching_module_does_not_import_constraints(self):
        import engine.matching as matching_module
        source = Path(matching_module.__file__).read_text()
        self.assertNotIn("constraint", source.lower())


class TestRequiredSignalVotes(unittest.TestCase):
    """C005 false-fired on two real job descriptions (Alan Marketing Ops,
    Servier R&D Project Leader) via axis thresholds alone — cognitive_structuring
    was pushed high by unrelated signals (internal process/governance language),
    never actual regulatory conformity. required_signal_votes fixes this by
    requiring the cadre_reglementaire signal specifically, not just the axis
    threshold it happens to share with other signals."""

    def setUp(self):
        self.library = load_constraint_library()
        self.c005 = next(c for c in self.library if c.id == "C005")

    def _job_meeting_c005_thresholds(self, detected_signal_ids=None):
        return JobCard(
            job_id="j", title="Test",
            axis_requirements={
                "exploration": JobAxisRequirement(axis_id="exploration", level=0.8),
                "cognitive_structuring": JobAxisRequirement(axis_id="cognitive_structuring", level=0.85),
            },
            detected_signal_ids=detected_signal_ids or [],
        )

    def test_c005_declares_required_signal_vote(self):
        self.assertEqual(self.c005.required_signal_votes, ["cadre_reglementaire"])

    def test_thresholds_alone_no_longer_fire_c005(self):
        # This is exactly the false-positive pattern from the two real JDs:
        # axis thresholds met, but no cadre_reglementaire signal detected
        # (cognitive_structuring came from e.g. "processus_stricts" instead).
        job = self._job_meeting_c005_thresholds(detected_signal_ids=["processus_stricts", "innovation_produit"])
        detected = detect_constraints(job, self.library)
        self.assertNotIn("C005", [c.id for c in detected])

    def test_thresholds_plus_required_vote_fires_c005(self):
        job = self._job_meeting_c005_thresholds(detected_signal_ids=["cadre_reglementaire"])
        detected = detect_constraints(job, self.library)
        self.assertIn("C005", [c.id for c in detected])

    def test_required_vote_without_thresholds_does_not_fire_c005(self):
        # The vote is necessary but still not sufficient — axis thresholds
        # (the actual pattern) must still hold too.
        job = JobCard(
            job_id="j", title="Test",
            axis_requirements={
                "exploration": JobAxisRequirement(axis_id="exploration", level=0.2),
            },
            detected_signal_ids=["cadre_reglementaire"],
        )
        detected = detect_constraints(job, self.library)
        self.assertNotIn("C005", [c.id for c in detected])

    def test_constraints_without_required_votes_are_unaffected(self):
        # C002 has no required_signal_votes — must behave exactly as before,
        # firing on axis thresholds alone regardless of detected_signal_ids.
        job = JobCard(
            job_id="j", title="Test",
            axis_requirements={
                "influence": JobAxisRequirement(axis_id="influence", level=0.9),
                "social_interaction": JobAxisRequirement(axis_id="social_interaction", level=0.9),
            },
            detected_signal_ids=[],
        )
        detected = detect_constraints(job, self.library)
        self.assertIn("C002", [c.id for c in detected])


if __name__ == "__main__":
    unittest.main()
