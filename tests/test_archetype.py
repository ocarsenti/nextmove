"""Tests for engine/archetype.py — Builder/Expert/Operator/Leader/Connecteur on 18 axes."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.registry import AxisRegistry, QuestionBank
from engine.scorer import ProfileScorer
from engine.archetype import compute_archetype_distribution, ARCHETYPE_AXIS_AFFINITY, ARCHETYPE_NAMES
from questionnaire.adaptive import AdaptiveQuestionnaire


def _build_full_answers(bank: QuestionBank, registry: AxisRegistry, score_value: str = "B") -> dict[str, str]:
    questionnaire = AdaptiveQuestionnaire(registry, bank)
    base = questionnaire.initial_sequence()
    return {qid: score_value for qid in base}


class TestAffinityTable(unittest.TestCase):

    def test_five_archetypes(self):
        self.assertEqual(set(ARCHETYPE_AXIS_AFFINITY.keys()), {"Builder", "Expert", "Operator", "Leader", "Connecteur"})

    def test_weights_sum_to_one(self):
        for name, weights in ARCHETYPE_AXIS_AFFINITY.items():
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=4, msg=f"{name} weights don't sum to 1.0")

    def test_no_meta_axis_in_any_signature(self):
        meta_axes = {"cognitive_granularity", "context_sensitivity"}
        for name, weights in ARCHETYPE_AXIS_AFFINITY.items():
            self.assertFalse(set(weights.keys()) & meta_axes, f"{name} uses a meta axis")

    def test_no_motivational_axis_in_any_signature(self):
        # value_orientation is the ontology's one clearly motivational ("why") axis;
        # archetypes here are deliberately behavioral-only ("how") — see engine/archetype.py docstring.
        for name, weights in ARCHETYPE_AXIS_AFFINITY.items():
            self.assertNotIn("value_orientation", weights, f"{name} mixes a motivational axis into a behavioral signature")


class TestArchitecturalIsolation(unittest.TestCase):
    """Archetypes must be a read-only interpretation layer: the tension engine
    (engine/matching.py) must never import or depend on engine/archetype.py,
    so archetype labels can never silently influence fit_score or tensions."""

    def test_matching_module_does_not_import_archetype(self):
        import engine.matching as matching_module
        source = Path(matching_module.__file__).read_text()
        self.assertNotIn("archetype", source.lower())


class TestComputeArchetypeDistribution(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.bank = QuestionBank()
        self.scorer = ProfileScorer(self.registry, self.bank)

    def test_full_profile_produces_five_scores_summing_to_100(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="B")
        profile = self.scorer.score("candidate_1", answers)
        dist = compute_archetype_distribution(profile, self.registry)
        self.assertEqual(len(dist.scores), 5)
        self.assertEqual({s.name for s in dist.scores}, set(ARCHETYPE_NAMES))
        self.assertAlmostEqual(sum(s.percentage for s in dist.scores), 100.0, places=1)

    def test_scores_sorted_descending(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="C")
        profile = self.scorer.score("candidate_2", answers)
        dist = compute_archetype_distribution(profile, self.registry)
        percentages = [s.percentage for s in dist.scores]
        self.assertEqual(percentages, sorted(percentages, reverse=True))

    def test_dominant_matches_top_score(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="C")
        profile = self.scorer.score("candidate_3", answers)
        dist = compute_archetype_distribution(profile, self.registry)
        self.assertEqual(dist.dominant, dist.scores[0].name)

    def test_empty_profile_is_unscored_not_crashing(self):
        profile = self.scorer.score("candidate_empty", {})
        dist = compute_archetype_distribution(profile, self.registry)
        self.assertTrue(dist.low_confidence)
        self.assertTrue(all(s.percentage == 0.0 for s in dist.scores) or dist.dominant == "Unscored")

    def test_masking_an_axis_renormalizes_that_archetypes_weights(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="C")
        profile_before = self.scorer.score("candidate_mask_before", answers)
        dist_before = compute_archetype_distribution(profile_before, self.registry)
        builder_before = next(s for s in dist_before.scores if s.name == "Builder")
        self.assertIn("rigor", builder_before.axes_used)

        self.registry.mask("rigor", reason="test")
        try:
            profile_after = self.scorer.score("candidate_mask_after", answers)
            dist_after = compute_archetype_distribution(profile_after, self.registry)
            builder_after = next(s for s in dist_after.scores if s.name == "Builder")
            self.assertNotIn("rigor", builder_after.axes_used)
            self.assertIn("rigor", builder_after.axes_missing)
            # Remaining weights must still sum to 1.0 post-renormalization —
            # verified indirectly: the archetype must still be scoreable.
            self.assertGreater(builder_after.percentage, 0.0)
        finally:
            self.registry.unmask("rigor")

    def test_all_signature_axes_masked_gives_zero_not_crash(self):
        answers = _build_full_answers(self.bank, self.registry, score_value="B")
        for axis_id in ARCHETYPE_AXIS_AFFINITY["Connecteur"]:
            self.registry.mask(axis_id, reason="test")
        try:
            profile = self.scorer.score("candidate_connecteur_masked", answers)
            dist = compute_archetype_distribution(profile, self.registry)
            connecteur = next(s for s in dist.scores if s.name == "Connecteur")
            self.assertEqual(connecteur.percentage, 0.0)
            self.assertEqual(connecteur.axes_used, [])
        finally:
            for axis_id in ARCHETYPE_AXIS_AFFINITY["Connecteur"]:
                self.registry.unmask(axis_id)

    def test_low_confidence_flagged_on_thin_answers(self):
        base_q = self.bank.base_questions_for("rigor")[0]
        profile = self.scorer.score("candidate_thin", {base_q.id: "B"})
        dist = compute_archetype_distribution(profile, self.registry)
        self.assertTrue(dist.low_confidence)


if __name__ == "__main__":
    unittest.main()
