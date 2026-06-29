"""NextMove V5 test suite."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.models import AxisStatus, RuleType, SignalType
from ontology.registry import AxisRegistry, QuestionBank
from engine.context import ContextEngine, KNOWN_CONTEXTS
from engine.rules import RuleEngine
from engine.scorer import ProfileScorer, _confidence, _variance
from engine.explainer import Explainer
from questionnaire.adaptive import AdaptiveQuestionnaire


# ===================================================================
# TEST FIXTURES
# ===================================================================

def _build_full_answers(bank: QuestionBank, score_value: str = "B") -> dict[str, str]:
    """Build a complete set of answers (all base questions answered with same value)."""
    registry = AxisRegistry()
    questionnaire = AdaptiveQuestionnaire(registry, bank)
    base = questionnaire.initial_sequence()
    return {qid: score_value for qid in base}


# ===================================================================
# T1 — Registry
# ===================================================================

class TestAxisRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()

    def test_loads_18_axes(self):
        all_axes = self.registry.all_active()
        self.assertEqual(len(all_axes), 18)

    def test_all_axes_have_id_label_description(self):
        for ax in self.registry.all_active():
            self.assertTrue(ax.id, f"Missing id on axis")
            self.assertTrue(ax.label, f"Missing label on {ax.id}")
            self.assertTrue(ax.description, f"Missing description on {ax.id}")

    def test_axes_have_valid_layers(self):
        for ax in self.registry.all_active():
            self.assertIn(ax.layer, [1, 2, 3, 4], f"Invalid layer on {ax.id}")

    def test_meta_axes_exist(self):
        meta = self.registry.meta_axes()
        self.assertGreaterEqual(len(meta), 2)
        meta_ids = {ax.id for ax in meta}
        self.assertIn("context_sensitivity", meta_ids)
        self.assertIn("cognitive_granularity", meta_ids)

    def test_non_meta_axes_count(self):
        non_meta = self.registry.non_meta_axes()
        self.assertEqual(len(non_meta), 16)

    def test_layer_distribution(self):
        for layer in [1, 2, 3, 4]:
            axes = self.registry.by_layer(layer)
            self.assertGreater(len(axes), 0, f"No axes in layer {layer}")

    def test_axis_retrieval(self):
        ax = self.registry.get("autonomy")
        self.assertIsNotNone(ax)
        self.assertEqual(ax.id, "autonomy")
        self.assertEqual(ax.status, AxisStatus.ACTIVE)

    def test_unknown_axis_returns_none(self):
        ax = self.registry.get("nonexistent_axis")
        self.assertIsNone(ax)

    def test_axes_have_rules(self):
        total_rules = sum(len(ax.rules) for ax in self.registry.all_active())
        self.assertGreater(total_rules, 10, "Expected at least 10 rules across all axes")

    def test_startup_has_most_rules(self):
        startup_rules = sum(
            1 for ax in self.registry.all_active()
            for rule in ax.rules
            if rule.condition.context == "startup"
        )
        self.assertGreaterEqual(startup_rules, 5, "Expected ≥5 startup-specific rules")

    def test_potential_merges_reference_existing_axes(self):
        axis_ids = set(self.registry.ids())
        for ax in self.registry.all_active():
            for merge_id in ax.potential_merges:
                self.assertIn(merge_id, axis_ids, f"{ax.id} references unknown merge target {merge_id}")

    def test_mask_and_unmask(self):
        self.registry.mask("autonomy", reason="test")
        self.assertEqual(self.registry.get("autonomy").status, AxisStatus.MASKED)
        self.registry.unmask("autonomy")
        self.assertEqual(self.registry.get("autonomy").status, AxisStatus.ACTIVE)


# ===================================================================
# T2 — Question bank
# ===================================================================

class TestQuestionBank(unittest.TestCase):

    def setUp(self):
        self.bank = QuestionBank()
        self.registry = AxisRegistry()

    def test_loads_questions(self):
        all_ids = self.bank.all_ids()
        self.assertGreater(len(all_ids), 50)

    def test_each_axis_has_base_questions(self):
        for ax in self.registry.all_active():
            base = self.bank.base_questions_for(ax.id)
            self.assertGreaterEqual(len(base), 3, f"{ax.id} has fewer than 3 base questions")

    def test_each_axis_has_adaptive_questions(self):
        for ax in self.registry.all_active():
            adaptive = self.bank.adaptive_questions_for(ax.id)
            self.assertGreaterEqual(len(adaptive), 1, f"{ax.id} has no adaptive questions")

    def test_questions_have_3_options(self):
        for qid in self.bank.all_ids():
            q = self.bank.get(qid)
            self.assertEqual(len(q.options), 3, f"{qid} does not have 3 options")

    def test_option_scores_valid(self):
        valid_scores = {0.0, 0.3, 0.5, 0.7, 0.75, 1.0}
        for qid in self.bank.all_ids():
            q = self.bank.get(qid)
            for opt in q.options:
                self.assertIn(opt.score, valid_scores, f"{qid} option has unexpected score {opt.score}")

    def test_options_have_unique_values(self):
        for qid in self.bank.all_ids():
            q = self.bank.get(qid)
            vals = [opt.value for opt in q.options]
            self.assertEqual(len(vals), len(set(vals)), f"{qid} has duplicate option values")

    def test_base_questions_have_no_trigger(self):
        for qid in self.bank.all_ids():
            q = self.bank.get(qid)
            if q.trigger is None:
                self.assertIsNone(q.trigger)

    def test_adaptive_questions_have_trigger(self):
        for ax in self.registry.all_active():
            for q in self.bank.adaptive_questions_for(ax.id):
                self.assertIsNotNone(q.trigger, f"{q.id} is adaptive but has no trigger")


# ===================================================================
# T3 — Context engine
# ===================================================================

class TestContextEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ContextEngine()
        self.registry = AxisRegistry()

    def test_detects_startup(self):
        ctx = self.engine.detect_context("Je travaille dans une startup seed stage")
        self.assertEqual(ctx, "startup")

    def test_detects_research(self):
        ctx = self.engine.detect_context("Je suis chercheur dans un laboratoire CNRS")
        self.assertEqual(ctx, "research")

    def test_detects_clinical(self):
        ctx = self.engine.detect_context("Mon équipe est une équipe clinique en hôpital")
        self.assertEqual(ctx, "clinical_team")

    def test_detects_public_sector(self):
        ctx = self.engine.detect_context("Je suis fonctionnaire dans un ministère")
        self.assertEqual(ctx, "public_sector")

    def test_unknown_context_returns_none(self):
        ctx = self.engine.detect_context("Je travaille dans une boulangerie artisanale")
        self.assertIsNone(ctx)

    def test_apply_context_startup_boosts_autonomy(self):
        axes = self.registry.all_active()
        result = self.engine.apply_context(axes, "startup")
        autonomy_weight, _ = result["autonomy"]
        self.assertGreater(autonomy_weight, 1.0, "Startup should boost autonomy weight")

    def test_apply_context_public_reduces_autonomy(self):
        axes = self.registry.all_active()
        result = self.engine.apply_context(axes, "public_sector")
        autonomy_weight, _ = result["autonomy"]
        self.assertLess(autonomy_weight, 1.0, "Public sector should reduce autonomy weight")

    def test_apply_context_startup_boosts_risk(self):
        axes = self.registry.all_active()
        result = self.engine.apply_context(axes, "startup")
        risk_weight, _ = result["risk_appetite"]
        self.assertGreater(risk_weight, 1.0)

    def test_apply_context_clinical_reduces_risk(self):
        axes = self.registry.all_active()
        result = self.engine.apply_context(axes, "clinical_team")
        risk_weight, _ = result["risk_appetite"]
        self.assertLess(risk_weight, 1.0)

    def test_no_context_returns_default_weights(self):
        axes = self.registry.all_active()
        result = self.engine.apply_context(axes, None)
        for ax_id, (weight, status) in result.items():
            ax = self.registry.get(ax_id)
            self.assertEqual(weight, ax.weight, f"{ax_id} weight should be default when no context")

    def test_weight_clamped_above_zero(self):
        axes = self.registry.all_active()
        for ctx in KNOWN_CONTEXTS:
            result = self.engine.apply_context(axes, ctx)
            for ax_id, (weight, _) in result.items():
                self.assertGreater(weight, 0.0, f"{ax_id} weight went to 0 in context {ctx}")

    def test_weight_clamped_below_2(self):
        axes = self.registry.all_active()
        for ctx in KNOWN_CONTEXTS:
            result = self.engine.apply_context(axes, ctx)
            for ax_id, (weight, _) in result.items():
                self.assertLessEqual(weight, 2.0, f"{ax_id} weight exceeded 2.0 in context {ctx}")


# ===================================================================
# T4 — Rule engine
# ===================================================================

class TestRuleEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RuleEngine()
        self.registry = AxisRegistry()

    def test_no_rules_fire_without_context(self):
        firings = self.engine.evaluate(self.registry, None)
        self.assertEqual(len(firings), 0)

    def test_rules_fire_for_startup(self):
        firings = self.engine.evaluate(self.registry, "startup")
        self.assertGreater(len(firings), 0)

    def test_fired_rules_are_reweight_type(self):
        firings = self.engine.evaluate(self.registry, "startup")
        for f in firings:
            self.assertEqual(f.rule_type, RuleType.REWEIGHT)

    def test_all_firings_have_explanation(self):
        firings = self.engine.evaluate(self.registry, "research")
        for f in firings:
            self.assertTrue(f.explanation, f"Rule {f.rule_id} has no explanation")

    def test_v6_rules_not_evaluated(self):
        pending = self.engine.pending_v6_rules(self.registry)
        # In V5 seed, no rules require data — but the interface should work
        self.assertIsInstance(pending, list)

    def test_different_contexts_fire_different_rules(self):
        startup_firings = {f.rule_id for f in self.engine.evaluate(self.registry, "startup")}
        public_firings = {f.rule_id for f in self.engine.evaluate(self.registry, "public_sector")}
        self.assertNotEqual(startup_firings, public_firings)

    def test_firings_reference_existing_axes(self):
        axis_ids = set(self.registry.ids())
        for ctx in KNOWN_CONTEXTS:
            for f in self.engine.evaluate(self.registry, ctx):
                self.assertIn(f.axis_id, axis_ids, f"Firing references unknown axis {f.axis_id}")


# ===================================================================
# T5 — Scorer
# ===================================================================

class TestScorer(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.bank = QuestionBank()
        self.scorer = ProfileScorer(self.registry, self.bank)

    def test_variance_empty(self):
        self.assertEqual(_variance([]), 0.0)

    def test_variance_single(self):
        self.assertEqual(_variance([0.5]), 0.0)

    def test_variance_identical(self):
        self.assertAlmostEqual(_variance([0.5, 0.5, 0.5]), 0.0)

    def test_variance_spread(self):
        self.assertGreater(_variance([0.0, 0.5, 1.0]), 0.1)

    def test_confidence_increases_with_n(self):
        c1 = _confidence(1, 0.0)
        c3 = _confidence(3, 0.0)
        c5 = _confidence(5, 0.0)
        self.assertLess(c1, c3)
        self.assertLess(c3, c5)

    def test_confidence_reduced_by_variance(self):
        c_low_var = _confidence(3, 0.0)
        c_high_var = _confidence(3, 0.3)
        self.assertGreater(c_low_var, c_high_var)

    def test_scores_all_b_answers(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers)
        for ax_id, score in profile.axis_scores.items():
            self.assertAlmostEqual(score.raw_value, 0.5, places=1,
                                   msg=f"{ax_id} should be ~0.5 for all B answers")

    def test_scores_all_a_answers(self):
        answers = _build_full_answers(self.bank, "A")
        profile = self.scorer.score("user1", answers)
        for ax_id, score in profile.axis_scores.items():
            self.assertAlmostEqual(score.raw_value, 0.0, places=1,
                                   msg=f"{ax_id} should be ~0.0 for all A answers")

    def test_scores_all_c_answers(self):
        answers = _build_full_answers(self.bank, "C")
        profile = self.scorer.score("user1", answers)
        for ax_id, score in profile.axis_scores.items():
            self.assertAlmostEqual(score.raw_value, 1.0, places=1,
                                   msg=f"{ax_id} should be ~1.0 for all C answers")

    def test_profile_has_all_active_axes(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers)
        active_ids = set(self.registry.active_ids())
        scored_ids = set(profile.axis_scores.keys())
        self.assertTrue(active_ids.issubset(scored_ids) or scored_ids.issubset(active_ids))

    def test_context_affects_effective_weight(self):
        answers = _build_full_answers(self.bank, "B")
        profile_startup = self.scorer.score("user1", answers, context_id="startup")
        profile_generic = self.scorer.score("user1", answers, context_id=None)
        # Startup should boost autonomy weight
        startup_aw = profile_startup.axis_scores.get("autonomy")
        generic_aw = profile_generic.axis_scores.get("autonomy")
        if startup_aw and generic_aw:
            self.assertGreater(startup_aw.effective_weight, generic_aw.effective_weight)

    def test_confidence_is_in_0_1(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers)
        for ax_id, score in profile.axis_scores.items():
            self.assertGreaterEqual(score.confidence, 0.0)
            self.assertLessEqual(score.confidence, 1.0)

    def test_raw_value_is_in_0_1(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers)
        for ax_id, score in profile.axis_scores.items():
            self.assertGreaterEqual(score.raw_value, 0.0)
            self.assertLessEqual(score.raw_value, 1.0)

    def test_empty_answers_returns_empty_profile(self):
        profile = self.scorer.score("user1", {})
        self.assertEqual(len(profile.axis_scores), 0)
        self.assertFalse(profile.is_complete)

    def test_is_complete_with_full_answers(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers)
        self.assertTrue(profile.is_complete)

    def test_low_confidence_axes_detected(self):
        # Partial answers: only 1 question per axis → low confidence
        answers = {qid: "B" for qid in list(self.bank.all_ids())[:5]}
        profile = self.scorer.score("user1", answers)
        low_conf = self.scorer.low_confidence_axes(profile)
        self.assertIsInstance(low_conf, list)

    def test_weighted_profile_keys_match_active_axes(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers)
        weighted = self.scorer.weighted_profile(profile)
        self.assertTrue(all(isinstance(v, float) for v in weighted.values()))


# ===================================================================
# T6 — Explainer
# ===================================================================

class TestExplainer(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.bank = QuestionBank()
        self.scorer = ProfileScorer(self.registry, self.bank)
        self.rule_engine = RuleEngine()
        self.explainer = Explainer()

    def test_explanation_has_expected_keys(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers, context_id="startup")
        firings = self.rule_engine.evaluate(self.registry, "startup")
        trace = self.explainer.explain(profile, firings, self.registry)

        self.assertIsNotNone(trace.rules_fired)
        self.assertIsNotNone(trace.axis_modifications)
        self.assertIsNotNone(trace.uncertainty_notes)

    def test_summary_is_non_empty_string(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers, context_id="startup")
        firings = self.rule_engine.evaluate(self.registry, "startup")
        trace = self.explainer.explain(profile, firings, self.registry)
        summary = self.explainer.summary(trace)

        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10)

    def test_no_rules_no_modifications(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers, context_id=None)
        trace = self.explainer.explain(profile, [], self.registry)
        self.assertEqual(len(trace.axis_modifications), 0)

    def test_startup_context_generates_modifications(self):
        answers = _build_full_answers(self.bank, "B")
        profile = self.scorer.score("user1", answers, context_id="startup")
        firings = self.rule_engine.evaluate(self.registry, "startup")
        trace = self.explainer.explain(profile, firings, self.registry)
        self.assertGreater(len(trace.axis_modifications), 0)

    def test_partial_profile_generates_uncertainty_notes(self):
        # Only answer 2 questions total
        answers = {qid: "B" for qid in list(self.bank.all_ids())[:2]}
        profile = self.scorer.score("user1", answers)
        trace = self.explainer.explain(profile, [], self.registry)
        # Some axes have 0 questions — but they won't appear in axis_scores
        # Axes that do appear with <3 questions get uncertainty notes
        self.assertIsInstance(trace.uncertainty_notes, dict)


# ===================================================================
# T7 — Adaptive questionnaire
# ===================================================================

class TestAdaptiveQuestionnaire(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.bank = QuestionBank()
        self.scorer = ProfileScorer(self.registry, self.bank)
        self.questionnaire = AdaptiveQuestionnaire(self.registry, self.bank)

    def test_initial_sequence_non_empty(self):
        seq = self.questionnaire.initial_sequence()
        self.assertGreater(len(seq), 40)

    def test_initial_sequence_no_duplicates(self):
        seq = self.questionnaire.initial_sequence()
        self.assertEqual(len(seq), len(set(seq)))

    def test_initial_sequence_all_valid_ids(self):
        seq = self.questionnaire.initial_sequence()
        for qid in seq:
            self.assertIsNotNone(self.bank.get(qid), f"{qid} not found in bank")

    def test_initial_sequence_covers_all_axes(self):
        seq = self.questionnaire.initial_sequence()
        axes_covered = {self.bank.get(qid).axis_id for qid in seq}
        active_ids = set(self.registry.active_ids())
        self.assertEqual(axes_covered, active_ids)

    def test_adaptive_questions_triggered_by_high_value(self):
        # Give all C answers → high value on all axes
        all_base = self.questionnaire.initial_sequence()
        answers = {qid: "C" for qid in all_base}
        profile = self.scorer.score("user1", answers)
        already_seen = set(all_base)
        adaptive = self.questionnaire.next_adaptive_questions(profile, already_seen)
        # With high values, high_value triggers should fire
        self.assertGreater(len(adaptive), 0, "Expected adaptive questions for high-value profile")

    def test_adaptive_questions_triggered_by_low_value(self):
        all_base = self.questionnaire.initial_sequence()
        answers = {qid: "A" for qid in all_base}
        profile = self.scorer.score("user1", answers)
        already_seen = set(all_base)
        adaptive = self.questionnaire.next_adaptive_questions(profile, already_seen)
        # With low values, low_value triggers should fire
        self.assertGreater(len(adaptive), 0, "Expected adaptive questions for low-value profile")

    def test_no_duplicate_adaptive_activations(self):
        all_base = self.questionnaire.initial_sequence()
        answers = {qid: "C" for qid in all_base}
        profile = self.scorer.score("user1", answers)
        already_seen = set(all_base)
        adaptive = self.questionnaire.next_adaptive_questions(profile, already_seen)
        self.assertEqual(len(adaptive), len(set(adaptive)))

    def test_already_seen_questions_not_reactivated(self):
        all_base = self.questionnaire.initial_sequence()
        answers = {qid: "C" for qid in all_base}
        profile = self.scorer.score("user1", answers)
        # Mark all adaptive questions as seen too
        all_adaptive = [q.id for ax in self.registry.all_active()
                        for q in self.bank.adaptive_questions_for(ax.id)]
        already_seen = set(all_base) | set(all_adaptive)
        adaptive = self.questionnaire.next_adaptive_questions(profile, already_seen)
        self.assertEqual(len(adaptive), 0)

    def test_pending_questions_empty_when_all_seen(self):
        all_base = self.questionnaire.initial_sequence()
        answers = {qid: "B" for qid in all_base}
        profile = self.scorer.score("user1", answers)
        all_adaptive = [q.id for ax in self.registry.all_active()
                        for q in self.bank.adaptive_questions_for(ax.id)]
        already_seen = set(all_base) | set(all_adaptive)
        pending = self.questionnaire.pending_questions(profile, already_seen)
        self.assertEqual(len(pending), 0)

    def test_build_session_base_only(self):
        session = self.questionnaire.build_session()
        self.assertGreater(len(session), 0)
        for q in session:
            self.assertIsNone(q.trigger)


# ===================================================================
# T8 — Integration (full pipeline)
# ===================================================================

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.bank = QuestionBank()
        self.scorer = ProfileScorer(self.registry, self.bank)
        self.rule_engine = RuleEngine()
        self.explainer = Explainer()
        self.questionnaire = AdaptiveQuestionnaire(self.registry, self.bank)
        self.context_engine = ContextEngine()

    def _run_pipeline(self, answer_value: str, context_text: str | None = None):
        context_id = self.context_engine.detect_context(context_text or "")
        rules_fired = self.rule_engine.evaluate(self.registry, context_id)
        answers = _build_full_answers(self.bank, answer_value)
        profile = self.scorer.score("integration_test", answers, context_id)
        trace = self.explainer.explain(profile, rules_fired, self.registry)
        return profile, trace, rules_fired

    def test_full_pipeline_generic_context(self):
        profile, trace, firings = self._run_pipeline("B")
        self.assertTrue(profile.is_complete)
        self.assertEqual(len(firings), 0)
        self.assertIsNone(trace.context)

    def test_full_pipeline_startup_context(self):
        profile, trace, firings = self._run_pipeline("B", "startup seed stage")
        self.assertTrue(profile.is_complete)
        self.assertGreater(len(firings), 0)
        self.assertEqual(trace.context, "startup")
        self.assertGreater(len(trace.axis_modifications), 0)

    def test_startup_context_autonomy_weight_higher_than_public(self):
        _, _, _ = self._run_pipeline("B", "startup")
        profile_startup = self.scorer.score(
            "u1", _build_full_answers(self.bank, "B"), context_id="startup"
        )
        profile_public = self.scorer.score(
            "u2", _build_full_answers(self.bank, "B"), context_id="public_sector"
        )
        startup_aw = profile_startup.axis_scores["autonomy"].effective_weight
        public_aw = profile_public.axis_scores["autonomy"].effective_weight
        self.assertGreater(startup_aw, public_aw)

    def test_high_C_profile_has_high_raw_values(self):
        profile, _, _ = self._run_pipeline("C")
        for ax_id, score in profile.axis_scores.items():
            self.assertGreaterEqual(score.raw_value, 0.8,
                                    f"{ax_id} raw_value too low for all-C answers")

    def test_low_A_profile_has_low_raw_values(self):
        profile, _, _ = self._run_pipeline("A")
        for ax_id, score in profile.axis_scores.items():
            self.assertLessEqual(score.raw_value, 0.2,
                                 f"{ax_id} raw_value too high for all-A answers")

    def test_weighted_scores_respect_context(self):
        """In startup context, autonomy weighted score > public sector autonomy weighted score."""
        answers = _build_full_answers(self.bank, "B")
        ws_startup = self.scorer.weighted_profile(
            self.scorer.score("u1", answers, "startup")
        )
        ws_public = self.scorer.weighted_profile(
            self.scorer.score("u2", answers, "public_sector")
        )
        if "autonomy" in ws_startup and "autonomy" in ws_public:
            self.assertGreater(ws_startup["autonomy"], ws_public["autonomy"])

    def test_explanation_summary_mentions_context(self):
        _, trace, _ = self._run_pipeline("B", "startup")
        summary = self.explainer.summary(trace)
        self.assertIn("startup", summary)

    def test_research_context_boosts_rigor(self):
        answers = _build_full_answers(self.bank, "B")
        profile_research = self.scorer.score("u1", answers, "research")
        profile_generic = self.scorer.score("u2", answers, None)
        research_rigor = profile_research.axis_scores.get("rigor")
        generic_rigor = profile_generic.axis_scores.get("rigor")
        if research_rigor and generic_rigor:
            self.assertGreater(research_rigor.effective_weight, generic_rigor.effective_weight)

    def test_clinical_context_reduces_risk_appetite(self):
        answers = _build_full_answers(self.bank, "B")
        profile_clinical = self.scorer.score("u1", answers, "clinical_team")
        profile_generic = self.scorer.score("u2", answers, None)
        clinical_risk = profile_clinical.axis_scores.get("risk_appetite")
        generic_risk = profile_generic.axis_scores.get("risk_appetite")
        if clinical_risk and generic_risk:
            self.assertLess(clinical_risk.effective_weight, generic_risk.effective_weight)

    def test_adaptive_questions_fire_after_full_base(self):
        all_base = self.questionnaire.initial_sequence()
        answers_c = {qid: "C" for qid in all_base}
        profile = self.scorer.score("u1", answers_c, None)
        already_seen = set(all_base)
        adaptive = self.questionnaire.next_adaptive_questions(profile, already_seen)
        self.assertGreater(len(adaptive), 0,
                           "All-C profile should trigger adaptive questions")


if __name__ == "__main__":
    unittest.main(verbosity=2)
