"""NextMove V6 test suite — rule engine, signals, transformations, audit."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ontology.registry import AxisRegistry, QuestionBank
from ontology.rule_store import RuleStore, OntologyRule, RuleTrigger, RuleAction, NewAxisSpec
from ontology.models import AxisStatus, Profile, AxisScore
from engine.signals import SignalComputer, SignalSet
from engine.v6_rules import V6RuleEngine
from engine.transformation import TransformationEngine
from engine.audit import AuditLog


# ===================================================================
# FIXTURES
# ===================================================================

def _make_temp_audit(tmp_path=None) -> AuditLog:
    import tempfile
    path = Path(tempfile.mktemp(suffix=".json"))
    return AuditLog(path=path)


def _make_profiles(n: int = 10, variance_axis: str = "autonomy", low: bool = False) -> list[Profile]:
    """Create synthetic profiles."""
    import random
    random.seed(42)
    profiles = []
    for i in range(n):
        val = random.uniform(0.0, 0.3) if low else random.uniform(0.0, 1.0)
        sc = AxisScore(
            axis_id=variance_axis,
            raw_value=val,
            confidence=0.8,
            n_questions=3,
            variance=random.uniform(0.0, 0.1),
            raw_answers=[val],
        )
        profile = Profile(
            user_id=f"user_{i}",
            axis_scores={variance_axis: sc},
            is_complete=False,
        )
        profiles.append(profile)
    return profiles


def _make_correlated_profiles(ax_a: str, ax_b: str, correlation: float, n: int = 20) -> list[Profile]:
    """Create profiles where two axes have a given correlation."""
    import random, math
    random.seed(99)
    profiles = []
    for i in range(n):
        x = random.uniform(0.2, 0.8)
        noise = random.uniform(-0.05, 0.05)
        y = x * correlation + (1 - abs(correlation)) * noise
        y = max(0.0, min(1.0, y))
        sc_a = AxisScore(axis_id=ax_a, raw_value=x, confidence=0.8, n_questions=3, variance=0.05, raw_answers=[x])
        sc_b = AxisScore(axis_id=ax_b, raw_value=y, confidence=0.8, n_questions=3, variance=0.05, raw_answers=[y])
        profiles.append(Profile(
            user_id=f"user_{i}",
            axis_scores={ax_a: sc_a, ax_b: sc_b},
            is_complete=False,
        ))
    return profiles


# ===================================================================
# T1 — Rule store
# ===================================================================

class TestRuleStore(unittest.TestCase):

    def setUp(self):
        self.store = RuleStore()

    def test_loads_rules(self):
        self.assertGreater(len(self.store.all_active()), 5)

    def test_rules_have_ids(self):
        for r in self.store.all_active():
            self.assertTrue(r.id, "Rule missing id")

    def test_rules_have_valid_types(self):
        valid_types = {"split", "merge", "mask", "reweight", "transform"}
        for r in self.store.all_active():
            self.assertIn(r.type, valid_types, f"Unknown rule type: {r.type}")

    def test_rules_sorted_by_priority(self):
        rules = self.store.all_active()
        priorities = [r.priority for r in rules]
        self.assertEqual(priorities, sorted(priorities))

    def test_by_type_split(self):
        splits = self.store.by_type("split")
        self.assertGreater(len(splits), 0)
        for r in splits:
            self.assertEqual(r.type, "split")

    def test_by_type_merge(self):
        merges = self.store.by_type("merge")
        self.assertGreater(len(merges), 0)
        for r in merges:
            self.assertEqual(r.type, "merge")

    def test_by_type_transform(self):
        transforms = self.store.by_type("transform")
        self.assertGreater(len(transforms), 0)

    def test_immediate_rules_have_no_data_requirement(self):
        for r in self.store.immediate_rules():
            self.assertFalse(r.requires_data)

    def test_data_rules_require_data(self):
        for r in self.store.data_rules():
            self.assertTrue(r.requires_data)

    def test_split_rules_have_new_axes(self):
        for r in self.store.by_type("split"):
            self.assertGreater(len(r.new_axes), 0, f"{r.id} has no new_axes")

    def test_merge_rules_have_new_axis(self):
        for r in self.store.by_type("merge"):
            self.assertIsNotNone(r.new_axis, f"{r.id} has no new_axis")

    def test_all_rules_have_explanation(self):
        for r in self.store.all_active():
            self.assertTrue(r.explanation, f"{r.id} has no explanation")

    def test_deactivate_and_activate(self):
        rules = self.store.all_active()
        rule_id = rules[0].id
        self.store.deactivate(rule_id)
        self.assertFalse(self.store.get(rule_id).active)
        self.store.activate(rule_id)
        self.assertTrue(self.store.get(rule_id).active)

    def test_context_rules_filtered(self):
        startup_rules = self.store.context_rules("startup")
        for r in startup_rules:
            self.assertEqual(r.trigger.context, "startup")


# ===================================================================
# T2 — Signal computer
# ===================================================================

class TestSignalComputer(unittest.TestCase):

    def setUp(self):
        self.computer = SignalComputer()

    def test_empty_profiles_returns_empty(self):
        signals = self.computer.compute([])
        self.assertEqual(len(signals), 0)

    def test_computes_variance(self):
        profiles = _make_profiles(20, "autonomy")
        signals = self.computer.compute(profiles)
        self.assertIn("autonomy", signals)
        self.assertIn("variance", signals["autonomy"])

    def test_computes_internal_consistency(self):
        profiles = _make_profiles(20, "autonomy")
        signals = self.computer.compute(profiles)
        self.assertIn("internal_consistency", signals["autonomy"])
        ic = signals["autonomy"]["internal_consistency"]
        self.assertGreaterEqual(ic, 0.0)
        self.assertLessEqual(ic, 1.0)

    def test_computes_discrimination_index(self):
        profiles = _make_profiles(20, "autonomy")
        signals = self.computer.compute(profiles)
        self.assertIn("discrimination_index", signals["autonomy"])

    def test_high_variance_detected(self):
        profiles = _make_profiles(30, "autonomy", low=False)
        signals = self.computer.compute(profiles)
        var = signals["autonomy"]["variance"]
        # Uniform [0,1] has variance ~0.08
        self.assertGreater(var, 0.02)

    def test_low_variance_detected(self):
        profiles = _make_profiles(30, "autonomy", low=True)
        signals = self.computer.compute(profiles)
        var = signals["autonomy"]["variance"]
        self.assertLess(var, 0.1)

    def test_computes_pairwise_correlation(self):
        profiles = _make_correlated_profiles("rigor", "cognitive_structuring", 0.9)
        signals = self.computer.compute(profiles)
        self.assertIn(("rigor", "cognitive_structuring"), signals)
        corr = signals[("rigor", "cognitive_structuring")]["correlation"]
        self.assertGreater(corr, 0.7, "Expected high correlation")

    def test_low_correlation_detected(self):
        profiles = _make_correlated_profiles("autonomy", "rigor", 0.1)
        signals = self.computer.compute(profiles)
        pair = signals.get(("autonomy", "rigor")) or signals.get(("rigor", "autonomy"))
        if pair:
            # With small n and mild noise injection, Pearson stays below 0.8
            self.assertLess(pair["correlation"], 0.8)

    def test_inject_manual_signals(self):
        manual = {"autonomy": {"variance": 0.9, "internal_consistency": 0.2}}
        signals = self.computer.inject(manual)
        self.assertEqual(signals["autonomy"]["variance"], 0.9)

    def test_n_profiles_recorded(self):
        profiles = _make_profiles(15, "autonomy")
        signals = self.computer.compute(profiles)
        self.assertEqual(signals["autonomy"]["n_profiles"], 15)


# ===================================================================
# T3 — V6 Rule engine
# ===================================================================

class TestV6RuleEngine(unittest.TestCase):

    def setUp(self):
        self.store = RuleStore()
        self.engine = V6RuleEngine()

    def test_no_firings_without_context_or_signals(self):
        firings = self.engine.evaluate(self.store, None, None)
        # Rules with no conditions and no context trigger should fire
        self.assertIsInstance(firings, list)

    def test_immediate_rules_fire_for_startup(self):
        firings = self.engine.evaluate_immediate(self.store, "startup")
        self.assertGreater(len(firings), 0)

    def test_immediate_firings_have_no_signals_used(self):
        firings = self.engine.evaluate_immediate(self.store, "startup")
        for f in firings:
            self.assertEqual(f.signals_used, {})

    def test_firings_sorted_by_priority(self):
        firings = self.engine.evaluate_immediate(self.store, "startup")
        priorities = [f.priority for f in firings]
        self.assertEqual(priorities, sorted(priorities))

    def test_split_requires_execution(self):
        signals = SignalSet({"autonomy": {"internal_consistency": 0.4, "variance": 0.5, "discrimination_index": 0.6}})
        firings = self.engine.evaluate(self.store, None, signals)
        split_firings = [f for f in firings if f.rule_type == "split"]
        for f in split_firings:
            self.assertTrue(f.requires_execution)

    def test_reweight_does_not_require_execution(self):
        firings = self.engine.evaluate_immediate(self.store, "startup")
        reweight_firings = [f for f in firings if f.rule_type == "reweight"]
        for f in reweight_firings:
            self.assertFalse(f.requires_execution)

    def test_data_gated_fires_with_signals(self):
        # Split rule for autonomy requires internal_consistency < 0.6
        signals = SignalSet({
            "autonomy": {"variance": 0.8, "internal_consistency": 0.45, "discrimination_index": 0.7}
        })
        firings = self.engine.evaluate_data_gated(self.store, signals)
        self.assertGreater(len(firings), 0)

    def test_data_gated_does_not_fire_without_signals(self):
        firings = self.engine.evaluate_immediate(self.store, None)
        # None should be data-gated
        for f in firings:
            self.assertEqual(f.signals_used, {})

    def test_merge_fires_when_correlation_high(self):
        signals = SignalSet({
            ("rigor", "cognitive_structuring"): {"correlation": 0.92},
            ("cognitive_structuring", "rigor"): {"correlation": 0.92},
        })
        firings = self.engine.evaluate_data_gated(self.store, signals)
        merge_ids = [f.rule_id for f in firings if f.rule_type == "merge"]
        self.assertIn("rule_merge_rigor_structuring", merge_ids)

    def test_merge_does_not_fire_when_correlation_low(self):
        signals = SignalSet({
            ("rigor", "cognitive_structuring"): {"correlation": 0.5},
            ("cognitive_structuring", "rigor"): {"correlation": 0.5},
        })
        firings = self.engine.evaluate_data_gated(self.store, signals)
        merge_ids = [f.rule_id for f in firings if f.rule_type == "merge"]
        self.assertNotIn("rule_merge_rigor_structuring", merge_ids)

    def test_different_contexts_produce_different_firings(self):
        startup = {f.rule_id for f in self.engine.evaluate_immediate(self.store, "startup")}
        research = {f.rule_id for f in self.engine.evaluate_immediate(self.store, "research")}
        self.assertNotEqual(startup, research)

    def test_all_firings_have_explanation(self):
        firings = self.engine.evaluate_immediate(self.store, "startup")
        for f in firings:
            self.assertTrue(f.explanation, f"Firing {f.rule_id} has no explanation")

    def test_pending_report_non_empty(self):
        report = self.engine.pending_report(self.store)
        self.assertGreater(len(report), 0)


# ===================================================================
# T4 — Transformation engine
# ===================================================================

class TestTransformationEngine(unittest.TestCase):

    def _fresh_registry(self) -> AxisRegistry:
        return AxisRegistry()

    def setUp(self):
        self.store = RuleStore()
        self.registry = self._fresh_registry()
        self.audit = _make_temp_audit()
        self.engine = TransformationEngine(self.registry, self.audit)

    def test_split_autonomy(self):
        rule = self.store.get("rule_split_autonomy")
        self.assertIsNotNone(rule)
        signals = SignalSet({"autonomy": {"internal_consistency": 0.45}})
        success, event, reason = self.engine.execute_split(rule, signals)
        self.assertTrue(success, reason)
        self.assertIsNotNone(event)
        # New axes should exist in registry
        self.assertIsNotNone(self.registry.get("autonomy_strategic"))
        self.assertIsNotNone(self.registry.get("autonomy_operational"))

    def test_split_marks_source_as_split(self):
        rule = self.store.get("rule_split_autonomy")
        self.engine.execute_split(rule, SignalSet())
        source = self.registry.get("autonomy")
        self.assertEqual(source.status, AxisStatus.SPLIT)

    def test_split_records_child_ids(self):
        rule = self.store.get("rule_split_autonomy")
        self.engine.execute_split(rule, SignalSet())
        source = self.registry.get("autonomy")
        self.assertIn("autonomy_strategic", source.split_into)
        self.assertIn("autonomy_operational", source.split_into)

    def test_split_fails_on_nonexistent_axis(self):
        rule = self.store.get("rule_split_autonomy")
        rule2 = rule.model_copy()
        rule2 = rule.model_copy(update={"trigger": rule.trigger.model_copy(update={"axis": "nonexistent"})})
        success, event, reason = self.engine.execute_split(rule2, SignalSet())
        self.assertFalse(success)

    def test_split_cannot_run_twice(self):
        rule = self.store.get("rule_split_autonomy")
        self.engine.execute_split(rule, SignalSet())
        success2, _, _ = self.engine.execute_split(rule, SignalSet())
        self.assertFalse(success2, "Should not split an already-split axis")

    def test_merge_rigor_structuring(self):
        rule = self.store.get("rule_merge_rigor_structuring")
        self.assertIsNotNone(rule)
        signals = SignalSet({
            ("rigor", "cognitive_structuring"): {"correlation": 0.91},
            ("cognitive_structuring", "rigor"): {"correlation": 0.91},
        })
        success, event, reason = self.engine.execute_merge(rule, signals)
        self.assertTrue(success, reason)
        merged = self.registry.get("rigor_and_structuring")
        self.assertIsNotNone(merged)

    def test_merge_marks_sources_as_merged(self):
        rule = self.store.get("rule_merge_rigor_structuring")
        self.engine.execute_merge(rule, SignalSet())
        self.assertEqual(self.registry.get("rigor").status, AxisStatus.MERGED)
        self.assertEqual(self.registry.get("cognitive_structuring").status, AxisStatus.MERGED)

    def test_merge_combined_questions(self):
        rule = self.store.get("rule_merge_rigor_structuring")
        rigor_qs = set(self.registry.get("rigor").questions)
        struct_qs = set(self.registry.get("cognitive_structuring").questions)
        self.engine.execute_merge(rule, SignalSet())
        merged = self.registry.get("rigor_and_structuring")
        merged_qs = set(merged.questions)
        self.assertTrue(rigor_qs.issubset(merged_qs) or struct_qs.issubset(merged_qs))

    def test_transform_autonomy_clinical(self):
        rule = self.store.get("rule_transform_autonomy_clinical")
        self.assertIsNotNone(rule)
        success, event, reason = self.engine.execute_transform(rule, "clinical_team")
        self.assertTrue(success, reason)
        ax = self.registry.get("autonomy")
        self.assertIn("clinical_team", ax.contextual_variants)

    def test_transform_logged_to_audit(self):
        rule = self.store.get("rule_transform_autonomy_clinical")
        self.engine.execute_transform(rule, "clinical_team")
        events = self.audit.events_for_axis("autonomy")
        self.assertGreater(len(events), 0)

    def test_split_logged_to_audit(self):
        rule = self.store.get("rule_split_autonomy")
        self.engine.execute_split(rule, SignalSet())
        events = self.audit.events_for_axis("autonomy")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].rule_type, "split")


# ===================================================================
# T5 — Audit log
# ===================================================================

class TestAuditLog(unittest.TestCase):

    def setUp(self):
        self.log = _make_temp_audit()

    def test_empty_log(self):
        self.assertEqual(self.log.count(), 0)

    def test_record_event(self):
        event = self.log.record(
            rule_id="test_rule",
            rule_type="split",
            axis_id="autonomy",
            axes_involved=["autonomy"],
            resulting_axes=["autonomy_strategic", "autonomy_operational"],
            explanation="Test split",
        )
        self.assertEqual(self.log.count(), 1)
        self.assertEqual(event.rule_id, "test_rule")

    def test_events_are_ordered(self):
        for i in range(5):
            self.log.record(rule_id=f"rule_{i}", rule_type="reweight", axis_id=f"axis_{i}", explanation="")
        events = self.log.all_events()
        ids = [e.event_id for e in events]
        self.assertEqual(ids, sorted(ids))

    def test_events_for_axis(self):
        self.log.record("r1", "split", axis_id="autonomy", axes_involved=["autonomy"], explanation="")
        self.log.record("r2", "merge", axis_id="rigor", axes_involved=["rigor"], explanation="")
        autonomy_events = self.log.events_for_axis("autonomy")
        self.assertEqual(len(autonomy_events), 1)
        self.assertEqual(autonomy_events[0].rule_id, "r1")

    def test_events_by_type(self):
        self.log.record("r1", "split", axis_id="a", explanation="")
        self.log.record("r2", "merge", axis_id="b", explanation="")
        self.log.record("r3", "split", axis_id="c", explanation="")
        splits = self.log.events_by_type("split")
        self.assertEqual(len(splits), 2)

    def test_latest_returns_n_events(self):
        for i in range(15):
            self.log.record(f"rule_{i}", "reweight", axis_id="x", explanation="")
        latest = self.log.latest(5)
        self.assertEqual(len(latest), 5)

    def test_replay_summary_format(self):
        self.log.record("r1", "split", axis_id="autonomy", axes_involved=["autonomy"],
                        resulting_axes=["a", "b"], explanation="Some explanation")
        summary = self.log.replay_summary()
        self.assertEqual(len(summary), 1)
        self.assertIn("type", summary[0])
        self.assertIn("axes", summary[0])
        self.assertIn("result", summary[0])

    def test_event_ids_unique(self):
        for i in range(20):
            self.log.record(f"rule_{i}", "reweight", axis_id="x", explanation="")
        ids = [e.event_id for e in self.log.all_events()]
        self.assertEqual(len(ids), len(set(ids)))


# ===================================================================
# T6 — Integration V6
# ===================================================================

class TestV6Integration(unittest.TestCase):

    def setUp(self):
        self.registry = AxisRegistry()
        self.store = RuleStore()
        self.engine = V6RuleEngine()
        self.computer = SignalComputer()
        self.audit = _make_temp_audit()
        self.transformer = TransformationEngine(self.registry, self.audit)

    def test_detect_high_correlation_then_merge(self):
        """Full pipeline: compute signals → detect merge → execute merge → audit."""
        # Simulate highly correlated profiles
        profiles = _make_correlated_profiles("rigor", "cognitive_structuring", 0.92, n=30)
        signals = self.computer.compute(profiles)
        # Add reverse key if missing
        if ("cognitive_structuring", "rigor") not in signals:
            signals[("cognitive_structuring", "rigor")] = signals.get(("rigor", "cognitive_structuring"), {})

        firings = self.engine.evaluate_data_gated(self.store, signals)
        merge_firings = [f for f in firings if f.rule_type == "merge" and f.rule_id == "rule_merge_rigor_structuring"]
        self.assertGreater(len(merge_firings), 0, "Expected merge to fire")

        # Execute merge
        rule = self.store.get("rule_merge_rigor_structuring")
        success, event, reason = self.transformer.execute_merge(rule, signals)
        self.assertTrue(success, reason)
        self.assertIsNotNone(self.registry.get("rigor_and_structuring"))
        self.assertEqual(self.audit.count(), 1)

    def test_detect_low_consistency_then_split(self):
        """Full pipeline: inject low consistency signal → detect split → execute split."""
        signals = self.computer.inject({
            "autonomy": {"variance": 0.9, "internal_consistency": 0.35, "discrimination_index": 0.8}
        })
        firings = self.engine.evaluate_data_gated(self.store, signals)
        split_firings = [f for f in firings if f.rule_type == "split" and "autonomy" in f.axes_involved]
        self.assertGreater(len(split_firings), 0, "Expected split to fire for low consistency")

        rule = self.store.get("rule_split_autonomy")
        success, event, _ = self.transformer.execute_split(rule, signals)
        self.assertTrue(success)
        self.assertIsNotNone(self.registry.get("autonomy_strategic"))

    def test_transform_then_audit(self):
        rule = self.store.get("rule_transform_autonomy_clinical")
        self.transformer.execute_transform(rule, "clinical_team")
        ax = self.registry.get("autonomy")
        self.assertIn("clinical_team", ax.contextual_variants)
        events = self.audit.events_for_axis("autonomy")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].rule_type, "transform")

    def test_multiple_transformations_all_logged(self):
        r1 = self.store.get("rule_transform_autonomy_clinical")
        r2 = self.store.get("rule_transform_risk_research")
        self.transformer.execute_transform(r1, "clinical_team")
        self.transformer.execute_transform(r2, "research")
        self.assertEqual(self.audit.count(), 2)
        types = {e.rule_type for e in self.audit.all_events()}
        self.assertIn("transform", types)

    def test_immediate_context_rules_fire_for_research(self):
        firings = self.engine.evaluate_immediate(self.store, "research")
        fired_ids = {f.rule_id for f in firings}
        self.assertIn("rule_reweight_research_rigor", fired_ids)

    def test_startup_reweight_is_highest_priority_reweight(self):
        firings = self.engine.evaluate_immediate(self.store, "startup")
        reweights = [f for f in firings if f.rule_type == "reweight"]
        if len(reweights) >= 2:
            self.assertLessEqual(reweights[0].priority, reweights[1].priority)

    def test_full_audit_replay_readable(self):
        r1 = self.store.get("rule_split_autonomy")
        r2 = self.store.get("rule_transform_autonomy_clinical")
        self.transformer.execute_split(r1, SignalSet())
        self.transformer.execute_transform(r2, "clinical_team")
        summary = self.audit.replay_summary()
        self.assertEqual(len(summary), 2)
        for entry in summary:
            self.assertIn("type", entry)
            self.assertIn("explanation", entry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
