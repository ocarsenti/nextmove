"""Golden regression tests — NextMove V5 scoring/archetype pipeline.

These fix a handful of input->output pairs (full questionnaire, known
answer patterns) and assert the results never silently drift. Unlike
tests/test_v5.py and tests/test_v6.py, which test individual mechanisms,
this file exists purely as a tripwire: if a change to the ontology,
scorer, weighting rules, or archetype signatures shifts these numbers,
that's either an intentional recalibration (update the fixture) or a
regression (fix the code) — never a silent change.

Do NOT "fix" a failing test here by copy-pasting the new output unless
you've confirmed the change was intentional.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from ontology.registry import AxisRegistry, QuestionBank
from engine.scorer import ProfileScorer
from engine.archetype import compute_archetype_distribution
from questionnaire.adaptive import AdaptiveQuestionnaire


@pytest.fixture(scope="module")
def engine_setup():
    registry = AxisRegistry()
    bank = QuestionBank()
    questionnaire = AdaptiveQuestionnaire(registry, bank)
    scorer = ProfileScorer(registry, bank)
    return registry, scorer, questionnaire.initial_sequence()


@pytest.mark.golden
class TestGoldenUniformAnswers:
    """All base questions answered identically — simplest possible fixture."""

    def test_all_A_profile_is_floor(self, engine_setup):
        registry, scorer, base = engine_setup
        profile = scorer.score("golden_all_A", {qid: "A" for qid in base}, None)
        weighted = scorer.weighted_profile(profile)
        assert profile.is_complete is True
        assert all(v == pytest.approx(0.0) for v in weighted.values())

    def test_all_C_profile_is_ceiling(self, engine_setup):
        registry, scorer, base = engine_setup
        profile = scorer.score("golden_all_C", {qid: "C" for qid in base}, None)
        weighted = scorer.weighted_profile(profile)
        assert all(v == pytest.approx(1.0) for v in weighted.values())

    def test_all_B_profile_is_midpoint_and_ties_on_archetype(self, engine_setup):
        registry, scorer, base = engine_setup
        profile = scorer.score("golden_all_B", {qid: "B" for qid in base}, None)
        weighted = scorer.weighted_profile(profile)
        assert all(v == pytest.approx(0.5) for v in weighted.values())

        arche = compute_archetype_distribution(profile, registry).model_dump()
        # A perfectly neutral profile ties all 5 archetypes at 20% —
        # dominant/secondary resolution is alphabetical-by-construction.
        assert all(s["percentage"] == pytest.approx(20.0) for s in arche["scores"])
        assert arche["dominant"] == "Builder"
        assert arche["secondary"] == "Expert"
        assert arche["low_confidence"] is False


@pytest.mark.golden
class TestGoldenMixedPattern:
    """A/B/C cycled across the base questionnaire — exercises real
    weighting and per-axis variance rather than a single flat value."""

    EXPECTED_WEIGHTED = {
        "ambiguity_tolerance": 0.6667,
        "autonomy": 0.3333,
        "behavioral_stability": 0.5,
        "cognitive_flexibility": 0.5,
        "cognitive_granularity": 0.3333,
        "cognitive_structuring": 0.3333,
        "context_sensitivity": 1.0,
        "decision_speed": 0.8333,
        "direction": 0.3333,
        "exploration": 0.3333,
        "influence": 0.3333,
        "persistence": 0.6667,
        "resilience": 0.5,
        "rigor": 0.6667,
        "risk_appetite": 0.6667,
        "situational_leadership": 0.5,
        "social_interaction": 0.3333,
        "value_orientation": 0.1667,
    }

    def _mixed_answers(self, base: list[str]) -> dict[str, str]:
        pattern = ["A", "B", "C"]
        return {qid: pattern[i % 3] for i, qid in enumerate(base)}

    def test_weighted_scores_match_fixture(self, engine_setup):
        registry, scorer, base = engine_setup
        profile = scorer.score("golden_mixed", self._mixed_answers(base), None)
        weighted = scorer.weighted_profile(profile)

        assert weighted.keys() == self.EXPECTED_WEIGHTED.keys()
        for axis_id, expected in self.EXPECTED_WEIGHTED.items():
            assert weighted[axis_id] == pytest.approx(expected, abs=1e-3), (
                f"Axis '{axis_id}' drifted: expected {expected}, got {weighted[axis_id]}"
            )

    def test_archetype_dominant_and_secondary_match_fixture(self, engine_setup):
        registry, scorer, base = engine_setup
        profile = scorer.score("golden_mixed_2", self._mixed_answers(base), None)
        arche = compute_archetype_distribution(profile, registry).model_dump()

        assert arche["dominant"] == "Builder"
        assert arche["secondary"] == "Explorer"

    def test_startup_context_shifts_decision_speed_and_risk_appetite_up(self, engine_setup):
        """Sanity check that context reweighting is still active — doesn't
        pin exact values (context rules are covered elsewhere) but asserts
        the direction of the known startup effect doesn't invert."""
        registry, scorer, base = engine_setup
        neutral = scorer.score("golden_ctx_neutral", self._mixed_answers(base), None)
        startup = scorer.score("golden_ctx_startup", self._mixed_answers(base), "startup")

        w_neutral = scorer.weighted_profile(neutral)
        w_startup = scorer.weighted_profile(startup)

        assert w_startup["decision_speed"] > w_neutral["decision_speed"]
        assert w_startup["risk_appetite"] > w_neutral["risk_appetite"]
