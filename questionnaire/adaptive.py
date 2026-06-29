"""Adaptive questionnaire engine — selects next questions based on uncertainty."""

from __future__ import annotations

from ontology.models import AxisStatus, Profile, Question
from ontology.registry import AxisRegistry, QuestionBank


_BASE_DISPLAY_ORDER = [
    # Round 1 — one question per axis, interleaved to prevent pattern detection
    "q_autonomy_1", "q_social_interaction_1", "q_rigor_1",
    "q_exploration_1", "q_behavioral_stability_1", "q_decision_speed_1",
    "q_cognitive_flexibility_1", "q_resilience_1", "q_cognitive_structuring_1",
    "q_risk_appetite_1", "q_direction_1", "q_ambiguity_tolerance_1",
    "q_influence_1", "q_persistence_1", "q_situational_leadership_1",
    "q_cognitive_granularity_1", "q_value_orientation_1", "q_context_sensitivity_1",
    # Round 2
    "q_rigor_2", "q_autonomy_2", "q_persistence_2",
    "q_social_interaction_2", "q_cognitive_flexibility_2", "q_exploration_2",
    "q_ambiguity_tolerance_2", "q_direction_2", "q_resilience_2",
    "q_cognitive_structuring_2", "q_behavioral_stability_2", "q_decision_speed_2",
    "q_value_orientation_2", "q_situational_leadership_2", "q_risk_appetite_2",
    "q_influence_2", "q_cognitive_granularity_2", "q_context_sensitivity_2",
    # Round 3
    "q_exploration_3", "q_behavioral_stability_3", "q_rigor_3",
    "q_direction_3", "q_social_interaction_3", "q_cognitive_flexibility_3",
    "q_resilience_3", "q_autonomy_3", "q_ambiguity_tolerance_3",
    "q_cognitive_structuring_3", "q_persistence_3", "q_influence_3",
    "q_situational_leadership_3", "q_decision_speed_3", "q_risk_appetite_3",
    "q_value_orientation_3", "q_cognitive_granularity_3", "q_context_sensitivity_3",
]


class AdaptiveQuestionnaire:
    """
    Manages question selection for a session.

    V5 logic:
    - Start with all base questions in interleaved order
    - After each answer, check if any adaptive questions should be activated
    - Activated questions are inserted after the current position
    - Avoids duplicates
    """

    def __init__(self, registry: AxisRegistry, bank: QuestionBank) -> None:
        self._registry = registry
        self._bank = bank

    def initial_sequence(self) -> list[str]:
        """Return base question IDs in interleaved order."""
        existing = {q for q in _BASE_DISPLAY_ORDER if self._bank.get(q) is not None}
        return [q for q in _BASE_DISPLAY_ORDER if q in existing]

    def next_adaptive_questions(
        self,
        profile: Profile,
        already_seen: set[str],
    ) -> list[str]:
        """
        Return adaptive question IDs that should be activated given current profile.
        Considers trigger conditions against current axis scores.
        """
        to_activate: list[str] = []

        for ax in self._registry.all_active():
            score = profile.axis_scores.get(ax.id)
            if score is None:
                continue

            for q in self._bank.adaptive_questions_for(ax.id):
                if q.id in already_seen:
                    continue
                if q.trigger is None:
                    continue

                trigger = q.trigger
                fired = False

                if trigger.condition == "high_value" and score.raw_value >= trigger.threshold:
                    fired = True
                elif trigger.condition == "low_value" and score.raw_value <= trigger.threshold:
                    fired = True
                elif trigger.condition == "high_uncertainty" and score.confidence < (1.0 - trigger.threshold):
                    fired = True

                if fired:
                    to_activate.append(q.id)

        return to_activate

    def activations_from_answer(
        self, question_id: str, already_seen: set[str]
    ) -> list[str]:
        """Return question IDs that this answered question activates."""
        q = self._bank.get(question_id)
        if q is None:
            return []
        return [qid for qid in q.activates if qid not in already_seen]

    def build_session(
        self,
        profile: Profile | None = None,
        already_seen: set[str] | None = None,
    ) -> list[Question]:
        """
        Build a full ordered question list for a session.
        - Base questions first (interleaved)
        - Adaptive questions appended based on current profile state
        """
        seen = already_seen or set()
        base = [
            q for qid in self.initial_sequence()
            if (q := self._bank.get(qid)) is not None and qid not in seen
        ]

        adaptive: list[Question] = []
        if profile:
            for qid in self.next_adaptive_questions(profile, seen | {q.id for q in base}):
                q = self._bank.get(qid)
                if q:
                    adaptive.append(q)

        return base + adaptive

    def pending_questions(
        self,
        profile: Profile,
        already_seen: set[str],
    ) -> list[str]:
        """
        Return IDs of questions not yet seen that should be asked.
        Combines base questions not yet seen + adaptive questions triggered.
        """
        base_not_seen = [
            qid for qid in self.initial_sequence()
            if qid not in already_seen
        ]
        adaptive = self.next_adaptive_questions(profile, already_seen)
        return base_not_seen + [q for q in adaptive if q not in base_not_seen]
