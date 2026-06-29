"""Profile scorer — computes continuous axis scores with confidence from answers."""

from __future__ import annotations

import math
from ontology.models import AxisScore, AxisStatus, Profile
from ontology.registry import AxisRegistry, QuestionBank
from engine.context import ContextEngine


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _confidence(n: int, variance: float) -> float:
    """
    Heuristic confidence [0, 1]:
    - n=1: 0.33 max
    - n=2: 0.60 max
    - n=3: 0.80 max
    - n=5+: ~0.95 max
    Reduced by variance (high variance = inconsistent answers = lower confidence).
    """
    base = 1.0 - math.exp(-0.7 * n)
    consistency_penalty = variance * 0.5
    return max(0.0, min(1.0, base - consistency_penalty))


class ProfileScorer:
    """Scores a profile from raw questionnaire answers."""

    def __init__(self, registry: AxisRegistry, bank: QuestionBank) -> None:
        self._registry = registry
        self._bank = bank
        self._context_engine = ContextEngine()

    def score(
        self,
        user_id: str,
        answers: dict[str, str],
        context_id: str | None = None,
    ) -> Profile:
        """
        answers: {question_id: answer_value ("A"/"B"/"C")}
        Returns a complete Profile with per-axis scores and confidence.
        """
        context_weights = self._context_engine.apply_context(
            self._registry.all_active(), context_id
        )

        axis_answers: dict[str, list[float]] = {ax.id: [] for ax in self._registry.all_active()}

        completed_questions: list[str] = []

        for qid, answer_value in answers.items():
            q = self._bank.get(qid)
            if q is None:
                continue
            if q.axis_id not in axis_answers:
                continue
            score_val = next(
                (opt.score for opt in q.options if opt.value == answer_value), None
            )
            if score_val is not None:
                axis_answers[q.axis_id].append(score_val)
                completed_questions.append(qid)

        axis_scores: dict[str, AxisScore] = {}
        for ax in self._registry.all_active():
            vals = axis_answers.get(ax.id, [])
            if not vals:
                continue
            raw_value = sum(vals) / len(vals)
            var = _variance(vals)
            conf = _confidence(len(vals), var)
            eff_weight, eff_status = context_weights.get(ax.id, (ax.weight, ax.status))

            axis_scores[ax.id] = AxisScore(
                axis_id=ax.id,
                raw_value=raw_value,
                confidence=conf,
                n_questions=len(vals),
                variance=var,
                raw_answers=vals,
                effective_weight=eff_weight,
                status=eff_status,
            )

        return Profile(
            user_id=user_id,
            axis_scores=axis_scores,
            context=context_id,
            registry_version=self._registry.version,
            completed_questions=completed_questions,
            is_complete=self._is_complete(axis_scores),
        )

    def _is_complete(self, scores: dict[str, AxisScore]) -> bool:
        """Profile is complete if all non-meta active axes have ≥3 questions answered."""
        axes = self._registry.non_meta_axes()
        for ax in axes:
            sc = scores.get(ax.id)
            if sc is None or sc.n_questions < ax.uncertainty_model.min_questions:
                return False
        return True

    def low_confidence_axes(self, profile: Profile) -> list[str]:
        """Return axis IDs where confidence < 0.6 — candidates for adaptive questions."""
        return [
            ax_id for ax_id, sc in profile.axis_scores.items()
            if sc.confidence < 0.6 and sc.status == AxisStatus.ACTIVE
        ]

    def weighted_profile(self, profile: Profile) -> dict[str, float]:
        """Return weighted scores (effective_weight * raw_value) per axis."""
        return {
            ax_id: sc.raw_value * sc.effective_weight
            for ax_id, sc in profile.axis_scores.items()
            if sc.status == AxisStatus.ACTIVE
        }
