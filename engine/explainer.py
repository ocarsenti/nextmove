"""Explainer — produces human-readable traces of all decisions made by the system."""

from __future__ import annotations

from ontology.models import ExplanationTrace, Profile, RuleFiring
from ontology.registry import AxisRegistry


class Explainer:
    """Builds an ExplanationTrace for a scored profile."""

    def explain(
        self,
        profile: Profile,
        rules_fired: list[RuleFiring],
        registry: AxisRegistry,
        adaptive_activations: list[str] | None = None,
    ) -> ExplanationTrace:
        axis_modifications: dict[str, str] = {}
        uncertainty_notes: dict[str, str] = {}

        for firing in rules_fired:
            if firing.axis_id not in axis_modifications:
                axis_modifications[firing.axis_id] = []
            axis_modifications[firing.axis_id].append(
                f"[{firing.rule_id}] {firing.effect_applied} — {firing.explanation}"
            )

        axis_modifications = {
            k: " | ".join(v) for k, v in axis_modifications.items()
        }

        for ax_id, score in profile.axis_scores.items():
            ax = registry.get(ax_id)
            label = ax.label if ax else ax_id
            n = score.n_questions
            conf = score.confidence
            var = score.variance

            if n < 3:
                uncertainty_notes[ax_id] = (
                    f"{label} : {n} question(s) répondue(s) — confidence {conf:.0%}. "
                    f"3 questions minimum recommandées pour fiabilité."
                )
            elif var > 0.15:
                uncertainty_notes[ax_id] = (
                    f"{label} : réponses inconsistantes (variance={var:.2f}) — confidence {conf:.0%}. "
                    f"Des questions adaptatives pourraient clarifier."
                )
            elif conf < 0.6:
                uncertainty_notes[ax_id] = (
                    f"{label} : confidence faible ({conf:.0%}) — profil partiel sur cet axe."
                )

        return ExplanationTrace(
            user_id=profile.user_id,
            context=profile.context,
            rules_fired=rules_fired,
            axis_modifications=axis_modifications,
            uncertainty_notes=uncertainty_notes,
            adaptive_activations=adaptive_activations or [],
        )

    def summary(self, trace: ExplanationTrace) -> str:
        """One-paragraph human-readable summary of the trace."""
        lines = []

        if trace.context:
            lines.append(f"Contexte détecté : {trace.context}.")

        if trace.rules_fired:
            rule_count = len(trace.rules_fired)
            axes_affected = {f.axis_id for f in trace.rules_fired}
            lines.append(
                f"{rule_count} règle(s) déclenchée(s), affectant {len(axes_affected)} axe(s) : "
                f"{', '.join(sorted(axes_affected))}."
            )
        else:
            lines.append("Aucune règle contextuelle déclenchée (contexte générique).")

        if trace.uncertainty_notes:
            axes_low = list(trace.uncertainty_notes.keys())
            lines.append(
                f"{len(axes_low)} axe(s) avec confidence insuffisante : {', '.join(axes_low)}."
            )

        if trace.adaptive_activations:
            lines.append(
                f"{len(trace.adaptive_activations)} question(s) adaptative(s) activée(s) : "
                f"{', '.join(trace.adaptive_activations)}."
            )

        return " ".join(lines)
