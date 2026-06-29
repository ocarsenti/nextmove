"""Rule engine — evaluates axis rules deterministically. V5: reweight + mask only.
V6 will add split/merge/creation using statistical signals from accumulated profiles."""

from __future__ import annotations

from ontology.models import Axis, AxisRule, AxisStatus, RuleFiring, RuleType, SignalType
from ontology.registry import AxisRegistry


class RuleEngine:
    """Evaluates rules against a context signal set and produces a list of firings."""

    def evaluate(
        self,
        registry: AxisRegistry,
        context_id: str | None,
        axis_signals: dict[str, dict] | None = None,
    ) -> list[RuleFiring]:
        """
        Evaluate all applicable rules. Returns list of RuleFiring.

        axis_signals: optional dict of {axis_id: {"variance": float, ...}} for V6+ rules.
                      Ignored in V5 — only context_match rules are evaluated.
        """
        firings: list[RuleFiring] = []

        for ax in registry.all_active():
            for rule in ax.rules:
                if rule.requires_data:
                    continue  # skip V6+ rules in V5

                firing = self._evaluate_rule(ax, rule, context_id, axis_signals or {})
                if firing:
                    firings.append(firing)

        return firings

    def _evaluate_rule(
        self,
        ax: Axis,
        rule: AxisRule,
        context_id: str | None,
        axis_signals: dict[str, dict],
    ) -> RuleFiring | None:
        cond = rule.condition
        eff = rule.effect

        if cond.signal == SignalType.CONTEXT_MATCH:
            if context_id != cond.context:
                return None
            return RuleFiring(
                rule_id=rule.rule_id,
                axis_id=ax.id,
                rule_type=eff.action,
                condition_met=f"context={context_id}",
                effect_applied=self._describe_effect(eff),
                explanation=rule.explanation,
            )

        if cond.signal == SignalType.AXIS_VALUE_THRESHOLD:
            # For V5: check if we have a signal for this axis
            sig = axis_signals.get(ax.id, {})
            val = sig.get("value")
            if val is None or cond.threshold is None:
                return None
            axis_val = val
            if axis_val >= cond.threshold:
                return RuleFiring(
                    rule_id=rule.rule_id,
                    axis_id=ax.id,
                    rule_type=eff.action,
                    condition_met=f"axis={ax.id} value={axis_val:.2f} >= threshold={cond.threshold}",
                    effect_applied=self._describe_effect(eff),
                    explanation=rule.explanation,
                )

        # V6+ signals (high_variance, high_correlation, low_discrimination) — not evaluated in V5
        return None

    def _describe_effect(self, eff) -> str:
        if eff.action == RuleType.REWEIGHT and eff.weight_modifier is not None:
            sign = "+" if eff.weight_modifier >= 0 else ""
            return f"reweight {sign}{eff.weight_modifier:.1f}"
        if eff.action == RuleType.MASK:
            return "masked"
        if eff.action == RuleType.SPLIT and eff.into:
            return f"split into {eff.into}"
        if eff.action == RuleType.MERGE and eff.target:
            return f"merge into {eff.target}"
        return str(eff.action.value)

    def pending_v6_rules(self, registry: AxisRegistry) -> list[dict]:
        """Return rules that require data signals (V6+ only) — for observability."""
        pending = []
        for ax in registry.all_active():
            for rule in ax.rules:
                if rule.requires_data:
                    pending.append({
                        "axis_id": ax.id,
                        "rule_id": rule.rule_id,
                        "signal": rule.condition.signal.value,
                        "explanation": rule.explanation,
                    })
        return pending
