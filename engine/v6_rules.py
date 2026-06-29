"""V6 Rule evaluator — priority-ordered, all rule types, signal-aware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ontology.rule_store import OntologyRule, RuleStore
from engine.signals import SignalSet


@dataclass
class V6RuleFiring:
    rule_id: str
    rule_type: str
    priority: int
    axis_id: Optional[str]
    axes_involved: list[str]
    condition_met: str
    action_description: str
    explanation: str
    requires_execution: bool     # True for split/merge/transform (structural changes)
    context: Optional[str] = None
    signals_used: dict = None

    def __post_init__(self):
        if self.signals_used is None:
            self.signals_used = {}


class V6RuleEngine:
    """
    Full V6 rule evaluator.
    - Priority-ordered evaluation
    - All rule types: split, merge, mask, reweight, transform
    - Signal-aware conditions (variance, correlation, discrimination)
    - Separates immediate firings (no data needed) from data-gated firings
    """

    def evaluate(
        self,
        store: RuleStore,
        context_id: Optional[str] = None,
        signals: Optional[SignalSet] = None,
    ) -> list[V6RuleFiring]:
        """
        Evaluate all active rules. Returns fired rules sorted by priority.
        Structural rules (split/merge/transform) are flagged requires_execution=True.
        """
        firings: list[V6RuleFiring] = []

        for rule in store.all_active():
            firing = self._evaluate_rule(rule, context_id, signals)
            if firing:
                firings.append(firing)

        return sorted(firings, key=lambda f: f.priority)

    def evaluate_immediate(
        self,
        store: RuleStore,
        context_id: Optional[str] = None,
    ) -> list[V6RuleFiring]:
        """Only evaluate rules that don't require data signals."""
        return [
            f for f in self.evaluate(store, context_id, None)
            if not self._is_data_gated(f)
        ]

    def evaluate_data_gated(
        self,
        store: RuleStore,
        signals: SignalSet,
        context_id: Optional[str] = None,
    ) -> list[V6RuleFiring]:
        """Only evaluate rules that require data signals."""
        return [
            f for f in self.evaluate(store, context_id, signals)
            if self._is_data_gated(f)
        ]

    def _is_data_gated(self, firing: V6RuleFiring) -> bool:
        return bool(firing.signals_used)

    def _evaluate_rule(
        self,
        rule: OntologyRule,
        context_id: Optional[str],
        signals: Optional[SignalSet],
    ) -> Optional[V6RuleFiring]:

        trigger = rule.trigger

        # Context gate — if rule has a context, it must match
        if trigger.context and trigger.context != context_id:
            return None

        axis_id = trigger.axis if trigger.axis and trigger.axis != "__any__" else None
        axes_involved = trigger.axes if trigger.axes else ([axis_id] if axis_id else [])

        signals_used: dict = {}
        conditions_met: list[str] = []

        # Evaluate conditions
        if trigger.conditions:
            if not self._evaluate_conditions(rule, signals, signals_used, conditions_met, axis_id, axes_involved):
                return None
        else:
            # No conditions → context match alone is sufficient
            if trigger.context:
                conditions_met.append(f"context={context_id}")
            elif not trigger.conditions:
                conditions_met.append("always")

        condition_str = " AND ".join(conditions_met) if conditions_met else "no-condition"

        return V6RuleFiring(
            rule_id=rule.id,
            rule_type=rule.type,
            priority=rule.priority,
            axis_id=axis_id,
            axes_involved=axes_involved,
            condition_met=condition_str,
            action_description=self._describe_action(rule),
            explanation=rule.explanation,
            requires_execution=(rule.type in ("split", "merge", "transform")),
            context=context_id,
            signals_used=signals_used,
        )

    def _evaluate_conditions(
        self,
        rule: OntologyRule,
        signals: Optional[SignalSet],
        signals_used: dict,
        conditions_met: list[str],
        axis_id: Optional[str],
        axes_involved: list[str],
    ) -> bool:
        if signals is None:
            return False  # data-gated rules can't fire without signals

        for cond in rule.trigger.conditions:
            metric = cond.metric

            if metric == "correlation":
                # Pairwise: need both axes
                if len(axes_involved) < 2:
                    return False
                a, b = axes_involved[0], axes_involved[1]
                sig = signals.get((a, b)) or signals.get((b, a))
                if sig is None:
                    return False
                val = sig.get("correlation")
                if not cond.evaluate(val):
                    return False
                signals_used[f"correlation_{a}_{b}"] = val
                conditions_met.append(f"correlation({a},{b})={val:.3f}{cond.operator}{cond.value}")

            elif metric in ("variance", "internal_consistency", "discrimination_index"):
                target = axis_id or (axes_involved[0] if axes_involved else None)
                if target == "__any__":
                    # Wildcard: check all axes — fire if ANY fails the condition
                    fired_for = None
                    for ax_id, ax_signals in signals.items():
                        if not isinstance(ax_id, str):
                            continue
                        val = ax_signals.get(metric)
                        if cond.evaluate(val):
                            fired_for = ax_id
                            signals_used[f"{metric}_{ax_id}"] = val
                            break
                    if not fired_for:
                        return False
                    conditions_met.append(f"{metric}({fired_for}){cond.operator}{cond.value}")
                else:
                    if target not in signals:
                        return False
                    val = signals[target].get(metric)
                    if not cond.evaluate(val):
                        return False
                    signals_used[f"{metric}_{target}"] = val
                    conditions_met.append(f"{metric}({target})={val:.3f}{cond.operator}{cond.value}")
            else:
                return False  # unknown metric

        return True

    def _describe_action(self, rule: OntologyRule) -> str:
        if rule.type == "split":
            targets = [spec.id for spec in rule.new_axes]
            return f"split into {targets}"
        if rule.type == "merge":
            target = rule.new_axis.id if rule.new_axis else "?"
            return f"merge into '{target}'"
        if rule.type == "mask":
            return f"mask (visible=False)"
        if rule.type == "reweight":
            mult = rule.action.weight_multiplier
            return f"reweight ×{mult}"
        if rule.type == "transform":
            label = rule.action.new_label or "?"
            return f"transform label→'{label}'"
        return rule.type

    def pending_report(self, store: RuleStore) -> list[dict]:
        """All active rules that haven't fired yet — for observability."""
        return [
            {
                "rule_id": r.id,
                "type": r.type,
                "priority": r.priority,
                "requires_data": r.requires_data,
                "explanation": r.explanation[:80] + "..." if len(r.explanation) > 80 else r.explanation,
            }
            for r in store.all_active()
        ]
