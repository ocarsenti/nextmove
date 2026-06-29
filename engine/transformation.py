"""Transformation engine — executes split, merge, transform rules on the axis registry (V6).

Every transformation is:
- deterministic
- logged to the audit trail
- reversible (registry can be restored from audit log)
"""

from __future__ import annotations

from typing import Optional

from ontology.models import Axis, AxisStatus, UncertaintyModel
from ontology.registry import AxisRegistry
from ontology.rule_store import OntologyRule, NewAxisSpec
from engine.audit import AuditLog, OntologyEvent
from engine.signals import SignalSet


class TransformationResult(dict):
    """Result of a transformation: {axis_id: change_description}"""
    pass


class TransformationEngine:
    """Executes structural ontology transformations with full audit trail."""

    def __init__(self, registry: AxisRegistry, audit_log: AuditLog) -> None:
        self._registry = registry
        self._audit = audit_log

    # ===================================================================
    # SPLIT
    # ===================================================================

    def execute_split(
        self,
        rule: OntologyRule,
        signals: Optional[SignalSet] = None,
        context: Optional[str] = None,
    ) -> tuple[bool, OntologyEvent | None, str]:
        """
        Split a source axis into child axes.
        Returns (success, event, reason).
        """
        source_id = rule.trigger.axis
        if not source_id or source_id == "__any__":
            return False, None, "Split rule requires a single source axis"

        source = self._registry.get(source_id)
        if source is None:
            return False, None, f"Source axis '{source_id}' not found"
        if source.status != AxisStatus.ACTIVE:
            return False, None, f"Source axis '{source_id}' is not active (status={source.status})"

        child_ids = []
        for spec in rule.new_axes:
            child = self._spec_to_axis(spec)
            child.questions = source.questions.copy()  # migrate questions to children
            self._registry.add_axis(child)
            child_ids.append(child.id)

        self._registry.register_split(source_id, child_ids)

        signals_used = {}
        if signals and source_id in signals:
            signals_used = signals[source_id]

        event = self._audit.record(
            rule_id=rule.id,
            rule_type="split",
            axis_id=source_id,
            axes_involved=[source_id],
            resulting_axes=child_ids,
            context=context,
            signals_used=signals_used,
            explanation=rule.explanation,
            reversible=rule.reversible,
        )

        return True, event, f"Split '{source_id}' into {child_ids}"

    # ===================================================================
    # MERGE
    # ===================================================================

    def execute_merge(
        self,
        rule: OntologyRule,
        signals: Optional[SignalSet] = None,
        context: Optional[str] = None,
    ) -> tuple[bool, OntologyEvent | None, str]:
        """
        Merge multiple source axes into a single target axis.
        Returns (success, event, reason).
        """
        source_ids = rule.trigger.axes
        if len(source_ids) < 2:
            return False, None, "Merge rule requires at least 2 source axes"

        sources = []
        for sid in source_ids:
            ax = self._registry.get(sid)
            if ax is None:
                return False, None, f"Source axis '{sid}' not found"
            if ax.status != AxisStatus.ACTIVE:
                return False, None, f"Source axis '{sid}' is not active"
            sources.append(ax)

        if rule.new_axis is None:
            return False, None, "Merge rule has no new_axis spec"

        merged = self._spec_to_axis(rule.new_axis)
        # Combine questions from all sources
        seen_questions: set[str] = set()
        for src in sources:
            for q in src.questions:
                if q not in seen_questions:
                    merged.questions.append(q)
                    seen_questions.add(q)

        self._registry.add_axis(merged)
        self._registry.register_merge(source_ids, merged.id)

        signals_used = {}
        if signals:
            for a, b in zip(source_ids, source_ids[1:]):
                key = (a, b)
                if key in signals:
                    signals_used[f"{a}_{b}_correlation"] = signals[key].get("correlation")

        event = self._audit.record(
            rule_id=rule.id,
            rule_type="merge",
            axes_involved=source_ids,
            resulting_axes=[merged.id],
            context=context,
            signals_used=signals_used,
            explanation=rule.explanation,
            reversible=rule.reversible,
        )

        return True, event, f"Merged {source_ids} into '{merged.id}'"

    # ===================================================================
    # TRANSFORM
    # ===================================================================

    def execute_transform(
        self,
        rule: OntologyRule,
        context: Optional[str] = None,
    ) -> tuple[bool, OntologyEvent | None, str]:
        """
        Transform an axis's label/description within a context.
        Context-local — does not affect the global axis definition.
        """
        source_id = rule.trigger.axis
        if not source_id:
            return False, None, "Transform rule requires a source axis"

        ax = self._registry.get(source_id)
        if ax is None:
            return False, None, f"Axis '{source_id}' not found"

        ctx_scope = rule.action.context_scope or context
        if ctx_scope:
            from ontology.models import ContextualVariant
            variant = ax.contextual_variants.get(ctx_scope) or ContextualVariant(
                context_id=ctx_scope,
                weight_modifier=1.0,
            )
            if rule.action.new_label:
                variant.description = (rule.action.new_label or "") + " — " + (rule.action.new_description or "")
            ax.contextual_variants[ctx_scope] = variant

        event = self._audit.record(
            rule_id=rule.id,
            rule_type="transform",
            axis_id=source_id,
            axes_involved=[source_id],
            resulting_axes=[source_id],
            context=context,
            signals_used={},
            explanation=rule.explanation,
            reversible=rule.reversible,
        )

        return True, event, (
            f"Transformed '{source_id}' definition for context '{ctx_scope}': "
            f"{rule.action.new_label}"
        )

    # ===================================================================
    # HELPERS
    # ===================================================================

    def _spec_to_axis(self, spec: NewAxisSpec) -> Axis:
        return Axis(
            id=spec.id,
            label=spec.label,
            description=spec.description,
            layer=spec.layer,
            parents=[spec.split_from] if spec.split_from else (spec.merged_from or []),
            split_from=spec.split_from,
            merged_into=None,
            weight=1.0,
            status=AxisStatus.ACTIVE,
            version=1,
            uncertainty_model=UncertaintyModel(
                type="heuristic", confidence=0.0, min_questions=3
            ),
        )
