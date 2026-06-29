"""Rule store — standalone registry of ontology transformation rules (V6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

_SEED_DIR = Path(__file__).parent


# ===================================================================
# RULE MODELS
# ===================================================================

class RuleCondition(BaseModel):
    metric: str                  # variance | correlation | internal_consistency | discrimination_index
    operator: str                # > | < | >= | <= | ==
    value: float

    def evaluate(self, signal_value: float | None) -> bool:
        if signal_value is None:
            return False
        ops = {">": signal_value > self.value, "<": signal_value < self.value,
               ">=": signal_value >= self.value, "<=": signal_value <= self.value,
               "==": abs(signal_value - self.value) < 1e-9}
        return ops.get(self.operator, False)


class RuleTrigger(BaseModel):
    axis: Optional[str] = None       # single axis (or "__any__" for wildcard)
    axes: list[str] = Field(default_factory=list)   # for merge rules
    context: Optional[str] = None
    conditions: list[RuleCondition] = Field(default_factory=list)


class RuleAction(BaseModel):
    result: Optional[object] = None          # str or list[str]
    visibility: Optional[bool] = None
    reason: Optional[str] = None
    weight_multiplier: Optional[float] = None
    new_label: Optional[str] = None
    new_description: Optional[str] = None
    context_scope: Optional[str] = None


class NewAxisSpec(BaseModel):
    id: str
    label: str
    description: str
    layer: int
    split_from: Optional[str] = None
    merged_from: Optional[list[str]] = None


class OntologyRule(BaseModel):
    id: str
    type: str                        # split | merge | mask | reweight | transform
    priority: int = 50
    trigger: RuleTrigger
    action: RuleAction
    new_axes: list[NewAxisSpec] = Field(default_factory=list)  # for split
    new_axis: Optional[NewAxisSpec] = None                      # for merge
    explanation: str = ""
    requires_data: bool = False
    reversible: bool = True
    active: bool = True


# ===================================================================
# RULE STORE
# ===================================================================

class RuleStore:
    """Persistent, versioned store of ontology transformation rules."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _SEED_DIR / "rules_seed.json"
        self._rules: dict[str, OntologyRule] = {}
        self.version: int = 1
        self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self.version = raw.get("version", 1)
        for r in raw["rules"]:
            rule = OntologyRule(**r)
            self._rules[rule.id] = rule

    def save(self, path: Optional[Path] = None) -> None:
        target = path or self._path
        payload = {
            "version": self.version,
            "rules": [r.model_dump() for r in self._rules.values()],
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Query ---

    def all_active(self) -> list[OntologyRule]:
        return sorted(
            [r for r in self._rules.values() if r.active],
            key=lambda r: r.priority,
        )

    def by_type(self, rule_type: str) -> list[OntologyRule]:
        return [r for r in self.all_active() if r.type == rule_type]

    def context_rules(self, context_id: str) -> list[OntologyRule]:
        return [r for r in self.all_active() if r.trigger.context == context_id]

    def data_rules(self) -> list[OntologyRule]:
        return [r for r in self.all_active() if r.requires_data]

    def immediate_rules(self) -> list[OntologyRule]:
        """Rules that can fire without data (context-only triggers)."""
        return [r for r in self.all_active() if not r.requires_data]

    def get(self, rule_id: str) -> Optional[OntologyRule]:
        return self._rules.get(rule_id)

    def all_ids(self) -> list[str]:
        return list(self._rules.keys())

    # --- Mutations ---

    def add(self, rule: OntologyRule) -> None:
        self._rules[rule.id] = rule
        self.version += 1

    def deactivate(self, rule_id: str) -> bool:
        if rule_id in self._rules:
            self._rules[rule_id].active = False
            return True
        return False

    def activate(self, rule_id: str) -> bool:
        if rule_id in self._rules:
            self._rules[rule_id].active = True
            return True
        return False
