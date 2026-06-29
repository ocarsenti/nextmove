"""Axis registry and question bank — persistent, versionable, hot-reloadable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .models import Axis, AxisRule, AxisStatus, Question, UncertaintyModel


_SEED_DIR = Path(__file__).parent


class AxisRegistry:
    """Live registry of all axes. Source of truth for the ontology."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _SEED_DIR / "axes_seed.json"
        self._axes: dict[str, Axis] = {}
        self.version: int = 1
        self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self.version = raw.get("version", 1)
        for a in raw["axes"]:
            # Rebuild nested objects
            rules = [AxisRule(**r) for r in a.pop("rules", [])]
            uncertainty = UncertaintyModel(**a.pop("uncertainty_model", {}))
            axis = Axis(**a, rules=rules, uncertainty_model=uncertainty)
            self._axes[axis.id] = axis

    def save(self, path: Optional[Path] = None) -> None:
        target = path or self._path
        payload = {
            "version": self.version,
            "axes": [self._serialize(ax) for ax in self._axes.values()],
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _serialize(self, ax: Axis) -> dict:
        d = ax.model_dump()
        d["rules"] = [r.model_dump() for r in ax.rules]
        d["uncertainty_model"] = ax.uncertainty_model.model_dump()
        return d

    # --- Query ---

    def get(self, axis_id: str) -> Optional[Axis]:
        return self._axes.get(axis_id)

    def all_active(self) -> list[Axis]:
        return [a for a in self._axes.values() if a.status == AxisStatus.ACTIVE]

    def by_layer(self, layer: int) -> list[Axis]:
        return [a for a in self.all_active() if a.layer == layer]

    def meta_axes(self) -> list[Axis]:
        return [a for a in self.all_active() if a.is_meta]

    def non_meta_axes(self) -> list[Axis]:
        return [a for a in self.all_active() if not a.is_meta]

    def with_potential_merge(self, axis_id: str) -> list[str]:
        ax = self.get(axis_id)
        return ax.potential_merges if ax else []

    # --- Mutations (tracked for V6 evolution) ---

    def mask(self, axis_id: str, reason: str) -> None:
        ax = self._axes.get(axis_id)
        if ax:
            ax.status = AxisStatus.MASKED

    def unmask(self, axis_id: str) -> None:
        ax = self._axes.get(axis_id)
        if ax and ax.status == AxisStatus.MASKED:
            ax.status = AxisStatus.ACTIVE

    def register_split(self, source_id: str, child_ids: list[str]) -> None:
        ax = self._axes.get(source_id)
        if ax:
            ax.status = AxisStatus.SPLIT
            ax.split_into = child_ids

    def register_merge(self, source_ids: list[str], target_id: str) -> None:
        for sid in source_ids:
            ax = self._axes.get(sid)
            if ax:
                ax.status = AxisStatus.MERGED
                ax.merged_into = target_id

    def add_axis(self, axis: Axis) -> None:
        self._axes[axis.id] = axis
        self.version += 1

    def ids(self) -> list[str]:
        return list(self._axes.keys())

    def active_ids(self) -> list[str]:
        return [a.id for a in self.all_active()]


class QuestionBank:
    """Persistent bank of questions, indexed by axis and trigger type."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _SEED_DIR / "questions_seed.json"
        self._questions: dict[str, Question] = {}
        self.version: int = 1
        self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self.version = raw.get("version", 1)
        for q in raw["questions"]:
            question = Question(**q)
            self._questions[question.id] = question

    def get(self, question_id: str) -> Optional[Question]:
        return self._questions.get(question_id)

    def base_questions_for(self, axis_id: str) -> list[Question]:
        """Return always-active (no trigger) questions for an axis."""
        return [
            q for q in self._questions.values()
            if q.axis_id == axis_id and q.trigger is None
        ]

    def adaptive_questions_for(self, axis_id: str) -> list[Question]:
        """Return trigger-dependent questions for an axis."""
        return [
            q for q in self._questions.values()
            if q.axis_id == axis_id and q.trigger is not None
        ]

    def questions_for_axes(self, axis_ids: list[str]) -> list[Question]:
        return [
            q for q in self._questions.values()
            if q.axis_id in axis_ids and q.trigger is None
        ]

    def all_ids(self) -> list[str]:
        return list(self._questions.keys())
