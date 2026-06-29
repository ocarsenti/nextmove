"""Audit log — immutable, append-only record of all ontology transformations (V6)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class OntologyEvent(BaseModel):
    event_id: str
    timestamp: str
    rule_id: str
    rule_type: str                        # split | merge | mask | reweight | transform
    axis_id: Optional[str] = None        # source axis
    axes_involved: list[str] = Field(default_factory=list)
    resulting_axes: list[str] = Field(default_factory=list)
    context: Optional[str] = None
    signals_used: dict = Field(default_factory=dict)
    explanation: str = ""
    reversible: bool = True
    reversed_by: Optional[str] = None    # event_id of the reversal event


class AuditLog:
    """
    Append-only log of all ontology transformation events.
    Persisted to JSON. Never mutated — only appended.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or Path(__file__).parent.parent / "data" / "audit" / "audit_log.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[OntologyEvent] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._events = [OntologyEvent(**e) for e in raw.get("events", [])]

    def _save(self) -> None:
        payload = {"events": [e.model_dump() for e in self._events]}
        self._path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append(self, event: OntologyEvent) -> None:
        self._events.append(event)
        self._save()

    def record(
        self,
        rule_id: str,
        rule_type: str,
        axis_id: Optional[str] = None,
        axes_involved: Optional[list[str]] = None,
        resulting_axes: Optional[list[str]] = None,
        context: Optional[str] = None,
        signals_used: Optional[dict] = None,
        explanation: str = "",
        reversible: bool = True,
    ) -> OntologyEvent:
        event = OntologyEvent(
            event_id=f"evt_{len(self._events):06d}_{rule_id}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            rule_id=rule_id,
            rule_type=rule_type,
            axis_id=axis_id,
            axes_involved=axes_involved or ([axis_id] if axis_id else []),
            resulting_axes=resulting_axes or [],
            context=context,
            signals_used=signals_used or {},
            explanation=explanation,
            reversible=reversible,
        )
        self.append(event)
        return event

    def all_events(self) -> list[OntologyEvent]:
        return list(self._events)

    def events_for_axis(self, axis_id: str) -> list[OntologyEvent]:
        return [e for e in self._events if axis_id in e.axes_involved or e.axis_id == axis_id]

    def events_by_type(self, rule_type: str) -> list[OntologyEvent]:
        return [e for e in self._events if e.rule_type == rule_type]

    def latest(self, n: int = 10) -> list[OntologyEvent]:
        return self._events[-n:]

    def count(self) -> int:
        return len(self._events)

    def replay_summary(self) -> list[dict]:
        """Human-readable summary of all events for inspection."""
        return [
            {
                "id": e.event_id,
                "at": e.timestamp,
                "type": e.rule_type,
                "rule": e.rule_id,
                "axes": e.axes_involved,
                "result": e.resulting_axes,
                "explanation": e.explanation[:100] + "..." if len(e.explanation) > 100 else e.explanation,
            }
            for e in self._events
        ]
