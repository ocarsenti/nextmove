"""Success conditions — candidate-facing "conditions de réussite", mirrors the
detect/narrative pattern already used in engine/constraints.py for job-side
constraints, but for positive framing of the candidate's own motivational
signals (see ontology/success_conditions_seed.json for the rule library).

This is presentation-only, same isolation guarantee as archetype.py's
display_percentage/display_mode: never used by matching.py, scorer.py, or
any decision-facing logic — just a traceable translation from axis_scores
to short candidate-facing phrases.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path

from ontology.models import AxisStatus, Profile

_OPERATORS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le}
_SEED_PATH = Path(__file__).resolve().parent.parent / "ontology" / "success_conditions_seed.json"

# PROVISIONAL — same caveat as DISPLAY_TEMPERATURE/DISPLAY_FLOOR/DISPLAY_MIN_GAP
# in engine/archetype.py: picked to behave sensibly on 5 synthetic personas,
# not calibrated on real beta data. Must be revisited together with those
# three once real distributions exist (see review notes 2026-08-02).
MIN_CONFIDENCE = 0.5   # rules resting on an axis below this confidence are not surfaced
TOP_N = 3               # how many conditions to keep for a post/badge

# Remark from 2026-08-02 review: "aucune condition tranchée identifiée" reads
# as the tool having failed, not as a trait of the person. Reframe the
# absence of a dominant condition as information, same move already made for
# archetype.py's "polyvalent" display_mode.
NO_SIGNAL_SUMMARY = (
    "Configuration polyvalente : plusieurs modes de fonctionnement compatibles selon le contexte. "
    "Aucune condition de réussite dominante ne ressort actuellement."
)


def _load_library() -> list[dict]:
    with open(_SEED_PATH, encoding="utf-8") as f:
        return json.load(f)["success_conditions"]


def detect_conditions(profile: Profile, library: list[dict] | None = None) -> list[dict]:
    """Every rule whose triggers are satisfied by this profile, each annotated with
    the evidence behind it: signal_strength (how extreme), confidence (how trustworthy),
    and evidence_strength = signal_strength * confidence (2026-08-02 review — intensity
    and reliability are different things, don't rank on intensity alone)."""
    if library is None:
        library = _load_library()

    fired = []
    for rule in library:
        axis_evidence = []
        satisfied = True
        for trigger in rule["triggers"]:
            score = profile.axis_scores.get(trigger["axis_id"])
            if score is None or score.status != AxisStatus.ACTIVE:
                satisfied = False
                break
            if not _OPERATORS[trigger["operator"]](score.raw_value, trigger["threshold"]):
                satisfied = False
                break
            axis_evidence.append(score)
        if not satisfied:
            continue

        # A rule is only as trustworthy as its least-confident triggering axis
        # (most rules here have a single trigger, so this is just that axis'
        # confidence — written as min() to stay correct for multi-axis rules too).
        signal_strength = min(abs(s.raw_value - 0.5) for s in axis_evidence)
        confidence = min(s.confidence for s in axis_evidence)
        fired.append({
            "id": rule["id"],
            "name": rule["name"],
            "signal_strength": round(signal_strength, 3),
            "confidence": round(confidence, 3),
            "evidence_strength": round(signal_strength * confidence, 3),
            "axis_evidence": {s.axis_id: s.raw_value for s in axis_evidence},
        })
    return fired


def build_conditions_narrative(
    profile: Profile,
    library: list[dict] | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    top_n: int = TOP_N,
) -> dict:
    """Presentation-only: {'display_summary': str, 'conditions': [...]}.
    conditions is the ranked, traceable evidence behind display_summary —
    each entry can answer "how do you know that?" with its exact axis/value."""
    fired = detect_conditions(profile, library)
    trustworthy = [f for f in fired if f["confidence"] > min_confidence]
    trustworthy.sort(key=lambda f: f["evidence_strength"], reverse=True)
    top = trustworthy[:top_n]

    if not top:
        return {"display_summary": NO_SIGNAL_SUMMARY, "conditions": []}

    display_summary = "Mes conditions de réussite : " + " ; ".join(c["name"] for c in top)
    return {"display_summary": display_summary, "conditions": top}
