"""Archetype distribution — Builder/Expert/Operator/Leader/Explorer, ported from
V4 (backend/questionnaire.py::ARCHETYPE_AXIS_AFFINITY) and refined for v5's
18-axis ontology.

Key differences from V4:
- V4 asked 4 dedicated vote questions and used a 6-axis affinity table only
  as a tie-breaker. v5 has no separate vote questions: the distribution is
  derived directly from the axis_scores the questionnaire already produces.
- Each archetype's signature now spans 6-8 of the 16 primary axes (the 2
  meta/modulator axes — cognitive_granularity, context_sensitivity — are
  excluded, since they modulate other axes rather than describing a trait
  themselves).
- Masked or absent axes are dropped from an archetype's signature and the
  remaining weights renormalized, rather than defaulting to 0.5 like V4
  did for missing axes. Concretely: if a job context masks e.g. `rigor`,
  the archetype reading for that context legitimately shifts to rely on
  whatever axes remain active — something V4's fixed 6-axis table could
  never do, since all 6 of its axes always existed.
"""
from __future__ import annotations

import math

from ontology.models import (
    ArchetypeDistribution,
    ArchetypeScore,
    AxisStatus,
    Profile,
)
from ontology.registry import AxisRegistry

CONFIDENCE_FLOOR = 0.6

# ---------------------------------------------------------------------
# PROVISIONAL — display-only differentiation layer
# ---------------------------------------------------------------------
# `percentage` (below) is raw_affinity's share of the total across the 5
# archetypes. Since every raw_affinity is a weighted AVERAGE bounded to
# [0,1] (weights sum to 1 per archetype — see ARCHETYPE_AXIS_AFFINITY),
# and several axes feed 3 of the 5 archetypes at once (autonomy, rigor,
# persistence, cognitive_structuring...), `percentage` mechanically
# clusters around 100/5=20% for most profiles, even ones with genuinely
# high axis scores — the shared axes lift several archetypes together
# instead of isolating one. Verified on synthetic personas 2026-08-02:
# a profile with autonomy=0.85 and persistence=0.80 still only reached
# 23.8% on its dominant archetype.
#
# display_percentage re-expresses the SAME raw_affinity values, centered
# on the profile's own mean and passed through a softmax — so it reflects
# how differentiated a person is RELATIVE TO THEMSELVES rather than
# against the absolute [0,1] scale. A genuinely flat profile (all 5
# raw_affinity close together) still comes out flat; a profile with real
# separation gets a wider, more legible spread. It changes nothing about
# ARCHETYPE_AXIS_AFFINITY or raw_affinity itself.
#
# DISPLAY_TEMPERATURE IS NOT CALIBRATED ON REAL DATA. Synthetic testing
# (independent-axes vs. halo/acquiescence-style correlated axes, n=300
# each) showed the SAME temperature produces a median dominant-secondary
# gap of 24pt under one population assumption and 8pt under the other —
# a 3x swing from the same formula, purely from an assumption about how
# real respondents actually answer. This constant must be recalibrated
# once real beta raw_affinity data exists (recommended: n>=50-100 real
# respondents), by picking T against the empirical distribution of gaps
# rather than any synthetic sample. Until then, treat display_percentage
# as illustrative, not something to build a public-facing badge/claim on.
#
# display_percentage is presentation-only: it must never feed into
# dominant/secondary/summary/low_confidence, matching, or any other
# decision-facing logic (see TestArchitecturalIsolation in
# tests/test_archetype.py for the equivalent guarantee on `percentage`).
DISPLAY_TEMPERATURE = 0.06  # PROVISIONAL — recalibrate on real beta data before any public-facing use


def _display_percentages(raw_affinity: dict[str, float], temperature: float = DISPLAY_TEMPERATURE) -> dict[str, float]:
    """Centered-softmax transform of raw_affinity — presentation only, see module docstring above."""
    if not raw_affinity or all(v == 0.0 for v in raw_affinity.values()):
        return {name: 0.0 for name in raw_affinity}
    values = list(raw_affinity.values())
    mean = sum(values) / len(values)
    exp_values = {name: math.exp((v - mean) / temperature) for name, v in raw_affinity.items()}
    total = sum(exp_values.values())
    return {name: round(v / total * 100, 1) for name, v in exp_values.items()}

# Each archetype's affinity to the 16 primary axes (meta axes excluded).
# Weights are non-negative and sum to 1.0 per archetype — same convention
# as V4, just spread over more axes for a finer signature.
#
# Deliberately behavioral-only: every axis here describes HOW someone acts
# (rigor, decision speed, social interaction...), never WHY (motivation,
# sense of purpose). `value_orientation` — the one clearly motivational
# axis in the ontology ("besoin de sens") — is excluded from every
# signature for that reason: mixing the two families inside one archetype
# creates false splits (e.g. two equally strong Leaders who differ only in
# how mission-driven they are would score as different archetypes for the
# wrong reason). `direction` was also removed from Expert specifically —
# an excellent specialist doesn't need to want to set direction for
# others; that's a Leader trait, not an Expert one.
ARCHETYPE_AXIS_AFFINITY: dict[str, dict[str, float]] = {
    "Builder": {
        "autonomy": 0.20,
        "persistence": 0.15,
        "rigor": 0.15,
        "decision_speed": 0.10,
        "ambiguity_tolerance": 0.10,
        "exploration": 0.10,
        "cognitive_structuring": 0.10,
        "resilience": 0.10,
    },
    "Expert": {
        "rigor": 0.30,
        "cognitive_structuring": 0.25,
        "autonomy": 0.20,
        "persistence": 0.25,
    },
    "Operator": {
        "behavioral_stability": 0.25,
        "social_interaction": 0.15,
        "rigor": 0.15,
        "cognitive_structuring": 0.15,
        "resilience": 0.15,
        "persistence": 0.15,
    },
    "Leader": {
        "influence": 0.30,
        "situational_leadership": 0.25,
        "direction": 0.20,
        "social_interaction": 0.15,
        "decision_speed": 0.10,
    },
    "Explorer": {
        "exploration": 0.25,
        "ambiguity_tolerance": 0.20,
        "cognitive_flexibility": 0.20,
        "risk_appetite": 0.15,
        "autonomy": 0.10,
        "decision_speed": 0.10,
    },
}

ARCHETYPE_NAMES = list(ARCHETYPE_AXIS_AFFINITY.keys())


def _usable_axes(profile: Profile, weights: dict[str, float]) -> dict[str, float]:
    """Weights for axes that are present and ACTIVE in this profile, renormalized to sum 1.0."""
    usable = {
        axis_id: w for axis_id, w in weights.items()
        if (score := profile.axis_scores.get(axis_id)) is not None
        and score.status == AxisStatus.ACTIVE
    }
    total = sum(usable.values())
    if total == 0:
        return {}
    return {axis_id: w / total for axis_id, w in usable.items()}


def compute_archetype_distribution(profile: Profile, registry: AxisRegistry) -> ArchetypeDistribution:
    raw_affinity: dict[str, float] = {}
    axes_used: dict[str, list[str]] = {}
    axes_missing: dict[str, list[str]] = {}
    confidence: dict[str, float] = {}

    for name, weights in ARCHETYPE_AXIS_AFFINITY.items():
        usable = _usable_axes(profile, weights)
        axes_used[name] = list(usable.keys())
        axes_missing[name] = [a for a in weights if a not in usable]

        if not usable:
            raw_affinity[name] = 0.0
            confidence[name] = 0.0
            continue

        raw_affinity[name] = sum(
            w * profile.axis_scores[axis_id].raw_value for axis_id, w in usable.items()
        )
        confidence[name] = sum(
            w * profile.axis_scores[axis_id].confidence for axis_id, w in usable.items()
        )

    total_affinity = sum(raw_affinity.values())
    if total_affinity > 0:
        percentages = {name: round(v / total_affinity * 100, 1) for name, v in raw_affinity.items()}
    else:
        percentages = {name: 0.0 for name in ARCHETYPE_NAMES}

    # Presentation-only layer — see module-level comment above DISPLAY_TEMPERATURE.
    # Does not affect percentages, dominant/secondary, summary, or low_confidence below.
    display_percentages = _display_percentages(raw_affinity)

    scores = [
        ArchetypeScore(
            name=name,
            percentage=percentages[name],
            display_percentage=display_percentages[name],
            confidence=round(confidence[name], 3),
            axes_used=axes_used[name],
            axes_missing=axes_missing[name],
        )
        for name in ARCHETYPE_NAMES
    ]
    scores.sort(key=lambda s: s.percentage, reverse=True)

    dominant = scores[0].name if scores[0].percentage > 0 else "Unscored"
    secondary = scores[1].name if len(scores) > 1 and scores[1].percentage >= 15 else None
    low_confidence = scores[0].confidence < CONFIDENCE_FLOOR if scores[0].percentage > 0 else True

    if scores[0].percentage == 0:
        summary = "Signal insuffisant sur tous les axes pertinents — aucun archétype préférentiel lisible."
    elif secondary:
        summary = f"Archétype préférentiel : {dominant} ({scores[0].percentage}%), avec une composante {secondary} ({scores[1].percentage}%)."
    else:
        summary = f"Archétype préférentiel : {dominant}, nettement dominant ({scores[0].percentage}%)."
    if scores[0].percentage > 0 and low_confidence:
        summary += " Confiance encore faible sur cette lecture — répondre à plus de questions l'affinera."

    return ArchetypeDistribution(
        user_id=profile.user_id,
        scores=scores,
        dominant=dominant,
        secondary=secondary,
        low_confidence=low_confidence,
        summary=summary,
    )
