"""Archetype distribution — Builder/Expert/Operator/Leader/Connecteur, ported from
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
#
# 2026-08-05: bumped 0.06 -> 0.10 as an interim, more conservative default
# while research/archetype_calibration_log.py accumulates real raw_affinity
# data (see its module docstring). 0.06 was polarizing enough that a
# modest real gap (e.g. Operator 26.8% vs. Connecteur 22.2% raw) rendered as
# 72% vs. 13% — legible, but risking more certainty than the underlying
# signal supports. 0.10 is still a guess, not a calibrated value: replace
# both once n>=50-100 real completed profiles exist in the calibration log.
DISPLAY_TEMPERATURE = 0.10  # PROVISIONAL — recalibrate on real beta data before any public-facing use

# Softmax alone can push a genuinely-present-but-not-dominant archetype
# toward ~0-2%, which reads as "this trait is absent" rather than "this
# trait exists but isn't dominant" — verified 2026-08-02: a manager
# persona with real Operator-relevant signal (behavioral_stability=0.7,
# social_interaction=0.8) still fell to Operator=4.0% post-softmax.
# DISPLAY_FLOOR raises anything under this threshold up to it, taking the
# difference proportionally from archetypes above the threshold — so the
# dominant stays clearly dominant, but no archetype reads as literally
# nonexistent. A profile with no real separation (all raw_affinity close
# together, nothing under the floor) is untouched by this step.
#
# DISPLAY_FLOOR IS NOT CALIBRATED ON REAL DATA either — same caveat as
# DISPLAY_TEMPERATURE above, must be set from real beta distributions,
# not guessed.
DISPLAY_FLOOR = 5.0  # PROVISIONAL — % minimum per archetype after softmax, before any public-facing use

# Even after DISPLAY_TEMPERATURE + DISPLAY_FLOOR, some profiles are
# genuinely balanced — not under-answered (see low_confidence below,
# which is a separate, existing check on data COVERAGE), but a real
# person whose answers just don't favor one archetype over another.
# Verified 2026-08-02: 4 "typed" personas showed a dominant-secondary
# display_percentage gap of 33-69pt; one deliberately balanced persona
# (full confidence=0.85, low_confidence=False) showed only 3pt.
# Presenting that person as "Connecteur" because it happens to edge out
# "Builder" by 3pt would misrepresent a real, positive trait (versatility)
# as a false certainty — and risks a different archetype coming out on a
# retest, undermining trust in the tool.
#
# DISPLAY_MIN_GAP gates which display_mode a profile gets:
#   - low_confidence=True                    -> "insufficient_signal"
#   - gap between top two display_percentage
#     below DISPLAY_MIN_GAP                  -> "polyvalent"
#   - otherwise                              -> "typed"
#
# 15.0 sits between the 3pt and 33-69pt clusters observed above, but —
# same caveat as DISPLAY_TEMPERATURE/DISPLAY_FLOOR — this is picked from
# 5 synthetic personas, not real data, and must be recalibrated once a
# real beta distribution of gaps exists.
DISPLAY_MIN_GAP = 15.0  # PROVISIONAL — minimum dominant/secondary display_percentage gap to call a profile "typed"

# Remark 2 (2026-08-02 review): a bare number ("59%") invites questions
# the model can't honestly answer — 59% of what, relative to what
# population, is this a probability? display_percentage is neither: it's
# a share relative to the person's OWN other archetype scores, not a
# population-relative measure. Tiers avoid implying a precision/referent
# that doesn't exist, while still conveying relative strength.
# PROVISIONAL, same caveat as the constants above — boundaries picked to
# fit the 5 synthetic personas (dominants 59-77% -> "forte", secondaries
# 8-26% -> "modérée"/"légère"), not calibrated on real distributions.
DISPLAY_TIERS = [
    (55.0, "forte"),
    (30.0, "marquée"),
    (15.0, "modérée"),
]  # below the lowest threshold -> "légère"


def _tier_label(pct: float) -> str:
    for threshold, label in DISPLAY_TIERS:
        if pct >= threshold:
            return label
    return "légère"


def _display_mode_and_summary(scores: list[ArchetypeScore], low_confidence: bool) -> tuple[str, str]:
    """Presentation-only classification + human-readable text — never used for
    dominant/secondary/summary/matching, see module docstring above."""
    by_display = sorted(scores, key=lambda s: s.display_percentage, reverse=True)
    top, runner_up = by_display[0], by_display[1]

    if low_confidence:
        return (
            "insufficient_signal",
            "Profil pas encore assez précis pour une lecture fiable — répondre à plus de questions l'affinera.",
        )

    gap = top.display_percentage - runner_up.display_percentage
    if gap < DISPLAY_MIN_GAP:
        return (
            "polyvalent",
            f"Configuration polyvalente — à l'aise aussi bien en {top.name} qu'en {runner_up.name}, "
            "sans mode de fonctionnement nettement dominant.",
        )

    # Remark 1 (2026-08-02 review): "Mon profil professionnel : Connecteur" reads
    # as a personality label ("I am a Connecteur"). "Configuration dominante"
    # keeps the sentence about what the model's reading says, not an identity
    # claim — same shift in framing throughout constraints.py/archetype.py.
    return (
        "typed",
        f"Configuration dominante : {top.name} (affinité {_tier_label(top.display_percentage)}), "
        f"avec une composante {runner_up.name} (affinité {_tier_label(runner_up.display_percentage)}).",
    )


def _display_percentages(raw_affinity: dict[str, float], temperature: float = DISPLAY_TEMPERATURE, floor: float = DISPLAY_FLOOR) -> dict[str, float]:
    """Centered-softmax transform of raw_affinity, floored — presentation only, see module docstring above."""
    if not raw_affinity or all(v == 0.0 for v in raw_affinity.values()):
        return {name: 0.0 for name in raw_affinity}
    values = list(raw_affinity.values())
    mean = sum(values) / len(values)
    exp_values = {name: math.exp((v - mean) / temperature) for name, v in raw_affinity.items()}
    total = sum(exp_values.values())
    pct = {name: v / total * 100 for name, v in exp_values.items()}

    below = {name: v for name, v in pct.items() if v < floor}
    above = {name: v for name, v in pct.items() if v >= floor}
    if below and above:
        deficit = sum(floor - v for v in below.values())
        above_total = sum(above.values())
        if deficit < above_total:
            pct = {
                name: floor if name in below else v - deficit * (v / above_total)
                for name, v in pct.items()
            }
        # else: raising every below-floor archetype to `floor` would require
        # taking more than 100% of what's above the floor combined (only
        # possible with many archetypes under the floor and a thin margin
        # above it) — skip flooring for this profile rather than risk a
        # negative percentage; fall through to the raw softmax values.
    # if everything is below the floor (fully flat profile) or everything
    # is above it, there's nothing to redistribute — leave pct as-is.

    return {name: round(v, 1) for name, v in pct.items()}

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
    # Renamed from "Explorer" 2026-09-07: the old signature (exploration/
    # ambiguity_tolerance/cognitive_flexibility/risk_appetite/autonomy/
    # decision_speed) lost on 10/10 real job descriptions tested 2026-07-20
    # (research finding, see git history on this dict) — its axes were the
    # sparsest in the signal vocabulary (1 signal each) AND it shared 3 of
    # its 6 axes with Builder, so it rarely won even when signals fired.
    # "Connecteur" redefines this slot around a genuinely distinct trait —
    # building/maintaining relationships across people, teams or
    # organizations — rather than trying to patch the old exploration/risk
    # framing. social_interaction/cognitive_flexibility/influence/
    # ambiguity_tolerance were chosen because each now has >=2 signals in
    # signals_seed.json (social_interaction was raised from 1 to 3 in the
    # same change, see coordination_multi_equipes/reseau_relationnel_cle/
    # interface_entre_equipes). Still overlaps Leader on influence and
    # Operator/Leader on social_interaction — full separation isn't
    # achievable with only 16 primary axes across 5 archetypes — but this
    # is a first cut, not empirically validated yet: same caveat as
    # DISPLAY_TEMPERATURE/DISPLAY_MIN_GAP above, must be checked against
    # real job descriptions the way the old Explorer signature was.
    "Connecteur": {
        "social_interaction": 0.35,
        "cognitive_flexibility": 0.25,
        "influence": 0.20,
        "ambiguity_tolerance": 0.20,
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
    distribution, _raw_affinity = compute_archetype_distribution_with_raw(profile, registry)
    return distribution


def compute_archetype_distribution_with_raw(
    profile: Profile, registry: AxisRegistry
) -> tuple[ArchetypeDistribution, dict[str, float]]:
    """Same computation as compute_archetype_distribution, but also returns
    the pre-normalization raw_affinity dict — needed by callers that want
    to log it for calibration (see research/archetype_calibration_log.py),
    since `percentage`/`display_percentage` are both derived FROM
    raw_affinity and aren't interchangeable with it for that purpose."""
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

    # Presentation-only layer — see module comment above DISPLAY_MIN_GAP.
    # Does not affect dominant/secondary/summary/low_confidence above.
    if scores[0].percentage == 0:
        display_mode, display_summary = "insufficient_signal", summary
    else:
        display_mode, display_summary = _display_mode_and_summary(scores, low_confidence)

    distribution = ArchetypeDistribution(
        user_id=profile.user_id,
        scores=scores,
        dominant=dominant,
        secondary=secondary,
        low_confidence=low_confidence,
        summary=summary,
        display_mode=display_mode,
        display_summary=display_summary,
    )
    return distribution, raw_affinity
