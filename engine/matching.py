"""Tension engine — compares a candidate Profile against a JobCard, axis by axis.

Direct port of the V4 principle (backend/engine.py::_build_profile_tensions,
_compute_tension_intensity): expose per-axis tension, never a hidden verdict.
Adapted to v5's ontology: 18 axes instead of 6, AxisScore.confidence taken
into account (a tension read off a low-confidence axis is flagged, not hidden),
and effective_weight/status (masked axes are skipped, not silently zeroed).

fit_score is a transparent by-product (weighted mean of 1 - distance), not a
recommendation — same "expose structure, never decide" philosophy as V4.
"""
from __future__ import annotations

from ontology.models import (
    AxisTension,
    JobCard,
    MatchResult,
    Profile,
    AxisStatus,
)
from ontology.registry import AxisRegistry

# Distance thresholds for intensity — mirrors V4's 3-level LOW/MEDIUM/HIGH
# scheme (backend/engine.py::_compute_tension_intensity), expressed on the
# continuous 0-1 scale used by AxisScore/JobAxisRequirement instead of V4's
# discrete LOW/MEDIUM/HIGH job levels.
LOW_THRESHOLD = 0.15
MEDIUM_THRESHOLD = 0.35

# Below this confidence, a tension reading is reported but excluded from
# fit_score and top_tensions — mirrors ProfileScorer.low_confidence_axes().
CONFIDENCE_FLOOR = 0.6

# Distances below this are treated as "aligned" rather than a direction —
# avoids flagging noise as a directional tension.
ALIGNMENT_EPSILON = 0.08


def _intensity(distance: float) -> str:
    if distance < LOW_THRESHOLD:
        return "low"
    if distance < MEDIUM_THRESHOLD:
        return "medium"
    return "high"


def _direction(candidate_value: float, job_level: float) -> str:
    diff = job_level - candidate_value
    if abs(diff) < ALIGNMENT_EPSILON:
        return "aligned"
    return "job_demands_more" if diff > 0 else "job_demands_less"


def _interpretation(label: str, candidate_value: float, job_level: float,
                     direction: str, low_confidence: bool) -> str:
    c_pct = round(candidate_value * 100)
    j_pct = round(job_level * 100)
    if low_confidence:
        return (f"Signal insuffisant sur « {label} » ({c_pct}% mesuré, confiance basse) — "
                f"la comparaison avec le poste ({j_pct}%) n'est pas fiable en l'état.")
    if direction == "aligned":
        return f"« {label} » aligné : profil {c_pct}%, poste {j_pct}%."
    if direction == "job_demands_more":
        return f"Tension sur « {label} » : le poste demande plus ({j_pct}%) que le profil n'en montre ({c_pct}%)."
    return f"Tension sur « {label} » : le poste demande moins ({j_pct}%) que le profil n'en montre ({c_pct}%)."


def compute_tensions(profile: Profile, job: JobCard, registry: AxisRegistry) -> MatchResult:
    """Compare a Profile to a JobCard, axis by axis. Never aggregates silently."""
    tensions: list[AxisTension] = []
    low_confidence_axes: list[str] = []
    skipped_axes: list[str] = []

    for axis_id, req in job.axis_requirements.items():
        axis = registry.get(axis_id)
        score = profile.axis_scores.get(axis_id)

        if score is None or score.status != AxisStatus.ACTIVE:
            skipped_axes.append(axis_id)
            continue

        label = axis.label if axis else axis_id
        distance = abs(score.raw_value - req.level)
        low_conf = score.confidence < CONFIDENCE_FLOOR
        direction = _direction(score.raw_value, req.level)
        intensity = _intensity(distance)

        if low_conf:
            low_confidence_axes.append(axis_id)

        tensions.append(AxisTension(
            axis_id=axis_id,
            axis_label=label,
            candidate_value=score.raw_value,
            candidate_confidence=score.confidence,
            job_level=req.level,
            importance=req.importance,
            distance=distance,
            intensity=intensity,
            direction=direction,
            low_confidence=low_conf,
            interpretation=_interpretation(label, score.raw_value, req.level, direction, low_conf),
        ))

    # fit_score: importance-weighted mean of (1 - distance), confident axes only.
    # Transparent by construction — every term is visible in `tensions` above.
    confident = [t for t in tensions if not t.low_confidence]
    weight_sum = sum(t.importance for t in confident)
    if weight_sum > 0:
        fit_score = sum((1 - t.distance) * t.importance for t in confident) / weight_sum
    else:
        fit_score = 0.0

    top_tensions = sorted(
        (t for t in confident if t.intensity in ("medium", "high")),
        key=lambda t: (t.distance * t.importance),
        reverse=True,
    )

    return MatchResult(
        user_id=profile.user_id,
        job_id=job.job_id,
        tensions=tensions,
        fit_score=round(fit_score, 4),
        top_tensions=[t.axis_id for t in top_tensions],
        low_confidence_axes=low_confidence_axes,
        skipped_axes=skipped_axes,
    )
