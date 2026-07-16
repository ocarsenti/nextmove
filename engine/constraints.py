"""Job constraint engine — detects structural constraints from a JobCard's axis
requirements (combinations, never a single axis), then assesses whether a
candidate Profile is compatible with each one.

Three-stage pipeline, matching the design discussed for NextMove:

    JobCard.axis_requirements  (dimensions of the job's context)
              |
    detect_constraints()        -> which patterns fire (business constraints)
              |
    build_job_narrative()       -> what this job structurally demands, in prose,
              |                     independent of any candidate
    assess_compatibility()      -> per constraint, is THIS candidate equipped
              |                     for it (not a raw distance — a readiness read
              |                     over the constraint's relevant axes)
    build_match_narrative()     -> "compatible with X and Y, Z risks friction"

Same isolation principle as engine/archetype.py: this module reads Profile
and JobCard, never writes to them, and is never imported by engine/matching.py
or engine/scorer.py — it cannot influence fit_score or raw tensions.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path

from ontology.models import (
    AxisStatus,
    ConstraintCompatibility,
    JobCard,
    JobConstraint,
    JobConstraintProfile,
    Profile,
)

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

# A candidate axis below this value is treated as a weak spot for whatever
# constraint relies on it — separate from AxisScore.confidence, which
# measures how *sure* we are of the reading, not how *high* it is.
SUFFICIENCY_FLOOR = 0.5
CONFIDENCE_FLOOR = 0.6

_SEED_PATH = Path(__file__).parent.parent / "ontology" / "constraints_seed.json"


def load_constraint_library(path: Path = _SEED_PATH) -> list[JobConstraint]:
    data = json.loads(path.read_text())
    return [JobConstraint(**c) for c in data["constraints"]]


def _trigger_holds(trigger, job: JobCard) -> bool:
    req = job.axis_requirements.get(trigger.axis_id)
    if req is None:
        return False
    return _OPS[trigger.operator](req.level, trigger.threshold)


def detect_constraints(job: JobCard, library: list[JobConstraint]) -> list[JobConstraint]:
    """Which constraints does this job's axis_requirements combination trigger?
    A constraint fires only if ALL of its triggers hold — it's a pattern
    across axes, never a single-axis reading."""
    return [c for c in library if all(_trigger_holds(t, job) for t in c.triggers)]


def build_job_narrative(job: JobCard, detected: list[JobConstraint]) -> str:
    if not detected:
        return f"« {job.title} » ne déclenche aucune contrainte structurelle marquante dans la bibliothèque actuelle."
    names = [c.name[0].lower() + c.name[1:] for c in detected]
    if len(names) == 1:
        body = names[0]
    else:
        body = ", ".join(names[:-1]) + " et " + names[-1]
    return f"« {job.title} » n'est pas un poste générique : c'est un environnement où il faut {body}."


def assess_compatibility(profile: Profile, constraint: JobConstraint) -> ConstraintCompatibility:
    checked: list[str] = []
    unavailable: list[str] = []
    weak: list[str] = []

    for axis_id in constraint.axes_involved:
        score = profile.axis_scores.get(axis_id)
        if score is None or score.status != AxisStatus.ACTIVE:
            unavailable.append(axis_id)
            continue
        if score.confidence < CONFIDENCE_FLOOR:
            unavailable.append(axis_id)
            continue
        checked.append(axis_id)
        if score.raw_value < SUFFICIENCY_FLOOR:
            weak.append(axis_id)

    if not checked:
        status = "unknown"
        narrative = f"Signal insuffisant pour évaluer la compatibilité avec « {constraint.name} »."
        risks_flagged: list[str] = []
    elif not weak:
        status = "aligned"
        narrative = f"Profil compatible avec « {constraint.name} »."
        risks_flagged = []
    else:
        status = "tension"
        narrative = f"« {constraint.name} » risque de générer de la friction — signal plus faible sur {', '.join(weak)}."
        risks_flagged = constraint.risks

    return ConstraintCompatibility(
        constraint_id=constraint.id,
        constraint_name=constraint.name,
        status=status,
        axes_checked=checked,
        axes_unavailable=unavailable,
        risks_flagged=risks_flagged,
        narrative=narrative,
    )


def build_match_narrative(compatibilities: list[ConstraintCompatibility]) -> str:
    if not compatibilities:
        return ""

    def _lc(name: str) -> str:
        return name[0].lower() + name[1:] if name else name

    aligned = [_lc(c.constraint_name) for c in compatibilities if c.status == "aligned"]
    tension = [_lc(c.constraint_name) for c in compatibilities if c.status == "tension"]
    unknown = [_lc(c.constraint_name) for c in compatibilities if c.status == "unknown"]

    parts = []
    if aligned:
        parts.append("compatible avec " + " et ".join(aligned) if len(aligned) <= 2 else
                      "compatible avec " + ", ".join(aligned[:-1]) + " et " + aligned[-1])
    if tension:
        parts.append(("mais " if aligned else "") + ("la contrainte " if len(tension) == 1 else "les contraintes ") +
                      (" et ".join(tension) if len(tension) <= 2 else ", ".join(tension[:-1]) + " et " + tension[-1]) +
                      (" risque" if len(tension) == 1 else " risquent") + " de générer de la friction")
    if unknown and not aligned and not tension:
        parts.append("signal encore trop faible pour se prononcer sur " + ", ".join(unknown))

    if not parts:
        return "Aucune contrainte détectée à comparer au profil."
    return ("Votre mode de fonctionnement est " + ", ".join(parts) + ".").replace("est compatible", "est compatible")


def analyze_job(job: JobCard, library: list[JobConstraint], profile: Profile | None = None) -> JobConstraintProfile:
    detected = detect_constraints(job, library)
    job_narrative = build_job_narrative(job, detected)

    compatibilities: list[ConstraintCompatibility] = []
    match_narrative = ""
    if profile is not None and detected:
        compatibilities = [assess_compatibility(profile, c) for c in detected]
        match_narrative = build_match_narrative(compatibilities)

    return JobConstraintProfile(
        job_id=job.job_id,
        job_title=job.title,
        detected_constraints=detected,
        job_narrative=job_narrative,
        compatibilities=compatibilities,
        match_narrative=match_narrative,
    )
