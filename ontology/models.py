"""Core ontology models — axes as first-class objects, rules as explicit structures."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ===================================================================
# ENUMS
# ===================================================================

class AxisStatus(str, Enum):
    ACTIVE = "active"
    MASKED = "masked"       # suppressed in this context
    MERGED = "merged"       # absorbed into another axis
    SPLIT = "split"         # decomposed into children
    DEPRECATED = "deprecated"


class RuleType(str, Enum):
    SPLIT = "split"
    MERGE = "merge"
    MASK = "mask"
    REWEIGHT = "reweight"


class SignalType(str, Enum):
    CONTEXT_MATCH = "context_match"
    AXIS_VALUE_THRESHOLD = "axis_value_threshold"
    HIGH_VARIANCE = "high_variance"           # V6+ (requires data)
    HIGH_CORRELATION = "high_correlation"     # V6+ (requires data)
    LOW_DISCRIMINATION = "low_discrimination" # V6+ (requires data)


class Layer(int, Enum):
    INDIVIDUAL_DYNAMICS = 1
    SOCIAL_INTERACTION = 2
    COGNITION = 3
    ADAPTIVE_DYNAMICS = 4


# ===================================================================
# RULES
# ===================================================================

class RuleCondition(BaseModel):
    signal: SignalType
    axis_ids: list[str] = Field(default_factory=list)
    context: Optional[str] = None
    threshold: Optional[float] = None


class RuleEffect(BaseModel):
    action: RuleType
    weight_modifier: Optional[float] = None   # for reweight (delta applied to weight)
    into: Optional[list[str]] = None          # for split (target axis IDs)
    target: Optional[str] = None             # for merge (target axis ID)


class AxisRule(BaseModel):
    rule_id: str
    condition: RuleCondition
    effect: RuleEffect
    explanation: str
    requires_data: bool = False              # True = V6+ only


# ===================================================================
# AXIS
# ===================================================================

class ContextualVariant(BaseModel):
    context_id: str
    weight_modifier: float = 1.0
    description: str = ""


class UncertaintyModel(BaseModel):
    type: str = "heuristic"
    confidence: float = 0.0
    min_questions: int = 3


class Axis(BaseModel):
    id: str
    label: str
    description: str
    layer: int                              # 1-4

    parents: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    is_meta: bool = False                   # modulators: context_sensitivity, granularity

    contexts: list[str] = Field(default_factory=list)  # empty = applies everywhere
    contextual_variants: dict[str, ContextualVariant] = Field(default_factory=dict)

    rules: list[AxisRule] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)

    weight: float = 1.0
    status: AxisStatus = AxisStatus.ACTIVE
    version: int = 1

    merged_into: Optional[str] = None
    split_from: Optional[str] = None
    split_into: Optional[list[str]] = None

    potential_merges: list[str] = Field(default_factory=list)
    potential_splits: list[dict] = Field(default_factory=list)

    uncertainty_model: UncertaintyModel = Field(default_factory=UncertaintyModel)


# ===================================================================
# QUESTIONS
# ===================================================================

class QuestionOption(BaseModel):
    label: str
    value: str
    score: float                            # 0.0, 0.5, or 1.0


class QuestionTrigger(BaseModel):
    axis_id: str
    condition: str                          # "high_value", "low_value", "high_uncertainty"
    threshold: float = 0.5


class Question(BaseModel):
    id: str
    text: str
    axis_id: str
    options: list[QuestionOption]
    trigger: Optional[QuestionTrigger] = None   # None = always shown (base question)
    activates: list[str] = Field(default_factory=list)
    difficulty: float = 0.5                     # 0-1, discriminability level
    discriminability: float = 0.5


# ===================================================================
# PROFILE / SCORING
# ===================================================================

class AxisScore(BaseModel):
    axis_id: str
    raw_value: float                        # mean of answers (0.0-1.0)
    confidence: float                       # 0.0-1.0 based on n_questions and variance
    n_questions: int
    variance: float = 0.0
    raw_answers: list[float] = Field(default_factory=list)
    effective_weight: float = 1.0           # after context rules applied
    status: AxisStatus = AxisStatus.ACTIVE  # may be masked by context


class Profile(BaseModel):
    user_id: str
    axis_scores: dict[str, AxisScore]
    context: Optional[str] = None
    registry_version: int = 1
    completed_questions: list[str] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
    is_complete: bool = False


# ===================================================================
# JOB CARD — persistent job-side counterpart to Profile
# ===================================================================
# A JobCard states, per axis, the level a role actually demands — filled
# by a recruiter/hiring manager (or extracted from a job description),
# independently of any candidate. Matching a Profile against a JobCard
# is then a pure comparison, not a black-box score: every axis produces
# an explicit tension (or its absence), the same philosophy V4 used for
# comparing two jobs, ported here to the 18-axis living ontology.

class JobAxisRequirement(BaseModel):
    axis_id: str
    level: float = Field(ge=0.0, le=1.0)        # required/expected level, same scale as AxisScore.raw_value
    importance: float = Field(default=1.0, ge=0.0, le=1.0)  # how much this axis matters for THIS job
    note: str = ""                              # optional human rationale for the level


class JobCard(BaseModel):
    job_id: str
    title: str
    context_id: Optional[str] = None            # links to ContextEngine's known contexts, if applicable
    description: str = ""
    axis_requirements: dict[str, JobAxisRequirement] = Field(default_factory=dict)
    detected_signal_ids: list[str] = Field(default_factory=list)  # from extraction, if any — empty for manually-built jobs
    version: int = 1


# ===================================================================
# TENSION / MATCHING — candidate Profile vs. JobCard, axis by axis
# ===================================================================

class AxisTension(BaseModel):
    axis_id: str
    axis_label: str
    candidate_value: float
    candidate_confidence: float
    job_level: float
    importance: float
    distance: float                              # abs(candidate_value - job_level), 0-1
    intensity: str                                # "low" | "medium" | "high"
    direction: str                                # "job_demands_more" | "job_demands_less" | "aligned"
    low_confidence: bool                          # True if candidate signal too thin to trust this reading
    interpretation: str


class MatchResult(BaseModel):
    user_id: str
    job_id: str
    tensions: list[AxisTension]
    fit_score: float                              # transparent weighted average, NOT a hidden verdict
    top_tensions: list[str]                        # axis_ids, highest intensity first (confident axes only)
    low_confidence_axes: list[str]                 # axes excluded from fit_score / top_tensions, and why
    skipped_axes: list[str]                        # axes in the job but absent/masked in the profile


# ===================================================================
# ARCHETYPE DISTRIBUTION — the 5-persona summary from V4, ported and
# refined onto the 18-axis ontology (see engine/archetype.py)
# ===================================================================
# V4 computed this from 4 discrete vote questions + a 6-axis tie-break
# table. v5 drops the separate vote questions entirely: the distribution
# is derived directly from the same axis_scores the questionnaire already
# produces, using a wider signature per archetype (6-8 of the 16 primary
# axes each, meta axes excluded). Masked/absent axes are excluded and the
# remaining weights renormalized — so the reading legitimately shifts with
# job context, which V4's fixed 6-axis table could never do.

class ArchetypeScore(BaseModel):
    name: str                     # "Builder" | "Expert" | "Operator" | "Leader" | "Connecteur"
    percentage: float             # 0-100, the 5 always sum to 100 (or to 0 if fully unscoreable)
    display_percentage: float     # 0-100, the 5 always sum to 100 — PRESENTATION ONLY, see engine/archetype.py
    confidence: float             # weighted mean confidence of the axes that contributed
    axes_used: list[str]          # which axes actually contributed (post masking/renormalization)
    axes_missing: list[str]       # axes in this archetype's signature that were masked/absent


class ArchetypeDistribution(BaseModel):
    user_id: str
    scores: list[ArchetypeScore]  # always 5, sorted descending by percentage
    dominant: str                 # name of the top archetype
    secondary: Optional[str]      # name of the runner-up, if it's not negligibly close to 0
    low_confidence: bool          # True if the dominant reading itself rests on thin signal
    summary: str                  # short human-readable description of the dominance pattern
    display_mode: str             # "typed" | "polyvalent" | "insufficient_signal" — PRESENTATION ONLY, see engine/archetype.py
    display_summary: str          # human-readable, presentation-only counterpart to `summary` — do not use for any decision-facing logic


# ===================================================================
# JOB CONSTRAINTS — qualitative diagnoses built from axis COMBINATIONS,
# never a single axis alone (see engine/constraints.py)
# ===================================================================
# Mirrors EvidenceAble's core move: don't show raw signals (confounding=8,
# selection_bias=5), produce a diagnosis ("this study can't identify a
# causal effect because..."). Here: don't show raw axis levels, produce a
# structural read of what the JOB actually demands ("influencing people
# without formal authority") — then, separately, whether the candidate's
# profile is compatible with that specific demand.
#
# A JobConstraint is detected from the JOB's axis_requirements (never from
# a candidate) — it describes a property of the role itself. Compatibility
# with a candidate is a distinct, later step.

class ConstraintTrigger(BaseModel):
    axis_id: str
    operator: str    # ">" | ">=" | "<" | "<="
    threshold: float  # 0-1, same scale as JobAxisRequirement.level


class JobConstraint(BaseModel):
    id: str
    name: str
    triggers: list[ConstraintTrigger]   # ALL must hold against job.axis_requirements (AND)
    axes_involved: list[str]            # candidate axes checked for compatibility with this constraint
    risks: list[str]                    # what tends to go wrong when a candidate is a poor fit for this constraint
    required_signal_votes: list[str] = Field(default_factory=list)
    # If non-empty: the axis-threshold triggers above are NECESSARY but no longer
    # SUFFICIENT — at least one of these signal_ids must also have been detected
    # in the job's source text (job.detected_signal_ids) for the constraint to
    # fire. Added after two real job descriptions both false-positived C005
    # (cognitive_structuring pushed high by an unrelated signal, e.g. internal
    # governance process language, not actual regulatory conformity) — the
    # axis alone can't distinguish which signal produced it, so for a
    # constraint whose NAME implies a specific textual origin, that origin
    # must be checked directly. Left empty (no change in behavior) for every
    # constraint that hasn't shown this failure mode.


class ConstraintCompatibility(BaseModel):
    constraint_id: str
    constraint_name: str
    status: str                # "aligned" | "tension" | "unknown"
    axes_checked: list[str]    # candidate axes actually usable (present, active, confident enough)
    axes_unavailable: list[str]  # axes_involved that were masked/absent/too uncertain to use
    risks_flagged: list[str]   # subset of constraint.risks judged relevant given the candidate's profile
    narrative: str


class JobConstraintProfile(BaseModel):
    job_id: str
    job_title: str
    detected_constraints: list[JobConstraint]
    job_narrative: str                          # what this job structurally demands, independent of any candidate
    compatibilities: list[ConstraintCompatibility]  # empty if no user_id / profile was supplied
    match_narrative: str                         # empty if no candidate compared


# ===================================================================
# SIGNAL EXTRACTION — the traceable layer between raw job-description
# text and axis_requirements (see engine/job_extraction.py)
# ===================================================================
# Splits what used to be a single opaque LLM call (text -> axis levels)
# into two inspectable steps:
#   1. LLM picks applicable signals from a FIXED, closed vocabulary
#      (ontology/signals_seed.json) and must quote the exact phrase in
#      the description that justifies each one — no free-form invention.
#   2. A deterministic Python function (signals_to_axes) maps signals to
#      axis levels via a fixed table, with no LLM involved. Same rule for
#      every job, every time — reproducible and editable without touching
#      a prompt.
# If an axis level looks wrong, you can now see WHY: which signal fired,
# and which sentence in the job description triggered it.

class AxisEffect(BaseModel):
    level: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)


class JobSignal(BaseModel):
    id: str
    label: str
    axis_effects: dict[str, AxisEffect]
    constraint_votes: list[str] = Field(default_factory=list)
    # Constraint ids this signal directly supports — read by
    # JobConstraint.required_signal_votes, not by the axis-threshold path.
    # A signal can contribute to axes AND vote for a constraint at the same
    # time; the two mechanisms are independent.


class DetectedSignal(BaseModel):
    signal_id: str
    source_phrase: str    # verbatim excerpt from the job description that justifies this signal


# ===================================================================
# EXPLANATION
# ===================================================================

class RuleFiring(BaseModel):
    rule_id: str
    axis_id: str
    rule_type: RuleType
    condition_met: str
    effect_applied: str
    explanation: str


class ExplanationTrace(BaseModel):
    user_id: str
    context: Optional[str]
    rules_fired: list[RuleFiring]
    axis_modifications: dict[str, str]      # axis_id → what changed and why
    uncertainty_notes: dict[str, str]       # axis_id → why confidence is low/high
    adaptive_activations: list[str]         # question IDs activated adaptively
