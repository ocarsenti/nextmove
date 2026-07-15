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
