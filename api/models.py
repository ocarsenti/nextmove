"""API request/response models."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class ScoreRequest(BaseModel):
    user_id: str
    answers: dict[str, str]          # {question_id: "A"|"B"|"C"}
    context_hint: Optional[str] = None   # free text for context detection, or explicit context_id


class ScoreResponse(BaseModel):
    user_id: str
    context_detected: Optional[str]
    axis_scores: dict[str, dict]     # {axis_id: AxisScore dict}
    weighted_scores: dict[str, float]
    is_complete: bool
    low_confidence_axes: list[str]
    pending_questions: list[str]
    explanation: dict
    archetype: dict                  # ArchetypeDistribution.model_dump()


class QuestionnaireResponse(BaseModel):
    questions: list[dict]
    total: int
    base_count: int
    adaptive_count: int


class AxisResponse(BaseModel):
    axes: list[dict]
    version: int
    count: int
    layers: dict[str, list[str]]


class RulesReport(BaseModel):
    active_rules: list[dict]
    pending_v6_rules: list[dict]
    context_id: Optional[str]


class NextQuestionsRequest(BaseModel):
    user_id: str
    answers: dict[str, str]
    already_seen: list[str] = []
    context_hint: Optional[str] = None


class NextQuestionsResponse(BaseModel):
    next_questions: list[dict]
    adaptive_activations: list[str]
    remaining_base: int


# ===================================================================
# JOB CARDS / MATCHING
# ===================================================================

class JobCardRequest(BaseModel):
    job_id: str
    title: str
    context_id: Optional[str] = None
    description: str = ""
    axis_requirements: dict[str, dict]   # {axis_id: {"level": float, "importance": float, "note": str}}
    detected_signal_ids: list[str] = []  # from a prior /jobs/extract call, if any — passed through as-is


class JobCardResponse(BaseModel):
    job_id: str
    title: str
    context_id: Optional[str]
    description: str
    axis_requirements: dict[str, dict]
    version: int


class MatchRequest(BaseModel):
    user_id: str
    answers: dict[str, str]
    job_id: str
    context_hint: Optional[str] = None


class MatchResponse(BaseModel):
    user_id: str
    job_id: str
    fit_score: float
    tensions: list[dict]
    top_tensions: list[str]
    low_confidence_axes: list[str]
    skipped_axes: list[str]
    archetype: dict
    job_constraints: dict          # JobConstraintProfile.model_dump()


class JobConstraintsResponse(BaseModel):
    job_id: str
    job_title: str
    detected_constraints: list[dict]
    job_narrative: str


class JobExtractRequest(BaseModel):
    description: str


class JobExtractResponse(BaseModel):
    axis_requirements: dict[str, dict]
    signals: list[dict]        # [{signal_id, label, source_phrase}] — the traceable intermediate layer


# ===================================================================
# TEST-RETEST STUDY
# ===================================================================

class RetestSaveRequest(BaseModel):
    user_id: str
    answers: dict[str, str]
    context_hint: Optional[str] = None
    participant_code: Optional[str] = None   # None on first passage; required on retest


class RetestSaveResponse(BaseModel):
    participant_code: str
    passage_number: int
    is_new_participant: bool


class RetestWithdrawRequest(BaseModel):
    participant_code: str


class RetestWithdrawResponse(BaseModel):
    deleted_passages: int


# ===================================================================
# JOB SIGNAL ANNOTATION — human-vs-LLM agreement (criterion validity)
# ===================================================================

class JobSignalListResponse(BaseModel):
    signals: list[dict]   # [{id, label}]


class JobAnnotationRequest(BaseModel):
    description: str
    human_signal_ids: list[str]
    annotator_note: str = ""


class JobAnnotationResponse(BaseModel):
    annotation_id: int
    n_total_annotations: int
