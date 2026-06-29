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
