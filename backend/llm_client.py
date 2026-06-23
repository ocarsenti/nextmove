"""
Ensemble LLM extraction for NEXTMOVE V4.

Runs N extractions (default 3) with temperature=0, then merges
axis levels via majority vote and archetype distributions via median.
Qualitative content comes from the first extraction.
"""
import hashlib
import json as _json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from anthropic import Anthropic

from models import AnalyzeRequest, ExtractionOutput
from prompts import SYSTEM_PROMPT, build_user_message

MODEL_ID = "claude-sonnet-4-6"
ENSEMBLE_RUNS = int(os.environ.get("NEXTMOVE_ENSEMBLE_RUNS", "3"))
AXES = ["autonomy", "uncertainty", "operational_load", "learning", "optionality", "relational"]
ARCHETYPES = ["Builder", "Expert", "Operator", "Leader", "Explorer"]

_client = Anthropic()


class _AxisDetail(BaseModel):
    level: str
    explanation: str


class _JobAxes(BaseModel):
    autonomy: _AxisDetail
    uncertainty: _AxisDetail
    operational_load: _AxisDetail
    learning: _AxisDetail
    optionality: _AxisDetail
    relational: _AxisDetail


class _ArchDist(BaseModel):
    Builder: int = Field(ge=0, le=100)
    Expert: int = Field(ge=0, le=100)
    Operator: int = Field(ge=0, le=100)
    Leader: int = Field(ge=0, le=100)
    Explorer: int = Field(ge=0, le=100)


class _JobPanel(BaseModel):
    title: str
    archetype_signature: str
    short_description: str


class _Tension(BaseModel):
    axis: str
    current: str
    opportunity: str
    interpretation: str


class _Optionality(BaseModel):
    doors_opened_current: list[str]
    doors_opened_opportunity: list[str]
    doors_closed: list[str]
    reversibility_risk: str
    switching_cost: str


class _UncertaintyItem(BaseModel):
    unknown: str
    impact: str


class _DecisionFraming(BaseModel):
    stay_meaning: str
    switch_meaning: str
    wait_meaning: str


class _Scenario(BaseModel):
    name: str
    description: str
    likelihood: str


class _Question(BaseModel):
    question: str
    impact: str


class _Raw(BaseModel):
    current_panel: _JobPanel
    opportunity_panel: _JobPanel
    current_axes: _JobAxes
    opportunity_axes: _JobAxes
    current_archetype: _ArchDist
    opportunity_archetype: _ArchDist
    tensions: list[_Tension]
    optionality: _Optionality
    uncertainty_items: list[_UncertaintyItem]
    decision_framing: _DecisionFraming
    current_pros: list[str]
    current_cons: list[str]
    opportunity_pros: list[str]
    opportunity_cons: list[str]
    current_scenarios: list[_Scenario]
    opportunity_scenarios: list[_Scenario]
    key_insights: list[str]
    decision_sensitive_questions: list[_Question]


def _single_extraction(user_msg: str) -> ExtractionOutput:
    response = _client.messages.create(
        model=MODEL_ID,
        max_tokens=8000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    parsed = _Raw.model_validate_json(raw_text)
    return ExtractionOutput(**parsed.model_dump())


def _majority_vote(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0]


def _median_int(values: list[int]) -> int:
    return sorted(values)[len(values) // 2]


def _merge_extractions(extractions: list[ExtractionOutput]) -> ExtractionOutput:
    base = extractions[0].model_copy(deep=True)

    for axis in AXES:
        levels_c = [getattr(e.current_axes, axis).level for e in extractions]
        levels_o = [getattr(e.opportunity_axes, axis).level for e in extractions]
        getattr(base.current_axes, axis).level = _majority_vote(levels_c)
        getattr(base.opportunity_axes, axis).level = _majority_vote(levels_o)

    for arch in ARCHETYPES:
        vals_c = [getattr(e.current_archetype, arch) for e in extractions]
        vals_o = [getattr(e.opportunity_archetype, arch) for e in extractions]
        setattr(base.current_archetype, arch, _median_int(vals_c))
        setattr(base.opportunity_archetype, arch, _median_int(vals_o))

    costs = [e.optionality.switching_cost for e in extractions]
    base.optionality.switching_cost = _majority_vote(costs)

    return base


def compute_reproducibility_hash(request: AnalyzeRequest) -> str:
    canonical = _json.dumps(request.model_dump(), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def extract_job_analysis(request: AnalyzeRequest) -> tuple[ExtractionOutput, str]:
    payload = {
        "current_job": request.current_job.model_dump(),
        "opportunity": request.opportunity.model_dump(),
        "profile": request.profile.model_dump(),
    }
    if request.clarifications:
        payload["clarifications"] = request.clarifications

    json_schema_instruction = (
        "\n\nIMPORTANT: Return your analysis as a single valid JSON object "
        "matching this exact schema:\n"
        + _json.dumps(_Raw.model_json_schema(), indent=2)
        + "\n\nReturn ONLY the JSON object, no markdown fences, no commentary."
    )

    user_msg = build_user_message(payload) + json_schema_instruction

    repro_hash = compute_reproducibility_hash(request)
    n = max(1, ENSEMBLE_RUNS)

    if n == 1:
        return _single_extraction(user_msg), repro_hash

    extractions: list[ExtractionOutput] = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_single_extraction, user_msg) for _ in range(n)]
        for fut in as_completed(futures):
            extractions.append(fut.result())

    return _merge_extractions(extractions), repro_hash
