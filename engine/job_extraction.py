"""Job description → axis requirements extraction, via Claude.

Free-text job description in, {axis_id: {level, importance}} out — the same
shape /jobs already accepts, so the caller pre-fills the existing manual
sliders instead of requiring the user to guess 16 axis values by hand.
Ported from NEXTMOVE-V4's llm_client.py extraction pattern (plain-text JSON
completion, no tool-use), adapted to v5's single-job / 16-axis model.
"""
from __future__ import annotations

import os

from anthropic import Anthropic
from pydantic import BaseModel, Field

from ontology.registry import AxisRegistry

MODEL_ID = "claude-sonnet-4-6"


class AxisExtraction(BaseModel):
    level: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)


class JobExtractionOutput(BaseModel):
    axes: dict[str, AxisExtraction]


def _build_system_prompt(registry: AxisRegistry) -> str:
    axis_lines = "\n".join(
        f"- {a.id}: {a.label} — {a.description}"
        for a in registry.non_meta_axes()
    )
    return f"""You are the job-description extraction module of NEXTMOVE, a career-fit \
engine. You are NOT a recommendation engine — you only extract structural \
requirements from a job description, mapped onto a fixed 16-axis behavioral \
ontology. You never judge, rank, or recommend the role.

For EACH of the 16 axes below, produce:
- level (0.0-1.0): how much this role structurally demands/expresses this trait, \
based only on what is stated or strongly implied in the description.
- importance (0.0-1.0): how much this axis matters for succeeding in this specific \
role (0 = irrelevant to this role, 1 = critical).

If the description gives no signal at all for an axis, set its importance to 0.0 \
rather than guessing a level.

AXES:
{axis_lines}

Return ONLY a JSON object of this exact shape, no markdown fences, no commentary:
{{"axes": {{"<axis_id>": {{"level": <float>, "importance": <float>}}, ...}}}}
"""


def extract_job_axes(description: str, registry: AxisRegistry) -> dict[str, dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=2000,
        temperature=0,
        system=_build_system_prompt(registry),
        messages=[{"role": "user", "content": description}],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    parsed = JobExtractionOutput.model_validate_json(raw_text)
    return {
        axis_id: {"level": v.level, "importance": v.importance}
        for axis_id, v in parsed.axes.items()
        if v.importance > 0
    }
