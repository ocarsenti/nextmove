"""Job description -> axis requirements, via an explicit, inspectable two-step
pipeline instead of one opaque LLM call.

    description (free text)
              |
    extract_signals()     <- LLM, but constrained to a FIXED vocabulary
              |              (ontology/signals_seed.json) and required to
              |              quote the exact phrase that justifies each pick.
              |              This is the only LLM-touched step.
              |
    signals_to_axes()     <- pure Python, deterministic, no LLM. Same signal
              |              list always produces the same axis_requirements.
              |
    axis_requirements

If a resulting axis level looks wrong, you can see exactly why: which
signal fired, and which sentence in the description triggered it — instead
of having to trust an LLM's implicit reasoning on faith.

extract_job_axes() ties both steps together and keeps the original
signature/behavior for callers that only want the end result (e.g. the
/jobs/extract API endpoint, which now also returns the intermediate
signals for display).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
from anthropic import Anthropic
from pydantic import BaseModel, Field

from ontology.models import DetectedSignal, JobSignal
from ontology.registry import AxisRegistry

MODEL_ID = "claude-sonnet-4-6"
_SIGNALS_SEED_PATH = Path(__file__).parent.parent / "ontology" / "signals_seed.json"


def load_signal_library(path: Path = _SIGNALS_SEED_PATH) -> list[JobSignal]:
    data = json.loads(path.read_text())
    return [JobSignal(**s) for s in data["signals"]]


# ---------------------------------------------------------------------------
# Step 1 — LLM, constrained to the fixed signal vocabulary
# ---------------------------------------------------------------------------

class _SignalPick(BaseModel):
    signal_id: str
    source_phrase: str


class _SignalExtractionOutput(BaseModel):
    signals: list[_SignalPick] = Field(default_factory=list)


def _build_signal_prompt(library: list[JobSignal]) -> str:
    lines = "\n".join(f"- {s.id}: {s.label}" for s in library)
    return f"""You are the signal-extraction module of NEXTMOVE, a career-fit engine. \
You are NOT a recommendation engine and you never invent signals outside the fixed \
list below — if nothing in the text matches a signal, simply don't include it.

Read the job description and pick every signal from this FIXED list that is \
explicitly stated or strongly implied in the text. For each one you pick, quote \
the EXACT phrase from the description (verbatim, not paraphrased) that justifies it.

SIGNALS:
{lines}

Return ONLY a JSON object of this exact shape, no markdown fences, no commentary:
{{"signals": [{{"signal_id": "<id from the list above>", "source_phrase": "<verbatim excerpt>"}}, ...]}}

If no signal applies, return {{"signals": []}}.
"""


def extract_signals(description: str, library: list[JobSignal]) -> list[DetectedSignal]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")

    valid_ids = {s.id for s in library}
    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=1500,
            temperature=0,
            system=_build_signal_prompt(library),
            messages=[{"role": "user", "content": description}],
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Claude signal extraction failed: {e}") from e

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    parsed = _SignalExtractionOutput.model_validate_json(raw_text)
    # Defensive: drop anything outside the fixed vocabulary rather than trust
    # the model never hallucinates an id — this is exactly the guardrail the
    # closed-vocabulary design is meant to provide.
    return [
        DetectedSignal(signal_id=p.signal_id, source_phrase=p.source_phrase)
        for p in parsed.signals
        if p.signal_id in valid_ids
    ]


# ---------------------------------------------------------------------------
# Step 2 — deterministic, no LLM
# ---------------------------------------------------------------------------

def signals_to_axes(signals: list[DetectedSignal], library: list[JobSignal]) -> dict[str, dict]:
    """Aggregation rule: when several signals push the same axis, take the MAX
    level and MAX importance across them — a job is treated as at least as
    demanding as its single most demanding piece of evidence, never diluted
    by averaging with weaker signals on the same axis. Deterministic: the
    same signal list always produces the same axis_requirements."""
    by_id = {s.id: s for s in library}
    result: dict[str, dict] = {}

    for detected in signals:
        signal = by_id.get(detected.signal_id)
        if signal is None:
            continue
        for axis_id, effect in signal.axis_effects.items():
            current = result.get(axis_id)
            if current is None or effect.level > current["level"]:
                level = effect.level
            else:
                level = current["level"]
            if current is None or effect.importance > current["importance"]:
                importance = effect.importance
            else:
                importance = current["importance"]
            result[axis_id] = {"level": level, "importance": importance}

    return result


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def extract_job_axes(description: str, registry: AxisRegistry) -> dict[str, dict]:
    """Backward-compatible: text in, axis_requirements out. Use
    extract_job_signals_and_axes() instead when the caller wants to display
    the intermediate signals too."""
    library = load_signal_library()
    signals = extract_signals(description, library)
    return signals_to_axes(signals, library)


def extract_job_signals_and_axes(description: str) -> tuple[list[DetectedSignal], dict[str, dict]]:
    library = load_signal_library()
    signals = extract_signals(description, library)
    axes = signals_to_axes(signals, library)
    return signals, axes
