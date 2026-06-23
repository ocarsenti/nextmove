"""
NEXTMOVE V4 — Decision Visualization Engine.

NOT a recommender system. NOT a scoring engine.
Exposes structure, tension, and consequence paths. Never decides.

Three model families:
  - Input: what the frontend sends (job descriptions + decision profile).
  - Extraction: what the LLM produces (job analysis, qualitative content).
  - Output: the Decision Card schema (8 sections, UI-ready JSON).
"""
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input — what the frontend sends
# ---------------------------------------------------------------------------

class CurrentJob(BaseModel):
    title: str
    description: str


class Opportunity(BaseModel):
    title: str
    description: str


class UserArchetype(BaseModel):
    Builder: int = Field(ge=0, le=100)
    Expert: int = Field(ge=0, le=100)
    Operator: int = Field(ge=0, le=100)
    Leader: int = Field(ge=0, le=100)
    Explorer: int = Field(ge=0, le=100)


class DecisionProfile(BaseModel):
    autonomy: int = Field(ge=1, le=5)
    uncertainty: int = Field(ge=1, le=5)
    operational_load: int = Field(ge=1, le=5)
    learning: int = Field(ge=1, le=5)
    optionality: int = Field(ge=1, le=5)
    relational: int = Field(ge=1, le=5)
    archetype: UserArchetype


class AnalyzeRequest(BaseModel):
    current_job: CurrentJob
    opportunity: Opportunity
    profile: DecisionProfile
    clarifications: str | None = None


# ---------------------------------------------------------------------------
# Extraction — what the LLM produces
# ---------------------------------------------------------------------------

Level = Literal["LOW", "MEDIUM", "HIGH"]
Intensity = Literal["low", "medium", "high"]
Beneficiary = Literal["current", "opportunity", "both", "neither"]
SwitchingCost = Literal["LOW", "MEDIUM", "HIGH"]


class AxisDetail(BaseModel):
    level: str
    explanation: str


class JobAxesExtraction(BaseModel):
    autonomy: AxisDetail
    uncertainty: AxisDetail
    operational_load: AxisDetail
    learning: AxisDetail
    optionality: AxisDetail
    relational: AxisDetail


class ArchetypeDistribution(BaseModel):
    Builder: int = Field(ge=0, le=100)
    Expert: int = Field(ge=0, le=100)
    Operator: int = Field(ge=0, le=100)
    Leader: int = Field(ge=0, le=100)
    Explorer: int = Field(ge=0, le=100)


class JobPanelExtraction(BaseModel):
    title: str
    archetype_signature: str
    short_description: str


class TensionExtraction(BaseModel):
    axis: str
    current: str
    opportunity: str
    interpretation: str


class OptionalityExtraction(BaseModel):
    doors_opened_current: list[str]
    doors_opened_opportunity: list[str]
    doors_closed: list[str]
    reversibility_risk: str
    switching_cost: str


class UncertaintyItemExtraction(BaseModel):
    unknown: str
    impact: str


class DecisionFramingExtraction(BaseModel):
    stay_meaning: str
    switch_meaning: str
    wait_meaning: str


class ScenarioExtraction(BaseModel):
    name: str
    description: str
    likelihood: str  # LOW/MEDIUM/HIGH


class DecisionQuestionExtraction(BaseModel):
    question: str
    impact: str  # LOW/MEDIUM/HIGH


class ExtractionOutput(BaseModel):
    current_panel: JobPanelExtraction
    opportunity_panel: JobPanelExtraction

    current_axes: JobAxesExtraction
    opportunity_axes: JobAxesExtraction

    current_archetype: ArchetypeDistribution
    opportunity_archetype: ArchetypeDistribution

    tensions: list[TensionExtraction]

    optionality: OptionalityExtraction
    uncertainty_items: list[UncertaintyItemExtraction]
    decision_framing: DecisionFramingExtraction

    current_pros: list[str]
    current_cons: list[str]
    opportunity_pros: list[str]
    opportunity_cons: list[str]
    current_scenarios: list[ScenarioExtraction]
    opportunity_scenarios: list[ScenarioExtraction]
    key_insights: list[str]
    decision_sensitive_questions: list[DecisionQuestionExtraction]


# ---------------------------------------------------------------------------
# Output — the Decision Card (8 sections)
# ---------------------------------------------------------------------------

class ProfileAxisAlignment(BaseModel):
    axis: str
    user_preference: Level
    current_job_level: Level
    opportunity_job_level: Level
    best_fit: Literal["current", "opportunity", "both", "neither"]
    interpretation: str


class Header(BaseModel):
    title: str = "DECISION MAP — TRADE-OFF VIEW"
    subtitle: str = "Current Job vs Opportunity"
    note: str = "No recommendation. Only structured trade-offs."
    reproducibility_hash: str = ""


class JobPanel(BaseModel):
    title: str
    archetype_signature: str
    short_description: str


class JobPanels(BaseModel):
    current: JobPanel
    opportunity: JobPanel


class AxisAlignment(BaseModel):
    level: Level
    explanation: str


class AlignmentGrid(BaseModel):
    autonomy: AxisAlignment
    uncertainty: AxisAlignment
    operational_load: AxisAlignment
    learning: AxisAlignment
    optionality: AxisAlignment
    relational: AxisAlignment


class ProfileAlignmentGrid(BaseModel):
    current: AlignmentGrid
    opportunity: AlignmentGrid


class DominanceSummary(BaseModel):
    primary: str
    secondary: str


class ArchetypeLayer(BaseModel):
    distribution: ArchetypeDistribution
    dominance_summary: DominanceSummary


class ArchetypeLayers(BaseModel):
    current: ArchetypeLayer
    opportunity: ArchetypeLayer


class Tension(BaseModel):
    axis: str
    current: str
    opportunity: str
    intensity: Intensity
    interpretation: str
    who_benefits: Beneficiary


class OptionalityMap(BaseModel):
    doors_opened_current: list[str]
    doors_opened_opportunity: list[str]
    doors_closed: list[str]
    reversibility_risk: str
    switching_cost: SwitchingCost


class UncertaintyItem(BaseModel):
    unknown: str
    impact: Level


class PathOption(BaseModel):
    label: str
    meaning: str


class DecisionFraming(BaseModel):
    A: PathOption
    B: PathOption
    C: PathOption


class UserArchetypeLayer(BaseModel):
    distribution: UserArchetype
    dominance_summary: DominanceSummary


class Scenario(BaseModel):
    name: str
    description: str
    likelihood: Level


class DecisionQuestion(BaseModel):
    question: str
    impact: Level


class DecisionCard(BaseModel):
    header: Header
    job_panels: JobPanels
    user_archetype: UserArchetypeLayer
    profile_alignment_grid: ProfileAlignmentGrid
    archetype_layer: ArchetypeLayers
    tension_map: list[Tension]
    profile_tensions: list[ProfileAxisAlignment]
    optionality_map: OptionalityMap
    uncertainty_block: list[UncertaintyItem]
    decision_framing: DecisionFraming
    current_pros: list[str]
    current_cons: list[str]
    opportunity_pros: list[str]
    opportunity_cons: list[str]
    current_scenarios: list[Scenario]
    opportunity_scenarios: list[Scenario]
    key_insights: list[str]
    decision_sensitive_questions: list[DecisionQuestion]


# ---------------------------------------------------------------------------
# Share
# ---------------------------------------------------------------------------

class ShareCreateRequest(BaseModel):
    current_job_title: str
    opportunity_title: str
    result: DecisionCard


class ShareCreateResponse(BaseModel):
    share_id: str
    url: str
