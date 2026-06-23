"""
Deterministic comparison engine — no scores, no ranking.

Transforms raw LLM extraction into the final Decision Card by computing:
  - Tension intensity (from axis level differences)
  - Who benefits (from axis comparison)
  - Dominance summary (from archetype distribution)
  - Validated levels (normalizing LLM output to strict enums)
"""
from models import (
    AlignmentGrid,
    ArchetypeDistribution,
    ArchetypeLayer,
    ArchetypeLayers,
    AxisAlignment,
    DecisionCard,
    DecisionFraming,
    DecisionProfile,
    DecisionQuestion,
    DominanceSummary,
    ExtractionOutput,
    Header,
    JobPanel,
    JobPanels,
    OptionalityMap,
    PathOption,
    ProfileAlignmentGrid,
    ProfileAxisAlignment,
    Scenario,
    Tension,
    UncertaintyItem,
    UserArchetype,
    UserArchetypeLayer,
)

LEVEL_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
VALID_LEVELS = {"LOW", "MEDIUM", "HIGH"}
VALID_INTENSITIES = {"low", "medium", "high"}
VALID_BENEFICIARIES = {"current", "opportunity", "both", "neither"}
VALID_SWITCHING_COSTS = {"LOW", "MEDIUM", "HIGH"}


def _normalize_level(raw: str) -> str:
    upper = raw.strip().upper()
    if upper in VALID_LEVELS:
        return upper
    return "MEDIUM"


def _normalize_intensity(raw: str) -> str:
    lower = raw.strip().lower()
    if lower in VALID_INTENSITIES:
        return lower
    return "medium"


def _normalize_switching_cost(raw: str) -> str:
    upper = raw.strip().upper()
    if upper in VALID_SWITCHING_COSTS:
        return upper
    return "MEDIUM"


def _compute_tension_intensity(current_level: str, opportunity_level: str) -> str:
    diff = abs(LEVEL_ORDER.get(current_level, 1) - LEVEL_ORDER.get(opportunity_level, 1))
    if diff == 0:
        return "low"
    if diff == 1:
        return "medium"
    return "high"


def _profile_to_level(value: int) -> str:
    if value <= 2:
        return "LOW"
    if value >= 4:
        return "HIGH"
    return "MEDIUM"


def _compute_who_benefits(axis: str, current_level: str, opportunity_level: str, profile_value: int | None = None) -> str:
    c = LEVEL_ORDER.get(current_level, 1)
    o = LEVEL_ORDER.get(opportunity_level, 1)
    if c == o:
        return "neither"

    if profile_value is not None:
        preferred = LEVEL_ORDER.get(_profile_to_level(profile_value), 1)
        dist_c = abs(c - preferred)
        dist_o = abs(o - preferred)
        if dist_c < dist_o:
            return "current"
        if dist_o < dist_c:
            return "opportunity"
        return "neither"

    higher_is_better = axis in ("autonomy", "optionality", "relational", "learning")
    lower_is_better = axis in ("operational_load", "uncertainty")

    if higher_is_better:
        return "opportunity" if o > c else "current"
    if lower_is_better:
        return "opportunity" if o < c else "current"
    return "both"


ALPHABETICAL_ORDER = {name: i for i, name in enumerate(sorted(["Builder", "Expert", "Operator", "Leader", "Explorer"]))}


def _compute_dominance_summary(dist) -> DominanceSummary:
    scores = {
        "Builder": dist.Builder,
        "Expert": dist.Expert,
        "Operator": dist.Operator,
        "Leader": dist.Leader,
        "Explorer": dist.Explorer,
    }
    # Deterministic tie-break: score desc, then alphabetical asc
    ranked = sorted(scores.items(), key=lambda x: (-x[1], ALPHABETICAL_ORDER[x[0]]))
    primary_name, primary_score = ranked[0]
    secondary_name = ranked[1][0]

    if primary_score >= 40:
        strength = "strong"
    elif primary_score >= 25:
        strength = "moderate"
    else:
        strength = "mixed"

    return DominanceSummary(
        primary=f"{primary_name} ({strength})",
        secondary=secondary_name,
    )


AXIS_TO_TENSION = {
    "autonomy": "autonomy vs structure",
    "uncertainty": "uncertainty exposure",
    "operational_load": "operational load pattern",
    "learning": "learning style shift",
    "optionality": "optionality expansion/contraction",
    "relational": "relational environment shift",
}

MANDATORY_TENSIONS = [
    "autonomy vs structure",
    "uncertainty exposure",
    "operational load pattern",
    "learning style shift",
    "archetype drift",
    "optionality expansion/contraction",
]

AXIS_LABELS = {
    "autonomy": "autonomy",
    "uncertainty": "uncertainty tolerance",
    "operational_load": "operational load tolerance",
    "learning": "learning velocity",
    "optionality": "optionality preference",
    "relational": "collaboration preference",
}


def _build_profile_tensions(profile: DecisionProfile, extraction: ExtractionOutput) -> list[ProfileAxisAlignment]:
    axes = ["autonomy", "uncertainty", "operational_load", "learning", "optionality", "relational"]
    tensions = []
    for axis in axes:
        pref_value = getattr(profile, axis)
        pref_level = _profile_to_level(pref_value)
        c_level = _normalize_level(getattr(extraction.current_axes, axis).level)
        o_level = _normalize_level(getattr(extraction.opportunity_axes, axis).level)

        pref_num = LEVEL_ORDER[pref_level]
        c_num = LEVEL_ORDER[c_level]
        o_num = LEVEL_ORDER[o_level]

        dist_c = abs(c_num - pref_num)
        dist_o = abs(o_num - pref_num)

        if dist_c == 0 and dist_o == 0:
            best_fit = "both"
        elif dist_c < dist_o:
            best_fit = "current"
        elif dist_o < dist_c:
            best_fit = "opportunity"
        else:
            best_fit = "neither"

        label = AXIS_LABELS.get(axis, axis)
        if best_fit == "both":
            interp = f"Both jobs match the user's {label} preference ({pref_level})."
        elif best_fit == "neither":
            interp = (f"User prefers {pref_level} {label}. "
                      f"Both jobs are equally distant (current: {c_level}, opportunity: {o_level}).")
        elif best_fit == "current":
            interp = (f"User prefers {pref_level} {label}. "
                      f"Current job ({c_level}) aligns better than opportunity ({o_level}).")
        else:
            interp = (f"User prefers {pref_level} {label}. "
                      f"Opportunity ({o_level}) aligns better than current job ({c_level}).")

        tensions.append(ProfileAxisAlignment(
            axis=axis,
            user_preference=pref_level,
            current_job_level=c_level,
            opportunity_job_level=o_level,
            best_fit=best_fit,
            interpretation=interp,
        ))
    return tensions


def build_decision_card(extraction: ExtractionOutput, profile: DecisionProfile, reproducibility_hash: str = "") -> DecisionCard:
    header = Header(reproducibility_hash=reproducibility_hash)

    user_archetype = UserArchetypeLayer(
        distribution=profile.archetype,
        dominance_summary=_compute_dominance_summary(profile.archetype),
    )

    job_panels = JobPanels(
        current=JobPanel(
            title=extraction.current_panel.title,
            archetype_signature=extraction.current_panel.archetype_signature,
            short_description=extraction.current_panel.short_description,
        ),
        opportunity=JobPanel(
            title=extraction.opportunity_panel.title,
            archetype_signature=extraction.opportunity_panel.archetype_signature,
            short_description=extraction.opportunity_panel.short_description,
        ),
    )

    axes = ["autonomy", "uncertainty", "operational_load", "learning", "optionality", "relational"]

    current_grid = {}
    opportunity_grid = {}
    for axis in axes:
        c_detail = getattr(extraction.current_axes, axis)
        o_detail = getattr(extraction.opportunity_axes, axis)
        current_grid[axis] = AxisAlignment(
            level=_normalize_level(c_detail.level),
            explanation=c_detail.explanation,
        )
        opportunity_grid[axis] = AxisAlignment(
            level=_normalize_level(o_detail.level),
            explanation=o_detail.explanation,
        )

    profile_alignment_grid = ProfileAlignmentGrid(
        current=AlignmentGrid(**current_grid),
        opportunity=AlignmentGrid(**opportunity_grid),
    )

    archetype_layer = ArchetypeLayers(
        current=ArchetypeLayer(
            distribution=extraction.current_archetype,
            dominance_summary=_compute_dominance_summary(extraction.current_archetype),
        ),
        opportunity=ArchetypeLayer(
            distribution=extraction.opportunity_archetype,
            dominance_summary=_compute_dominance_summary(extraction.opportunity_archetype),
        ),
    )

    tension_index = {}
    for t in extraction.tensions:
        tension_index[t.axis.lower().strip()] = t

    tension_map = []
    for mandatory in MANDATORY_TENSIONS:
        raw = tension_index.pop(mandatory, None)
        if mandatory == "archetype drift":
            c_dom = _compute_dominance_summary(extraction.current_archetype).primary.split(" (")[0]
            o_dom = _compute_dominance_summary(extraction.opportunity_archetype).primary.split(" (")[0]
            if raw:
                intensity = "high" if c_dom != o_dom else "low"
                who = "neither" if c_dom == o_dom else "both"
                tension_map.append(Tension(
                    axis=mandatory,
                    current=raw.current,
                    opportunity=raw.opportunity,
                    intensity=intensity,
                    interpretation=raw.interpretation,
                    who_benefits=who,
                ))
            else:
                tension_map.append(Tension(
                    axis=mandatory,
                    current=f"Dominant: {c_dom}",
                    opportunity=f"Dominant: {o_dom}",
                    intensity="high" if c_dom != o_dom else "low",
                    interpretation=f"Shift from {c_dom} to {o_dom}" if c_dom != o_dom else "No archetype drift",
                    who_benefits="both" if c_dom != o_dom else "neither",
                ))
        elif raw:
            matched_axis = None
            for axis, tension_name in AXIS_TO_TENSION.items():
                if tension_name == mandatory:
                    matched_axis = axis
                    break

            if matched_axis:
                c_level = _normalize_level(getattr(extraction.current_axes, matched_axis).level)
                o_level = _normalize_level(getattr(extraction.opportunity_axes, matched_axis).level)
                intensity = _compute_tension_intensity(c_level, o_level)
                prof_val = getattr(profile, matched_axis, None)
                who = _compute_who_benefits(matched_axis, c_level, o_level, prof_val)
            else:
                intensity = "medium"
                who = "both"

            tension_map.append(Tension(
                axis=mandatory,
                current=raw.current,
                opportunity=raw.opportunity,
                intensity=intensity,
                interpretation=raw.interpretation,
                who_benefits=who,
            ))
        else:
            matched_axis = None
            for axis, tension_name in AXIS_TO_TENSION.items():
                if tension_name == mandatory:
                    matched_axis = axis
                    break

            if matched_axis:
                c_detail = getattr(extraction.current_axes, matched_axis)
                o_detail = getattr(extraction.opportunity_axes, matched_axis)
                c_level = _normalize_level(c_detail.level)
                o_level = _normalize_level(o_detail.level)
                prof_val = getattr(profile, matched_axis, None)
                tension_map.append(Tension(
                    axis=mandatory,
                    current=f"{c_level}: {c_detail.explanation}",
                    opportunity=f"{o_level}: {o_detail.explanation}",
                    intensity=_compute_tension_intensity(c_level, o_level),
                    interpretation=f"Shift from {c_level} to {o_level}",
                    who_benefits=_compute_who_benefits(matched_axis, c_level, o_level, prof_val),
                ))

    for remaining_axis, raw in tension_index.items():
        tension_map.append(Tension(
            axis=raw.axis,
            current=raw.current,
            opportunity=raw.opportunity,
            intensity="medium",
            interpretation=raw.interpretation,
            who_benefits="both",
        ))

    optionality_map = OptionalityMap(
        doors_opened_current=extraction.optionality.doors_opened_current,
        doors_opened_opportunity=extraction.optionality.doors_opened_opportunity,
        doors_closed=extraction.optionality.doors_closed,
        reversibility_risk=extraction.optionality.reversibility_risk,
        switching_cost=_normalize_switching_cost(extraction.optionality.switching_cost),
    )

    uncertainty_block = [
        UncertaintyItem(
            unknown=item.unknown,
            impact=_normalize_level(item.impact),
        )
        for item in extraction.uncertainty_items
    ]

    decision_framing = DecisionFraming(
        A=PathOption(label="Stay", meaning=extraction.decision_framing.stay_meaning),
        B=PathOption(label="Switch", meaning=extraction.decision_framing.switch_meaning),
        C=PathOption(label="Wait", meaning=extraction.decision_framing.wait_meaning),
    )

    # --- New V3-ported fields ---
    current_pros = extraction.current_pros
    current_cons = extraction.current_cons
    opportunity_pros = extraction.opportunity_pros
    opportunity_cons = extraction.opportunity_cons

    current_scenarios = [
        Scenario(
            name=s.name,
            description=s.description,
            likelihood=_normalize_level(s.likelihood),
        )
        for s in extraction.current_scenarios
    ]
    opportunity_scenarios = [
        Scenario(
            name=s.name,
            description=s.description,
            likelihood=_normalize_level(s.likelihood),
        )
        for s in extraction.opportunity_scenarios
    ]

    key_insights = extraction.key_insights

    decision_sensitive_questions = [
        DecisionQuestion(
            question=q.question,
            impact=_normalize_level(q.impact),
        )
        for q in extraction.decision_sensitive_questions
    ]

    profile_tensions = _build_profile_tensions(profile, extraction)

    return DecisionCard(
        header=header,
        job_panels=job_panels,
        user_archetype=user_archetype,
        profile_alignment_grid=profile_alignment_grid,
        archetype_layer=archetype_layer,
        tension_map=tension_map,
        profile_tensions=profile_tensions,
        optionality_map=optionality_map,
        uncertainty_block=uncertainty_block,
        decision_framing=decision_framing,
        current_pros=current_pros,
        current_cons=current_cons,
        opportunity_pros=opportunity_pros,
        opportunity_cons=opportunity_cons,
        current_scenarios=current_scenarios,
        opportunity_scenarios=opportunity_scenarios,
        key_insights=key_insights,
        decision_sensitive_questions=decision_sensitive_questions,
    )
