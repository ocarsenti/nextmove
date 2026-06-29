"""NextMove V5 — FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ontology.registry import AxisRegistry, QuestionBank
from engine.context import ContextEngine, KNOWN_CONTEXTS
from engine.rules import RuleEngine
from engine.scorer import ProfileScorer
from engine.explainer import Explainer
from questionnaire.adaptive import AdaptiveQuestionnaire
from api.models import (
    ScoreRequest, ScoreResponse,
    QuestionnaireResponse, AxisResponse, RulesReport,
    NextQuestionsRequest, NextQuestionsResponse,
)

app = FastAPI(
    title="NextMove V5",
    description="Living ontology engine for professional preference modeling",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Singletons ---
_registry = AxisRegistry()
_bank = QuestionBank()
_context_engine = ContextEngine()
_rule_engine = RuleEngine()
_scorer = ProfileScorer(_registry, _bank)
_explainer = Explainer()
_questionnaire = AdaptiveQuestionnaire(_registry, _bank)


# ===================================================================
# AXES
# ===================================================================

@app.get("/axes", response_model=AxisResponse)
def get_axes():
    """Return the full active axis registry."""
    axes = _registry.all_active()
    layers: dict[str, list[str]] = {}
    for ax in axes:
        layer_key = str(ax.layer)
        layers.setdefault(layer_key, []).append(ax.id)

    return AxisResponse(
        axes=[ax.model_dump() for ax in axes],
        version=_registry.version,
        count=len(axes),
        layers=layers,
    )


@app.get("/axes/{axis_id}")
def get_axis(axis_id: str):
    ax = _registry.get(axis_id)
    if ax is None:
        raise HTTPException(status_code=404, detail=f"Axis '{axis_id}' not found")
    return ax.model_dump()


# ===================================================================
# QUESTIONNAIRE
# ===================================================================

@app.get("/questionnaire", response_model=QuestionnaireResponse)
def get_questionnaire():
    """Return base questionnaire (no adaptive questions — session-independent)."""
    base_ids = _questionnaire.initial_sequence()
    questions = [_bank.get(qid) for qid in base_ids if _bank.get(qid)]

    return QuestionnaireResponse(
        questions=[q.model_dump() for q in questions],
        total=len(questions),
        base_count=len(questions),
        adaptive_count=0,
    )


@app.post("/questionnaire/next", response_model=NextQuestionsResponse)
def get_next_questions(req: NextQuestionsRequest):
    """
    Given current answers, return which questions to ask next.
    Used for progressive/adaptive questionnaire sessions.
    """
    context_id = _context_engine.detect_context(req.context_hint or "") or req.context_hint
    if context_id and context_id not in KNOWN_CONTEXTS:
        context_id = None

    profile = _scorer.score(req.user_id, req.answers, context_id)
    already_seen = set(req.already_seen) | set(req.answers.keys())

    pending = _questionnaire.pending_questions(profile, already_seen)
    adaptive = _questionnaire.next_adaptive_questions(profile, already_seen)

    next_q_ids = pending[:5]  # return at most 5 at a time
    next_questions = [_bank.get(qid).model_dump() for qid in next_q_ids if _bank.get(qid)]

    base_remaining = len([q for q in _questionnaire.initial_sequence() if q not in already_seen])

    return NextQuestionsResponse(
        next_questions=next_questions,
        adaptive_activations=adaptive,
        remaining_base=base_remaining,
    )


# ===================================================================
# SCORING
# ===================================================================

@app.post("/score", response_model=ScoreResponse)
def score_profile(req: ScoreRequest):
    """Compute a profile from questionnaire answers."""
    # Detect or resolve context
    context_id: str | None = None
    if req.context_hint:
        context_id = _context_engine.detect_context(req.context_hint)
        if context_id is None and req.context_hint in KNOWN_CONTEXTS:
            context_id = req.context_hint

    # Evaluate rules
    rules_fired = _rule_engine.evaluate(_registry, context_id)

    # Compute profile
    profile = _scorer.score(req.user_id, req.answers, context_id)

    # Explanation
    trace = _explainer.explain(profile, rules_fired, _registry)

    # Pending questions
    already_seen = set(req.answers.keys())
    pending = _questionnaire.pending_questions(profile, already_seen)
    low_conf = _scorer.low_confidence_axes(profile)
    weighted = _scorer.weighted_profile(profile)

    return ScoreResponse(
        user_id=profile.user_id,
        context_detected=context_id,
        axis_scores={k: v.model_dump() for k, v in profile.axis_scores.items()},
        weighted_scores=weighted,
        is_complete=profile.is_complete,
        low_confidence_axes=low_conf,
        pending_questions=pending,
        explanation={
            "rules_fired": [f.model_dump() for f in rules_fired],
            "axis_modifications": trace.axis_modifications,
            "uncertainty_notes": trace.uncertainty_notes,
            "adaptive_activations": trace.adaptive_activations,
            "summary": _explainer.summary(trace),
        },
    )


# ===================================================================
# RULES
# ===================================================================

@app.get("/rules")
def get_rules(context: str | None = None):
    """Inspect all rules — which are active, which are pending V6+."""
    fired = _rule_engine.evaluate(_registry, context)
    pending_v6 = _rule_engine.pending_v6_rules(_registry)

    return RulesReport(
        active_rules=[f.model_dump() for f in fired],
        pending_v6_rules=pending_v6,
        context_id=context,
    )


# ===================================================================
# CONTEXTS
# ===================================================================

@app.get("/contexts")
def get_contexts():
    """Return known contexts."""
    return {
        ctx_id: {
            "label": ctx["label"],
            "description": ctx["description"],
        }
        for ctx_id, ctx in KNOWN_CONTEXTS.items()
    }


# ===================================================================
# HEALTH
# ===================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "axes": len(_registry.all_active()),
        "questions": len(_bank.all_ids()),
        "registry_version": _registry.version,
    }
