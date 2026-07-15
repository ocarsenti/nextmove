"""NextMove V5/V6 — FastAPI application."""

from __future__ import annotations

from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ontology.registry import AxisRegistry, QuestionBank
from ontology.rule_store import RuleStore, OntologyRule
from ontology.models import JobCard, JobAxisRequirement
from engine.context import ContextEngine, KNOWN_CONTEXTS
from engine.rules import RuleEngine
from engine.scorer import ProfileScorer
from engine.explainer import Explainer
from engine.signals import SignalComputer
from engine.v6_rules import V6RuleEngine
from engine.transformation import TransformationEngine
from engine.matching import compute_tensions
from engine.audit import AuditLog
from questionnaire.adaptive import AdaptiveQuestionnaire
from api.models import (
    ScoreRequest, ScoreResponse,
    QuestionnaireResponse, AxisResponse, RulesReport,
    NextQuestionsRequest, NextQuestionsResponse,
    JobCardRequest, JobCardResponse, MatchRequest, MatchResponse,
)

app = FastAPI(
    title="NextMove V6",
    description="Living ontology engine — rule-driven, interpretable, evolvable",
    version="6.0.0",
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
_rule_store = RuleStore()
_context_engine = ContextEngine()
_rule_engine = RuleEngine()
_v6_rule_engine = V6RuleEngine()
_scorer = ProfileScorer(_registry, _bank)
_explainer = Explainer()
_questionnaire = AdaptiveQuestionnaire(_registry, _bank)
_signal_computer = SignalComputer()
_audit_log = AuditLog()
_transformation_engine = TransformationEngine(_registry, _audit_log)
_job_store: dict[str, JobCard] = {}  # in-memory — mirrors the rest of v5 (no persistence layer yet)


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
# JOB CARDS
# ===================================================================

@app.post("/jobs", response_model=JobCardResponse)
def create_job(req: JobCardRequest):
    """Create or update a job card — the job-side counterpart to a candidate Profile."""
    for axis_id in req.axis_requirements:
        if _registry.get(axis_id) is None:
            raise HTTPException(status_code=400, detail=f"Unknown axis_id: {axis_id}")

    existing = _job_store.get(req.job_id)
    version = existing.version + 1 if existing else 1

    axis_requirements = {
        axis_id: JobAxisRequirement(axis_id=axis_id, **fields)
        for axis_id, fields in req.axis_requirements.items()
    }
    job = JobCard(
        job_id=req.job_id,
        title=req.title,
        context_id=req.context_id,
        description=req.description,
        axis_requirements=axis_requirements,
        version=version,
    )
    _job_store[req.job_id] = job
    return JobCardResponse(
        job_id=job.job_id,
        title=job.title,
        context_id=job.context_id,
        description=job.description,
        axis_requirements={k: v.model_dump() for k, v in job.axis_requirements.items()},
        version=job.version,
    )


@app.get("/jobs", response_model=list[JobCardResponse])
def list_jobs():
    return [
        JobCardResponse(
            job_id=j.job_id, title=j.title, context_id=j.context_id,
            description=j.description,
            axis_requirements={k: v.model_dump() for k, v in j.axis_requirements.items()},
            version=j.version,
        )
        for j in _job_store.values()
    ]


@app.get("/jobs/{job_id}", response_model=JobCardResponse)
def get_job(job_id: str):
    job = _job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return JobCardResponse(
        job_id=job.job_id, title=job.title, context_id=job.context_id,
        description=job.description,
        axis_requirements={k: v.model_dump() for k, v in job.axis_requirements.items()},
        version=job.version,
    )


# ===================================================================
# MATCHING — candidate Profile vs. JobCard, axis by axis (V4 tension
# principle ported to the 18-axis living ontology — see engine/matching.py)
# ===================================================================

@app.post("/match", response_model=MatchResponse)
def match_profile_to_job(req: MatchRequest):
    """Score a candidate's answers, then compare the resulting Profile to a JobCard.

    Not a hidden verdict: every axis produces an explicit, inspectable tension.
    fit_score is a transparent by-product of those tensions, not a recommendation.
    """
    job = _job_store.get(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {req.job_id}")

    context_id: str | None = None
    if req.context_hint:
        context_id = _context_engine.detect_context(req.context_hint)
        if context_id is None and req.context_hint in KNOWN_CONTEXTS:
            context_id = req.context_hint
    elif job.context_id:
        context_id = job.context_id

    profile = _scorer.score(req.user_id, req.answers, context_id)
    result = compute_tensions(profile, job, _registry)

    return MatchResponse(
        user_id=result.user_id,
        job_id=result.job_id,
        fit_score=result.fit_score,
        tensions=[t.model_dump() for t in result.tensions],
        top_tensions=result.top_tensions,
        low_confidence_axes=result.low_confidence_axes,
        skipped_axes=result.skipped_axes,
    )


# ===================================================================
# RULES (V5 — axis-embedded)
# ===================================================================

@app.get("/rules")
def get_rules(context: Optional[str] = None):
    """Inspect V5 axis-embedded rules — which fire for a given context."""
    fired = _rule_engine.evaluate(_registry, context)
    pending_v6 = _rule_engine.pending_v6_rules(_registry)

    return RulesReport(
        active_rules=[f.model_dump() for f in fired],
        pending_v6_rules=pending_v6,
        context_id=context,
    )


# ===================================================================
# RULES V6 — standalone rule store
# ===================================================================

class SignalInjectRequest(BaseModel):
    signals: dict
    context: Optional[str] = None


@app.get("/v6/rules")
def get_v6_rules(context: Optional[str] = None):
    """Return all V6 rules from the standalone rule store."""
    rules = _rule_store.all_active()
    immediate = _v6_rule_engine.evaluate_immediate(_rule_store, context)
    return {
        "total_rules": len(rules),
        "immediate_firings": [
            {"rule_id": f.rule_id, "type": f.rule_type, "priority": f.priority,
             "condition": f.condition_met, "action": f.action_description}
            for f in immediate
        ],
        "all_rules": [
            {"id": r.id, "type": r.type, "priority": r.priority,
             "requires_data": r.requires_data, "active": r.active,
             "explanation": r.explanation}
            for r in rules
        ],
        "context": context,
    }


@app.post("/v6/rules/evaluate")
def evaluate_v6_rules_with_signals(req: SignalInjectRequest):
    """
    Evaluate V6 rules with injected signals.
    Signals format: {"axis_id": {"variance": 0.8, "internal_consistency": 0.4, ...}}
    Correlation signals: {"axis_a__axis_b": {"correlation": 0.9}}
    """
    from engine.signals import SignalSet
    # Convert string keys like "rigor__cognitive_structuring" to tuple keys
    raw = req.signals
    signals = SignalSet()
    for k, v in raw.items():
        if "__" in k:
            parts = k.split("__", 1)
            signals[(parts[0], parts[1])] = v
            signals[(parts[1], parts[0])] = v
        else:
            signals[k] = v

    all_firings = _v6_rule_engine.evaluate(_rule_store, req.context, signals)
    return {
        "firings": [
            {
                "rule_id": f.rule_id,
                "type": f.rule_type,
                "priority": f.priority,
                "condition_met": f.condition_met,
                "action": f.action_description,
                "requires_execution": f.requires_execution,
                "explanation": f.explanation,
                "signals_used": f.signals_used,
            }
            for f in all_firings
        ],
        "structural_changes_pending": [
            f.rule_id for f in all_firings if f.requires_execution
        ],
    }


@app.post("/v6/transform/split/{rule_id}")
def execute_split(rule_id: str, signals: Optional[dict] = None):
    """Execute a split rule on the axis registry."""
    rule = _rule_store.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")
    if rule.type != "split":
        raise HTTPException(status_code=400, detail=f"Rule '{rule_id}' is not a split rule")

    from engine.signals import SignalSet
    signal_set = SignalSet(signals or {})
    success, event, reason = _transformation_engine.execute_split(rule, signal_set)
    if not success:
        raise HTTPException(status_code=409, detail=reason)

    return {
        "success": True,
        "reason": reason,
        "event_id": event.event_id if event else None,
        "new_axes": [spec.id for spec in rule.new_axes],
    }


@app.post("/v6/transform/merge/{rule_id}")
def execute_merge(rule_id: str, signals: Optional[dict] = None):
    """Execute a merge rule on the axis registry."""
    rule = _rule_store.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")
    if rule.type != "merge":
        raise HTTPException(status_code=400, detail=f"Rule '{rule_id}' is not a merge rule")

    from engine.signals import SignalSet
    signal_set = SignalSet(signals or {})
    success, event, reason = _transformation_engine.execute_merge(rule, signal_set)
    if not success:
        raise HTTPException(status_code=409, detail=reason)

    return {
        "success": True,
        "reason": reason,
        "event_id": event.event_id if event else None,
        "merged_into": rule.new_axis.id if rule.new_axis else None,
    }


@app.post("/v6/transform/transform/{rule_id}")
def execute_transform(rule_id: str, context: Optional[str] = None):
    """Execute a transform rule (context-local axis redefinition)."""
    rule = _rule_store.get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")
    if rule.type != "transform":
        raise HTTPException(status_code=400, detail=f"Rule '{rule_id}' is not a transform rule")

    success, event, reason = _transformation_engine.execute_transform(rule, context)
    if not success:
        raise HTTPException(status_code=409, detail=reason)

    return {
        "success": True,
        "reason": reason,
        "event_id": event.event_id if event else None,
    }


# ===================================================================
# AUDIT LOG
# ===================================================================

@app.get("/v6/audit")
def get_audit_log(limit: int = 20):
    """Return recent audit log entries."""
    return {
        "total_events": _audit_log.count(),
        "events": _audit_log.replay_summary()[-limit:],
    }


@app.get("/v6/audit/axis/{axis_id}")
def get_audit_for_axis(axis_id: str):
    """Return all audit events for a specific axis."""
    events = _audit_log.events_for_axis(axis_id)
    return {
        "axis_id": axis_id,
        "event_count": len(events),
        "events": [e.model_dump() for e in events],
    }


# ===================================================================
# AXIS CRUD (V6)
# ===================================================================

@app.post("/v6/axes/{axis_id}/mask")
def mask_axis(axis_id: str, reason: str = "manual"):
    """Mask an axis (make it invisible without deleting it)."""
    ax = _registry.get(axis_id)
    if ax is None:
        raise HTTPException(status_code=404, detail=f"Axis '{axis_id}' not found")
    _registry.mask(axis_id, reason)
    _audit_log.record(
        rule_id="manual_mask",
        rule_type="mask",
        axis_id=axis_id,
        axes_involved=[axis_id],
        explanation=f"Manual mask: {reason}",
    )
    return {"success": True, "axis_id": axis_id, "status": "masked"}


@app.post("/v6/axes/{axis_id}/unmask")
def unmask_axis(axis_id: str):
    """Unmask a previously masked axis."""
    ax = _registry.get(axis_id)
    if ax is None:
        raise HTTPException(status_code=404, detail=f"Axis '{axis_id}' not found")
    _registry.unmask(axis_id)
    _audit_log.record(
        rule_id="manual_unmask",
        rule_type="reweight",
        axis_id=axis_id,
        axes_involved=[axis_id],
        explanation="Manual unmask — axis restored to active",
    )
    return {"success": True, "axis_id": axis_id, "status": "active"}


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
        "version": "6.0.0",
        "axes_active": len(_registry.all_active()),
        "questions": len(_bank.all_ids()),
        "registry_version": _registry.version,
        "rule_store_version": _rule_store.version,
        "v6_rules": len(_rule_store.all_active()),
        "audit_events": _audit_log.count(),
    }
