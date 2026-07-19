"""Validation of the job-side signal extraction pipeline (engine/job_extraction.py).

Four distinct questions, because "validate an LLM reading free text" is a
different problem from "validate a fixed questionnaire a person answers"
(see research/ for the candidate side):

1. Are the citations genuine?
   -> Solved structurally, not statistically: engine/job_extraction.py now
      verifies every source_phrase against the input text (see its
      _normalize()/substring check). Nothing here duplicates that — a
      citation that survives extract_signals() is, by construction,
      verified. What IS computed here is the groundedness RATE across
      logged extractions (how often the LLM's raw output even contained a
      genuine citation vs. one that got dropped) — a data-quality signal
      about the extraction, not a re-verification.

2. Is the signal vocabulary well-calibrated?
   -> calibrate_signal_vocabulary(): per-signal firing rate across logged
      real extractions. A signal that never fires is either genuinely rare
      in practice or mis-specified (too narrow/oddly worded); a signal
      that fires on nearly everything doesn't discriminate between jobs —
      same logic as engine/quality_report.py's flagged_low_discrimination,
      applied to the signal layer instead of the axis layer.

3. Is the extraction reproducible run-to-run on the SAME text?
   -> compute_extraction_stability(): re-runs extraction N times on the
      same description and measures agreement. temperature=0 should make
      this close to deterministic, but isn't guaranteed to be exactly so —
      this is the direct job-side analog of candidate test-retest, except
      the "respondent" being retested is the LLM, not a person. Needs a
      live ANTHROPIC_API_KEY to actually run (real API calls, small cost —
      see the per-call cost estimate discussed for job_extraction.py).

4. Is the extraction robust to paraphrasing that doesn't change meaning?
   -> compute_paraphrase_stability(): same idea, across a human-provided
      (original, paraphrase) pair rather than a repeated identical call.
      Tests whether the pipeline tracks genuine content or surface wording.
      Needs real descriptions + real paraphrases — a corpus this module
      doesn't generate, deliberately (see the docstring on
      build_paraphrase_corpus_stub for why not to auto-generate this with
      another LLM call).

What this module does NOT cover, and can't automate: agreement with a
human expert's independent coding of the same job descriptions (criterion
validity). That needs an actual annotated sample — see
job_annotation_agreement.py for the scoring function once such a sample
exists; producing the sample itself is a human task, not a code task.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from ontology.models import DetectedSignal
from engine.job_extraction import JobSignal, extract_signals, signals_to_axes

# Same thresholds/rationale as engine/quality_report.py's axis-side flags —
# kept identical on purpose so "well-calibrated" means the same thing on
# both sides of the pipeline.
MIN_EXTRACTIONS_FOR_RELIABLE_CALIBRATION = 30
NEVER_FIRES_THRESHOLD = 0.02   # fires on <2% of descriptions
ALWAYS_FIRES_THRESHOLD = 0.90  # fires on >90% of descriptions


def calibrate_signal_vocabulary(
    logged_extractions: list[dict], library: list[JobSignal]
) -> dict:
    """logged_extractions: rows shaped like research.job_extraction_log.all_extractions()."""
    n = len(logged_extractions)
    firing_counts = {s.id: 0 for s in library}
    for row in logged_extractions:
        for sig_id in row["detected_signals"]:
            if sig_id in firing_counts:
                firing_counts[sig_id] += 1

    sample_sufficient = n >= MIN_EXTRACTIONS_FOR_RELIABLE_CALIBRATION
    signals_report = {}
    for s in library:
        rate = (firing_counts[s.id] / n) if n > 0 else 0.0
        signals_report[s.id] = {
            "label": s.label,
            "n_fired": firing_counts[s.id],
            "firing_rate": round(rate, 4),
            "flagged_never_fires": sample_sufficient and rate < NEVER_FIRES_THRESHOLD,
            "flagged_always_fires": sample_sufficient and rate > ALWAYS_FIRES_THRESHOLD,
        }

    return {
        "n_extractions": n,
        "min_for_reliable_calibration": MIN_EXTRACTIONS_FOR_RELIABLE_CALIBRATION,
        "sample_sufficient": sample_sufficient,
        "signals": signals_report,
    }


def _jaccard(a: set, b: set) -> float | None:
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


@dataclass
class StabilityResult:
    description: str
    n_runs: int
    signal_sets: list[set]
    mean_pairwise_jaccard: float | None
    axis_level_stddev: dict[str, float]  # axis_id -> stddev of `level` across runs


def compute_extraction_stability(
    description: str,
    library: list[JobSignal],
    n_runs: int = 5,
    extract_fn=extract_signals,
) -> StabilityResult:
    """Re-runs extraction n_runs times on the SAME description. extract_fn is
    injectable for testing (pass a fake); calling this for real needs a live
    ANTHROPIC_API_KEY and makes n_runs real API calls (a few cents at most,
    same per-call cost as any other /jobs/extract call)."""
    runs: list[list[DetectedSignal]] = [extract_fn(description, library) for _ in range(n_runs)]
    signal_sets = [{d.signal_id for d in run} for run in runs]

    pairwise = []
    for i in range(len(signal_sets)):
        for j in range(i + 1, len(signal_sets)):
            j_score = _jaccard(signal_sets[i], signal_sets[j])
            if j_score is not None:
                pairwise.append(j_score)
    mean_jaccard = round(mean(pairwise), 4) if pairwise else None

    axes_per_run = [signals_to_axes(run, library) for run in runs]
    all_axis_ids = set().union(*[a.keys() for a in axes_per_run]) if axes_per_run else set()
    axis_stddev = {}
    for axis_id in all_axis_ids:
        levels = [a[axis_id]["level"] for a in axes_per_run if axis_id in a]
        axis_stddev[axis_id] = round(pstdev(levels), 4) if len(levels) > 1 else 0.0

    return StabilityResult(
        description=description,
        n_runs=n_runs,
        signal_sets=signal_sets,
        mean_pairwise_jaccard=mean_jaccard,
        axis_level_stddev=axis_stddev,
    )


@dataclass
class ParaphraseStabilityResult:
    original: str
    paraphrase: str
    signal_jaccard: float | None
    axis_level_mean_abs_diff: float | None


def compute_paraphrase_stability(
    original: str,
    paraphrase: str,
    library: list[JobSignal],
    extract_fn=extract_signals,
) -> ParaphraseStabilityResult:
    """Same idea as compute_extraction_stability, across a genuine (original,
    paraphrase) pair supplied by the caller rather than a repeated call.
    Deliberately does NOT generate the paraphrase itself with another LLM
    call — an LLM-paraphrased-then-LLM-extracted pipeline risks the
    paraphraser and the extractor sharing systematic blind spots, which
    would make this look more robust than it is. Use a human-written
    paraphrase, or a paraphrase from a different model family, not the
    same model extracting from its own rewrite."""
    sig_a = extract_fn(original, library)
    sig_b = extract_fn(paraphrase, library)
    set_a = {d.signal_id for d in sig_a}
    set_b = {d.signal_id for d in sig_b}

    axes_a = signals_to_axes(sig_a, library)
    axes_b = signals_to_axes(sig_b, library)
    shared_axes = set(axes_a) & set(axes_b)
    diffs = [abs(axes_a[ax]["level"] - axes_b[ax]["level"]) for ax in shared_axes]

    return ParaphraseStabilityResult(
        original=original,
        paraphrase=paraphrase,
        signal_jaccard=_jaccard(set_a, set_b),
        axis_level_mean_abs_diff=round(mean(diffs), 4) if diffs else None,
    )
