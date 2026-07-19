"""Human-vs-LLM agreement — criterion validity for the signal extraction.

The one piece of the job-side validation protocol that can't be automated:
someone has to read a sample of real job descriptions and independently
say which signals they'd flag, by hand, from the same closed vocabulary
the LLM uses. This module only scores the comparison once that sample
exists — it doesn't and can't produce the sample itself.

Annotation format (one JSON object per description):

    {
      "description": "...",
      "human_signal_ids": ["cadre_reglementaire", "pilotage_projet"]
    }

A reasonable starting sample size: 20-30 real descriptions, ideally
spanning different sectors/seniority levels, annotated by someone who
didn't write the signal vocabulary (Olivier himself risks unconsciously
annotating toward what he knows the extractor looks for) — a colleague
or a second, blinded pass is worth more here than a larger sample
annotated by the same person who designed the signals.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.job_extraction import JobSignal, extract_signals


@dataclass
class SignalAgreement:
    signal_id: str
    true_positive: int   # human flagged it, LLM flagged it
    false_positive: int  # LLM flagged it, human didn't
    false_negative: int  # human flagged it, LLM didn't
    true_negative: int   # neither flagged it

    @property
    def precision(self) -> float | None:
        denom = self.true_positive + self.false_positive
        return round(self.true_positive / denom, 4) if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.true_positive + self.false_negative
        return round(self.true_positive / denom, 4) if denom else None


def score_agreement(
    annotations: list[dict], library: list[JobSignal], extract_fn=extract_signals
) -> dict:
    """annotations: list of {"description": str, "human_signal_ids": list[str]}."""
    all_ids = [s.id for s in library]
    counts = {sid: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for sid in all_ids}

    for row in annotations:
        human_set = set(row["human_signal_ids"])
        llm_detected = extract_fn(row["description"], library)
        llm_set = {d.signal_id for d in llm_detected}

        for sid in all_ids:
            in_human = sid in human_set
            in_llm = sid in llm_set
            if in_human and in_llm:
                counts[sid]["tp"] += 1
            elif in_llm and not in_human:
                counts[sid]["fp"] += 1
            elif in_human and not in_llm:
                counts[sid]["fn"] += 1
            else:
                counts[sid]["tn"] += 1

    per_signal = {
        sid: SignalAgreement(
            signal_id=sid,
            true_positive=c["tp"], false_positive=c["fp"],
            false_negative=c["fn"], true_negative=c["tn"],
        )
        for sid, c in counts.items()
    }

    total_tp = sum(c["tp"] for c in counts.values())
    total_fp = sum(c["fp"] for c in counts.values())
    total_fn = sum(c["fn"] for c in counts.values())
    overall_precision = round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) else None
    overall_recall = round(total_tp / (total_tp + total_fn), 4) if (total_tp + total_fn) else None

    return {
        "n_annotated": len(annotations),
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "per_signal": {
            sid: {
                "precision": a.precision, "recall": a.recall,
                "tp": a.true_positive, "fp": a.false_positive, "fn": a.false_negative,
            }
            for sid, a in per_signal.items()
        },
    }
