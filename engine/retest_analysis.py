"""Test-retest reliability analysis for the reproducibility study.

Computes, per axis, how stable `raw_value` (the unweighted 0.0-1.0 mean of
answers — unaffected by context/weight, so a clean read on the underlying
self-report) is between a participant's first and second passage.

No numpy/scipy available in this venv, so ICC and Pearson r are implemented
directly. ICC(1,1) (one-way random effects, single measurement — Shrout &
Fleiss / McGraw & Wong) is used rather than a two-way model: passage 1 vs
passage 2 are just two occasions of the same measurement, not two raters
with a distinct systematic effect to separate out.
"""

from __future__ import annotations

import json
import math

from engine.retest_store import RetestStore

# Below this many pairs, ICC/correlation estimates are too noisy to act on —
# the report still computes them but flags the sample as insufficient.
MIN_PAIRS_FOR_RELIABLE_REPORT = 20


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / math.sqrt(vx * vy)


def _icc_1_1(xs: list[float], ys: list[float]) -> float | None:
    """ICC(1,1), k=2 occasions: (MSB - MSW) / (MSB + MSW)."""
    n = len(xs)
    if n < 2:
        return None
    grand_mean = (sum(xs) + sum(ys)) / (2 * n)
    subject_means = [(x + y) / 2 for x, y in zip(xs, ys)]
    msb = 2 * sum((m - grand_mean) ** 2 for m in subject_means) / (n - 1)
    msw = sum(
        (x - m) ** 2 + (y - m) ** 2 for x, y, m in zip(xs, ys, subject_means)
    ) / n
    if msb + msw == 0:
        return 1.0
    return (msb - msw) / (msb + msw)


def _icc_band(icc: float | None) -> str:
    """Koo & Li (2016) conventional thresholds."""
    if icc is None:
        return "n/a"
    if icc < 0.5:
        return "faible"
    if icc < 0.75:
        return "modérée"
    if icc < 0.9:
        return "bonne"
    return "excellente"


def compute_retest_report(store: RetestStore, tolerance: float = 0.10) -> dict:
    pairs = store.paired_first_two()
    n_pairs = len(pairs)

    per_axis: dict[str, list[tuple[float, float]]] = {}
    for first, second in pairs:
        ax1 = json.loads(first["axis_scores_json"])
        ax2 = json.loads(second["axis_scores_json"])
        for axis_id, data1 in ax1.items():
            data2 = ax2.get(axis_id)
            if data2 is None:
                continue
            per_axis.setdefault(axis_id, []).append(
                (data1["raw_value"], data2["raw_value"])
            )

    axes_report: dict[str, dict] = {}
    for axis_id, values in sorted(per_axis.items()):
        xs = [v[0] for v in values]
        ys = [v[1] for v in values]
        icc = _icc_1_1(xs, ys)
        within_tol = sum(1 for x, y in zip(xs, ys) if abs(x - y) <= tolerance)
        axes_report[axis_id] = {
            "n": len(values),
            "icc": icc,
            "icc_band": _icc_band(icc),
            "pearson_r": _pearson(xs, ys),
            "mean_abs_delta": sum(abs(x - y) for x, y in zip(xs, ys)) / len(values),
            "pct_within_tolerance": within_tol / len(values),
        }

    return {
        "n_pairs": n_pairs,
        "min_pairs_for_reliable_report": MIN_PAIRS_FOR_RELIABLE_REPORT,
        "sample_sufficient": n_pairs >= MIN_PAIRS_FOR_RELIABLE_REPORT,
        "tolerance": tolerance,
        "axes": axes_report,
    }
