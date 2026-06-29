"""Signal computer — computes statistical signals from accumulated profiles (V6).

In V6, signals are computed on demand from a set of profiles.
In V7+, this will run continuously as profiles accumulate.
"""

from __future__ import annotations

import math
from typing import Optional

from ontology.models import Profile


class SignalSet(dict):
    """
    {
      "axis_id": {
        "variance": float,
        "internal_consistency": float,  # 1 - variance (normalized)
        "discrimination_index": float,
        "mean": float,
        "n_profiles": int,
      },
      ("axis_a", "axis_b"): {
        "correlation": float,
      }
    }
    """
    pass


class SignalComputer:
    """Computes statistical signals from a collection of profiles."""

    def compute(self, profiles: list[Profile]) -> SignalSet:
        """Compute all signals from a list of profiles."""
        signals = SignalSet()

        if not profiles:
            return signals

        # Collect per-axis values across profiles
        axis_values: dict[str, list[float]] = {}
        for profile in profiles:
            for ax_id, score in profile.axis_scores.items():
                axis_values.setdefault(ax_id, []).append(score.raw_value)

        # Per-axis signals
        for ax_id, values in axis_values.items():
            n = len(values)
            if n < 2:
                continue
            mean = sum(values) / n
            var = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(var)

            # Internal consistency: how consistent are answers across profiles
            # High variance among people = axis discriminates well (good)
            # Internal consistency here = mean per-profile answer variance
            per_profile_vars = []
            for profile in profiles:
                sc = profile.axis_scores.get(ax_id)
                if sc and sc.variance is not None:
                    per_profile_vars.append(sc.variance)

            internal_consistency = 1.0 - (
                sum(per_profile_vars) / len(per_profile_vars)
                if per_profile_vars else 0.0
            )

            # Discrimination index: how well the axis separates profiles
            # Simple proxy: inter-profile std (higher = more discriminating)
            discrimination_index = min(1.0, std * 2)

            signals[ax_id] = {
                "mean": round(mean, 4),
                "variance": round(var, 4),
                "std": round(std, 4),
                "internal_consistency": round(max(0.0, min(1.0, internal_consistency)), 4),
                "discrimination_index": round(discrimination_index, 4),
                "n_profiles": n,
            }

        # Pairwise correlations
        axis_ids = list(axis_values.keys())
        for i, ax_a in enumerate(axis_ids):
            for ax_b in axis_ids[i + 1:]:
                corr = self._pearson(axis_values[ax_a], axis_values[ax_b])
                if corr is not None:
                    signals[(ax_a, ax_b)] = {"correlation": round(corr, 4)}
                    signals[(ax_b, ax_a)] = {"correlation": round(corr, 4)}

        return signals

    def inject(self, manual_signals: dict) -> SignalSet:
        """Accept manually provided signals (for testing / demo without real profiles)."""
        return SignalSet(manual_signals)

    def _pearson(self, xs: list[float], ys: list[float]) -> Optional[float]:
        paired = [(x, y) for x, y in zip(xs, ys)]
        n = len(paired)
        if n < 3:
            return None
        mx = sum(x for x, _ in paired) / n
        my = sum(y for _, y in paired) / n
        num = sum((x - mx) * (y - my) for x, y in paired)
        denom = math.sqrt(
            sum((x - mx) ** 2 for x, _ in paired) *
            sum((y - my) ** 2 for _, y in paired)
        )
        return num / denom if denom > 1e-9 else None
