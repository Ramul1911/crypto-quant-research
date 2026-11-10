from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import (
    BaseExpert,
    ExpertOutput,
    FeaturePanel,
    bounded_strength,
    require_fields,
    rolling_zscore,
)


@dataclass(frozen=True)
class FundingExpert(BaseExpert):
    """Fade unusually crowded funding when its edge clears trading costs."""

    zscore_window: int = 24 * 30
    min_periods: int = 24 * 7
    extreme_z: float = 2.0
    round_trip_cost: float = 0.0008
    expected_funding_intervals: int = 3
    min_abs_funding_rate: float = 0.0001

    def generate(self, panel: FeaturePanel) -> ExpertOutput:
        (funding_observations,) = require_fields(panel, ("funding",))
        if self.expected_funding_intervals <= 0:
            raise ValueError("expected_funding_intervals must be positive")
        if self.extreme_z <= 0.0:
            raise ValueError("extreme_z must be positive")

        # Funding is sparse (normally every eight hours).  Forward fill means
        # each bar sees only the most recently published observation.
        last_known = funding_observations.ffill()
        zscore = rolling_zscore(
            last_known,
            self.zscore_window,
            self.min_periods,
        )

        required_rate = max(
            self.min_abs_funding_rate,
            self.round_trip_cost / self.expected_funding_intervals,
        )
        extreme = zscore.abs() >= self.extreme_z
        cost_effective = last_known.abs() >= required_rate
        active = extreme & cost_effective

        excess_z = (zscore.abs() - self.extreme_z).clip(lower=0.0)
        strength = bounded_strength(excess_z)
        scores = (-np.sign(zscore) * strength).where(active, 0.0)

        cost_margin = (
            last_known.abs().div(required_rate).sub(1.0).clip(lower=0.0)
        )
        confidence = (strength * bounded_strength(cost_margin)).where(active, 0.0)
        confidence = confidence.clip(0.0, 1.0)

        required_rate_frame = last_known.notna().astype(float) * required_rate
        return ExpertOutput(
            scores=scores,
            confidence=confidence,
            diagnostics={
                "last_known_funding": last_known,
                "funding_zscore": zscore,
                "extreme": extreme.astype(float),
                "cost_effective": cost_effective.astype(float),
                "required_abs_funding_rate": required_rate_frame,
            },
        )
