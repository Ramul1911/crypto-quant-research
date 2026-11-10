from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BaseExpert, ExpertOutput, FeaturePanel, require_fields


DEFAULT_FIXED_COST_HURDLE = max(0.0001, 0.0008 / 3.0)
SLOW_CSMOM_HORIZONS = (168, 336, 720)


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1, skipna=True)
    std = frame.std(axis=1, skipna=True, ddof=0)
    usable = std.where(std > np.finfo(float).eps)
    return frame.sub(mean, axis=0).div(usable, axis=0)


@dataclass(frozen=True)
class FundingCarryExpert(BaseExpert):
    """Rank last-known funding only when one event clears round-trip costs."""

    fixed_cost_hurdle: float = DEFAULT_FIXED_COST_HURDLE

    def generate(self, panel: FeaturePanel) -> ExpertOutput:
        if self.fixed_cost_hurdle < 0.0:
            raise ValueError("fixed_cost_hurdle cannot be negative")
        (funding_observations,) = require_fields(panel, ("funding",))
        last_known = funding_observations.ffill()
        eligible = last_known.abs() > self.fixed_cost_hurdle

        # Positive funding is paid by longs, so it is a short signal. Negative
        # funding is received by longs, so it is a long signal.
        scores = (-last_known).where(eligible, 0.0)
        excess = (last_known.abs() - self.fixed_cost_hurdle).clip(lower=0.0)
        confidence = excess.div(
            excess + self.fixed_cost_hurdle
        ).where(eligible, 0.0)
        if self.fixed_cost_hurdle == 0.0:
            confidence = eligible.astype(float)

        hurdle = pd.DataFrame(
            self.fixed_cost_hurdle,
            index=last_known.index,
            columns=last_known.columns,
        ).where(last_known.notna())
        return ExpertOutput(
            scores=scores,
            confidence=confidence.clip(0.0, 1.0),
            diagnostics={
                "last_known_funding": last_known,
                "cost_eligible": eligible.astype(float),
                "fixed_cost_hurdle": hurdle,
            },
        )


@dataclass(frozen=True)
class SlowCrossSectionalMomentumExpert(BaseExpert):
    """Equal mean of causal cross-sectional 7d, 14d and 30d momentum z-scores."""

    horizons: tuple[int, ...] = SLOW_CSMOM_HORIZONS

    def generate(self, panel: FeaturePanel) -> ExpertOutput:
        (close,) = require_fields(panel, ("close",))
        if self.horizons != SLOW_CSMOM_HORIZONS:
            raise ValueError(
                f"horizons are frozen at {SLOW_CSMOM_HORIZONS}"
            )
        valid_close = close.where(close > 0.0)
        log_close = np.log(valid_close)
        returns = {
            horizon: log_close - log_close.shift(horizon)
            for horizon in self.horizons
        }
        zscores = {
            horizon: _cross_sectional_zscore(values)
            for horizon, values in returns.items()
        }
        # Frozen equal weighting requires all three horizons. Averaging only the
        # available horizons would silently turn this into a different strategy
        # for short-history listings.
        scores = sum(zscores.values()) / float(len(zscores))
        scores = scores.reindex(index=close.index, columns=close.columns)
        confidence = (
            sum(values.notna().astype(float) for values in zscores.values())
            / float(len(zscores))
        )
        confidence = confidence.eq(1.0).astype(float).where(scores.notna(), 0.0)

        diagnostics: dict[str, pd.DataFrame] = {}
        for horizon in self.horizons:
            diagnostics[f"log_return_{horizon}h"] = returns[horizon]
            diagnostics[f"cross_sectional_zscore_{horizon}h"] = zscores[horizon]
        return ExpertOutput(
            scores=scores,
            confidence=confidence.clip(0.0, 1.0),
            diagnostics=diagnostics,
        )
