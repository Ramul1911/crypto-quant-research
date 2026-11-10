from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import BaseExpert, ExpertOutput, FeaturePanel, require_fields


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0)
    return frame.sub(mean, axis=0).div(
        std.where(std > np.finfo(float).eps),
        axis=0,
    )


@dataclass(frozen=True)
class TrendExpert(BaseExpert):
    """Causal absolute and cross-sectional multi-horizon trend expert."""

    horizons: tuple[int, ...] = (12, 24, 72, 168)
    volatility_window: int = 72
    min_volatility_periods: int = 24

    def generate(self, panel: FeaturePanel) -> ExpertOutput:
        (close,) = require_fields(panel, ("close",))
        if any(horizon <= 1 for horizon in self.horizons):
            raise ValueError("trend horizons must be greater than one")

        log_close = np.log(close.where(close > 0.0))
        hourly_returns = log_close.diff()
        volatility = hourly_returns.rolling(
            self.volatility_window,
            min_periods=self.min_volatility_periods,
        ).std(ddof=0)
        volatility = volatility.where(volatility > np.finfo(float).eps)

        momentum_parts: list[pd.DataFrame] = []
        breakout_parts: list[pd.DataFrame] = []
        relative_parts: list[pd.DataFrame] = []
        direction_parts: list[pd.DataFrame] = []
        availability_parts: list[pd.DataFrame] = []

        for horizon in self.horizons:
            log_momentum = log_close.diff(horizon)
            normalized = log_momentum.div(volatility * np.sqrt(horizon))
            momentum_parts.append(np.tanh(normalized))
            relative_parts.append(np.tanh(_cross_sectional_zscore(log_momentum)))

            rolling_low = close.rolling(horizon, min_periods=horizon).min()
            rolling_high = close.rolling(horizon, min_periods=horizon).max()
            span = (rolling_high - rolling_low).where(
                (rolling_high - rolling_low) > np.finfo(float).eps
            )
            breakout = 2.0 * close.sub(rolling_low).div(span) - 1.0
            breakout_parts.append(breakout.clip(-1.0, 1.0))
            direction_parts.append(np.sign(log_momentum))
            availability_parts.append(log_momentum.notna().astype(float))

        momentum = sum(momentum_parts) / len(momentum_parts)
        breakout = sum(breakout_parts) / len(breakout_parts)
        cross_sectional = sum(relative_parts) / len(relative_parts)
        consistency = abs(sum(direction_parts) / len(direction_parts))
        availability = sum(availability_parts) / len(availability_parts)

        raw_score = 0.50 * momentum + 0.30 * breakout + 0.20 * cross_sectional
        scores = (raw_score * consistency).clip(-1.0, 1.0)
        confidence = (
            consistency
            * availability
            * (0.35 + 0.65 * raw_score.abs().clip(0.0, 1.0))
        ).clip(0.0, 1.0)
        scores = scores.where(availability > 0.0)
        confidence = confidence.where(scores.notna(), 0.0)

        return ExpertOutput(
            scores=scores,
            confidence=confidence,
            diagnostics={
                "momentum": momentum,
                "breakout": breakout,
                "cross_sectional_momentum": cross_sectional,
                "trend_consistency": consistency,
                "hourly_volatility": volatility,
            },
        )
