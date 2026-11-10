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
class TrendV21Profile:
    """A fixed, non-fitted set of multi-horizon momentum weights."""

    name: str
    horizon_weights: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must not be empty")
        if not self.horizon_weights:
            raise ValueError("profile must contain at least one horizon")

        horizons = [horizon for horizon, _ in self.horizon_weights]
        weights = [weight for _, weight in self.horizon_weights]
        if any(horizon <= 1 for horizon in horizons):
            raise ValueError("trend horizons must be greater than one")
        if len(set(horizons)) != len(horizons):
            raise ValueError("trend horizons must be unique")
        if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("profile weights must be finite and non-negative")
        if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("profile weights must sum to one")

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(horizon for horizon, _ in self.horizon_weights)


FAST = TrendV21Profile(
    "FAST",
    ((12, 0.40), (24, 0.30), (72, 0.20), (168, 0.10)),
)
BALANCED = TrendV21Profile(
    "BALANCED",
    ((12, 0.20), (24, 0.30), (72, 0.30), (168, 0.20)),
)
SLOW = TrendV21Profile(
    "SLOW",
    ((12, 0.10), (24, 0.15), (72, 0.30), (168, 0.45)),
)


@dataclass(frozen=True)
class TrendExpertV21(BaseExpert):
    """Stable causal trend signal with regime filtering and hysteresis."""

    profile: TrendV21Profile = BALANCED
    volatility_window: int = 72
    min_volatility_periods: int = 24
    quality_window: int = 48
    short_volatility_window: int = 12
    long_volatility_window: int = 72
    vol_of_vol_window: int = 48
    acceleration_horizon: int = 6
    long_entry_threshold: float = 0.18
    short_entry_threshold: float = 0.26
    long_exit_threshold: float = 0.08
    short_exit_threshold: float = 0.12
    minimum_hold_bars: int = 12
    short_max_scale: float = 0.65

    def __post_init__(self) -> None:
        if not isinstance(self.profile, TrendV21Profile):
            raise TypeError("profile must be a TrendV21Profile")
        windows = (
            self.volatility_window,
            self.min_volatility_periods,
            self.quality_window,
            self.short_volatility_window,
            self.long_volatility_window,
            self.vol_of_vol_window,
            self.acceleration_horizon,
        )
        if any(window <= 1 for window in windows):
            raise ValueError("rolling windows must be greater than one")
        if self.min_volatility_periods > self.volatility_window:
            raise ValueError("minimum volatility periods exceed window")
        if self.minimum_hold_bars < 0:
            raise ValueError("minimum_hold_bars must be non-negative")
        if not 0.0 < self.short_max_scale <= 1.0:
            raise ValueError("short_max_scale must be in (0, 1]")
        if not 0.0 <= self.long_exit_threshold < self.long_entry_threshold:
            raise ValueError("long exit threshold must be below entry threshold")
        if not 0.0 <= self.short_exit_threshold < self.short_entry_threshold:
            raise ValueError("short exit threshold must be below entry threshold")

    def generate(self, panel: FeaturePanel) -> ExpertOutput:
        close, high, low = require_fields(panel, ("close", "high", "low"))
        if ((close <= 0.0) | (high <= 0.0) | (low <= 0.0)).any().any():
            raise ValueError("close, high, and low must be positive")

        log_close = np.log(close)
        returns = log_close.diff()
        volatility = returns.rolling(
            self.volatility_window,
            min_periods=self.min_volatility_periods,
        ).std(ddof=0)
        mean_absolute_return = returns.abs().rolling(
            self.volatility_window,
            min_periods=self.min_volatility_periods,
        ).mean()
        effective_volatility = volatility.clip(
            lower=mean_absolute_return * 0.10,
        ).where(lambda value: value > np.finfo(float).eps)

        momentum = close * 0.0
        breakout = close * 0.0
        cross_sectional = close * 0.0
        signed_consistency = close * 0.0
        availability = close * 0.0

        for horizon, weight in self.profile.horizon_weights:
            log_momentum = log_close.diff(horizon)
            normalized = log_momentum.div(
                effective_volatility * np.sqrt(horizon),
            )
            momentum = momentum + weight * np.tanh(normalized)
            cross_sectional = cross_sectional + weight * np.tanh(
                _cross_sectional_zscore(log_momentum)
            )

            rolling_low = low.rolling(horizon, min_periods=horizon).min()
            rolling_high = high.rolling(horizon, min_periods=horizon).max()
            span = (rolling_high - rolling_low).where(
                (rolling_high - rolling_low) > np.finfo(float).eps
            )
            location = 2.0 * close.sub(rolling_low).div(span) - 1.0
            breakout = breakout + weight * location.clip(-1.0, 1.0)
            signed_consistency = signed_consistency + weight * np.sign(log_momentum)
            availability = availability + weight * log_momentum.notna().astype(float)

        consistency = signed_consistency.abs().clip(0.0, 1.0)
        raw_score = (
            0.50 * momentum + 0.30 * breakout + 0.20 * cross_sectional
        ).clip(-1.0, 1.0)

        net_change = log_close.diff(self.quality_window).abs()
        path_change = returns.abs().rolling(
            self.quality_window,
            min_periods=self.quality_window,
        ).sum()
        efficiency_ratio = net_change.div(
            path_change.where(path_change > np.finfo(float).eps)
        ).clip(0.0, 1.0)
        efficiency_quality = (
            (efficiency_ratio - 0.12) / 0.38
        ).clip(0.0, 1.0)

        squared_returns = returns.pow(2)
        short_vol = squared_returns.rolling(
            self.short_volatility_window,
            min_periods=self.short_volatility_window,
        ).mean().pow(0.5)
        long_vol = squared_returns.rolling(
            self.long_volatility_window,
            min_periods=self.long_volatility_window,
        ).mean().pow(0.5)
        vol_ratio = short_vol.div(
            long_vol.where(long_vol > np.finfo(float).eps)
        )
        vol_of_vol = short_vol.rolling(
            self.vol_of_vol_window,
            min_periods=self.vol_of_vol_window,
        ).std(ddof=0)
        baseline_vol_of_vol = vol_of_vol.rolling(
            self.long_volatility_window,
            min_periods=self.min_volatility_periods,
        ).mean()
        vol_of_vol_ratio = vol_of_vol.div(
            baseline_vol_of_vol.where(
                baseline_vol_of_vol > np.finfo(float).eps
            )
        ).fillna(1.0)
        vol_shock = (
            ((vol_ratio - 1.25) / 1.25).clip(lower=0.0)
            + ((vol_of_vol_ratio - 1.50) / 2.00).clip(lower=0.0)
        ).clip(0.0, 1.0)
        trend_quality = (
            efficiency_quality * (1.0 - vol_shock)
        ).clip(0.0, 1.0)

        short_acceleration = returns.rolling(
            self.acceleration_horizon,
            min_periods=self.acceleration_horizon,
        ).sum().div(
            long_vol * np.sqrt(self.acceleration_horizon)
        )
        acceleration_excess = (
            short_acceleration.abs() - 3.0
        ).clip(lower=0.0)
        acceleration_scale = (1.0 / (1.0 + acceleration_excess)).clip(
            lower=0.30,
            upper=1.0,
        )

        evidence = (
            raw_score
            * consistency
            * trend_quality
            * acceleration_scale
        ).where(availability >= 1.0, 0.0).fillna(0.0)
        scores, hysteresis_state = self._apply_hysteresis(evidence)
        confidence = (
            consistency
            * trend_quality
            * availability
            * raw_score.abs()
        ).clip(0.0, 1.0)
        confidence = confidence.where(hysteresis_state.ne(0.0), 0.0).fillna(0.0)

        return ExpertOutput(
            scores=scores,
            confidence=confidence,
            diagnostics={
                "raw_score": raw_score,
                "momentum": momentum,
                "breakout": breakout,
                "cross_sectional_momentum": cross_sectional,
                "trend_consistency": consistency,
                "trend_quality": trend_quality,
                "efficiency_ratio": efficiency_ratio,
                "vol_shock": vol_shock,
                "acceleration_scale": acceleration_scale,
                "hysteresis_state": hysteresis_state,
            },
        )

    def _apply_hysteresis(
        self,
        evidence: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        scores = pd.DataFrame(0.0, index=evidence.index, columns=evidence.columns)
        states = pd.DataFrame(0.0, index=evidence.index, columns=evidence.columns)

        for column in evidence.columns:
            state = 0
            held_bars = 0
            for row, value in enumerate(evidence[column].to_numpy(dtype=float)):
                value = value if np.isfinite(value) else 0.0
                if state == 0:
                    if value >= self.long_entry_threshold:
                        state = 1
                        held_bars = 0
                    elif value <= -self.short_entry_threshold:
                        state = -1
                        held_bars = 0
                else:
                    held_bars += 1
                    exit_threshold = (
                        self.long_exit_threshold
                        if state > 0
                        else self.short_exit_threshold
                    )
                    aligned_strength = state * value
                    if (
                        held_bars >= self.minimum_hold_bars
                        and aligned_strength < exit_threshold
                    ):
                        state = 0
                        held_bars = 0

                states.iat[row, states.columns.get_loc(column)] = float(state)
                if state > 0:
                    scores.iat[row, scores.columns.get_loc(column)] = min(
                        1.0,
                        max(value, self.long_exit_threshold),
                    )
                elif state < 0:
                    scores.iat[row, scores.columns.get_loc(column)] = -min(
                        self.short_max_scale,
                        max(-value, self.short_exit_threshold),
                    )

        return scores, states
