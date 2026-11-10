from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import (
    BaseExpert,
    ExpertOutput,
    FeaturePanel,
    bounded_strength,
    require_fields,
    rolling_zscore,
)


def _maximum(frames: list[pd.DataFrame]) -> pd.DataFrame:
    result = frames[0].copy()
    for frame in frames[1:]:
        # fmax keeps the available anomaly when a conventional z-score is
        # undefined because the historical window had zero variance.
        result = result.combine(frame, np.fmax)
    return result


@dataclass(frozen=True)
class ShockExpert(BaseExpert):
    """Detect historical liquidation-like candles and fade the impulse."""

    baseline_window: int = 24 * 14
    min_periods: int = 24
    volatility_window: int = 72
    extreme_threshold: float = 3.0

    def generate(self, panel: FeaturePanel) -> ExpertOutput:
        (
            open_,
            high,
            low,
            close,
            quote_volume,
            trades,
            taker_buy_quote,
        ) = require_fields(
            panel,
            (
                "open",
                "high",
                "low",
                "close",
                "quote_volume",
                "trades",
                "taker_buy_quote",
            ),
        )
        if self.extreme_threshold <= 0.0:
            raise ValueError("extreme_threshold must be positive")

        epsilon = np.finfo(float).eps
        returns = close.pct_change()
        candle_range = high.sub(low).div(open_.abs().where(open_.abs() > epsilon))
        upper_wick = high.sub(pd.DataFrame(
            np.maximum(open_.to_numpy(), close.to_numpy()),
            index=close.index,
            columns=close.columns,
        )).clip(lower=0.0)
        lower_wick = pd.DataFrame(
            np.minimum(open_.to_numpy(), close.to_numpy()),
            index=close.index,
            columns=close.columns,
        ).sub(low).clip(lower=0.0)
        wick_scale = high.sub(low).where(high.sub(low) > epsilon)
        wick_reversal = lower_wick.sub(upper_wick).div(wick_scale)

        taker_imbalance = (
            2.0
            * taker_buy_quote.div(quote_volume.where(quote_volume.abs() > epsilon))
            - 1.0
        ).clip(-1.0, 1.0)
        realized_volatility = returns.rolling(
            self.volatility_window,
            min_periods=min(self.min_periods, self.volatility_window),
        ).std(ddof=0)

        range_z = rolling_zscore(
            candle_range,
            self.baseline_window,
            self.min_periods,
            exclude_current=True,
        )
        volume_z = rolling_zscore(
            np.log1p(quote_volume.clip(lower=0.0)),
            self.baseline_window,
            self.min_periods,
            exclude_current=True,
        )
        trades_z = rolling_zscore(
            np.log1p(trades.clip(lower=0.0)),
            self.baseline_window,
            self.min_periods,
            exclude_current=True,
        )
        volatility_z = rolling_zscore(
            realized_volatility,
            self.baseline_window,
            self.min_periods,
            exclude_current=True,
        )

        # Ratios retain sensitivity when synthetic or quiet historical windows
        # have near-zero variance, where a conventional z-score is undefined.
        def ratio_anomaly(value: pd.DataFrame) -> pd.DataFrame:
            baseline = value.shift(1).rolling(
                self.baseline_window,
                min_periods=self.min_periods,
            ).mean()
            return value.div(baseline.where(baseline.abs() > epsilon)).sub(1.0)

        range_anomaly = _maximum([range_z, ratio_anomaly(candle_range)])
        volume_anomaly = _maximum([volume_z, ratio_anomaly(quote_volume)])
        trades_anomaly = _maximum([trades_z, ratio_anomaly(trades)])
        imbalance_anomaly = taker_imbalance.abs() * self.extreme_threshold
        shock_proxy = _maximum(
            [
                range_anomaly,
                volume_anomaly,
                trades_anomaly,
                volatility_z,
                imbalance_anomaly,
            ]
        ).clip(lower=0.0)

        current_scale = returns.rolling(
            self.volatility_window,
            min_periods=min(self.min_periods, self.volatility_window),
        ).std(ddof=0).where(lambda value: value > epsilon)
        return_impulse = returns.div(current_scale).clip(-6.0, 6.0)
        reversal_direction = np.tanh(
            -0.65 * return_impulse
            + 0.20 * wick_reversal
            - 0.15 * taker_imbalance
        )

        active = shock_proxy >= self.extreme_threshold
        excess = (shock_proxy - self.extreme_threshold).clip(lower=0.0)
        strength = bounded_strength(excess)
        scores = (reversal_direction * strength).where(active, 0.0)
        confidence = (strength * reversal_direction.abs()).where(active, 0.0)

        return ExpertOutput(
            scores=scores.clip(-1.0, 1.0),
            confidence=confidence.clip(0.0, 1.0),
            diagnostics={
                "shock_proxy": shock_proxy,
                "range_anomaly": range_anomaly,
                "volume_anomaly": volume_anomaly,
                "trade_count_anomaly": trades_anomaly,
                "taker_imbalance": taker_imbalance,
                "realized_volatility": realized_volatility,
                "reversal_direction": reversal_direction,
            },
        )
