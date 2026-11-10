from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from crypto_portfolio_v1.experts import (
    BALANCED,
    FAST,
    SLOW,
    TrendExpertV21,
    TrendV21Profile,
)


def _panel_from_close(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    previous = close.shift(1).fillna(close.iloc[0])
    high = pd.DataFrame(
        np.maximum(previous, close) * 1.002,
        index=close.index,
        columns=close.columns,
    )
    low = pd.DataFrame(
        np.minimum(previous, close) * 0.998,
        index=close.index,
        columns=close.columns,
    )
    return {"close": close, "high": high, "low": low}


def _persistent_trend_panel(rows: int = 320) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=rows, freq="h")
    time = np.arange(rows)
    close = pd.DataFrame(
        {
            "UP": 100.0 * np.exp(0.002 * time),
            "DOWN": 100.0 * np.exp(-0.002 * time),
        },
        index=index,
    )
    return _panel_from_close(close)


def test_fixed_profiles_are_valid_and_balanced_is_default() -> None:
    for profile in (FAST, BALANCED, SLOW):
        assert sum(weight for _, weight in profile.horizon_weights) == pytest.approx(
            1.0
        )
        assert len(profile.horizons) == len(set(profile.horizons))

    assert TrendExpertV21().profile is BALANCED
    with pytest.raises(ValueError, match="sum to one"):
        TrendV21Profile("INVALID", ((12, 0.7), (24, 0.2)))
    with pytest.raises(ValueError, match="unique"):
        TrendV21Profile("INVALID", ((12, 0.5), (12, 0.5)))
    with pytest.raises(ValueError, match="greater than one"):
        TrendV21Profile("INVALID", ((1, 1.0),))


def test_persistent_up_and_down_trends_have_asymmetric_signals() -> None:
    output = TrendExpertV21().predict(_persistent_trend_panel())
    final = output.scores.iloc[-1]

    assert final["UP"] > 0.0
    assert final["DOWN"] < 0.0
    assert abs(final["DOWN"]) <= TrendExpertV21().short_max_scale
    assert abs(final["DOWN"]) < final["UP"]
    assert output.diagnostics["hysteresis_state"].iloc[-1].to_dict() == {
        "UP": 1.0,
        "DOWN": -1.0,
    }


def test_choppy_market_is_suppressed_by_trend_quality() -> None:
    rows = 320
    index = pd.date_range("2025-01-01", periods=rows, freq="h")
    alternating_returns = np.where(np.arange(rows) % 2 == 0, 0.012, -0.012)
    close_values = 100.0 * np.exp(np.cumsum(alternating_returns))
    close = pd.DataFrame({"CHOP": close_values}, index=index)

    output = TrendExpertV21().predict(_panel_from_close(close))

    assert output.diagnostics["trend_quality"].iloc[-50:].max().max() < 0.10
    assert output.scores.iloc[-50:].eq(0.0).all().all()
    assert output.diagnostics["hysteresis_state"].iloc[-50:].eq(0.0).all().all()


def test_hysteresis_holds_through_weakening_then_exits() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="h")
    evidence = pd.DataFrame(
        {"A": [0.0, 0.30, 0.05, -0.20, 0.05, 0.0, 0.0, 0.0]},
        index=index,
    )
    expert = TrendExpertV21(minimum_hold_bars=3)

    scores, states = expert._apply_hysteresis(evidence)

    assert states["A"].tolist()[:5] == [0.0, 1.0, 1.0, 1.0, 0.0]
    assert scores.loc[index[2], "A"] >= expert.long_exit_threshold
    assert scores.loc[index[3], "A"] >= expert.long_exit_threshold
    assert scores.loc[index[4], "A"] == 0.0


def test_future_mutation_does_not_change_past_outputs_or_diagnostics() -> None:
    panel = _persistent_trend_panel(360)
    cutoff = panel["close"].index[260]
    original = TrendExpertV21().predict(panel)

    mutated = {name: frame.copy() for name, frame in panel.items()}
    future = mutated["close"].index > cutoff
    mutated["close"].loc[future] *= np.linspace(0.3, 3.0, future.sum())[:, None]
    mutated["high"].loc[future] *= 5.0
    mutated["low"].loc[future] *= 0.2
    changed = TrendExpertV21().predict(mutated)

    assert_frame_equal(original.scores.loc[:cutoff], changed.scores.loc[:cutoff])
    assert_frame_equal(
        original.confidence.loc[:cutoff],
        changed.confidence.loc[:cutoff],
    )
    for name in (
        "raw_score",
        "trend_consistency",
        "trend_quality",
        "vol_shock",
        "acceleration_scale",
        "hysteresis_state",
    ):
        assert_frame_equal(
            original.diagnostics[name].loc[:cutoff],
            changed.diagnostics[name].loc[:cutoff],
        )


def test_required_diagnostics_are_aligned() -> None:
    panel = _persistent_trend_panel()
    output = TrendExpertV21().predict(panel)
    required = {
        "raw_score",
        "trend_consistency",
        "trend_quality",
        "vol_shock",
        "acceleration_scale",
        "hysteresis_state",
    }

    assert required <= output.diagnostics.keys()
    for frame in output.diagnostics.values():
        assert frame.index.equals(panel["close"].index)
        assert frame.columns.equals(panel["close"].columns)
