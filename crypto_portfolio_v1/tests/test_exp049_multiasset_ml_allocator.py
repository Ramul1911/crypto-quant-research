from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import crypto_portfolio_v1.exp049_multiasset_ml_allocator as exp049


def test_momentum_and_volatility_are_causal_under_future_mutation() -> None:
    index = pd.date_range("2024-01-01", periods=500, freq="h")
    close = pd.DataFrame(
        {
            "A": 100.0 * np.exp(np.arange(500) * 0.001),
            "B": 100.0 * np.exp(np.sin(np.arange(500) / 20.0) * 0.1),
        },
        index=index,
    )
    cutoff = index[350]
    changed = close.copy()
    changed.loc[changed.index > cutoff, "A"] *= 4.0
    changed.loc[changed.index > cutoff, "B"] *= 0.2

    assert_frame_equal(
        exp049.momentum_direction(close).loc[:cutoff],
        exp049.momentum_direction(changed).loc[:cutoff],
    )
    assert_frame_equal(
        exp049.annualized_volatility(close).loc[:cutoff],
        exp049.annualized_volatility(changed).loc[:cutoff],
    )


def test_equal_risk_targets_have_unit_raw_gross_and_daily_schedule() -> None:
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    columns = ["A", "B", "C", "D"]
    direction = pd.DataFrame(1.0, index=index, columns=columns)
    volatility = pd.DataFrame(
        np.tile([0.4, 0.6, 0.8, 1.0], (len(index), 1)),
        index=index,
        columns=columns,
    )

    targets = exp049.build_equal_risk_targets(direction, volatility)

    assert np.asarray(targets.index.hour == 0).all()
    assert np.allclose(targets.abs().sum(axis=1), 1.0)
    assert (targets["A"] > targets["D"]).all()


def test_exact_label_uses_strictly_later_entry_and_24h_exit() -> None:
    index = pd.date_range("2024-01-01", periods=110, freq="15min")
    open_prices = pd.DataFrame(
        {"A": 100.0 * 1.001 ** np.arange(len(index))},
        index=index,
    )
    panel = {
        "open": open_prices,
        "funding": pd.DataFrame(0.0, index=index, columns=["A"]),
    }
    timestamp = pd.Timestamp("2024-01-01 00:00:00")
    decisions = pd.DatetimeIndex([timestamp])
    direction = pd.DataFrame({"A": [1.0]}, index=decisions)

    labels = exp049.exact_forward_net_labels(decisions, direction, panel)
    entry_i = int(index.searchsorted(timestamp, side="right"))
    exit_i = int(
        index.searchsorted(
            timestamp + pd.Timedelta(hours=exp049.LABEL_HORIZON_HOURS),
            side="right",
        )
    )
    expected = (
        open_prices.iloc[exit_i, 0] / open_prices.iloc[entry_i, 0]
        - 1.0
        - exp049.ROUND_TRIP_COST
    )

    assert labels.at[timestamp, "A"] == expected


def test_ml_allocator_selects_only_top_four_predictions() -> None:
    class FakeModel:
        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return features["score"].to_numpy()

    timestamp = pd.Timestamp("2025-01-01")
    symbols = [f"S{i}" for i in range(8)]
    table = pd.DataFrame(
        {
            "timestamp": [timestamp] * 8,
            "symbol": symbols,
            "score": np.arange(8, dtype=float),
            "annual_volatility": np.ones(8),
        }
    )
    direction = pd.DataFrame(1.0, index=[timestamp], columns=symbols)
    volatility = pd.DataFrame(1.0, index=[timestamp], columns=symbols)

    targets = exp049.build_ml_fold_targets(
        FakeModel(),  # type: ignore[arg-type]
        table,
        ["score"],
        direction,
        volatility,
        "2025-01-01",
        "2025-01-01 23:59:59",
    )

    assert int(targets.ne(0.0).sum(axis=1).iloc[0]) == exp049.TOP_K
    assert targets.loc[timestamp, ["S4", "S5", "S6", "S7"]].gt(0.0).all()
    assert targets.loc[timestamp].abs().sum() == 1.0
