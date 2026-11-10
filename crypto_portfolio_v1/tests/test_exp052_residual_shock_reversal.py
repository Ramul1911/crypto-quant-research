from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import crypto_portfolio_v1.exp052_residual_shock_reversal as exp052


def _synthetic_market(periods: int = 700) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    btc_return = 0.001 * np.sin(np.arange(periods) / 13.0)
    btc = 100.0 * np.exp(np.cumsum(btc_return))
    alt = 50.0 * np.exp(np.cumsum(1.4 * btc_return + 0.0005))
    close = pd.DataFrame({"BTCUSDT": btc, "ALTUSDT": alt}, index=index)
    volume = pd.DataFrame(
        {"BTCUSDT": 1_000.0, "ALTUSDT": 500.0},
        index=index,
    )
    return close, volume


def test_residual_features_are_causal_under_future_mutation() -> None:
    close, volume = _synthetic_market()
    cutoff = close.index[550]
    changed_close = close.copy()
    changed_volume = volume.copy()
    changed_close.loc[changed_close.index > cutoff, "ALTUSDT"] *= 3.0
    changed_volume.loc[changed_volume.index > cutoff, "ALTUSDT"] *= 10.0

    original = exp052.build_residual_features(close, volume)
    changed = exp052.build_residual_features(changed_close, changed_volume)

    for original_frame, changed_frame in zip(original, changed):
        assert_frame_equal(
            original_frame.loc[:cutoff],
            changed_frame.loc[:cutoff],
        )


def test_event_positions_hold_exactly_12h_and_cap_active_names() -> None:
    index = pd.date_range("2025-01-01", periods=25, freq="h")
    columns = ["BTCUSDT", "A", "B", "C", "D", "E"]
    shock = pd.DataFrame(0.0, index=index, columns=columns)
    volume_z = pd.DataFrame(0.0, index=index, columns=columns)
    shock.loc[index[0], ["A", "B", "C", "D", "E"]] = [5.0, 4.0, 3.5, 3.0, 2.6]
    volume_z.loc[index[0], ["A", "B", "C", "D", "E"]] = 2.0

    positions = exp052.build_event_positions(shock, volume_z)

    assert positions.iloc[0].ne(0.0).sum() == 4
    assert positions.iloc[1].ne(0.0).sum() == 4
    assert positions.iloc[2].ne(0.0).sum() == 4
    assert positions.iloc[3].eq(0.0).all()
    assert positions["E"].eq(0.0).all()
    assert positions.ne(0.0).sum(axis=1).max() <= exp052.MAX_ACTIVE_ALTS


def test_reversal_and_continuation_use_opposite_events_and_beta_hedge() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="4h")
    columns = ["BTCUSDT", "A", "B"]
    positions = pd.DataFrame(
        [[0.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
        index=index,
        columns=columns,
    )
    volatility = pd.DataFrame(0.5, index=index, columns=columns)
    beta = pd.DataFrame(
        [[1.0, 1.5, 0.5], [1.0, 1.5, 0.5]],
        index=index,
        columns=columns,
    )

    reversal = exp052.build_beta_neutral_targets(
        positions,
        volatility,
        beta,
        reversal=True,
    )
    continuation = exp052.build_beta_neutral_targets(
        positions,
        volatility,
        beta,
        reversal=False,
    )

    assert np.allclose(reversal.iloc[0], -continuation.iloc[0])
    assert np.isclose(float((reversal.iloc[0] * beta.iloc[0]).sum()), 0.0)
    assert reversal.iloc[0].abs().max() <= 0.30
    assert reversal.iloc[0].abs().sum() <= 1.0
    assert reversal.iloc[1].eq(0.0).all()
