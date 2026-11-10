from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import crypto_portfolio_v1.exp051_top50_trend_quality as exp051


def test_trend_efficiency_rewards_directional_path_and_is_bounded() -> None:
    index = pd.date_range("2024-01-01", periods=400, freq="h")
    straight = 100.0 * np.exp(np.arange(400) * 0.001)
    noisy = 100.0 * np.exp(np.sin(np.arange(400) * np.pi / 2.0) * 0.02)
    close = pd.DataFrame({"straight": straight, "noisy": noisy}, index=index)

    quality = exp051.trend_efficiency(close)

    assert quality.min().min() >= 0.0
    assert quality.max().max() <= 1.0
    assert quality.iloc[-1]["straight"] > quality.iloc[-1]["noisy"]
    assert quality.iloc[-1]["straight"] > 0.99


def test_trend_efficiency_is_causal_under_future_mutation() -> None:
    index = pd.date_range("2024-01-01", periods=400, freq="h")
    close = pd.DataFrame(
        {
            "A": 100.0 * np.exp(np.arange(400) * 0.001),
            "B": 100.0 * np.exp(np.sin(np.arange(400) / 10.0) * 0.05),
        },
        index=index,
    )
    cutoff = index[300]
    changed = close.copy()
    changed.loc[changed.index > cutoff] *= np.linspace(
        0.2,
        4.0,
        int((changed.index > cutoff).sum()),
    )[:, None]

    assert_frame_equal(
        exp051.trend_efficiency(close).loc[:cutoff],
        exp051.trend_efficiency(changed).loc[:cutoff],
    )


def test_quality_buckets_select_opposite_quartiles_at_unit_gross() -> None:
    index = pd.date_range("2025-01-01", periods=49, freq="h")
    columns = [f"S{i}" for i in range(8)]
    direction = pd.DataFrame(1.0, index=index, columns=columns)
    volatility = pd.DataFrame(1.0, index=index, columns=columns)
    quality = pd.DataFrame(
        np.tile(np.arange(8, dtype=float), (len(index), 1)),
        index=index,
        columns=columns,
    )

    high = exp051.build_quality_bucket_targets(
        direction,
        volatility,
        quality,
        "high",
    )
    low = exp051.build_quality_bucket_targets(
        direction,
        volatility,
        quality,
        "low",
    )

    assert high.ne(0.0).sum(axis=1).eq(2).all()
    assert low.ne(0.0).sum(axis=1).eq(2).all()
    assert high[["S6", "S7"]].gt(0.0).all().all()
    assert low[["S0", "S1"]].gt(0.0).all().all()
    assert np.allclose(high.abs().sum(axis=1), 1.0)
    assert np.allclose(low.abs().sum(axis=1), 1.0)


def test_daily_rank_ic_detects_monotonic_quality_return_relation() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    columns = [f"S{i}" for i in range(8)]
    values = np.tile(np.arange(8, dtype=float), (len(index), 1))
    quality = pd.DataFrame(values, index=index, columns=columns)
    labels = pd.DataFrame(values * 0.01, index=index, columns=columns)

    ic = exp051.daily_rank_ic(quality, labels)

    assert len(ic) == 3
    assert np.allclose(ic.to_numpy(), 1.0)
