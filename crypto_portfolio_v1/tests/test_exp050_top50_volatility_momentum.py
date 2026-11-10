from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import crypto_portfolio_v1.exp050_top50_volatility_momentum as exp050


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=49, freq="h")
    columns = [f"S{i}" for i in range(8)]
    direction = pd.DataFrame(1.0, index=index, columns=columns)
    volatility = pd.DataFrame(
        np.tile(np.arange(1.0, 9.0), (len(index), 1)),
        index=index,
        columns=columns,
    )
    return direction, volatility


def test_high_and_low_buckets_use_causal_cross_sectional_quartiles() -> None:
    direction, volatility = _panels()

    high = exp050.build_volatility_bucket_targets(direction, volatility, "high")
    low = exp050.build_volatility_bucket_targets(direction, volatility, "low")

    assert high.ne(0.0).sum(axis=1).eq(2).all()
    assert low.ne(0.0).sum(axis=1).eq(2).all()
    assert high[["S6", "S7"]].gt(0.0).all().all()
    assert low[["S0", "S1"]].gt(0.0).all().all()
    assert np.allclose(high.abs().sum(axis=1), 1.0)
    assert np.allclose(low.abs().sum(axis=1), 1.0)


def test_bucket_targets_do_not_change_before_future_volatility_mutation() -> None:
    direction, volatility = _panels()
    cutoff = volatility.index[30]
    changed = volatility.copy()
    changed.loc[changed.index > cutoff, "S0"] = 1000.0

    original = exp050.build_volatility_bucket_targets(direction, volatility, "high")
    mutated = exp050.build_volatility_bucket_targets(direction, changed, "high")

    assert_frame_equal(original.loc[:cutoff], mutated.loc[:cutoff])
