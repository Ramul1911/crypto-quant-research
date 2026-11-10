from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from crypto_portfolio_v1.experts import FundingExpert, ShockExpert, TrendExpert
from crypto_portfolio_v1.v2_features import V2_BAR_FIELDS, build_v2_feature_panel


def _synthetic_panel(rows: int = 260) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=rows, freq="h")
    symbols = ["UP", "DOWN"]
    up = 100.0 * np.exp(np.arange(rows) * 0.002)
    down = 100.0 * np.exp(-np.arange(rows) * 0.002)
    close = pd.DataFrame({"UP": up, "DOWN": down}, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.DataFrame(
        np.maximum(open_, close) * 1.005,
        index=index,
        columns=symbols,
    )
    low = pd.DataFrame(
        np.minimum(open_, close) * 0.995,
        index=index,
        columns=symbols,
    )
    quote_volume = pd.DataFrame(1_000_000.0, index=index, columns=symbols)
    trades = pd.DataFrame(10_000.0, index=index, columns=symbols)
    panel = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": quote_volume / close,
        "quote_volume": quote_volume,
        "trades": trades,
        "taker_buy_base": 0.5 * quote_volume / close,
        "taker_buy_quote": 0.5 * quote_volume,
        "funding": pd.DataFrame(0.0, index=index, columns=symbols),
    }
    return panel


def test_feature_panel_alignment_and_no_funding_backfill() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="h")
    bars_by_symbol = {}
    for offset, symbol in enumerate(("A", "B")):
        close = pd.Series(100.0 + offset + np.arange(len(index)), index=index)
        bars_by_symbol[symbol] = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 10.0,
                "quote_volume": 1_000.0,
                "trades": 20.0,
                "taker_buy_base": 4.0,
                "taker_buy_quote": 450.0,
            }
        )
    funding = {
        "A": pd.DataFrame(
            {"funding_rate": [0.001, 0.002]},
            index=index[[2, 5]],
        )
    }

    panel = build_v2_feature_panel(
        bars_by_symbol,
        funding,
        symbols=("A", "B"),
    )

    expected_fields = set(V2_BAR_FIELDS) | {"funding"}
    assert set(panel) == expected_fields
    for frame in panel.values():
        assert frame.index.equals(index)
        assert list(frame.columns) == ["A", "B"]
    assert panel["funding"].loc[index[0], "A"] != panel["funding"].loc[index[0], "A"]
    assert panel["funding"].loc[index[4], "A"] == pytest.approx(0.001)
    assert panel["funding"].loc[index[5], "A"] == pytest.approx(0.002)
    assert panel["funding"]["B"].isna().all()


def test_trend_expected_signs_and_output_shape() -> None:
    panel = _synthetic_panel()
    output = TrendExpert().predict(panel)

    assert output.scores.shape == panel["close"].shape
    assert output.confidence.shape == panel["close"].shape
    assert output.scores.iloc[-1]["UP"] > 0.0
    assert output.scores.iloc[-1]["DOWN"] < 0.0
    assert output.confidence.iloc[-1].between(0.0, 1.0).all()


def test_funding_only_fades_cost_effective_extremes() -> None:
    panel = _synthetic_panel()
    panel["funding"].iloc[-1, 0] = 0.01
    panel["funding"].iloc[-1, 1] = -0.01
    output = FundingExpert().predict(panel)

    assert (output.scores.iloc[:-1] == 0.0).all().all()
    assert output.scores.iloc[-1]["UP"] < 0.0
    assert output.scores.iloc[-1]["DOWN"] > 0.0
    assert output.diagnostics["extreme"].iloc[-1].eq(1.0).all()


def test_shock_reverses_crash_and_squeeze() -> None:
    panel = _synthetic_panel()
    last = panel["close"].index[-1]
    panel["open"].loc[last] = 100.0
    panel["close"].loc[last, "UP"] = 65.0
    panel["close"].loc[last, "DOWN"] = 145.0
    panel["high"].loc[last] = [102.0, 150.0]
    panel["low"].loc[last] = [60.0, 98.0]
    panel["quote_volume"].loc[last] = 20_000_000.0
    panel["trades"].loc[last] = 200_000.0
    panel["taker_buy_quote"].loc[last] = [1_000_000.0, 19_000_000.0]

    output = ShockExpert().predict(panel)

    assert output.scores.loc[last, "UP"] > 0.0
    assert output.scores.loc[last, "DOWN"] < 0.0
    assert output.diagnostics["shock_proxy"].loc[last].gt(3.0).all()


@pytest.mark.parametrize("expert", [TrendExpert(), FundingExpert(), ShockExpert()])
def test_experts_are_causal_under_future_mutation(expert: object) -> None:
    panel = _synthetic_panel(300)
    panel["funding"].iloc[210, 0] = 0.01
    panel["funding"].iloc[210, 1] = -0.01
    cutoff = panel["close"].index[220]
    original = expert.predict(panel).scores  # type: ignore[attr-defined]

    mutated = {name: frame.copy() for name, frame in panel.items()}
    future = mutated["close"].index > cutoff
    for name, frame in mutated.items():
        if name in {"high", "quote_volume", "trades", "taker_buy_quote"}:
            frame.loc[future] = frame.loc[future] * 100.0 + 1.0
        elif name == "low":
            frame.loc[future] = frame.loc[future] * 0.01
        else:
            frame.loc[future] = frame.loc[future] * -7.0 - 13.0
    changed = expert.predict(mutated).scores  # type: ignore[attr-defined]

    assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])
