from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_portfolio_v1.trade_analytics import (
    analyze_trades,
    summarize_trade_ledger,
)
from crypto_portfolio_v1.v2_backtest import run_v2_backtest
from crypto_portfolio_v1.v2_config import V2BacktestConfig


def _config(**overrides: float) -> V2BacktestConfig:
    values = {
        "fee_rate": 0.001,
        "slippage_rate": 0.0005,
        "annual_vol_target": 10.0,
        "max_symbol_weight": 2.0,
        "max_gross_exposure": 4.0,
        "max_abs_net_exposure": 4.0,
        "max_cluster_exposure": 4.0,
    }
    values.update(overrides)
    return V2BacktestConfig(**values)


def test_exact_attribution_flip_funding_and_episode_fields() -> None:
    times = pd.date_range("2026-01-01 09:15", periods=3, freq="15min")
    signals = pd.DataFrame(
        {"BTC": [1.0, -1.0, 0.0]},
        index=times - pd.Timedelta(minutes=1),
    )
    prices = pd.DataFrame({"BTC": [100.0, 110.0, 100.0]}, index=times)
    funding = pd.DataFrame({"BTC": [0.0, 0.0, 0.01]}, index=times)
    config = _config()

    equity, weights, _ = run_v2_backtest(
        signals, prices, funding_rates=funding, config=config
    )
    result = analyze_trades(equity, weights, prices, funding, config)
    attribution = result["attribution"]
    trades = result["trades"]

    for column in ("market_pnl", "funding_pnl", "fee", "slippage"):
        actual = attribution.groupby(level="timestamp")[column].sum()
        assert actual.to_numpy() == pytest.approx(equity[column].to_numpy())
    initial = equity.iloc[0]["equity"] - attribution.xs(times[0])["net_pnl"].sum()
    assert equity.iloc[-1]["equity"] == pytest.approx(
        initial + attribution["net_pnl"].sum()
    )
    assert list(trades["side"]) == ["long", "short"]
    assert list(trades["exit_reason"]) == ["flip", "flat"]
    assert trades["closed"].all()
    assert trades.iloc[0]["gross_market_pnl"] > 0
    assert trades.iloc[1]["gross_market_pnl"] > 0
    assert trades.iloc[1]["funding_pnl"] > 0
    assert trades["fee"].sum() == pytest.approx(equity["fee"].sum())
    assert trades["slippage"].sum() == pytest.approx(equity["slippage"].sum())
    assert {"mfe_pct", "mae_pct", "return_on_max_notional_pct"}.issubset(
        trades.columns
    )


def test_trade_summary_metrics_distinguish_trade_and_monthly_win_rate() -> None:
    trades = pd.DataFrame(
        {
            "trade_id": ["a", "b", "c"],
            "fold": ["one", "one", "two"],
            "symbol": ["BTC", "BTC", "ETH"],
            "side": ["long", "short", "long"],
            "closed": [True, True, True],
            "net_pnl": [2.0, -1.0, 0.0],
            "duration_hours": [1.0, 2.0, 3.0],
            "fee": [0.1, 0.1, 0.1],
            "slippage": [0.05, 0.05, 0.05],
            "funding_pnl": [0.0, 0.2, 0.0],
        }
    )
    result = summarize_trade_ledger(trades)
    summary = result["overall"]

    assert summary["trades"] == 3
    assert summary["wins"] == summary["losses"] == summary["breakeven"] == 1
    assert summary["win_rate"] == pytest.approx(1 / 3)
    assert summary["profit_factor"] == pytest.approx(2.0)
    assert summary["payoff_ratio"] == pytest.approx(2.0)
    assert summary["expectancy_per_trade"] == pytest.approx(1 / 3)
    assert summary["average_duration_hours"] == pytest.approx(2.0)
    assert summary["monthly_win_rate"] is None
    assert set(result["by_symbol"].index) == {"BTC", "ETH"}
    assert set(result["by_fold"].index) == {"one", "two"}
    assert np.isfinite(summary["cost_to_gross_profit"])


def test_trade_summary_return_excursion_streak_and_side_metrics() -> None:
    trades = pd.DataFrame(
        {
            "trade_id": list("abcdef"),
            "fold": ["one"] * 6,
            "symbol": ["BTC"] * 6,
            "side": ["long", "short", "long", "short", "long", "short"],
            "entry_timestamp": pd.date_range("2026-01-01", periods=6, freq="h"),
            "closed": [True] * 6,
            "net_pnl": [1.0, 2.0, -1.0, -2.0, -3.0, 4.0],
            "return_on_max_notional_pct": [1.0, 2.0, -1.0, -2.0, -3.0, 4.0],
            "mfe_pct": [2.0, 3.0, 0.5, 0.2, 0.1, 5.0],
            "mae_pct": [-0.1, -0.2, -2.0, -3.0, -4.0, -0.5],
            "duration_hours": [1.0] * 6,
            "exit_reason": ["flat", "flip", "flat", "end_of_fold", "flat", "flip"],
            "fee": [0.0] * 6,
            "slippage": [0.0] * 6,
            "funding_pnl": [0.0] * 6,
        }
    )

    summary = summarize_trade_ledger(trades)["overall"]

    assert summary["net_pnl_total"] == pytest.approx(1.0)
    assert summary["long_trade_count"] == summary["short_trade_count"] == 3
    assert summary["long_net_pnl"] == pytest.approx(-3.0)
    assert summary["short_net_pnl"] == pytest.approx(4.0)
    assert summary["max_consecutive_wins"] == 2
    assert summary["max_consecutive_losses"] == 3
    assert summary["average_return_on_max_notional_pct"] == pytest.approx(1 / 6)
    assert summary["median_return_on_max_notional_pct"] == pytest.approx(0.0)
    assert summary["winning_average_return_pct"] == pytest.approx(7 / 3)
    assert summary["losing_average_return_pct"] == pytest.approx(-2.0)
    assert summary["average_mfe_pct"] == pytest.approx(1.8)
    assert summary["average_mae_pct"] == pytest.approx(-9.8 / 6)
    assert summary["exit_reason_flat_count"] == 3
    assert summary["exit_reason_flip_count"] == 2
    assert summary["exit_reason_end_of_fold_count"] == 1
