from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_portfolio_v1.v2_backtest import (
    constrain_target_weights,
    run_v2_backtest,
    stitch_equity_curves,
)
from crypto_portfolio_v1.v2_config import V2BacktestConfig


def _index(*times: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(time) for time in times])


def _unlimited(**overrides: float) -> V2BacktestConfig:
    values = {
        "fee_rate": 0.0,
        "slippage_rate": 0.0,
        "annual_vol_target": 10.0,
        "max_symbol_weight": 2.0,
        "max_gross_exposure": 4.0,
        "max_abs_net_exposure": 4.0,
        "max_cluster_exposure": 4.0,
    }
    values.update(overrides)
    return V2BacktestConfig(**values)


def test_signal_at_t_fills_only_at_next_available_15m_open() -> None:
    signals = pd.DataFrame(
        {"BTC": [1.0]}, index=_index("2026-01-01 10:00")
    )
    prices = pd.DataFrame(
        {"BTC": [100.0, 110.0, 121.0]},
        index=_index(
            "2026-01-01 10:00",
            "2026-01-01 10:15",
            "2026-01-01 10:30",
        ),
    )

    equity, weights, _ = run_v2_backtest(
        signals, prices, config=_unlimited()
    )

    assert weights.loc["2026-01-01 10:00", "BTC"] == 0.0
    assert weights.loc["2026-01-01 10:15", "BTC"] == pytest.approx(1.0)
    assert equity.loc["2026-01-01 10:15", "market_pnl"] == 0.0
    assert equity.loc["2026-01-01 10:30", "market_pnl"] == pytest.approx(0.1)
    assert equity.loc[
        "2026-01-01 10:15", "executed_signal_timestamp"
    ] == pd.Timestamp("2026-01-01 10:00")


def test_sparse_execution_uses_first_strictly_later_open() -> None:
    signals = pd.DataFrame(
        {"BTC": [1.0]}, index=_index("2026-01-01 10:00")
    )
    prices = pd.DataFrame(
        {"BTC": [100.0, 100.0]},
        index=_index("2026-01-01 10:00", "2026-01-01 10:30"),
    )
    equity, _, _ = run_v2_backtest(signals, prices, config=_unlimited())
    assert pd.isna(equity.iloc[0]["executed_signal_timestamp"])
    assert equity.iloc[1]["executed_signal_timestamp"] == signals.index[0]


def test_fee_slippage_turnover_and_accounting_are_exact() -> None:
    signals = pd.DataFrame(
        {"BTC": [1.0, 0.0]},
        index=_index("2026-01-01 09:00", "2026-01-01 10:00"),
    )
    prices = pd.DataFrame(
        {"BTC": [100.0, 100.0]},
        index=_index("2026-01-01 09:15", "2026-01-01 10:15"),
    )
    config = _unlimited(fee_rate=0.01, slippage_rate=0.005)

    equity, _, stats = run_v2_backtest(signals, prices, config=config)

    assert equity["turnover_notional"].sum() > 1.9
    assert equity["fee"].sum() == pytest.approx(
        equity["turnover_notional"].sum() * config.fee_rate
    )
    assert equity["slippage"].sum() == pytest.approx(
        equity["turnover_notional"].sum() * config.slippage_rate
    )
    expected = 1.0 + equity["market_pnl"].sum() + equity["funding_pnl"].sum()
    expected -= equity["fee"].sum() + equity["slippage"].sum()
    assert equity.iloc[-1]["equity"] == pytest.approx(expected)
    assert stats["max_abs_accounting_error"] < 1e-10


def test_optional_final_liquidation_closes_and_charges_exact_costs() -> None:
    signals = pd.DataFrame(
        {"BTC": [1.0]}, index=_index("2026-01-01 09:00")
    )
    prices = pd.DataFrame(
        {"BTC": [100.0, 110.0]},
        index=_index("2026-01-01 09:15", "2026-01-01 09:30"),
    )
    config = _unlimited(fee_rate=0.01, slippage_rate=0.005)

    equity, weights, stats = run_v2_backtest(
        signals, prices, config=config, liquidate_at_end=True
    )

    assert weights.iloc[-1]["BTC"] == 0.0
    assert equity.iloc[-1]["end_liquidation"]
    assert stats["liquidated_at_end"]
    assert equity.iloc[-1]["turnover_notional"] > 1.0
    assert equity.iloc[-1]["fee"] == pytest.approx(
        equity.iloc[-1]["turnover_notional"] * config.fee_rate
    )
    assert equity.iloc[-1]["slippage"] == pytest.approx(
        equity.iloc[-1]["turnover_notional"] * config.slippage_rate
    )
    assert equity["accounting_error"].abs().max() < 1e-10


@pytest.mark.parametrize(
    ("weight", "expected_sign"),
    [(1.0, -1.0), (-1.0, 1.0)],
)
def test_positive_funding_is_paid_by_longs_and_received_by_shorts(
    weight: float, expected_sign: float
) -> None:
    signals = pd.DataFrame(
        {"BTC": [weight]}, index=_index("2026-01-01 09:00")
    )
    prices = pd.DataFrame(
        {"BTC": [100.0, 100.0]},
        index=_index("2026-01-01 09:15", "2026-01-01 09:30"),
    )
    funding = pd.DataFrame(
        {"BTC": [0.0, 0.01]}, index=prices.index
    )

    equity, _, _ = run_v2_backtest(
        signals, prices, funding_rates=funding, config=_unlimited()
    )

    assert np.sign(equity.iloc[-1]["funding_pnl"]) == expected_sign
    assert equity.iloc[-1]["funding_pnl"] == pytest.approx(-weight * 0.01)


def test_all_portfolio_constraints_hold() -> None:
    symbols = ["BTC", "ETH", "SOL", "DOGE"]
    target = pd.Series([0.8, 0.8, -0.8, -0.8], index=symbols)
    vol = pd.Series(0.80, index=symbols)
    corr = pd.DataFrame(0.0, index=symbols, columns=symbols)
    np.fill_diagonal(corr.values, 1.0)
    config = V2BacktestConfig(
        annual_vol_target=0.20,
        max_symbol_weight=0.25,
        max_gross_exposure=0.70,
        max_abs_net_exposure=0.10,
        max_cluster_exposure=0.30,
        correlation_clusters={
            "BTC": "large",
            "ETH": "large",
            "SOL": "alts",
            "DOGE": "alts",
        },
    )

    result = constrain_target_weights(
        target, config=config, annual_vol=vol, correlation=corr
    )

    assert result.abs().max() <= config.max_symbol_weight + 1e-12
    assert result.abs().sum() <= config.max_gross_exposure + 1e-12
    assert abs(result.sum()) <= config.max_abs_net_exposure + 1e-12
    assert result.loc[["BTC", "ETH"]].abs().sum() <= 0.30 + 1e-12
    projected_vol = np.sqrt(
        result.to_numpy() @ (np.eye(4) * 0.80**2) @ result.to_numpy()
    )
    assert projected_vol <= config.annual_vol_target + 1e-12


def test_terminal_drawdown_closes_positions_and_reports_time_to_kill() -> None:
    signals = pd.DataFrame(
        {"BTC": [1.0]}, index=_index("2026-01-01 09:00")
    )
    prices = pd.DataFrame(
        {"BTC": [100.0, 70.0, 200.0]},
        index=_index(
            "2026-01-01 09:15",
            "2026-01-01 09:30",
            "2026-01-01 09:45",
        ),
    )

    equity, weights, stats = run_v2_backtest(
        signals, prices, config=_unlimited(), fold_start="2026-01-01 09:00"
    )

    assert stats["killed"] is True
    assert stats["kill_timestamp"] == "2026-01-01 09:30:00"
    assert stats["time_to_kill_hours"] == pytest.approx(0.5)
    assert weights.loc["2026-01-01 09:30", "BTC"] == 0.0
    assert weights.loc["2026-01-01 09:45", "BTC"] == 0.0
    assert equity.iloc[-1]["equity"] == pytest.approx(0.70)
    assert equity["accounting_error"].abs().max() < 1e-10


def test_calendar_active_months_and_continuous_stitching() -> None:
    signals = pd.DataFrame(
        {"BTC": [1.0]}, index=_index("2026-02-01 00:00")
    )
    prices = pd.DataFrame(
        {"BTC": [100.0, 100.0, 110.0]},
        index=_index(
            "2026-01-31 23:45",
            "2026-02-01 00:15",
            "2026-02-01 00:30",
        ),
    )
    equity, _, stats = run_v2_backtest(
        signals, prices, config=_unlimited()
    )

    assert list(stats["calendar_monthly_returns"].index.astype(str)) == [
        "2026-01",
        "2026-02",
    ]
    assert list(stats["active_monthly_returns"].index.astype(str)) == ["2026-02"]

    fold_one = equity.iloc[:1].copy()
    fold_two = equity.iloc[1:].copy()
    fold_two["equity"] /= fold_two["equity"].iloc[0]
    stitched = stitch_equity_curves([fold_one, fold_two], ["one", "two"])
    assert stitched.index.is_monotonic_increasing
    assert stitched.loc[fold_two.index[0], "equity"] == pytest.approx(
        fold_one["equity"].iloc[-1]
    )
    assert set(stitched["fold"]) == {"one", "two"}
