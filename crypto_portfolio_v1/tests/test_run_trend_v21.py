from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import crypto_portfolio_v1.run_trend_v21 as trend_v21
from crypto_portfolio_v1.run_trend_v21 import (
    TREND_V21_CONFIG,
    _run_fold_backtest,
    _save_combined_trade_analytics,
    _save_fold_trade_analytics,
    build_correlation_clusters,
    calculate_portfolio_diagnostics,
    correlation_aware_target_weights,
    select_validation_profile,
    trailing_correlation_matrix,
)


def _metrics(
    *,
    killed: bool = False,
    median: float = 0.0,
    profitable: float = 0.5,
    mean: float = 0.0,
    mdd: float = -10.0,
) -> dict[str, float | bool]:
    return {
        "killed": killed,
        "median_monthly_return_pct": median,
        "profitable_month_share": profitable,
        "mean_monthly_return_pct": mean,
        "mdd_pct": mdd,
    }


def _correlated_close(rows: int = 225) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h")
    rng = np.random.default_rng(7)
    common = rng.normal(0.0002, 0.006, rows)
    returns = {
        "A": common + rng.normal(0.0, 0.0002, rows),
        "B": common + rng.normal(0.0, 0.0002, rows),
        "C": -common + rng.normal(0.0, 0.0002, rows),
        "D": -common + rng.normal(0.0, 0.0002, rows),
    }
    return pd.DataFrame(
        {
            symbol: 100.0 * np.exp(np.cumsum(values))
            for symbol, values in returns.items()
        },
        index=index,
    )


def test_profile_selection_uses_only_supplied_validation_metrics() -> None:
    assert "test" not in inspect.signature(select_validation_profile).parameters
    validation = {
        "FAST": _metrics(killed=True, median=100.0),
        "BALANCED": _metrics(median=3.0, profitable=0.50, mean=4.0, mdd=-20.0),
        "SLOW": _metrics(median=3.0, profitable=0.75, mean=1.0, mdd=-5.0),
    }

    selected, score = select_validation_profile(validation)

    assert selected == "SLOW"
    assert score[:3] == pytest.approx((1.0, 3.0, 0.75))
    # A hypothetical test winner is deliberately not an input to the selector.
    hypothetical_test = {"FAST": _metrics(median=999.0)}
    assert hypothetical_test["FAST"]["median_monthly_return_pct"] == 999.0
    assert select_validation_profile(validation)[0] == selected


def test_correlation_clusters_use_connected_components() -> None:
    correlation = pd.DataFrame(
        [
            [1.00, 0.80, 0.10, 0.00],
            [0.80, 1.00, -0.76, 0.00],
            [0.10, -0.76, 1.00, 0.20],
            [0.00, 0.00, 0.20, 1.00],
        ],
        index=list("ABCD"),
        columns=list("ABCD"),
    )

    clusters = build_correlation_clusters(correlation, threshold=0.75)

    assert clusters["A"] == clusters["B"] == clusters["C"]
    assert clusters["D"] != clusters["A"]


def test_trailing_correlation_is_causal_under_future_mutation() -> None:
    close = _correlated_close()
    cutoff = close.index[190]
    original = trailing_correlation_matrix(close, cutoff, window_hours=168)
    changed = close.copy()
    future = changed.index > cutoff
    changed.loc[future, "A"] *= np.linspace(0.2, 4.0, future.sum())
    changed.loc[future, "C"] *= np.linspace(5.0, 0.3, future.sum())

    mutated = trailing_correlation_matrix(changed, cutoff, window_hours=168)

    assert original is not None
    assert mutated is not None
    assert_frame_equal(original, mutated)


def test_correlation_aware_weights_enforce_all_caps_and_abstain_if_unknown() -> None:
    close = _correlated_close()
    timestamp = close.index[-1]
    scores = pd.DataFrame(
        [[0.9, 0.8, -0.9, -0.8]],
        index=pd.DatetimeIndex([timestamp]),
        columns=close.columns,
    )
    confidence = pd.DataFrame(1.0, index=scores.index, columns=scores.columns)
    annual_volatility = pd.DataFrame(
        [[0.30, 0.35, 0.30, 0.35]],
        index=scores.index,
        columns=scores.columns,
    )

    weights = correlation_aware_target_weights(
        scores,
        confidence,
        annual_volatility,
        close,
    )
    row = weights.iloc[0]
    correlation = trailing_correlation_matrix(close, timestamp)

    assert correlation is not None
    clusters = build_correlation_clusters(correlation)
    for label in set(clusters.values()):
        members = [symbol for symbol, cluster in clusters.items() if cluster == label]
        assert row.loc[members].abs().sum() <= 0.45 + 1e-12
    assert row.abs().max() <= TREND_V21_CONFIG.max_symbol_weight + 1e-12
    assert row.abs().sum() <= TREND_V21_CONFIG.max_gross_exposure + 1e-12
    assert abs(row.sum()) <= TREND_V21_CONFIG.max_abs_net_exposure + 1e-12
    covariance = (
        np.outer(annual_volatility.iloc[0], annual_volatility.iloc[0])
        * correlation.to_numpy()
    )
    projected_volatility = float(np.sqrt(max(row.to_numpy() @ covariance @ row.to_numpy(), 0.0)))
    assert projected_volatility <= TREND_V21_CONFIG.annual_vol_target + 1e-12

    unknown = close.copy()
    unknown["D"] = 100.0
    abstained = correlation_aware_target_weights(
        scores,
        confidence,
        annual_volatility,
        unknown,
    )
    assert abstained.eq(0.0).all().all()


def test_fold_backtest_always_requests_final_liquidation(monkeypatch) -> None:
    captured: dict[str, object] = {}
    index = pd.date_range("2026-01-01", periods=2, freq="15min")
    empty_equity = pd.DataFrame(index=index)
    empty_weights = pd.DataFrame(0.0, index=index, columns=["BTC"])

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return empty_equity, empty_weights, {}

    monkeypatch.setattr(trend_v21, "run_v2_backtest", fake_run)
    targets = pd.DataFrame(
        {"BTC": [0.2]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01")]),
    )
    opens = pd.DataFrame({"BTC": [100.0, 101.0]}, index=index)
    funding = pd.DataFrame(0.0, index=index, columns=["BTC"])

    _run_fold_backtest(
        targets,
        opens,
        funding,
        config=TREND_V21_CONFIG,
        annual_volatility=pd.DataFrame(),
        fold_start="2026-01-01",
    )

    assert captured["liquidate_at_end"] is True


def test_fold_trade_outputs_are_jsonable_and_all_trades_close(
    tmp_path, monkeypatch
) -> None:
    signal_time = pd.Timestamp("2026-01-01 00:00")
    execution_index = pd.date_range("2026-01-01 00:15", periods=4, freq="15min")
    targets = pd.DataFrame({"BTC": [0.20]}, index=[signal_time])
    opens = pd.DataFrame(
        {"BTC": [100.0, 101.0, 102.0, 103.0]},
        index=execution_index,
    )
    funding = pd.DataFrame(0.0, index=execution_index, columns=["BTC"])
    volatility = pd.DataFrame(
        {"BTC": [0.20]},
        index=[signal_time],
    )
    equity, weights, stats = _run_fold_backtest(
        targets,
        opens,
        funding,
        config=TREND_V21_CONFIG,
        annual_volatility=volatility,
        fold_start=str(signal_time),
    )
    parquet_paths: list[str] = []
    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, *args, **kwargs: parquet_paths.append(str(path)),
    )

    prefix = tmp_path / "fold_one_trend_v21"
    analytics = _save_fold_trade_analytics(
        prefix,
        "fold_one",
        equity,
        weights,
        opens,
        funding,
        TREND_V21_CONFIG,
    )
    saved_summary = json.loads(
        (tmp_path / "fold_one_trend_v21_trade_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert stats["liquidated_at_end"] is True
    assert weights.iloc[-1].eq(0.0).all()
    assert analytics["overall"]["open_trades"] == 0
    assert analytics["trades"]["closed"].all()
    assert saved_summary["open_trades"] == 0
    assert parquet_paths == [str(tmp_path / "fold_one_trend_v21_attribution.parquet")]
    json.dumps(saved_summary)


def test_combined_trade_ledger_scales_to_sequential_stitched_capital(tmp_path) -> None:
    columns = {
        "trade_id": ["trade"],
        "symbol": ["BTC"],
        "side": ["long"],
        "closed": [True],
        "entry_timestamp": [pd.Timestamp("2026-01-01")],
        "exit_timestamp": [pd.Timestamp("2026-01-01 00:15")],
        "duration_hours": [0.25],
        "entry_abs_notional": [1.0],
        "max_abs_notional": [1.0],
        "average_abs_notional": [1.0],
        "gross_market_pnl": [0.1],
        "funding_pnl": [0.0],
        "fee": [0.0],
        "slippage": [0.0],
        "net_pnl": [0.1],
        "return_on_max_notional_pct": [10.0],
        "mfe_pct": [10.0],
        "mae_pct": [0.0],
        "exit_reason": ["end_of_fold"],
    }
    fold_one = pd.DataFrame({**columns, "fold": ["one"]})
    fold_two_columns = dict(columns)
    fold_two_columns["gross_market_pnl"] = [0.2]
    fold_two_columns["net_pnl"] = [0.2]
    fold_two_columns["return_on_max_notional_pct"] = [20.0]
    fold_two_columns["mfe_pct"] = [20.0]
    fold_two_columns["entry_timestamp"] = [pd.Timestamp("2026-01-01 00:30")]
    fold_two_columns["exit_timestamp"] = [pd.Timestamp("2026-01-01 00:45")]
    fold_two = pd.DataFrame({**fold_two_columns, "fold": ["two"]})
    index_one = pd.date_range("2026-01-01", periods=2, freq="15min")
    index_two = pd.date_range("2026-01-01 00:30", periods=2, freq="15min")
    curve_one = pd.DataFrame({"equity": [1.0, 1.1]}, index=index_one)
    curve_two = pd.DataFrame({"equity": [1.0, 1.2]}, index=index_two)
    stitched = pd.DataFrame(
        {"equity": [1.0, 1.1, 1.1, 1.32]},
        index=index_one.append(index_two),
    )

    result = _save_combined_trade_analytics(
        tmp_path,
        "synthetic",
        [{"trades": fold_one}, {"trades": fold_two}],
        [curve_one, curve_two],
        stitched,
    )

    assert result["trades"]["capital_scale"].tolist() == pytest.approx([1.0, 1.1])
    assert result["trades"]["entry_abs_notional"].tolist() == pytest.approx([1.0, 1.1])
    assert result["trades"]["return_on_max_notional_pct"].tolist() == [10.0, 20.0]
    assert result["trades"]["net_pnl"].sum() == pytest.approx(0.32)
    assert abs(result["reconciliation"]["exact_error"]) <= result["reconciliation"][
        "strict_tolerance"
    ]


def test_portfolio_diagnostics_cover_risk_exposure_costs_and_flat_case() -> None:
    index = pd.date_range("2026-01-01", periods=4 * 24 * 3, freq="15min")
    equity = np.concatenate(
        [
            np.linspace(1.0, 1.1, 4 * 24),
            np.linspace(1.1, 0.9, 4 * 24),
            np.linspace(0.9, 1.2, 4 * 24),
        ]
    )
    frame = pd.DataFrame(
        {
            "equity": equity,
            "gross_exposure": 0.5,
            "net_exposure": np.tile([-0.1, 0.1], len(index) // 2),
            "turnover_notional": 0.01,
            "fee": 0.001,
            "slippage": 0.002,
            "funding_pnl": -0.0001,
        },
        index=index,
    )

    metrics = calculate_portfolio_diagnostics(frame)

    assert metrics["maximum_drawdown"] < 0.0
    assert metrics["maximum_drawdown_recovery_timestamp"] is not None
    assert metrics["annualized_volatility"] > 0.0
    assert metrics["daily_cvar_5pct"] <= metrics["daily_var_5pct"] + 1e-12
    assert metrics["average_gross_exposure"] == pytest.approx(0.5)
    assert metrics["average_absolute_net_exposure"] == pytest.approx(0.1)
    assert metrics["total_turnover_notional"] == pytest.approx(len(index) * 0.01)
    assert metrics["total_fees"] == pytest.approx(len(index) * 0.001)

    flat = pd.DataFrame({"equity": 1.0}, index=index)
    flat_metrics = calculate_portfolio_diagnostics(flat)
    assert flat_metrics["annualized_volatility"] == 0.0
    assert np.isnan(flat_metrics["sharpe_0pct_rf"])
    assert np.isnan(flat_metrics["downside_sortino_0pct_rf"])
