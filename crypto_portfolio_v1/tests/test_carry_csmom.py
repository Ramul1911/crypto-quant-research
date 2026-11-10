from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import crypto_portfolio_v1.run_carry_csmom as runner
from crypto_portfolio_v1.experts import (
    FundingCarryExpert,
    SlowCrossSectionalMomentumExpert,
)
from crypto_portfolio_v1.experts.carry_csmom import DEFAULT_FIXED_COST_HURDLE
from crypto_portfolio_v1.v2_config import V2BacktestConfig


def _close_panel(rows: int = 760, symbols: int = 6) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=rows, freq="h")
    columns = [f"S{i}" for i in range(symbols)]
    slopes = np.linspace(-0.002, 0.003, symbols)
    close = pd.DataFrame(
        {
            symbol: 100.0 * np.exp(slopes[position] * np.arange(rows))
            for position, symbol in enumerate(columns)
        },
        index=index,
    )
    return {
        "close": close,
        "funding": pd.DataFrame(np.nan, index=index, columns=columns),
    }


def test_funding_carry_uses_last_known_rate_and_fixed_cost_hurdle() -> None:
    panel = _close_panel(rows=4, symbols=4)
    funding = panel["funding"]
    funding.iloc[1] = [
        DEFAULT_FIXED_COST_HURDLE + 0.0001,
        -DEFAULT_FIXED_COST_HURDLE - 0.0001,
        DEFAULT_FIXED_COST_HURDLE,
        -DEFAULT_FIXED_COST_HURDLE,
    ]

    output = FundingCarryExpert().generate(panel)

    assert output.scores.iloc[0].eq(0.0).all()
    assert output.diagnostics["last_known_funding"].iloc[0].isna().all()
    assert output.scores.iloc[1, 0] < 0.0
    assert output.scores.iloc[1, 1] > 0.0
    assert output.scores.iloc[1, 2:].eq(0.0).all()
    assert output.scores.iloc[3].equals(output.scores.iloc[1])
    assert output.diagnostics["last_known_funding"].iloc[3, 0] == pytest.approx(
        funding.iloc[1, 0]
    )


def test_slow_csmom_is_equal_mean_of_frozen_cross_sectional_zscores() -> None:
    panel = _close_panel()
    output = SlowCrossSectionalMomentumExpert().generate(panel)
    timestamp = panel["close"].index[-1]
    diagnostics = output.diagnostics
    expected = sum(
        diagnostics[f"cross_sectional_zscore_{horizon}h"].loc[timestamp]
        for horizon in (168, 336, 720)
    ) / 3.0

    pd.testing.assert_series_equal(
        output.scores.loc[timestamp],
        expected,
        check_names=False,
    )
    assert set(diagnostics) == {
        "log_return_168h",
        "cross_sectional_zscore_168h",
        "log_return_336h",
        "cross_sectional_zscore_336h",
        "log_return_720h",
        "cross_sectional_zscore_720h",
    }
    assert output.scores.iloc[:720].isna().all().all()


@pytest.mark.parametrize(
    "expert",
    [FundingCarryExpert(), SlowCrossSectionalMomentumExpert()],
)
def test_new_experts_are_causal_under_future_mutation(expert: object) -> None:
    panel = _close_panel()
    panel["funding"].iloc[730] = np.linspace(-0.003, 0.003, 6)
    cutoff = panel["close"].index[735]
    original = expert.generate(panel).scores  # type: ignore[attr-defined]
    changed = {name: frame.copy() for name, frame in panel.items()}
    future = changed["close"].index > cutoff
    changed["close"].loc[future] *= np.linspace(0.2, 3.0, future.sum())[:, None]
    changed["funding"].loc[future] = 0.5

    mutated = expert.generate(changed).scores  # type: ignore[attr-defined]

    assert_frame_equal(original.loc[:cutoff], mutated.loc[:cutoff])


def test_frozen_decision_schedules_and_selection_counts() -> None:
    index = pd.date_range("2026-01-01", periods=49, freq="h")
    columns = [f"S{i}" for i in range(12)]
    row = np.arange(-6, 6, dtype=float)
    scores = pd.DataFrame(np.tile(row, (len(index), 1)), index=index, columns=columns)

    carry = runner.ranked_equal_side_targets(
        scores,
        decision_frequency="8h",
        long_count=2,
        short_count=2,
    )
    csmom = runner.ranked_equal_side_targets(
        scores,
        decision_frequency="daily_utc",
        long_count=5,
        short_count=5,
    )

    assert list(carry.index.hour) == [0, 8, 16, 0, 8, 16, 0]
    assert (carry.gt(0.0).sum(axis=1) == 2).all()
    assert (carry.lt(0.0).sum(axis=1) == 2).all()
    assert list(csmom.index.hour) == [0, 0, 0]
    assert (csmom.gt(0.0).sum(axis=1) == 5).all()
    assert (csmom.lt(0.0).sum(axis=1) == 5).all()
    assert float(carry.abs().max().max()) <= V2BacktestConfig().max_symbol_weight
    assert float(csmom.abs().sum(axis=1).max()) <= V2BacktestConfig().max_gross_exposure


def test_funding_published_on_right_edge_is_known_at_that_decision() -> None:
    index = pd.date_range("2026-01-01 07:00", periods=5, freq="15min")
    columns = ["A"]
    panel = {
        "open": pd.DataFrame([10, 11, 12, 13, 14], index=index, columns=columns),
        "high": pd.DataFrame([11, 12, 13, 14, 15], index=index, columns=columns),
        "low": pd.DataFrame([9, 10, 11, 12, 13], index=index, columns=columns),
        "close": pd.DataFrame([10, 11, 12, 13, 14], index=index, columns=columns),
        "volume": pd.DataFrame(1.0, index=index, columns=columns),
        "funding": pd.DataFrame(
            [np.nan, np.nan, np.nan, np.nan, 0.002],
            index=index,
            columns=columns,
        ),
    }

    hourly = runner.build_hourly_signal_panel(panel)

    assert hourly["close"].loc["2026-01-01 08:00", "A"] == 13
    assert hourly["funding"].loc["2026-01-01 08:00", "A"] == pytest.approx(0.002)


def test_inverted_carry_core_is_exact_target_sign_inversion() -> None:
    panel = _close_panel(rows=25, symbols=8)
    panel["funding"].iloc[0] = np.array(
        [-0.004, -0.003, -0.002, -0.0015, 0.0015, 0.002, 0.003, 0.004]
    )

    targets = runner.build_frozen_arm_targets(
        panel,
        long_count=runner.CORE_LONG_COUNT,
        short_count=runner.CORE_SHORT_COUNT,
    )

    assert_frame_equal(targets["inverted_carry"], -targets["funding_carry"])


def test_frozen_top50_manifest_rejects_symbol_or_window_drift(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = {
        "eligible_symbols": [f"S{i}" for i in range(37)],
        "test_start": runner.TEST_START,
        "test_end_requested": runner.TEST_END,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert len(runner.load_frozen_top50_manifest(path)["eligible_symbols"]) == 37

    manifest["eligible_symbols"].pop()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="37 symbols"):
        runner.load_frozen_top50_manifest(path)


def test_runner_uses_default_v2_config_and_mandatory_liquidation(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    index = pd.date_range("2026-01-01 00:15", periods=2, freq="15min")

    def fake_backtest(*args, **kwargs):
        captured.update(kwargs)
        equity = pd.DataFrame(
            {
                "equity": [1.0, 1.0],
                "gross_exposure": [0.0, 0.0],
                "turnover_notional": [0.0, 0.0],
            },
            index=index,
        )
        weights = pd.DataFrame(0.0, index=index, columns=["A"])
        return equity, weights, {"liquidated_at_end": True}

    monkeypatch.setattr(
        "crypto_portfolio_v1.run_trend_v21.run_v2_backtest",
        fake_backtest,
    )
    targets = pd.DataFrame({"A": [0.2]}, index=[pd.Timestamp("2026-01-01")])
    opens = pd.DataFrame({"A": [100.0, 101.0]}, index=index)
    funding = pd.DataFrame(0.0, index=index, columns=["A"])

    runner._run_fold_backtest(
        targets,
        opens,
        funding,
        config=runner.DEFAULT_CONFIG,
        annual_volatility=pd.DataFrame({"A": [0.2]}, index=targets.index),
        fold_start="2026-01-01",
    )

    assert runner.DEFAULT_CONFIG == V2BacktestConfig()
    assert captured["liquidate_at_end"] is True
