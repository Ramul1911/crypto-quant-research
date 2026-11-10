from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .config import StrategyConfig

EPS = 1e-12


def _allocate_side(
    symbols: list[str],
    annual_vol: pd.Series,
    gross: float,
    direction: float,
    max_symbol_weight: float,
) -> pd.Series:
    if not symbols or gross <= 0:
        return pd.Series(dtype=float)
    inverse_vol = 1.0 / annual_vol.loc[symbols].clip(lower=0.20)
    weights = inverse_vol / inverse_vol.sum() * gross
    weights = weights.clip(upper=max_symbol_weight)
    return direction * weights


def target_weights(
    score: pd.Series,
    annual_vol: pd.Series,
    dollar_volume_24h: pd.Series,
    market_trend: float,
    config: StrategyConfig,
) -> pd.Series:
    eligible = (
        score.notna()
        & annual_vol.notna()
        & dollar_volume_24h.notna()
        & (dollar_volume_24h >= 10_000_000.0)
    )
    valid_score = score[eligible]
    longs = (
        valid_score[valid_score > 0]
        .nlargest(config.long_count)
        .index.tolist()
    )
    shorts = (
        valid_score[valid_score < 0]
        .nsmallest(config.short_count)
        .index.tolist()
    )

    if market_trend > 0:
        long_gross, short_gross = 0.90, 0.60
    elif market_trend < 0:
        long_gross, short_gross = 0.60, 0.90
    else:
        long_gross = short_gross = 0.75

    long_weights = _allocate_side(
        longs,
        annual_vol,
        long_gross,
        1.0,
        config.max_symbol_weight,
    )
    short_weights = _allocate_side(
        shorts,
        annual_vol,
        short_gross,
        -1.0,
        config.max_symbol_weight,
    )
    weights = pd.concat([long_weights, short_weights])
    weights = weights.groupby(level=0).sum()

    if weights.empty:
        return weights
    projected_vol = float(
        np.sqrt(np.sum((weights * annual_vol.loc[weights.index]) ** 2))
    )
    vol_scale = float(
        np.clip(config.annual_vol_target / (projected_vol + EPS), 0.35, 1.50)
    )
    leverage_cap = config.gross_leverage / (weights.abs().sum() + EPS)
    weights *= min(vol_scale, leverage_cap)
    return weights.clip(
        lower=-config.max_symbol_weight,
        upper=config.max_symbol_weight,
    )


def risk_multiplier(drawdown: float, config: StrategyConfig) -> float:
    if drawdown <= -config.max_drawdown_flat:
        return 0.0
    if drawdown <= -config.max_drawdown_soft_3:
        return 0.25
    if drawdown <= -config.max_drawdown_soft_2:
        return 0.50
    if drawdown <= -config.max_drawdown_soft_1:
        return 0.70
    return 1.0


def run_backtest(
    features: dict[str, pd.DataFrame | pd.Series],
    config: StrategyConfig = StrategyConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    returns = features["returns"]
    score = features["score"]
    annual_vol = features["annual_vol"]
    dollar_volume = features["dollar_volume_24h"]
    funding = features["funding"]
    market_trend = features["market_trend"]
    assert isinstance(returns, pd.DataFrame)
    assert isinstance(score, pd.DataFrame)
    assert isinstance(annual_vol, pd.DataFrame)
    assert isinstance(dollar_volume, pd.DataFrame)
    assert isinstance(funding, pd.DataFrame)
    assert isinstance(market_trend, pd.Series)

    symbols = list(returns.columns)
    index = returns.index
    current = pd.Series(0.0, index=symbols)
    equity = 1.0
    peak = 1.0
    killed = False
    total_turnover = 0.0
    total_cost = 0.0
    equity_rows: list[dict[str, Any]] = []
    weight_rows: list[pd.Series] = []
    first_valid = config.min_history_hours

    for i, timestamp in enumerate(index):
        hourly_return = returns.iloc[i].fillna(0.0)
        funding_rate = funding.iloc[i].fillna(0.0)
        market_pnl = float((current * hourly_return).sum())
        funding_pnl = float((-current * funding_rate).sum())
        equity *= max(1.0 + market_pnl + funding_pnl, 1e-8)
        peak = max(peak, equity)
        drawdown = equity / (peak + EPS) - 1.0

        turnover = 0.0
        trading_cost = 0.0
        should_rebalance = (
            i >= first_valid
            and i % config.rebalance_every_hours == 0
        )
        if drawdown <= -config.max_drawdown_flat:
            killed = True

        if killed and current.abs().sum() > 0:
            target = pd.Series(0.0, index=symbols)
            turnover = float((target - current).abs().sum())
            trading_cost = turnover * (
                config.fee_rate + config.slippage_rate
            )
            equity *= max(1.0 - trading_cost, 1e-8)
            current = target
        elif should_rebalance and not killed:
            proposed = target_weights(
                score.iloc[i],
                annual_vol.iloc[i],
                dollar_volume.iloc[i],
                float(market_trend.iloc[i]),
                config,
            )
            target = pd.Series(0.0, index=symbols)
            target.loc[proposed.index] = proposed
            target *= risk_multiplier(drawdown, config)
            turnover = float((target - current).abs().sum())
            trading_cost = turnover * (
                config.fee_rate + config.slippage_rate
            )
            equity *= max(1.0 - trading_cost, 1e-8)
            current = target

        total_turnover += turnover
        total_cost += trading_cost
        peak = max(peak, equity)
        drawdown = equity / (peak + EPS) - 1.0
        equity_rows.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "drawdown": drawdown,
                "gross_exposure": float(current.abs().sum()),
                "net_exposure": float(current.sum()),
                "turnover": turnover,
                "trading_cost": trading_cost,
                "market_pnl": market_pnl,
                "funding_pnl": funding_pnl,
                "killed": killed,
            }
        )
        row = current.copy()
        row.name = timestamp
        weight_rows.append(row)

    equity_frame = pd.DataFrame(equity_rows).set_index("timestamp")
    weights_frame = pd.DataFrame(weight_rows)
    monthly_equity = equity_frame["equity"].groupby(
        equity_frame.index.to_period("M")
    ).last()
    monthly_returns = monthly_equity.pct_change()
    if len(monthly_returns):
        first_month = monthly_equity.index[0]
        first_rows = equity_frame[
            equity_frame.index.to_period("M") == first_month
        ]
        monthly_returns.loc[first_month] = (
            first_rows["equity"].iloc[-1] / first_rows["equity"].iloc[0] - 1.0
        )

    stats = {
        "config": asdict(config),
        "start": str(equity_frame.index.min()),
        "end": str(equity_frame.index.max()),
        "return_pct": (equity - 1.0) * 100.0,
        "mdd_pct": float(equity_frame["drawdown"].min()) * 100.0,
        "total_turnover": total_turnover,
        "cost_drag_simple_pct": total_cost * 100.0,
        "killed": killed,
        "mean_monthly_return_pct": float(monthly_returns.mean()) * 100.0,
        "median_monthly_return_pct": float(monthly_returns.median()) * 100.0,
        "profitable_month_share": float((monthly_returns > 0).mean()),
        "months_ge_10pct": int((monthly_returns >= 0.10).sum()),
        "months_le_minus_10pct": int((monthly_returns <= -0.10).sum()),
        "monthly_returns_pct": {
            str(period): float(value) * 100.0
            for period, value in monthly_returns.items()
        },
    }
    return equity_frame, weights_frame, stats

