from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v2_config import V2BacktestConfig

EPS = 1e-12


def drawdown_multiplier(drawdown: float, config: V2BacktestConfig) -> float:
    """Return the risk budget for a non-positive drawdown."""
    depth = max(0.0, -float(drawdown))
    if depth >= config.terminal_drawdown:
        return 0.0
    if depth >= config.drawdown_level_3:
        return config.drawdown_multiplier_3
    if depth >= config.drawdown_level_2:
        return config.drawdown_multiplier_2
    if depth >= config.drawdown_level_1:
        return config.drawdown_multiplier_1
    return 1.0


def _reduce_net(weights: pd.Series, net_cap: float) -> pd.Series:
    result = weights.copy()
    net = float(result.sum())
    if abs(net) <= net_cap + EPS:
        return result
    if net > net_cap:
        positive = result.clip(lower=0.0)
        excess = net - net_cap
        if positive.sum() > 0:
            result.loc[positive > 0] *= max(
                0.0, 1.0 - excess / float(positive.sum())
            )
    else:
        negative = -result.clip(upper=0.0)
        excess = -net - net_cap
        if negative.sum() > 0:
            result.loc[negative > 0] *= max(
                0.0, 1.0 - excess / float(negative.sum())
            )
    return result


def constrain_target_weights(
    target: pd.Series | Mapping[str, float],
    config: V2BacktestConfig = V2BacktestConfig(),
    annual_vol: pd.Series | Mapping[str, float] | None = None,
    correlation: pd.DataFrame | None = None,
    clusters: Mapping[str, str] | None = None,
) -> pd.Series:
    """Apply symbol, cluster, net, gross and ex-ante volatility limits.

    ``annual_vol`` and ``correlation`` must be information available at the
    signal timestamp. Correlation is interpreted as a correlation matrix.
    """
    weights = pd.Series(target, dtype=float).replace([np.inf, -np.inf], np.nan)
    weights = weights.fillna(0.0)
    if not weights.index.is_unique:
        raise ValueError("target symbols must be unique")
    weights = weights.clip(
        lower=-config.max_symbol_weight, upper=config.max_symbol_weight
    )

    cluster_map = config.correlation_clusters if clusters is None else clusters
    if cluster_map:
        labels = pd.Series(
            {symbol: cluster_map.get(symbol, symbol) for symbol in weights.index}
        )
        for label in labels.unique():
            members = labels.index[labels == label]
            exposure = float(weights.loc[members].abs().sum())
            if exposure > config.max_cluster_exposure + EPS:
                weights.loc[members] *= config.max_cluster_exposure / exposure

    weights = _reduce_net(weights, config.max_abs_net_exposure)
    gross = float(weights.abs().sum())
    if gross > config.max_gross_exposure + EPS:
        weights *= config.max_gross_exposure / gross

    if annual_vol is not None and config.annual_vol_target > 0:
        vol = pd.Series(annual_vol, dtype=float).reindex(weights.index)
        if (vol.dropna() < 0).any():
            raise ValueError("annual volatility cannot be negative")
        usable = vol.notna() & (vol > 0)
        weights.loc[~usable] = 0.0
        if usable.any():
            used_symbols = weights.index[usable]
            if correlation is None:
                covariance = np.diag(np.square(vol.loc[used_symbols].to_numpy()))
            else:
                corr = correlation.reindex(
                    index=used_symbols, columns=used_symbols
                )
                if corr.isna().any().any():
                    raise ValueError("correlation matrix is missing target symbols")
                covariance = (
                    np.outer(
                        vol.loc[used_symbols].to_numpy(),
                        vol.loc[used_symbols].to_numpy(),
                    )
                    * corr.to_numpy(dtype=float)
                )
            vector = weights.loc[used_symbols].to_numpy(dtype=float)
            variance = float(vector @ covariance @ vector)
            if variance < -1e-10:
                raise ValueError("correlation matrix produces negative variance")
            projected_vol = float(np.sqrt(max(variance, 0.0)))
            if projected_vol > config.annual_vol_target + EPS:
                weights *= config.annual_vol_target / projected_vol

    # Scaling above cannot violate an earlier cap, but roundoff should not leak.
    weights[np.abs(weights) < EPS] = 0.0
    return weights


def _validate_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy().astype(float)
    if not isinstance(result.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    if not result.index.is_monotonic_increasing or not result.index.is_unique:
        raise ValueError(f"{name} index must be sorted and unique")
    if not result.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    return result


def _information_at(
    values: pd.DataFrame | pd.Series | Mapping[str, float] | None,
    timestamp: pd.Timestamp,
) -> pd.Series | Mapping[str, float] | None:
    if values is None or not isinstance(values, (pd.DataFrame, pd.Series)):
        return values
    if isinstance(values.index, pd.DatetimeIndex):
        location = values.index.searchsorted(timestamp, side="right") - 1
        if location < 0:
            return None
        row = values.iloc[location]
        return row if isinstance(row, pd.Series) else values
    return values


def _trade_to_target(
    equity: float,
    notionals: pd.Series,
    target_weights: pd.Series,
    config: V2BacktestConfig,
) -> tuple[float, pd.Series, float, float, float]:
    """Trade exact dollar notionals and solve the small cost/equity fixed point."""
    rate = config.transaction_cost_rate
    post_cost_equity = equity
    for _ in range(100):
        target_notionals = target_weights * post_cost_equity
        traded = float((target_notionals - notionals).abs().sum())
        updated = equity - traded * rate
        if abs(updated - post_cost_equity) <= config.accounting_tolerance:
            post_cost_equity = updated
            break
        post_cost_equity = updated
    else:
        raise RuntimeError("transaction-cost accounting did not converge")
    if post_cost_equity <= 0:
        raise RuntimeError("transaction costs exhausted portfolio equity")
    target_notionals = target_weights * post_cost_equity
    traded = float((target_notionals - notionals).abs().sum())
    fee = traded * config.fee_rate
    slippage = traded * config.slippage_rate
    exact_equity = equity - fee - slippage
    if abs(exact_equity - post_cost_equity) > 10 * config.accounting_tolerance:
        raise AssertionError("trade-cost fixed point violated")
    return exact_equity, target_notionals, traded, fee, slippage


def _monthly_returns(
    equity_frame: pd.DataFrame, initial_equity: float
) -> tuple[pd.Series, pd.Series]:
    periods = equity_frame.index.to_period("M")
    values: dict[pd.Period, float] = {}
    prior = initial_equity
    for period, group in equity_frame.groupby(periods, sort=True):
        end = float(group["equity"].iloc[-1])
        values[period] = end / prior - 1.0
        prior = end
    calendar = pd.Series(values, dtype=float)
    active_periods = equity_frame.loc[
        (equity_frame["gross_exposure"] > EPS)
        | (equity_frame["turnover_notional"] > EPS)
    ].index.to_period("M").unique()
    active = calendar.loc[calendar.index.isin(active_periods)]
    return calendar, active


def run_v2_backtest(
    signal_weights: pd.DataFrame,
    open_prices_15m: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    config: V2BacktestConfig = V2BacktestConfig(),
    annual_vol: pd.DataFrame | pd.Series | Mapping[str, float] | None = None,
    correlation: pd.DataFrame | None = None,
    clusters: Mapping[str, str] | None = None,
    initial_equity: float = 1.0,
    fold_start: pd.Timestamp | str | None = None,
    liquidate_at_end: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Simulate target weights using strictly later 15-minute opens.

    Signals are decisions stamped at completed hourly candles. A decision at
    ``t`` is eligible only at the first available execution open strictly after
    ``t``. Prices are open-to-open and funding uses the perpetual convention:
    positive funding is paid by longs and received by shorts.
    """
    signals = _validate_frame(signal_weights, "signal_weights")
    prices = _validate_frame(open_prices_15m, "open_prices_15m")
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if prices.empty:
        raise ValueError("open_prices_15m cannot be empty")
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("execution prices must be finite and positive")
    symbols = list(prices.columns)
    unknown = signals.columns.difference(symbols)
    if len(unknown):
        raise ValueError(f"signals contain symbols without prices: {list(unknown)}")
    signals = signals.reindex(columns=symbols, fill_value=0.0)

    funding = (
        pd.DataFrame(0.0, index=prices.index, columns=symbols)
        if funding_rates is None
        else _validate_frame(funding_rates, "funding_rates")
        .reindex(index=prices.index, columns=symbols)
        .fillna(0.0)
    )

    scheduled: dict[int, tuple[pd.Timestamp, pd.Series]] = {}
    for timestamp, target in signals.iterrows():
        fill_index = int(prices.index.searchsorted(timestamp, side="right"))
        if fill_index < len(prices.index):
            # If several signals map to one sparse execution open, only the
            # latest information available at that open is actionable.
            scheduled[fill_index] = (timestamp, target)
    for fill_index, (signal_time, _) in scheduled.items():
        if not signal_time < prices.index[fill_index]:
            raise AssertionError("lookahead detected in execution schedule")

    equity = float(initial_equity)
    peak = equity
    notionals = pd.Series(0.0, index=symbols)
    base_target = pd.Series(0.0, index=symbols)
    previous_price: pd.Series | None = None
    previous_equity = equity
    killed = False
    kill_timestamp: pd.Timestamp | None = None
    prior_risk_multiplier = 1.0
    equity_rows: list[dict[str, Any]] = []
    weight_rows: list[pd.Series] = []

    for i, (timestamp, price) in enumerate(prices.iterrows()):
        market_pnl = 0.0
        if previous_price is not None:
            asset_return = price / previous_price - 1.0
            market_pnl = float((notionals * asset_return).sum())
            notionals = notionals * (1.0 + asset_return)
        funding_pnl = float((-notionals * funding.loc[timestamp]).sum())
        equity += market_pnl + funding_pnl
        if equity <= 0:
            raise RuntimeError("portfolio equity became non-positive")
        peak = max(peak, equity)
        pretrade_drawdown = equity / peak - 1.0
        risk_mult = drawdown_multiplier(pretrade_drawdown, config)

        turnover_notional = fee = slippage = 0.0
        end_liquidation = False
        executed_signal: pd.Timestamp | None = None
        terminal_now = risk_mult == 0.0
        target_to_trade: pd.Series | None = None

        if terminal_now and not killed:
            killed = True
            kill_timestamp = timestamp
            target_to_trade = pd.Series(0.0, index=symbols)
        elif not killed and i in scheduled:
            signal_time, raw_target = scheduled[i]
            signal_vol = _information_at(annual_vol, signal_time)
            base_target = constrain_target_weights(
                raw_target,
                config=config,
                annual_vol=signal_vol,
                correlation=correlation,
                clusters=clusters,
            ).reindex(symbols, fill_value=0.0)
            target_to_trade = base_target * risk_mult
            executed_signal = signal_time
        elif not killed and abs(risk_mult - prior_risk_multiplier) > EPS:
            target_to_trade = base_target * risk_mult

        if target_to_trade is not None:
            equity, notionals, traded, trade_fee, trade_slippage = _trade_to_target(
                equity, notionals, target_to_trade, config
            )
            turnover_notional += traded
            fee += trade_fee
            slippage += trade_slippage

        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        if not killed and drawdown <= -config.terminal_drawdown:
            killed = True
            kill_timestamp = timestamp
            equity, notionals, traded, trade_fee, trade_slippage = _trade_to_target(
                equity, notionals, pd.Series(0.0, index=symbols), config
            )
            turnover_notional += traded
            fee += trade_fee
            slippage += trade_slippage
            drawdown = equity / peak - 1.0

        if liquidate_at_end and i == len(prices.index) - 1 and notionals.abs().sum() > EPS:
            equity, notionals, traded, trade_fee, trade_slippage = _trade_to_target(
                equity, notionals, pd.Series(0.0, index=symbols), config
            )
            turnover_notional += traded
            fee += trade_fee
            slippage += trade_slippage
            end_liquidation = True
            drawdown = equity / peak - 1.0

        weights = notionals / equity
        accounting_change = equity - previous_equity
        accounting_components = market_pnl + funding_pnl - fee - slippage
        accounting_error = accounting_change - accounting_components
        if abs(accounting_error) > 10 * config.accounting_tolerance:
            raise AssertionError(
                f"accounting invariant failed at {timestamp}: {accounting_error}"
            )
        row_turnover = turnover_notional / max(
            equity + fee + slippage, config.accounting_tolerance
        )
        equity_rows.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "drawdown": drawdown,
                "gross_exposure": float(weights.abs().sum()),
                "net_exposure": float(weights.sum()),
                "turnover": row_turnover,
                "turnover_notional": turnover_notional,
                "fee": fee,
                "slippage": slippage,
                "trading_cost": fee + slippage,
                "market_pnl": market_pnl,
                "funding_pnl": funding_pnl,
                "accounting_error": accounting_error,
                "risk_multiplier": 0.0 if killed else risk_mult,
                "executed_signal_timestamp": executed_signal,
                "killed": killed,
                "end_liquidation": end_liquidation,
            }
        )
        weight_row = weights.copy()
        weight_row.name = timestamp
        weight_rows.append(weight_row)
        previous_price = price
        previous_equity = equity
        prior_risk_multiplier = 0.0 if killed else risk_mult

    equity_frame = pd.DataFrame(equity_rows).set_index("timestamp")
    weights_frame = pd.DataFrame(weight_rows)
    calendar_monthly, active_monthly = _monthly_returns(
        equity_frame, initial_equity
    )
    start_for_kill = (
        pd.Timestamp(fold_start) if fold_start is not None else prices.index[0]
    )
    time_to_kill = (
        kill_timestamp - start_for_kill if kill_timestamp is not None else None
    )
    stats: dict[str, Any] = {
        "config": asdict(config),
        "start": str(prices.index[0]),
        "end": str(prices.index[-1]),
        "initial_equity": initial_equity,
        "final_equity": equity,
        "return_pct": (equity / initial_equity - 1.0) * 100.0,
        "mdd_pct": float(equity_frame["drawdown"].min()) * 100.0,
        "total_turnover": float(equity_frame["turnover"].sum()),
        "total_turnover_notional": float(
            equity_frame["turnover_notional"].sum()
        ),
        "total_fee": float(equity_frame["fee"].sum()),
        "total_slippage": float(equity_frame["slippage"].sum()),
        "liquidated_at_end": bool(equity_frame["end_liquidation"].iloc[-1]),
        "killed": killed,
        "kill_timestamp": None if kill_timestamp is None else str(kill_timestamp),
        "time_to_kill": time_to_kill,
        "time_to_kill_hours": (
            None if time_to_kill is None else time_to_kill.total_seconds() / 3600.0
        ),
        "calendar_monthly_returns": calendar_monthly,
        "active_monthly_returns": active_monthly,
        "calendar_monthly_returns_pct": {
            str(period): float(value) * 100.0
            for period, value in calendar_monthly.items()
        },
        "active_monthly_returns_pct": {
            str(period): float(value) * 100.0
            for period, value in active_monthly.items()
        },
        "max_abs_accounting_error": float(
            equity_frame["accounting_error"].abs().max()
        ),
    }
    return equity_frame, weights_frame, stats


def stitch_equity_curves(
    curves: Sequence[pd.DataFrame],
    fold_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scale independent fold curves into one continuous OOS equity path."""
    if fold_names is not None and len(fold_names) != len(curves):
        raise ValueError("fold_names length must match curves")
    stitched: list[pd.DataFrame] = []
    previous_end: float | None = None
    last_timestamp: pd.Timestamp | None = None
    for i, curve in enumerate(curves):
        if curve.empty:
            continue
        if "equity" not in curve:
            raise ValueError("every curve must contain an equity column")
        part = curve.copy()
        if not isinstance(part.index, pd.DatetimeIndex):
            raise TypeError("curve index must be a DatetimeIndex")
        if not part.index.is_monotonic_increasing or not part.index.is_unique:
            raise ValueError("curve index must be sorted and unique")
        if last_timestamp is not None and part.index[0] <= last_timestamp:
            raise ValueError("fold curves must be strictly chronological")
        first = float(part["equity"].iloc[0])
        if first <= 0:
            raise ValueError("fold equity must be positive")
        scale = 1.0 if previous_end is None else previous_end / first
        part["equity"] = part["equity"] * scale
        if "market_pnl" in part:
            for column in (
                "market_pnl",
                "funding_pnl",
                "fee",
                "slippage",
                "trading_cost",
                "turnover_notional",
                "accounting_error",
            ):
                if column in part:
                    part[column] = part[column] * scale
        part["fold"] = fold_names[i] if fold_names is not None else str(i)
        stitched.append(part)
        previous_end = float(part["equity"].iloc[-1])
        last_timestamp = part.index[-1]
    if not stitched:
        return pd.DataFrame()
    result = pd.concat(stitched)
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1.0
    return result


# Compatibility with the V1 module's conventional entry-point name.
run_backtest = run_v2_backtest
