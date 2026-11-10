from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .v2_config import V2BacktestConfig

EPS = 1e-12


def _fold_values(equity_frame: pd.DataFrame) -> pd.Series:
    if "fold" in equity_frame:
        return equity_frame["fold"].astype(str)
    return pd.Series("all", index=equity_frame.index, dtype=object)


def reconstruct_trade_attribution(
    equity_frame: pd.DataFrame,
    weights_frame: pd.DataFrame,
    open_prices_15m: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    config: V2BacktestConfig = V2BacktestConfig(),
) -> pd.DataFrame:
    """Reconstruct the exact symbol-level accounting behind a V2 run.

    The returned long frame is indexed by ``timestamp, symbol``.  A fold
    boundary resets prior prices and notionals, so independently initialized
    walk-forward folds can safely be concatenated before calling this function.
    """
    required = {"equity", "market_pnl", "funding_pnl", "fee", "slippage"}
    missing = required.difference(equity_frame.columns)
    if missing:
        raise ValueError(f"equity_frame is missing columns: {sorted(missing)}")
    if not equity_frame.index.equals(weights_frame.index):
        raise ValueError("equity_frame and weights_frame indexes must match")
    symbols = list(weights_frame.columns)
    prices = open_prices_15m.reindex(index=equity_frame.index, columns=symbols)
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("execution prices must exist and be positive at every row")
    funding = (
        pd.DataFrame(0.0, index=equity_frame.index, columns=symbols)
        if funding_rates is None
        else funding_rates.reindex(index=equity_frame.index, columns=symbols).fillna(0.0)
    )
    folds = _fold_values(equity_frame)
    rows: list[dict[str, Any]] = []

    for fold in folds.drop_duplicates():
        locations = np.flatnonzero(folds.to_numpy() == fold)
        previous_post = pd.Series(0.0, index=symbols)
        previous_price: pd.Series | None = None
        fold_initial_equity: float | None = None
        cumulative_components = 0.0
        for location in locations:
            timestamp = equity_frame.index[location]
            price = prices.iloc[location]
            equity_row = equity_frame.iloc[location]
            if previous_price is None:
                prior_notional = pd.Series(0.0, index=symbols)
                market_pnl = pd.Series(0.0, index=symbols)
                pretrade_notional = prior_notional.copy()
            else:
                asset_return = price / previous_price - 1.0
                prior_notional = previous_post
                market_pnl = prior_notional * asset_return
                pretrade_notional = prior_notional * (1.0 + asset_return)
            funding_pnl = -pretrade_notional * funding.iloc[location]
            posttrade_notional = weights_frame.iloc[location] * float(equity_row["equity"])
            traded_notional = (posttrade_notional - pretrade_notional).abs()
            fee = traded_notional * config.fee_rate
            slippage = traded_notional * config.slippage_rate
            net_pnl = market_pnl + funding_pnl - fee - slippage

            checks = {
                "market_pnl": market_pnl,
                "funding_pnl": funding_pnl,
                "fee": fee,
                "slippage": slippage,
            }
            for column, values in checks.items():
                error = float(values.sum()) - float(equity_row[column])
                if abs(error) > 10 * config.accounting_tolerance:
                    raise AssertionError(
                        f"{column} attribution failed at {timestamp}: {error}"
                    )
            row_components = float(net_pnl.sum())
            if fold_initial_equity is None:
                fold_initial_equity = float(equity_row["equity"]) - row_components
            cumulative_components += row_components
            equity_error = (
                fold_initial_equity
                + cumulative_components
                - float(equity_row["equity"])
            )
            if abs(equity_error) > 10 * config.accounting_tolerance:
                raise AssertionError(
                    f"equity attribution failed at {timestamp}: {equity_error}"
                )

            for symbol in symbols:
                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "fold": fold,
                        "price": float(price[symbol]),
                        "previous_notional": float(prior_notional[symbol]),
                        "market_pnl": float(market_pnl[symbol]),
                        "pretrade_notional": float(pretrade_notional[symbol]),
                        "funding_rate": float(funding.iloc[location][symbol]),
                        "funding_pnl": float(funding_pnl[symbol]),
                        "posttrade_notional": float(posttrade_notional[symbol]),
                        "notional_change": float(
                            posttrade_notional[symbol] - pretrade_notional[symbol]
                        ),
                        "traded_notional": float(traded_notional[symbol]),
                        "fee": float(fee[symbol]),
                        "slippage": float(slippage[symbol]),
                        "net_pnl": float(net_pnl[symbol]),
                        "end_liquidation": bool(
                            equity_row.get("end_liquidation", False)
                        ),
                    }
                )
            previous_post = posttrade_notional
            previous_price = price

    result = pd.DataFrame(rows)
    return result.set_index(["timestamp", "symbol"]).sort_index()


def _new_episode(
    trade_id: str,
    fold: str,
    symbol: str,
    side: str,
    timestamp: pd.Timestamp,
    price: float,
    notional: float,
) -> dict[str, Any]:
    return {
        "trade_id": trade_id,
        "fold": fold,
        "symbol": symbol,
        "side": side,
        "entry_timestamp": timestamp,
        "entry_value": notional,
        "entry_quantity": notional / price,
        "exit_value": 0.0,
        "exit_quantity": 0.0,
        "entry_abs_notional": notional,
        "max_abs_notional": notional,
        "notional_samples": [notional],
        "gross_market_pnl": 0.0,
        "funding_pnl": 0.0,
        "fee": 0.0,
        "slippage": 0.0,
        "cumulative": 0.0,
        "path": [0.0],
    }


def _add_pnl(
    episode: dict[str, Any],
    market: float = 0.0,
    funding: float = 0.0,
    fee: float = 0.0,
    slippage: float = 0.0,
) -> None:
    episode["gross_market_pnl"] += market
    episode["funding_pnl"] += funding
    episode["fee"] += fee
    episode["slippage"] += slippage
    episode["cumulative"] += market + funding - fee - slippage
    episode["path"].append(episode["cumulative"])


def _finish_episode(
    episode: dict[str, Any],
    timestamp: pd.Timestamp,
    price: float,
    exit_reason: str,
    closed: bool,
) -> dict[str, Any]:
    max_notional = max(float(episode["max_abs_notional"]), EPS)
    entry_quantity = max(float(episode["entry_quantity"]), EPS)
    exit_quantity = float(episode["exit_quantity"])
    net = float(episode["cumulative"])
    path = np.asarray(episode["path"], dtype=float)
    return {
        "trade_id": episode["trade_id"],
        "fold": episode["fold"],
        "symbol": episode["symbol"],
        "side": episode["side"],
        "entry_timestamp": episode["entry_timestamp"],
        "exit_timestamp": timestamp,
        "entry_price": float(episode["entry_value"]) / entry_quantity,
        "exit_price": (
            float(episode["exit_value"]) / exit_quantity if exit_quantity > EPS else price
        ),
        "duration_hours": (timestamp - episode["entry_timestamp"]).total_seconds()
        / 3600.0,
        "entry_abs_notional": float(episode["entry_abs_notional"]),
        "max_abs_notional": float(episode["max_abs_notional"]),
        "average_abs_notional": float(np.mean(episode["notional_samples"])),
        "gross_market_pnl": float(episode["gross_market_pnl"]),
        "funding_pnl": float(episode["funding_pnl"]),
        "fee": float(episode["fee"]),
        "slippage": float(episode["slippage"]),
        "net_pnl": net,
        "return_on_max_notional_pct": net / max_notional * 100.0,
        "mfe_pct": float(path.max()) / max_notional * 100.0,
        "mae_pct": float(path.min()) / max_notional * 100.0,
        "exit_reason": exit_reason,
        "closed": closed,
    }


def build_trade_ledger(attribution: pd.DataFrame) -> pd.DataFrame:
    """Convert exact row attribution into symbol position episodes."""
    frame = attribution.reset_index().sort_values(["fold", "timestamp", "symbol"])
    episodes: list[dict[str, Any]] = []
    counters: dict[tuple[str, str], int] = {}
    columns = list(frame.columns)
    for (fold, symbol), group in frame.groupby(["fold", "symbol"], sort=False):
        active: dict[str, Any] | None = None
        last_timestamp: pd.Timestamp | None = None
        last_price = np.nan
        for row in group.itertuples(index=False, name="Attribution"):
            values = dict(zip(columns, row))
            timestamp = pd.Timestamp(values["timestamp"])
            price = float(values["price"])
            pre = float(values["pretrade_notional"])
            post = float(values["posttrade_notional"])
            fee = float(values["fee"])
            slippage = float(values["slippage"])
            market = float(values["market_pnl"])
            funding = float(values["funding_pnl"])
            end_liquidation = bool(values["end_liquidation"])
            last_timestamp, last_price = timestamp, price

            if active is not None:
                active["notional_samples"].append(abs(pre))
                active["max_abs_notional"] = max(active["max_abs_notional"], abs(pre))
                _add_pnl(active, market=market, funding=funding)
            elif abs(market) > EPS or abs(funding) > EPS:
                raise AssertionError("PnL exists without an active position episode")

            same_side = pre * post > 0
            is_flip = pre * post < 0
            if is_flip:
                close_share = abs(pre) / (abs(pre) + abs(post))
                close_fee, close_slippage = fee * close_share, slippage * close_share
                if active is None:
                    raise AssertionError("flip exists without an active episode")
                active["exit_value"] += abs(pre)
                active["exit_quantity"] += abs(pre) / price
                _add_pnl(active, fee=close_fee, slippage=close_slippage)
                episodes.append(_finish_episode(active, timestamp, price, "flip", True))
                counters[(fold, symbol)] = counters.get((fold, symbol), 0) + 1
                trade_id = f"{fold}:{symbol}:{counters[(fold, symbol)]}"
                active = _new_episode(
                    trade_id,
                    fold,
                    symbol,
                    "long" if post > 0 else "short",
                    timestamp,
                    price,
                    abs(post),
                )
                _add_pnl(
                    active,
                    fee=fee - close_fee,
                    slippage=slippage - close_slippage,
                )
            elif abs(pre) <= EPS and abs(post) > EPS:
                counters[(fold, symbol)] = counters.get((fold, symbol), 0) + 1
                trade_id = f"{fold}:{symbol}:{counters[(fold, symbol)]}"
                active = _new_episode(
                    trade_id,
                    fold,
                    symbol,
                    "long" if post > 0 else "short",
                    timestamp,
                    price,
                    abs(post),
                )
                _add_pnl(active, fee=fee, slippage=slippage)
            elif abs(pre) > EPS and abs(post) <= EPS:
                if active is None:
                    raise AssertionError("close exists without an active episode")
                active["exit_value"] += abs(pre)
                active["exit_quantity"] += abs(pre) / price
                _add_pnl(active, fee=fee, slippage=slippage)
                reason = "end_of_fold" if end_liquidation else "flat"
                episodes.append(_finish_episode(active, timestamp, price, reason, True))
                active = None
            elif same_side:
                if active is None:
                    raise AssertionError("position exists without an active episode")
                change = abs(post) - abs(pre)
                if change > EPS:
                    active["entry_value"] += change
                    active["entry_quantity"] += change / price
                    active["max_abs_notional"] = max(
                        active["max_abs_notional"], abs(post)
                    )
                elif change < -EPS:
                    active["exit_value"] += -change
                    active["exit_quantity"] += -change / price
                _add_pnl(active, fee=fee, slippage=slippage)
            elif abs(fee) > EPS or abs(slippage) > EPS:
                raise AssertionError("cost exists without a position change")

        if active is not None and last_timestamp is not None:
            episodes.append(
                _finish_episode(
                    active, last_timestamp, float(last_price), "end_of_fold", False
                )
            )

    return pd.DataFrame(episodes)


def reconstruct_trade_ledger(
    equity_frame: pd.DataFrame,
    weights_frame: pd.DataFrame,
    open_prices_15m: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    config: V2BacktestConfig = V2BacktestConfig(),
) -> pd.DataFrame:
    attribution = reconstruct_trade_attribution(
        equity_frame, weights_frame, open_prices_15m, funding_rates, config
    )
    return build_trade_ledger(attribution)


def _summary_row(trades: pd.DataFrame) -> dict[str, Any]:
    closed = trades.loc[trades["closed"].astype(bool)].copy()
    pnl = closed["net_pnl"].astype(float)
    wins = pnl[pnl > EPS]
    losses = pnl[pnl < -EPS]
    breakeven = pnl[(pnl >= -EPS) & (pnl <= EPS)]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    long = closed.loc[closed["side"] == "long", "net_pnl"]
    short = closed.loc[closed["side"] == "short", "net_pnl"]

    def win_rate(values: Iterable[float]) -> float:
        series = pd.Series(values, dtype=float)
        return float((series > EPS).mean()) if len(series) else np.nan

    def profit_factor(values: Iterable[float]) -> float:
        series = pd.Series(values, dtype=float)
        profit = float(series[series > EPS].sum())
        loss = float(series[series < -EPS].sum())
        return (
            profit / abs(loss)
            if loss < -EPS
            else (np.inf if profit > EPS else np.nan)
        )

    returns = pd.to_numeric(
        closed.get(
            "return_on_max_notional_pct",
            pd.Series(np.nan, index=closed.index),
        ),
        errors="coerce",
    )
    winning_returns = returns.loc[pnl > EPS].dropna()
    losing_returns = returns.loc[pnl < -EPS].dropna()
    mfe = pd.to_numeric(
        closed.get("mfe_pct", pd.Series(np.nan, index=closed.index)),
        errors="coerce",
    ).dropna()
    mae = pd.to_numeric(
        closed.get("mae_pct", pd.Series(np.nan, index=closed.index)),
        errors="coerce",
    ).dropna()
    chronological = closed.copy()
    chronological_column = (
        "exit_timestamp"
        if "exit_timestamp" in chronological
        else "entry_timestamp"
    )
    if chronological_column in chronological:
        chronological["_chronological_order"] = pd.to_datetime(
            chronological[chronological_column], errors="coerce"
        )
        chronological = chronological.sort_values(
            ["_chronological_order", "trade_id"],
            kind="stable",
            na_position="last",
        )
    outcomes = np.sign(chronological["net_pnl"].to_numpy(dtype=float))
    max_wins = max_losses = current_wins = current_losses = 0
    for outcome in outcomes:
        if outcome > 0:
            current_wins += 1
            current_losses = 0
        elif outcome < 0:
            current_losses += 1
            current_wins = 0
        else:
            current_wins = current_losses = 0
        max_wins = max(max_wins, current_wins)
        max_losses = max(max_losses, current_losses)
    exit_reasons = (
        closed.get("exit_reason", pd.Series(index=closed.index, dtype=object))
        .value_counts()
        .to_dict()
    )
    net_total = float(pnl.sum())
    top_gross_profit = float(wins.nlargest(10).sum())
    top_net_pnl = float(pnl.nlargest(10).sum())

    return {
        "trades": int(len(closed)),
        "open_trades": int((~trades["closed"].astype(bool)).sum()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "breakeven": int(len(breakeven)),
        "win_rate": win_rate(pnl),
        "trade_win_rate": win_rate(pnl),
        "long_win_rate": win_rate(long),
        "short_win_rate": win_rate(short),
        "net_pnl_total": net_total,
        "long_trade_count": int(len(long)),
        "short_trade_count": int(len(short)),
        "long_net_pnl": float(long.sum()),
        "short_net_pnl": float(short.sum()),
        "long_profit_factor": profit_factor(long),
        "short_profit_factor": profit_factor(short),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor(pnl),
        "average_win": float(wins.mean()) if len(wins) else np.nan,
        "median_win": float(wins.median()) if len(wins) else np.nan,
        "average_loss": float(losses.mean()) if len(losses) else np.nan,
        "median_loss": float(losses.median()) if len(losses) else np.nan,
        "payoff_ratio": (
            float(wins.mean()) / abs(float(losses.mean()))
            if len(wins) and len(losses)
            else np.nan
        ),
        "expectancy_per_trade": float(pnl.mean()) if len(pnl) else np.nan,
        "average_duration_hours": (
            float(closed["duration_hours"].mean()) if len(closed) else np.nan
        ),
        "median_duration_hours": (
            float(closed["duration_hours"].median()) if len(closed) else np.nan
        ),
        "average_return_on_max_notional_pct": (
            float(returns.mean()) if returns.notna().any() else np.nan
        ),
        "median_return_on_max_notional_pct": (
            float(returns.median()) if returns.notna().any() else np.nan
        ),
        "winning_average_return_pct": (
            float(winning_returns.mean()) if len(winning_returns) else np.nan
        ),
        "losing_average_return_pct": (
            float(losing_returns.mean()) if len(losing_returns) else np.nan
        ),
        "return_p05_pct": (
            float(returns.quantile(0.05)) if returns.notna().any() else np.nan
        ),
        "return_p25_pct": (
            float(returns.quantile(0.25)) if returns.notna().any() else np.nan
        ),
        "return_p50_pct": (
            float(returns.quantile(0.50)) if returns.notna().any() else np.nan
        ),
        "return_p75_pct": (
            float(returns.quantile(0.75)) if returns.notna().any() else np.nan
        ),
        "return_p95_pct": (
            float(returns.quantile(0.95)) if returns.notna().any() else np.nan
        ),
        "average_mfe_pct": float(mfe.mean()) if len(mfe) else np.nan,
        "median_mfe_pct": float(mfe.median()) if len(mfe) else np.nan,
        "average_mae_pct": float(mae.mean()) if len(mae) else np.nan,
        "median_mae_pct": float(mae.median()) if len(mae) else np.nan,
        "max_consecutive_wins": int(max_wins),
        "max_consecutive_losses": int(max_losses),
        "exit_reason_flat_count": int(exit_reasons.get("flat", 0)),
        "exit_reason_flip_count": int(exit_reasons.get("flip", 0)),
        "exit_reason_end_of_fold_count": int(exit_reasons.get("end_of_fold", 0)),
        "top_10_trade_share_of_gross_profit": (
            top_gross_profit / gross_profit if gross_profit > EPS else np.nan
        ),
        "top_10_trade_share_of_net_pnl": (
            top_net_pnl / net_total if net_total > EPS else np.nan
        ),
        "fees": float(trades["fee"].sum()),
        "slippage": float(trades["slippage"].sum()),
        "funding": float(trades["funding_pnl"].sum()),
        "cost_to_gross_profit": (
            float((trades["fee"] + trades["slippage"]).sum()) / gross_profit
            if gross_profit > EPS
            else np.nan
        ),
        "best_trade": (
            closed.loc[pnl.idxmax(), "trade_id"] if len(closed) else None
        ),
        "best_trade_pnl": float(pnl.max()) if len(pnl) else np.nan,
        "worst_trade": (
            closed.loc[pnl.idxmin(), "trade_id"] if len(closed) else None
        ),
        "worst_trade_pnl": float(pnl.min()) if len(pnl) else np.nan,
        "win_rate_definition": "closed profitable trades / closed trades",
        "monthly_win_rate": None,
    }


def summarize_trade_ledger(trades: pd.DataFrame) -> dict[str, Any]:
    """Return overall, per-symbol and per-fold trade (not monthly) metrics."""
    if trades.empty:
        return {"overall": {}, "by_symbol": pd.DataFrame(), "by_fold": pd.DataFrame()}
    by_symbol = pd.DataFrame(
        {name: _summary_row(group) for name, group in trades.groupby("symbol")}
    ).T
    by_symbol.index.name = "symbol"
    by_fold = pd.DataFrame(
        {name: _summary_row(group) for name, group in trades.groupby("fold")}
    ).T
    by_fold.index.name = "fold"
    return {
        "overall": _summary_row(trades),
        "by_symbol": by_symbol,
        "by_fold": by_fold,
    }


def analyze_trades(
    equity_frame: pd.DataFrame,
    weights_frame: pd.DataFrame,
    open_prices_15m: pd.DataFrame,
    funding_rates: pd.DataFrame | None = None,
    config: V2BacktestConfig = V2BacktestConfig(),
) -> dict[str, Any]:
    """Build exact attribution, episode ledger and all summary views."""
    attribution = reconstruct_trade_attribution(
        equity_frame, weights_frame, open_prices_15m, funding_rates, config
    )
    trades = build_trade_ledger(attribution)
    summaries = summarize_trade_ledger(trades)
    return {"attribution": attribution, "trades": trades, **summaries}


summarize_trades = summarize_trade_ledger
