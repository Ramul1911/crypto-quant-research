from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import RAW_DIR, ROOT
from .experts import BALANCED, FAST, SLOW, TrendExpert, TrendExpertV21, TrendV21Profile
from .run_hybrid import scores_to_target_weights as original_scores_to_target_weights
from .trade_analytics import analyze_trades, summarize_trade_ledger
from .v2_backtest import constrain_target_weights, run_v2_backtest
from .v2_config import V2BacktestConfig
from .v2_features import load_v2_feature_panel
from .universe_eligibility import DEFAULT_ELIGIBILITY_PATH, resolve_universe_symbols
from .v2_protocol import DEVELOPMENT_FOLDS, HISTORICAL_CORE, FoldSpec
from .walk_forward import stitch_oos_equity, summarize_walk_forward

DECISION_HOURS = 4
VOLATILITY_WINDOW_HOURS = 24 * 30
CORRELATION_WINDOW_HOURS = 168
CORRELATION_THRESHOLD = 0.75
REPORT_DIR = ROOT / "reports" / "trend_v21"
PROFILES = (FAST, BALANCED, SLOW)
CRYPTO_15M_PERIODS_PER_YEAR = 4 * 24 * 365
RECONCILIATION_TOLERANCE = 1e-10

# This lexicographic objective is protocol, not a statistic inferred from results.
# Higher is always better: survival first, then robust monthly performance, then MDD.
VALIDATION_OBJECTIVE = (
    "no_kill",
    "median_monthly_return_pct",
    "profitable_month_share",
    "mean_monthly_return_pct",
    "mdd_pct",
)

TREND_V21_CONFIG = V2BacktestConfig(max_cluster_exposure=0.45)
ORIGINAL_TREND_CONFIG = V2BacktestConfig()


def validation_selection_score(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the predeclared validation-only lexicographic selection score."""
    monthly = metrics.get("calendar_monthly_returns")
    if isinstance(monthly, pd.Series):
        monthly_values = monthly.astype(float)
        median = float(monthly_values.median()) * 100.0
        profitable_share = float((monthly_values > 0.0).mean())
        mean = float(monthly_values.mean()) * 100.0
    else:
        median = float(metrics["median_monthly_return_pct"])
        profitable_share = float(metrics["profitable_month_share"])
        mean = float(metrics["mean_monthly_return_pct"])
    values = (
        0.0 if bool(metrics.get("killed", False)) else 1.0,
        median,
        profitable_share,
        mean,
        float(metrics["mdd_pct"]),
    )
    return tuple(value if np.isfinite(value) else -np.inf for value in values)


def select_validation_profile(
    validation_metrics: Mapping[str, Mapping[str, Any]],
    *,
    profile_order: Sequence[str] = tuple(profile.name for profile in PROFILES),
) -> tuple[str, tuple[float, ...]]:
    """Select from supplied validation metrics; no test object is accepted."""
    if not validation_metrics:
        raise ValueError("validation_metrics cannot be empty")
    unknown = set(validation_metrics).difference(profile_order)
    if unknown:
        raise ValueError(f"unknown profiles: {sorted(unknown)}")
    order = {name: position for position, name in enumerate(profile_order)}
    ranked = [
        (validation_selection_score(metrics), -order[name], name)
        for name, metrics in validation_metrics.items()
    ]
    score, _, name = max(ranked)
    return name, score


def trailing_correlation_matrix(
    close: pd.DataFrame,
    timestamp: pd.Timestamp | str,
    *,
    window_hours: int = CORRELATION_WINDOW_HOURS,
) -> pd.DataFrame | None:
    """Calculate correlation from the last completed hourly returns at or before t."""
    if window_hours < 2:
        raise ValueError("window_hours must be at least two")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must use a DatetimeIndex")
    if not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError("close index must be sorted and unique")
    timestamp = pd.Timestamp(timestamp)
    history = close.loc[close.index <= timestamp].astype(float)
    if len(history) < window_hours + 1:
        return None
    returns = np.log(history.where(history > 0.0)).diff().iloc[-window_hours:]
    if len(returns) != window_hours or returns.isna().any().any():
        return None
    correlation = returns.corr()
    if (
        correlation.shape != (len(close.columns), len(close.columns))
        or not np.isfinite(correlation.to_numpy(dtype=float)).all()
    ):
        return None
    return correlation.reindex(index=close.columns, columns=close.columns)


def build_correlation_clusters(
    correlation: pd.DataFrame,
    *,
    threshold: float = CORRELATION_THRESHOLD,
) -> dict[str, str]:
    """Build deterministic connected components from absolute correlations."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not correlation.index.equals(correlation.columns):
        raise ValueError("correlation index and columns must match")
    if correlation.index.has_duplicates:
        raise ValueError("correlation symbols must be unique")
    values = correlation.astype(float)
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("correlation matrix must be complete and finite")

    symbols = [str(symbol) for symbol in values.index]
    parent = {symbol: symbol for symbol in symbols}

    def root(symbol: str) -> str:
        while parent[symbol] != symbol:
            parent[symbol] = parent[parent[symbol]]
            symbol = parent[symbol]
        return symbol

    def union(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left_position, left in enumerate(symbols):
        for right_position in range(left_position + 1, len(symbols)):
            right = symbols[right_position]
            if abs(float(values.iloc[left_position, right_position])) >= threshold:
                union(left, right)
    roots = {symbol: root(symbol) for symbol in symbols}
    labels = {name: f"cluster_{position}" for position, name in enumerate(sorted(set(roots.values())))}
    return {symbol: labels[roots[symbol]] for symbol in symbols}


def _validate_aligned(
    scores: pd.DataFrame,
    confidence: pd.DataFrame,
    annual_volatility: pd.DataFrame,
) -> None:
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("scores must use a DatetimeIndex")
    for name, frame in (
        ("confidence", confidence),
        ("annual_volatility", annual_volatility),
    ):
        if not frame.index.equals(scores.index) or not frame.columns.equals(scores.columns):
            raise ValueError(f"{name} must be aligned with scores")


def correlation_aware_target_weights(
    scores: pd.DataFrame,
    confidence: pd.DataFrame,
    annual_volatility: pd.DataFrame,
    close_history: pd.DataFrame,
    *,
    config: V2BacktestConfig = TREND_V21_CONFIG,
    decision_hours: int = DECISION_HOURS,
    long_count: int = 2,
    short_count: int = 2,
    correlation_window_hours: int = CORRELATION_WINDOW_HOURS,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Convert V2.1 scores to causal correlation-constrained 4-hour targets."""
    _validate_aligned(scores, confidence, annual_volatility)
    if decision_hours <= 0:
        raise ValueError("decision_hours must be positive")
    if long_count < 0 or short_count < 0:
        raise ValueError("long_count and short_count cannot be negative")
    if set(scores.columns).difference(close_history.columns):
        raise ValueError("close_history is missing score symbols")

    effective = (
        scores.astype(float).clip(-1.0, 1.0)
        * confidence.astype(float).clip(0.0, 1.0)
    ).replace([np.inf, -np.inf], np.nan)
    decision_mask = (
        (scores.index.hour % decision_hours == 0)
        & (scores.index.minute == 0)
        & (scores.index.second == 0)
    )
    rows: list[pd.Series] = []
    gross_per_side = config.max_gross_exposure / 2.0

    for timestamp in scores.index[decision_mask]:
        signal = effective.loc[timestamp].dropna()
        volatility = annual_volatility.loc[timestamp].astype(float)
        volatility = volatility.replace([np.inf, -np.inf], np.nan).where(lambda x: x > 0.0)
        signal = signal.loc[signal.index.intersection(volatility.dropna().index)]
        longs = list(signal[signal > 0.0].nlargest(long_count).index)
        shorts = list(signal[signal < 0.0].nsmallest(short_count).index)
        selected = longs + shorts
        raw = pd.Series(0.0, index=scores.columns, dtype=float, name=timestamp)

        correlation = trailing_correlation_matrix(
            close_history.reindex(columns=scores.columns),
            timestamp,
            window_hours=correlation_window_hours,
        )
        # Never substitute future/full-sample or identity correlation when unknown.
        if not selected or correlation is None:
            rows.append(raw)
            continue
        selected_correlation = correlation.reindex(index=selected, columns=selected)
        if selected_correlation.isna().any().any():
            rows.append(raw)
            continue

        for side in (longs, shorts):
            if not side:
                continue
            strength = signal.loc[side].abs()
            inverse_volatility = 1.0 / volatility.loc[side]
            allocation = strength * inverse_volatility
            total = float(allocation.sum())
            if total > 0.0 and np.isfinite(total):
                raw.loc[side] = (
                    np.sign(signal.loc[side]) * allocation / total * gross_per_side
                )

        clusters = build_correlation_clusters(
            selected_correlation,
            threshold=correlation_threshold,
        )
        selected_volatility = volatility.where(raw.ne(0.0))
        constrained = constrain_target_weights(
            raw,
            config=config,
            annual_vol=selected_volatility,
            correlation=selected_correlation,
            clusters=clusters,
        ).reindex(scores.columns, fill_value=0.0)
        constrained.name = timestamp
        rows.append(constrained)

    if not rows:
        return pd.DataFrame(columns=scores.columns, index=scores.index[:0], dtype=float)
    return pd.DataFrame(rows, columns=scores.columns)


def _annualized_volatility(close: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(close.where(close > 0.0)).diff()
    return returns.rolling(
        VOLATILITY_WINDOW_HOURS,
        min_periods=24 * 7,
    ).std(ddof=0) * np.sqrt(24 * 365)


def _slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[
        (frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))
    ].copy()


def _execution_data(
    panel_15m: Mapping[str, pd.DataFrame],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens = _slice(panel_15m["open"], start, end)
    valid = opens.notna().all(axis=1) & (opens > 0.0).all(axis=1)
    opens = opens.loc[valid]
    if opens.empty:
        raise ValueError(f"no complete 15m execution data in {start} through {end}")
    known_funding = panel_15m["funding"].reindex(opens.index).ffill()
    event_mask = (
        (opens.index.hour % 8 == 0)
        & (opens.index.minute == 0)
        & (opens.index.second == 0)
    )
    funding = known_funding.copy()
    funding.loc[~np.asarray(event_mask), :] = 0.0
    return opens, funding.fillna(0.0)


def _profile_by_name(name: str) -> TrendV21Profile:
    return next(profile for profile in PROFILES if profile.name == name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.Series):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_stats(path: Path, stats: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(stats), indent=2), encoding="utf-8")


def _run_fold_backtest(
    targets: pd.DataFrame,
    opens: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    config: V2BacktestConfig,
    annual_volatility: pd.DataFrame,
    fold_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run an independently evaluated fold with all positions closed."""
    return run_v2_backtest(
        targets,
        opens,
        funding_rates=funding,
        config=config,
        annual_vol=annual_volatility,
        fold_start=fold_start,
        liquidate_at_end=True,
    )


def _save_fold_trade_analytics(
    prefix: Path,
    fold_name: str,
    equity: pd.DataFrame,
    weights: pd.DataFrame,
    opens: pd.DataFrame,
    funding: pd.DataFrame,
    config: V2BacktestConfig,
) -> dict[str, Any]:
    """Analyze and persist one fold without allowing cross-fold episodes."""
    tagged_equity = equity.copy()
    tagged_equity["fold"] = fold_name
    analytics = analyze_trades(tagged_equity, weights, opens, funding, config)
    if int(analytics["overall"].get("open_trades", 0)) != 0:
        raise AssertionError(f"{fold_name} contains open trades after final liquidation")
    analytics["attribution"].to_parquet(f"{prefix}_attribution.parquet")
    analytics["trades"].to_csv(f"{prefix}_trades.csv", index=False)
    _write_stats(Path(f"{prefix}_trade_summary.json"), analytics["overall"])
    return analytics


def _monthly_rates(
    portfolio_metrics: Mapping[str, Any],
    fold_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose calendar and active monthly rates separately from trade win rate."""
    result = dict(portfolio_metrics)
    result["monthly_win_rate"] = float(result["profitable_month_share"])
    active_returns = [
        float(value)
        for stats in fold_stats
        for value in stats.get("active_monthly_returns", pd.Series(dtype=float))
    ]
    result["active_monthly_win_rate"] = (
        float(np.mean(np.asarray(active_returns) > 0.0)) if active_returns else None
    )
    return result


def calculate_portfolio_diagnostics(
    stitched_equity: pd.DataFrame,
    *,
    initial_equity: float = 1.0,
) -> dict[str, Any]:
    """Calculate risk, exposure, cost and calendar diagnostics for stitched OOS."""
    if stitched_equity.empty or "equity" not in stitched_equity:
        raise ValueError("stitched_equity must contain equity observations")
    equity = pd.to_numeric(stitched_equity["equity"], errors="raise").astype(float)
    if (
        not isinstance(equity.index, pd.DatetimeIndex)
        or not equity.index.is_monotonic_increasing
        or not equity.index.is_unique
    ):
        raise ValueError("stitched equity index must be a sorted unique DatetimeIndex")
    if initial_equity <= 0.0 or (equity <= 0.0).any():
        raise ValueError("equity values must be positive")

    period_returns = equity.pct_change()
    period_returns.iloc[0] = equity.iloc[0] / initial_equity - 1.0
    period_returns = period_returns.replace([np.inf, -np.inf], np.nan).dropna()
    elapsed_years = (
        (equity.index[-1] - equity.index[0]).total_seconds()
        / (365.0 * 24.0 * 3600.0)
    )
    if len(equity) == 1:
        elapsed_years = 1.0 / CRYPTO_15M_PERIODS_PER_YEAR
    total_return = float(equity.iloc[-1] / initial_equity)
    cagr = (
        total_return ** (1.0 / elapsed_years) - 1.0
        if elapsed_years > 0.0 and total_return > 0.0
        else np.nan
    )
    period_std = float(period_returns.std(ddof=0)) if len(period_returns) else 0.0
    annualized_volatility = period_std * np.sqrt(CRYPTO_15M_PERIODS_PER_YEAR)
    mean_period_return = float(period_returns.mean()) if len(period_returns) else 0.0
    sharpe = (
        mean_period_return / period_std * np.sqrt(CRYPTO_15M_PERIODS_PER_YEAR)
        if period_std > 0.0
        else np.nan
    )
    downside = np.minimum(period_returns.to_numpy(dtype=float), 0.0)
    downside_deviation = (
        float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    )
    sortino = (
        mean_period_return
        / downside_deviation
        * np.sqrt(CRYPTO_15M_PERIODS_PER_YEAR)
        if downside_deviation > 0.0
        else np.nan
    )

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough_timestamp = pd.Timestamp(drawdown.idxmin())
    peak_timestamp = pd.Timestamp(equity.loc[:trough_timestamp].idxmax())
    peak_equity = float(equity.loc[peak_timestamp])
    recovery_candidates = equity.loc[equity.index > trough_timestamp]
    recovery_candidates = recovery_candidates.loc[recovery_candidates >= peak_equity]
    max_drawdown = float(drawdown.loc[trough_timestamp])
    recovery_timestamp = (
        peak_timestamp
        if max_drawdown == 0.0
        else (
            pd.Timestamp(recovery_candidates.index[0])
            if len(recovery_candidates)
            else None
        )
    )
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0.0 else np.nan

    daily_equity = equity.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change()
    daily_returns.iloc[0] = daily_equity.iloc[0] / initial_equity - 1.0
    daily_returns = daily_returns.dropna()
    var_5 = float(daily_returns.quantile(0.05)) if len(daily_returns) else np.nan
    tail = (
        daily_returns.loc[daily_returns <= var_5]
        if np.isfinite(var_5)
        else daily_returns.iloc[:0]
    )
    cvar_5 = float(tail.mean()) if len(tail) else np.nan

    month_ends = equity.groupby(equity.index.to_period("M")).last()
    monthly_returns = month_ends.pct_change()
    monthly_returns.iloc[0] = month_ends.iloc[0] / initial_equity - 1.0
    best_month = monthly_returns.idxmax() if len(monthly_returns) else None
    worst_month = monthly_returns.idxmin() if len(monthly_returns) else None

    def column_stat(column: str, operation: str) -> float:
        if column not in stitched_equity:
            return 0.0
        values = pd.to_numeric(stitched_equity[column], errors="coerce").fillna(0.0)
        return float(getattr(values, operation)())

    return {
        "elapsed_years": float(elapsed_years),
        "cagr": float(cagr),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_0pct_rf": float(sharpe),
        "downside_sortino_0pct_rf": float(sortino),
        "calmar": float(calmar),
        "maximum_drawdown": max_drawdown,
        "maximum_drawdown_pct": max_drawdown * 100.0,
        "maximum_drawdown_peak_timestamp": peak_timestamp,
        "maximum_drawdown_trough_timestamp": trough_timestamp,
        "maximum_drawdown_recovery_timestamp": recovery_timestamp,
        "maximum_drawdown_duration_days": (
            trough_timestamp - peak_timestamp
        ).total_seconds()
        / 86400.0,
        "maximum_drawdown_recovery_days": (
            (recovery_timestamp - trough_timestamp).total_seconds() / 86400.0
            if recovery_timestamp is not None
            else None
        ),
        "daily_var_5pct": var_5,
        "daily_cvar_5pct": cvar_5,
        "best_day": daily_returns.idxmax() if len(daily_returns) else None,
        "best_day_return": float(daily_returns.max()) if len(daily_returns) else np.nan,
        "worst_day": daily_returns.idxmin() if len(daily_returns) else None,
        "worst_day_return": float(daily_returns.min()) if len(daily_returns) else np.nan,
        "average_gross_exposure": column_stat("gross_exposure", "mean"),
        "maximum_gross_exposure": column_stat("gross_exposure", "max"),
        "average_absolute_net_exposure": (
            float(
                pd.to_numeric(
                    stitched_equity["net_exposure"], errors="coerce"
                ).abs().mean()
            )
            if "net_exposure" in stitched_equity
            else 0.0
        ),
        "total_turnover_notional": column_stat("turnover_notional", "sum"),
        "total_fees": column_stat("fee", "sum"),
        "total_slippage": column_stat("slippage", "sum"),
        "total_funding": column_stat("funding_pnl", "sum"),
        "best_month": best_month,
        "best_month_return": (
            float(monthly_returns.loc[best_month]) if best_month is not None else np.nan
        ),
        "worst_month": worst_month,
        "worst_month_return": (
            float(monthly_returns.loc[worst_month]) if worst_month is not None else np.nan
        ),
        "monthly_return_standard_deviation": (
            float(monthly_returns.std(ddof=0)) if len(monthly_returns) else np.nan
        ),
    }


def _save_combined_trade_analytics(
    report_dir: Path,
    strategy_name: str,
    fold_analytics: Sequence[Mapping[str, Any]],
    fold_curves: Sequence[pd.DataFrame],
    stitched_equity: pd.DataFrame,
    *,
    initial_equity: float = 1.0,
) -> dict[str, Any]:
    """Scale closed fold episodes to stitched capital, concatenate, and reconcile."""
    if len(fold_analytics) != len(fold_curves):
        raise ValueError("fold analytics and curves must have equal lengths")
    monetary_columns = (
        "entry_abs_notional",
        "max_abs_notional",
        "average_abs_notional",
        "gross_market_pnl",
        "funding_pnl",
        "fee",
        "slippage",
        "net_pnl",
    )
    scaled_trades: list[pd.DataFrame] = []
    fold_scales: list[dict[str, Any]] = []
    stitched_capital = float(initial_equity)
    for analytics, curve in zip(fold_analytics, fold_curves):
        local = analytics["trades"].copy()
        if not local.empty:
            fold_name = str(local["fold"].iloc[0])
        else:
            attribution = analytics.get("attribution")
            if not isinstance(attribution, pd.DataFrame) or "fold" not in attribution:
                raise ValueError("empty fold ledger must include tagged attribution")
            fold_name = str(attribution["fold"].iloc[0])
        scale = stitched_capital / initial_equity
        local["capital_scale"] = scale
        local["fold_capital_scale"] = scale
        for column in monetary_columns:
            if column in local:
                local[column] = pd.to_numeric(local[column], errors="raise") * scale
        scaled_fold_pnl = (
            float(local["net_pnl"].sum()) if "net_pnl" in local else 0.0
        )
        expected_fold_pnl = float(curve["equity"].iloc[-1] - initial_equity) * scale
        fold_error = scaled_fold_pnl - expected_fold_pnl
        tolerance = RECONCILIATION_TOLERANCE * max(1.0, abs(stitched_capital))
        if abs(fold_error) > tolerance:
            raise AssertionError(
                f"{strategy_name} {fold_name} scaled trade reconciliation failed: "
                f"{fold_error}"
            )
        fold_scales.append(
            {
                "fold": fold_name,
                "capital_scale": scale,
                "stitched_starting_capital": stitched_capital,
                "scaled_trade_net_pnl": scaled_fold_pnl,
                "stitched_fold_equity_change": expected_fold_pnl,
                "reconciliation_error": fold_error,
            }
        )
        stitched_capital = float(curve["equity"].iloc[-1]) * scale
        scaled_trades.append(local)

    trades = pd.concat(scaled_trades, ignore_index=True)
    summaries = summarize_trade_ledger(trades)
    if int(summaries["overall"].get("open_trades", 0)) != 0:
        raise AssertionError(f"{strategy_name} combined ledger contains open trades")
    ledger_net_pnl = (
        float(trades["net_pnl"].sum()) if "net_pnl" in trades else 0.0
    )
    stitched_equity_change = float(
        stitched_equity["equity"].iloc[-1] - initial_equity
    )
    reconciliation_error = ledger_net_pnl - stitched_equity_change
    tolerance = RECONCILIATION_TOLERANCE * max(1.0, abs(initial_equity))
    if abs(reconciliation_error) > tolerance:
        raise AssertionError(
            f"{strategy_name} combined trade reconciliation failed: "
            f"{reconciliation_error}"
        )
    trades.to_csv(report_dir / f"{strategy_name}_trades.csv", index=False)
    summaries["by_symbol"].to_csv(
        report_dir / f"{strategy_name}_trade_summary_by_symbol.csv"
    )
    summaries["by_fold"].to_csv(
        report_dir / f"{strategy_name}_trade_summary_by_fold.csv"
    )
    return {
        "trades": trades,
        **summaries,
        "fold_capital_scales": fold_scales,
        "reconciliation": {
            "combined_trade_net_pnl": ledger_net_pnl,
            "stitched_final_equity_minus_initial": stitched_equity_change,
            "exact_error": reconciliation_error,
            "strict_tolerance": tolerance,
            "reconciled": True,
        },
    }


def _backtest_v21_split(
    output: Any,
    close: pd.DataFrame,
    annual_volatility: pd.DataFrame,
    panel_15m: Mapping[str, pd.DataFrame],
    start: str,
    end: str,
    *,
    config: V2BacktestConfig = TREND_V21_CONFIG,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    scores = _slice(output.scores, start, end)
    confidence = _slice(output.confidence, start, end)
    volatility = annual_volatility.reindex(scores.index)
    targets = correlation_aware_target_weights(
        scores,
        confidence,
        volatility,
        close,
        config=config,
    )
    opens, funding = _execution_data(panel_15m, start, end)
    equity, weights, stats = _run_fold_backtest(
        targets,
        opens,
        funding,
        config=config,
        annual_volatility=volatility,
        fold_start=start,
    )
    return targets, equity, weights, stats, opens, funding


def _slice_panel_symbols(
    panel: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
) -> dict[str, pd.DataFrame]:
    selected = tuple(str(symbol) for symbol in symbols)
    if not selected:
        raise ValueError("symbols cannot be empty")
    close = panel["close"]
    missing = [symbol for symbol in selected if symbol not in close.columns]
    if missing:
        raise KeyError(f"panel is missing symbols: {missing}")
    return {
        name: frame.reindex(columns=selected)
        for name, frame in panel.items()
    }


def run_trend_v21_experiment(
    *,
    raw_dir: Path = RAW_DIR,
    report_dir: Path = REPORT_DIR,
    folds: Sequence[FoldSpec] = DEVELOPMENT_FOLDS,
    symbols: Sequence[str] = HISTORICAL_CORE,
    config: V2BacktestConfig = TREND_V21_CONFIG,
) -> dict[str, Any]:
    """Run nested validation profile selection and untouched next-fold OOS tests."""
    report_dir.mkdir(parents=True, exist_ok=True)
    selected_symbols = tuple(str(symbol) for symbol in symbols)
    v21_config = config
    panel = load_v2_feature_panel(
        selected_symbols, interval="1h", raw_dir=raw_dir
    )
    panel_15m = load_v2_feature_panel(
        selected_symbols, interval="15m", raw_dir=raw_dir
    )
    close = panel["close"].astype(float)
    annual_volatility = _annualized_volatility(close)

    # All outputs are causal transforms. Profile choice below sees validation only.
    profile_outputs = {
        profile.name: TrendExpertV21(profile=profile).generate(panel)
        for profile in PROFILES
    }
    original_output = TrendExpert().generate(panel)
    available_15m_start = panel_15m["open"].dropna(how="all").index.min()

    validation_rows: list[dict[str, Any]] = []
    selected_curves: list[pd.DataFrame] = []
    original_curves: list[pd.DataFrame] = []
    selected_fold_stats: list[dict[str, Any]] = []
    original_fold_stats: list[dict[str, Any]] = []
    selected_profiles: dict[str, str] = {}
    completed_folds: list[str] = []
    selected_trade_analytics: list[dict[str, Any]] = []
    original_trade_analytics: list[dict[str, Any]] = []

    for fold in folds:
        if pd.Timestamp(fold.test_end) < available_15m_start:
            continue
        candidate_stats: dict[str, dict[str, Any]] = {}
        for profile in PROFILES:
            _, _, _, stats, _, _ = _backtest_v21_split(
                profile_outputs[profile.name],
                close,
                annual_volatility,
                panel_15m,
                fold.validation_start,
                fold.validation_end,
                config=v21_config,
            )
            candidate_stats[profile.name] = stats
            score = validation_selection_score(stats)
            monthly = stats["calendar_monthly_returns"].astype(float)
            validation_rows.append(
                {
                    "fold": fold.name,
                    "profile": profile.name,
                    "killed": bool(stats["killed"]),
                    "median_monthly_return_pct": float(monthly.median()) * 100.0,
                    "profitable_month_share": float((monthly > 0.0).mean()),
                    "mean_monthly_return_pct": float(monthly.mean()) * 100.0,
                    "mdd_pct": float(stats["mdd_pct"]),
                    "return_pct": float(stats["return_pct"]),
                    "selection_score": json.dumps(_jsonable(score)),
                }
            )

        selected_name, selected_score = select_validation_profile(candidate_stats)
        selected_profiles[fold.name] = selected_name
        for row in validation_rows:
            if row["fold"] == fold.name:
                row["selected"] = row["profile"] == selected_name

        targets, equity, weights, stats, opens, funding = _backtest_v21_split(
            profile_outputs[selected_name],
            close,
            annual_volatility,
            panel_15m,
            fold.test_start,
            fold.test_end,
            config=v21_config,
        )
        stats = {
            **stats,
            "selected_profile": selected_name,
            "validation_selection_score": selected_score,
        }
        prefix = report_dir / f"{fold.name}_trend_v21"
        targets.to_csv(f"{prefix}_targets.csv")
        equity.to_csv(f"{prefix}_equity.csv")
        weights.to_csv(f"{prefix}_weights.csv")
        _write_stats(Path(f"{prefix}_stats.json"), stats)
        selected_trade_analytics.append(
            _save_fold_trade_analytics(
                prefix,
                fold.name,
                equity,
                weights,
                opens,
                funding,
                v21_config,
            )
        )

        original_scores = _slice(original_output.scores, fold.test_start, fold.test_end)
        original_confidence = _slice(
            original_output.confidence,
            fold.test_start,
            fold.test_end,
        )
        original_volatility = annual_volatility.reindex(original_scores.index)
        original_targets = original_scores_to_target_weights(
            original_scores,
            original_confidence,
            original_volatility,
        )
        original_equity, original_weights, original_stats = _run_fold_backtest(
            original_targets,
            opens,
            funding,
            config=ORIGINAL_TREND_CONFIG,
            annual_volatility=original_volatility,
            fold_start=fold.test_start,
        )
        original_prefix = report_dir / f"{fold.name}_original_trend"
        original_targets.to_csv(f"{original_prefix}_targets.csv")
        original_equity.to_csv(f"{original_prefix}_equity.csv")
        original_weights.to_csv(f"{original_prefix}_weights.csv")
        _write_stats(Path(f"{original_prefix}_stats.json"), original_stats)
        original_trade_analytics.append(
            _save_fold_trade_analytics(
                original_prefix,
                fold.name,
                original_equity,
                original_weights,
                opens,
                funding,
                ORIGINAL_TREND_CONFIG,
            )
        )

        selected_curves.append(equity)
        original_curves.append(original_equity)
        selected_fold_stats.append(stats)
        original_fold_stats.append(original_stats)
        completed_folds.append(fold.name)

    if not completed_folds:
        raise ValueError("no development fold overlaps available 15m data")

    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(report_dir / "validation_profiles.csv", index=False)
    stitched_v21 = stitch_oos_equity(
        selected_curves,
        completed_folds,
        initial_equities=1.0,
        expected_frequency="15min",
    )
    stitched_original = stitch_oos_equity(
        original_curves,
        completed_folds,
        initial_equities=1.0,
        expected_frequency="15min",
    )
    stitched_v21.to_csv(report_dir / "stitched_trend_v21_equity.csv")
    stitched_original.to_csv(report_dir / "stitched_original_trend_equity.csv")

    v21_summary = _monthly_rates(
        summarize_walk_forward(
            stitched_v21,
            selected_fold_stats,
            baseline_equity=stitched_original,
        ),
        selected_fold_stats,
    )
    original_summary = _monthly_rates(
        summarize_walk_forward(
            stitched_original,
            original_fold_stats,
        ),
        original_fold_stats,
    )
    v21_summary.update(calculate_portfolio_diagnostics(stitched_v21))
    original_summary.update(calculate_portfolio_diagnostics(stitched_original))
    v21_trades = _save_combined_trade_analytics(
        report_dir,
        "trend_v21",
        selected_trade_analytics,
        selected_curves,
        stitched_v21,
    )
    original_trades = _save_combined_trade_analytics(
        report_dir,
        "original_trend",
        original_trade_analytics,
        original_curves,
        stitched_original,
    )
    v21_summary["overall_trade_metrics"] = v21_trades["overall"]
    v21_summary["trade_ledger_reconciliation"] = v21_trades["reconciliation"]
    v21_summary["fold_capital_scales"] = v21_trades["fold_capital_scales"]
    original_summary["overall_trade_metrics"] = original_trades["overall"]
    original_summary["trade_ledger_reconciliation"] = original_trades["reconciliation"]
    original_summary["fold_capital_scales"] = original_trades["fold_capital_scales"]
    report = {
        "experiment": "Trend Expert V2.1 nested validation walk-forward",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": selected_symbols,
        "excluded_from_historical_core": [
            symbol for symbol in HISTORICAL_CORE if symbol not in selected_symbols
        ],
        "folds": completed_folds,
        "profiles": [profile.name for profile in PROFILES],
        "validation_objective": {
            "ordering": VALIDATION_OBJECTIVE,
            "direction": "lexicographic descending",
            "tie_break": [profile.name for profile in PROFILES],
            "test_used_for_selection": False,
        },
        "selected_profiles": selected_profiles,
        "decision_hours": DECISION_HOURS,
        "correlation_window_hours": CORRELATION_WINDOW_HOURS,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "backtest_config": asdict(v21_config),
        "strategies": {
            "trend_v21": v21_summary,
            "original_trend": original_summary,
        },
        "acceptance": v21_summary["acceptance"],
        "trend_v21_vs_original": {
            "trend_v21_return_pct": v21_summary["return_pct"],
            "original_trend_return_pct": original_summary["return_pct"],
            "excess_return_pct": v21_summary["baseline_excess_return_pct"],
        },
    }
    _write_stats(report_dir / "summary.json", report)
    complete_metrics = {
        "experiment": report["experiment"],
        "created_at_utc": report["created_at_utc"],
        "folds": completed_folds,
        "selected_profiles": selected_profiles,
        "validation_metrics": validation_frame.to_dict(orient="records"),
        "strategies": {
            "trend_v21": {
                "portfolio_metrics": v21_summary,
                "overall_trade_metrics": v21_trades["overall"],
                "trade_ledger_reconciliation": v21_trades["reconciliation"],
                "fold_capital_scales": v21_trades["fold_capital_scales"],
                "per_symbol_records": v21_trades["by_symbol"]
                .reset_index()
                .to_dict(orient="records"),
                "per_fold_records": v21_trades["by_fold"]
                .reset_index()
                .to_dict(orient="records"),
            },
            "original_trend": {
                "portfolio_metrics": original_summary,
                "overall_trade_metrics": original_trades["overall"],
                "trade_ledger_reconciliation": original_trades["reconciliation"],
                "fold_capital_scales": original_trades["fold_capital_scales"],
                "per_symbol_records": original_trades["by_symbol"]
                .reset_index()
                .to_dict(orient="records"),
                "per_fold_records": original_trades["by_fold"]
                .reset_index()
                .to_dict(orient="records"),
            },
        },
        "metric_definitions": {
            "return_pct": "Compounded portfolio return after fees, slippage, funding, and final liquidation.",
            "monthly_win_rate": "Profitable calendar months divided by evaluated calendar months.",
            "active_monthly_win_rate": "Profitable active fold-month observations divided by active fold-month observations; null when unavailable.",
            "trade_win_rate": "Profitable closed position episodes divided by all closed position episodes.",
            "open_trades": "Position episodes not closed inside their independently evaluated fold; required to be zero.",
            "fold_boundary": "Trade episodes are never merged across independently initialized OOS folds.",
            "capital_scale": "Sequential stitched starting capital divided by fold-local initial capital; all monetary trade-ledger fields are multiplied by this scale.",
            "trade_ledger_reconciliation": "Scaled combined closed-trade net PnL minus stitched final equity minus initial equity; required within the recorded strict tolerance.",
            "return_on_max_notional_pct": "Closed-trade net PnL divided by maximum absolute notional during the episode, expressed as percent.",
            "mfe_pct": "Maximum cumulative net episode PnL divided by maximum absolute episode notional, expressed as percent.",
            "mae_pct": "Minimum cumulative net episode PnL divided by maximum absolute episode notional, expressed as percent.",
            "trade_streaks": "Maximum consecutive strictly profitable or strictly losing closed trades ordered by exit timestamp; breakeven trades reset either streak.",
            "exit_reason_counts": "Counts of closed episodes ending by flat target, direct side flip, or forced end-of-fold liquidation.",
            "top_10_trade_share_of_gross_profit": "Net PnL from the ten largest winning trades divided by gross profit; null without gross profit.",
            "top_10_trade_share_of_net_pnl": "Net PnL from the ten largest trades divided by positive total net PnL; null when total net PnL is not positive.",
            "cagr": "Compounded annual growth rate over elapsed wall-clock years.",
            "annualized_volatility": "Population standard deviation of 15-minute equity returns annualized by sqrt(4*24*365).",
            "sharpe_0pct_rf": "Mean 15-minute equity return divided by its population standard deviation, annualized by sqrt(4*24*365), with zero risk-free rate.",
            "downside_sortino_0pct_rf": "Mean 15-minute return divided by root-mean-square downside return, annualized by sqrt(4*24*365), with zero target and risk-free rate.",
            "calmar": "CAGR divided by absolute maximum drawdown; null when maximum drawdown is zero.",
            "daily_var_5pct": "Empirical fifth percentile of end-of-day equity returns.",
            "daily_cvar_5pct": "Mean end-of-day equity return at or below the empirical fifth percentile.",
            "maximum_drawdown_duration_days": "Calendar days from the prior equity peak to the maximum-drawdown trough.",
            "maximum_drawdown_recovery_days": "Calendar days from the maximum-drawdown trough until equity regains its prior peak; null if unrecovered.",
            "exposure": "Gross and net exposure diagnostics are time averages/maxima of stitched 15-minute observations.",
            "cost_totals": "Sums of stitched monetary turnover, fees, slippage, and signed funding PnL.",
            "monthly_returns": "Returns between end-of-month equity observations, with the first month measured from initial equity.",
        },
    }
    _write_stats(report_dir / "complete_metrics.json", complete_metrics)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Trend Expert V2.1 with nested validation selection."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--universe",
        choices=("historical_core", "historical_expanded"),
        default="historical_core",
        help=(
            "historical_core uses the frozen eight-symbol baseline; "
            "historical_expanded uses core plus the locked eligible midcaps."
        ),
    )
    parser.add_argument(
        "--eligibility-path",
        type=Path,
        default=DEFAULT_ELIGIBILITY_PATH,
        help="Eligibility manifest for --universe historical_expanded.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Explicit universe override; defaults to the selected --universe.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Symbols to drop from the selected universe before evaluation.",
    )
    parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=TREND_V21_CONFIG.max_gross_exposure,
        help="Gross leverage cap for Trend V2.1 (default 1.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.symbols is not None:
        symbols = tuple(
            symbol for symbol in args.symbols if symbol not in set(args.exclude)
        )
    else:
        symbols = resolve_universe_symbols(
            args.universe,
            eligibility_path=args.eligibility_path or DEFAULT_ELIGIBILITY_PATH,
            exclude=args.exclude,
        )
    config = V2BacktestConfig(
        max_cluster_exposure=TREND_V21_CONFIG.max_cluster_exposure,
        max_gross_exposure=float(args.max_gross_exposure),
    )
    report = run_trend_v21_experiment(
        raw_dir=args.raw_dir,
        report_dir=args.report_dir,
        symbols=symbols,
        config=config,
    )
    result = report["strategies"]["trend_v21"]
    original = report["strategies"]["original_trend"]
    print(
        f"symbols={','.join(report['symbols'])} "
        f"excluded={','.join(report['excluded_from_historical_core']) or 'none'}",
        flush=True,
    )
    print(
        f"trend_v21 return={result['return_pct']:+.2f}% "
        f"MDD={result['mdd_pct']:.2f}% "
        f"decision={report['acceptance']['decision']}",
        flush=True,
    )
    print(
        f"original_trend return={original['return_pct']:+.2f}% "
        f"MDD={original['mdd_pct']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
