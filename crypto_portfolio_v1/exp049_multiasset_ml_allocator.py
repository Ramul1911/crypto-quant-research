from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import RAW_DIR, REPORT_DIR
from .run_carry_csmom import _run_arm_windows, build_hourly_signal_panel
from .run_trend_v21 import _jsonable
from .v2_config import V2BacktestConfig
from .v2_features import load_v2_feature_panel
from .v2_protocol import DEVELOPMENT_FOLDS, HISTORICAL_CORE

EXPERIMENT_ID = "EXP-049"
REPORT_PATH = REPORT_DIR / "exp049_multiasset_ml_allocator"
DECISION_HOUR_UTC = 0
MOMENTUM_HORIZON_HOURS = 24
LABEL_HORIZON_HOURS = 24
PURGE_HOURS = 25
VOL_WINDOW_HOURS = 168
TOP_K = 4
ROUND_TRIP_COST = 2.0 * (0.0004 + 0.0002)
ANNUALIZATION = np.sqrt(24.0 * 365.0)

PORTFOLIO_CONFIG = V2BacktestConfig(
    annual_vol_target=0.50,
    max_symbol_weight=0.30,
    max_gross_exposure=1.00,
    max_abs_net_exposure=1.00,
    max_cluster_exposure=1.00,
)
INDIVIDUAL_CONFIG = V2BacktestConfig(
    annual_vol_target=0.50,
    max_symbol_weight=1.00,
    max_gross_exposure=1.00,
    max_abs_net_exposure=1.00,
    max_cluster_exposure=1.00,
)


def decision_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    mask = (
        (index.hour == DECISION_HOUR_UTC)
        & (index.minute == 0)
        & (index.second == 0)
    )
    return index[mask]


def annualized_volatility(close: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(close.where(close > 0.0)).diff()
    return returns.rolling(
        VOL_WINDOW_HOURS,
        min_periods=VOL_WINDOW_HOURS,
    ).std(ddof=0) * ANNUALIZATION


def momentum_direction(close: pd.DataFrame) -> pd.DataFrame:
    momentum = close / close.shift(MOMENTUM_HORIZON_HOURS) - 1.0
    return np.sign(momentum).where(momentum.notna())


def _normalize_inverse_vol(
    direction: pd.Series,
    volatility: pd.Series,
    selected: pd.Index | list[str] | None = None,
) -> pd.Series:
    target = pd.Series(0.0, index=direction.index, dtype=float)
    usable = (
        direction.notna()
        & direction.ne(0.0)
        & volatility.notna()
        & volatility.gt(0.0)
    )
    if selected is not None:
        usable &= direction.index.isin(selected)
    if not usable.any():
        return target
    target.loc[usable] = direction.loc[usable] / volatility.loc[usable]
    gross = float(target.abs().sum())
    if gross > 0.0:
        target /= gross
    return target


def build_equal_risk_targets(
    direction: pd.DataFrame,
    volatility: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        _normalize_inverse_vol(direction.loc[timestamp], volatility.loc[timestamp])
        .rename(timestamp)
        for timestamp in decision_index(direction.index)
    ]
    return pd.DataFrame(rows, columns=direction.columns, dtype=float)


def build_individual_targets(
    direction: pd.DataFrame,
    volatility: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for timestamp in decision_index(direction.index):
        row = pd.Series(0.0, index=direction.columns, name=timestamp)
        side = direction.at[timestamp, symbol]
        vol = volatility.at[timestamp, symbol]
        if pd.notna(side) and side != 0.0 and pd.notna(vol) and vol > 0.0:
            row.loc[symbol] = float(side / vol)
        rows.append(row)
    return pd.DataFrame(rows, columns=direction.columns, dtype=float)


def exact_forward_net_labels(
    decision_times: pd.DatetimeIndex,
    direction: pd.DataFrame,
    panel_15m: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Calculate 24h side-adjusted PnL using executable 15m opens and funding."""
    opens = panel_15m["open"].astype(float)
    known_funding = panel_15m["funding"].reindex(opens.index).ffill()
    event_mask = (
        (opens.index.hour % 8 == 0)
        & (opens.index.minute == 0)
        & (opens.index.second == 0)
    )
    funding_events = known_funding.copy()
    funding_events.loc[~np.asarray(event_mask), :] = 0.0
    funding_events = funding_events.fillna(0.0)
    labels = pd.DataFrame(np.nan, index=decision_times, columns=opens.columns, dtype=float)

    for timestamp in decision_times:
        entry_i = int(opens.index.searchsorted(timestamp, side="right"))
        exit_time = timestamp + pd.Timedelta(hours=LABEL_HORIZON_HOURS)
        exit_i = int(opens.index.searchsorted(exit_time, side="right"))
        if entry_i >= len(opens.index) or exit_i >= len(opens.index):
            continue
        entry = opens.iloc[entry_i]
        exit_price = opens.iloc[exit_i]
        valid = entry.notna() & exit_price.notna() & entry.gt(0.0) & exit_price.gt(0.0)
        side = direction.reindex(index=[timestamp], columns=opens.columns).iloc[0]
        funding_sum = funding_events.iloc[entry_i + 1 : exit_i + 1].sum(axis=0)
        net = side * (exit_price / entry - 1.0) - side * funding_sum - ROUND_TRIP_COST
        labels.loc[timestamp, valid & side.notna() & side.ne(0.0)] = net.loc[
            valid & side.notna() & side.ne(0.0)
        ]
    return labels


def build_model_table(
    hourly: Mapping[str, pd.DataFrame],
    volatility: pd.DataFrame,
    direction: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    close = hourly["close"].astype(float)
    volume = hourly["volume"].astype(float)
    funding = hourly["funding"].astype(float).ffill()
    decisions = decision_index(close.index)

    momentum = {
        horizon: close / close.shift(horizon) - 1.0
        for horizon in (12, 24, 72, 168)
    }
    vol_features = {
        horizon: np.log(close.where(close > 0.0))
        .diff()
        .rolling(horizon, min_periods=horizon)
        .std(ddof=0)
        * ANNUALIZATION
        for horizon in (24, 72, 168)
    }
    log_volume = np.log1p(volume.clip(lower=0.0))
    volume_z = (
        log_volume - log_volume.rolling(168, min_periods=168).mean()
    ) / log_volume.rolling(168, min_periods=168).std(ddof=0).replace(0.0, np.nan)
    funding_mean_24 = funding.rolling(24, min_periods=1).mean()
    cross_rank_24 = momentum[24].rank(axis=1, pct=True) - 0.5
    cross_rank_168 = momentum[168].rank(axis=1, pct=True) - 0.5
    btc_mom_24 = momentum[24]["BTCUSDT"]
    btc_mom_168 = momentum[168]["BTCUSDT"]
    btc_vol_168 = vol_features[168]["BTCUSDT"]

    feature_names = [
        "mom_12h",
        "mom_24h",
        "mom_72h",
        "mom_168h",
        "vol_24h",
        "vol_72h",
        "vol_168h",
        "volume_z_168h",
        "funding_last",
        "funding_mean_24h",
        "cross_rank_mom_24h",
        "cross_rank_mom_168h",
        "btc_mom_24h",
        "btc_mom_168h",
        "btc_vol_168h",
        "direction",
        "signed_mom_24h",
        "signed_mom_168h",
    ]

    blocks: list[pd.DataFrame] = []
    for symbol in close.columns:
        side = direction[symbol].reindex(decisions)
        block = pd.DataFrame(
            {
                "mom_12h": momentum[12][symbol].reindex(decisions),
                "mom_24h": momentum[24][symbol].reindex(decisions),
                "mom_72h": momentum[72][symbol].reindex(decisions),
                "mom_168h": momentum[168][symbol].reindex(decisions),
                "vol_24h": vol_features[24][symbol].reindex(decisions),
                "vol_72h": vol_features[72][symbol].reindex(decisions),
                "vol_168h": vol_features[168][symbol].reindex(decisions),
                "volume_z_168h": volume_z[symbol].reindex(decisions),
                "funding_last": funding[symbol].reindex(decisions),
                "funding_mean_24h": funding_mean_24[symbol].reindex(decisions),
                "cross_rank_mom_24h": cross_rank_24[symbol].reindex(decisions),
                "cross_rank_mom_168h": cross_rank_168[symbol].reindex(decisions),
                "btc_mom_24h": btc_mom_24.reindex(decisions),
                "btc_mom_168h": btc_mom_168.reindex(decisions),
                "btc_vol_168h": btc_vol_168.reindex(decisions),
                "direction": side,
                "signed_mom_24h": momentum[24][symbol].reindex(decisions) * side,
                "signed_mom_168h": momentum[168][symbol].reindex(decisions) * side,
                "annual_volatility": volatility[symbol].reindex(decisions),
                "label_net_return": labels[symbol].reindex(decisions),
                "symbol": symbol,
            },
            index=decisions,
        )
        block.index.name = "timestamp"
        blocks.append(block.reset_index())
    table = pd.concat(blocks, ignore_index=True)
    return table, feature_names


def fit_allocator(
    table: pd.DataFrame,
    feature_names: list[str],
    train_end: str,
) -> lgb.LGBMRegressor:
    cutoff = pd.Timestamp(train_end)
    label_known = table["timestamp"] + pd.Timedelta(hours=PURGE_HOURS) <= cutoff
    train = table.loc[label_known].dropna(subset=feature_names + ["label_net_return"])
    if len(train) < 1000:
        raise ValueError(f"only {len(train)} allocator training rows through {train_end}")
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=150,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        random_state=49,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train[feature_names], train["label_net_return"])
    return model


def build_ml_fold_targets(
    model: lgb.LGBMRegressor,
    table: pd.DataFrame,
    feature_names: list[str],
    direction: pd.DataFrame,
    volatility: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    fold_rows = table.loc[
        table["timestamp"].between(start_ts, end_ts)
    ].dropna(subset=feature_names + ["annual_volatility"])
    fold_rows = fold_rows.copy()
    fold_rows["prediction"] = model.predict(fold_rows[feature_names])

    rows: list[pd.Series] = []
    for timestamp, group in fold_rows.groupby("timestamp", sort=True):
        ranked = group.sort_values("prediction", ascending=False)
        selected = list(ranked.head(TOP_K)["symbol"])
        target = _normalize_inverse_vol(
            direction.loc[timestamp],
            volatility.loc[timestamp],
            selected=selected,
        )
        target.name = timestamp
        rows.append(target)
    return pd.DataFrame(rows, columns=direction.columns, dtype=float)


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")


def main() -> None:
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    print("Loading HISTORICAL_CORE 15m data...", flush=True)
    panel_15m = load_v2_feature_panel(HISTORICAL_CORE, interval="15m", raw_dir=RAW_DIR)
    hourly = build_hourly_signal_panel(panel_15m)
    close = hourly["close"].astype(float)
    volatility = annualized_volatility(close)
    direction = momentum_direction(close)
    decisions = decision_index(close.index)
    labels = exact_forward_net_labels(decisions, direction, panel_15m)
    table, feature_names = build_model_table(hourly, volatility, direction, labels)

    equal_risk_targets = build_equal_risk_targets(direction, volatility)
    ml_folds: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    for fold in DEVELOPMENT_FOLDS:
        print(f"Fitting allocator for {fold.name} through {fold.train_end}...", flush=True)
        model = fit_allocator(table, feature_names, fold.train_end)
        fold_targets = build_ml_fold_targets(
            model,
            table,
            feature_names,
            direction,
            volatility,
            fold.test_start,
            fold.test_end,
        )
        ml_folds.append(fold_targets)
        for feature, importance in zip(feature_names, model.feature_importances_):
            importance_rows.append(
                {"fold": fold.name, "feature": feature, "importance": int(importance)}
            )
    ml_targets = pd.concat(ml_folds).sort_index()
    if not ml_targets.index.is_unique:
        raise AssertionError("ML fold targets overlap")

    windows = [
        (fold.name, fold.test_start, fold.test_end)
        for fold in DEVELOPMENT_FOLDS
    ]
    summaries: dict[str, dict[str, Any]] = {}
    for arm, targets, config in (
        ("equal_risk", equal_risk_targets, PORTFOLIO_CONFIG),
        ("ml_top4", ml_targets, PORTFOLIO_CONFIG),
    ):
        print(f"Backtesting {arm}...", flush=True)
        summary = _run_arm_windows(
            arm=arm,
            targets=targets,
            annual_volatility=volatility,
            panel_15m=panel_15m,
            windows=windows,
            report_dir=REPORT_PATH,
            config=config,
            primary=True,
        )
        summary["experiment_id"] = EXPERIMENT_ID
        summary["evaluation"] = "historical_core_walk_forward"
        _write_summary(REPORT_PATH / arm / "summary.json", summary)
        summaries[arm] = summary

    individual_rows: list[dict[str, Any]] = []
    for symbol in HISTORICAL_CORE:
        arm = f"individual_{symbol.lower()}"
        print(f"Backtesting {arm}...", flush=True)
        targets = build_individual_targets(direction, volatility, symbol)
        summary = _run_arm_windows(
            arm=arm,
            targets=targets,
            annual_volatility=volatility,
            panel_15m=panel_15m,
            windows=windows,
            report_dir=REPORT_PATH,
            config=INDIVIDUAL_CONFIG,
            primary=True,
        )
        summary["experiment_id"] = EXPERIMENT_ID
        summary["evaluation"] = "individual_symbol_diagnostic"
        _write_summary(REPORT_PATH / arm / "summary.json", summary)
        summaries[arm] = summary
        individual_rows.append(
            {
                "symbol": symbol,
                "return_pct": summary["return_pct"],
                "mdd_pct": summary["mdd_pct"],
                "profitable_folds": sum(
                    float(value) > 0.0
                    for value in summary["fold_returns_pct"].values()
                ),
                "profit_factor": summary["trades"]["profit_factor"],
                "turnover": summary["portfolio_diagnostics"]["total_turnover_notional"],
            }
        )

    ml = summaries["ml_top4"]
    equal = summaries["equal_risk"]
    ml_pf = ml["trades"]["profit_factor"]
    decision = {
        "positive_stitched_return": bool(float(ml["return_pct"]) > 0.0),
        "profitable_folds_at_least_4_of_5": bool(
            sum(float(value) > 0.0 for value in ml["fold_returns_pct"].values()) >= 4
        ),
        "beats_equal_risk_return": bool(
            float(ml["return_pct"]) > float(equal["return_pct"])
        ),
        "mdd_not_worse_than_25pct": bool(float(ml["mdd_pct"]) >= -25.0),
        "profit_factor_at_least_1_2": bool(
            ml_pf is not None and np.isfinite(float(ml_pf)) and float(ml_pf) >= 1.2
        ),
        "no_terminal_kill": bool(int(ml["fold_kills"]) == 0),
    }
    decision["accepted"] = bool(all(decision.values()))
    decision["decision"] = "GO" if decision["accepted"] else "NO-GO"

    pd.DataFrame(importance_rows).to_csv(
        REPORT_PATH / "feature_importance_by_fold.csv",
        index=False,
    )
    pd.DataFrame(individual_rows).sort_values(
        "return_pct",
        ascending=False,
    ).to_csv(REPORT_PATH / "individual_symbol_comparison.csv", index=False)
    consolidated = {
        "experiment_id": EXPERIMENT_ID,
        "universe": list(HISTORICAL_CORE),
        "feature_names": feature_names,
        "portfolio_config": asdict(PORTFOLIO_CONFIG),
        "individual_config": asdict(INDIVIDUAL_CONFIG),
        "equal_risk": summaries["equal_risk"],
        "ml_top4": summaries["ml_top4"],
        "individual_symbols": individual_rows,
        "decision": decision,
    }
    _write_summary(REPORT_PATH / "decision_summary.json", consolidated)
    print(
        f"equal-risk={equal['return_pct']:+.2f}% MDD={equal['mdd_pct']:.2f}% | "
        f"ML={ml['return_pct']:+.2f}% MDD={ml['mdd_pct']:.2f}% "
        f"PF={ml_pf} -> {decision['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
