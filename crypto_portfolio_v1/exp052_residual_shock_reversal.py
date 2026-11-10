from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RAW_DIR, REPORT_DIR
from .exp049_multiasset_ml_allocator import PORTFOLIO_CONFIG, annualized_volatility
from .run_carry_csmom import _run_arm_windows, build_hourly_signal_panel
from .run_trend_v21 import _jsonable
from .v2_features import load_v2_feature_panel
from .v2_protocol import DEVELOPMENT_FOLDS, HISTORICAL_CORE

EXPERIMENT_ID = "EXP-052"
REPORT_PATH = REPORT_DIR / "exp052_residual_shock_reversal"
BTC = "BTCUSDT"
BETA_WINDOW = 168
SHOCK_HORIZON = 12
SHOCK_THRESHOLD = 2.5
VOLUME_Z_THRESHOLD = 1.0
DECISION_INTERVAL_HOURS = 4
HOLD_HOURS = 12
MAX_ACTIVE_ALTS = 4


def decision_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    mask = (
        (index.hour % DECISION_INTERVAL_HOURS == 0)
        & (index.minute == 0)
        & (index.second == 0)
    )
    return index[mask]


def build_residual_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return lagged BTC betas, standardized residual shock and volume z-score."""
    if BTC not in close:
        raise KeyError(f"{BTC} is required as the market hedge")
    log_close = np.log(close.where(close > 0.0))
    returns = log_close.diff()
    btc_return = returns[BTC]
    covariance = returns.rolling(
        BETA_WINDOW,
        min_periods=BETA_WINDOW,
    ).cov(btc_return, ddof=0)
    btc_variance = btc_return.rolling(
        BETA_WINDOW,
        min_periods=BETA_WINDOW,
    ).var(ddof=0)
    beta = covariance.div(btc_variance.replace(0.0, np.nan), axis=0).shift(1)
    residual = returns.sub(beta.mul(btc_return, axis=0))
    prior_residual_std = residual.rolling(
        BETA_WINDOW,
        min_periods=BETA_WINDOW,
    ).std(ddof=0).shift(1)
    shock = residual.rolling(
        SHOCK_HORIZON,
        min_periods=SHOCK_HORIZON,
    ).sum().div(prior_residual_std * np.sqrt(float(SHOCK_HORIZON)))

    log_volume = np.log1p(volume.clip(lower=0.0))
    prior_volume_mean = log_volume.rolling(
        BETA_WINDOW,
        min_periods=BETA_WINDOW,
    ).mean().shift(1)
    prior_volume_std = log_volume.rolling(
        BETA_WINDOW,
        min_periods=BETA_WINDOW,
    ).std(ddof=0).shift(1)
    volume_z = (log_volume - prior_volume_mean).div(
        prior_volume_std.replace(0.0, np.nan)
    )
    beta[BTC] = 1.0
    shock[BTC] = np.nan
    volume_z[BTC] = np.nan
    return beta, shock, volume_z


def build_event_positions(
    shock: pd.DataFrame,
    volume_z: pd.DataFrame,
) -> pd.DataFrame:
    """Create frozen 12h continuation positions for at most four alt events."""
    decisions = decision_index(shock.index)
    alt_symbols = [symbol for symbol in shock.columns if symbol != BTC]
    active: dict[str, tuple[float, pd.Timestamp]] = {}
    rows: list[pd.Series] = []
    for timestamp in decisions:
        row = pd.Series(0.0, index=shock.columns, name=timestamp)
        expired = [
            symbol
            for symbol, (_, exit_time) in active.items()
            if timestamp >= exit_time
        ]
        for symbol in expired:
            del active[symbol]

        slots = MAX_ACTIVE_ALTS - len(active)
        if slots > 0:
            current_shock = shock.loc[timestamp, alt_symbols]
            current_volume = volume_z.loc[timestamp, alt_symbols]
            eligible = (
                current_shock.abs().ge(SHOCK_THRESHOLD)
                & current_volume.ge(VOLUME_Z_THRESHOLD)
                & ~current_shock.index.isin(active)
            )
            candidates = current_shock.loc[eligible].abs().nlargest(slots).index
            for symbol in candidates:
                side = float(np.sign(current_shock.loc[symbol]))
                if side != 0.0:
                    active[symbol] = (
                        side,
                        timestamp + pd.Timedelta(hours=HOLD_HOURS),
                    )

        for symbol, (side, _) in active.items():
            row.loc[symbol] = side
        rows.append(row)
    return pd.DataFrame(rows, columns=shock.columns, dtype=float)


def build_beta_neutral_targets(
    positions: pd.DataFrame,
    volatility: pd.DataFrame,
    beta: pd.DataFrame,
    *,
    reversal: bool,
) -> pd.DataFrame:
    """Inverse-vol weight active events and add a lagged-beta BTC hedge."""
    rows: list[pd.Series] = []
    direction_multiplier = -1.0 if reversal else 1.0
    for timestamp, position in positions.iterrows():
        target = pd.Series(0.0, index=positions.columns, name=timestamp)
        alt_symbols = [
            symbol
            for symbol in positions.columns
            if symbol != BTC and position.loc[symbol] != 0.0
        ]
        vol = volatility.reindex(index=[timestamp], columns=positions.columns).iloc[0]
        current_beta = beta.reindex(
            index=[timestamp],
            columns=positions.columns,
        ).iloc[0]
        usable = [
            symbol
            for symbol in alt_symbols
            if pd.notna(vol.loc[symbol])
            and vol.loc[symbol] > 0.0
            and pd.notna(current_beta.loc[symbol])
        ]
        if usable:
            target.loc[usable] = (
                direction_multiplier * position.loc[usable] / vol.loc[usable]
            )
            alt_gross = float(target.loc[usable].abs().sum())
            if alt_gross > 0.0:
                target.loc[usable] /= alt_gross
                target.loc[BTC] = -float(
                    (target.loc[usable] * current_beta.loc[usable]).sum()
                )
                gross = float(target.abs().sum())
                if gross > 1.0:
                    target /= gross
                largest = float(target.abs().max())
                if largest > PORTFOLIO_CONFIG.max_symbol_weight:
                    target *= PORTFOLIO_CONFIG.max_symbol_weight / largest
        rows.append(target)
    return pd.DataFrame(rows, columns=positions.columns, dtype=float)


def event_diagnostics(positions: pd.DataFrame) -> dict[str, Any]:
    prior = positions.shift(1).fillna(0.0)
    starts = positions.ne(0.0) & prior.eq(0.0)
    per_symbol = starts.sum(axis=0).astype(int)
    return {
        "event_count": int(per_symbol.sum()),
        "events_by_symbol": {
            str(symbol): int(count)
            for symbol, count in per_symbol.items()
            if int(count) > 0
        },
        "active_decision_share": float(positions.ne(0.0).any(axis=1).mean()),
        "mean_active_alts": float(positions.ne(0.0).sum(axis=1).mean()),
        "max_active_alts": int(positions.ne(0.0).sum(axis=1).max()),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def main() -> None:
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    panel_15m = load_v2_feature_panel(HISTORICAL_CORE, interval="15m", raw_dir=RAW_DIR)
    hourly = build_hourly_signal_panel(panel_15m)
    close = hourly["close"].astype(float)
    volume = hourly["volume"].astype(float)
    volatility = annualized_volatility(close)
    beta, shock, volume_z = build_residual_features(close, volume)
    positions = build_event_positions(shock, volume_z)
    positions.to_csv(REPORT_PATH / "event_positions.csv")

    reversal_targets = build_beta_neutral_targets(
        positions,
        volatility,
        beta,
        reversal=True,
    )
    continuation_targets = build_beta_neutral_targets(
        positions,
        volatility,
        beta,
        reversal=False,
    )
    windows = [
        (fold.name, fold.test_start, fold.test_end)
        for fold in DEVELOPMENT_FOLDS
    ]
    summaries: dict[str, dict[str, Any]] = {}
    for arm, targets in (
        ("residual_reversal", reversal_targets),
        ("residual_continuation", continuation_targets),
    ):
        print(f"Backtesting {arm}...", flush=True)
        summary = _run_arm_windows(
            arm=arm,
            targets=targets,
            annual_volatility=volatility,
            panel_15m=panel_15m,
            windows=windows,
            report_dir=REPORT_PATH,
            config=PORTFOLIO_CONFIG,
            primary=True,
        )
        summary["experiment_id"] = EXPERIMENT_ID
        summary["evaluation"] = "historical_core_residual_shock"
        _write_json(REPORT_PATH / arm / "summary.json", summary)
        summaries[arm] = summary

    reversal = summaries["residual_reversal"]
    continuation = summaries["residual_continuation"]
    reversal_pf = reversal["trades"]["profit_factor"]
    positive_folds = sum(
        float(value) > 0.0 for value in reversal["fold_returns_pct"].values()
    )
    checks = {
        "positive_return_after_costs": bool(reversal["return_pct"] > 0.0),
        "positive_folds_at_least_3_of_5": bool(positive_folds >= 3),
        "profit_factor_at_least_1_20": bool(
            reversal_pf is not None and float(reversal_pf) >= 1.20
        ),
        "mdd_not_worse_than_minus_25": bool(reversal["mdd_pct"] >= -25.0),
        "no_terminal_kill": bool(reversal["fold_kills"] == 0),
        "beats_continuation_by_10pp": bool(
            reversal["return_pct"] >= continuation["return_pct"] + 10.0
        ),
    }
    checks["go"] = bool(all(checks.values()))
    result = {
        "experiment_id": EXPERIMENT_ID,
        "universe": list(HISTORICAL_CORE),
        "parameters": {
            "beta_window_hours": BETA_WINDOW,
            "shock_horizon_hours": SHOCK_HORIZON,
            "shock_threshold": SHOCK_THRESHOLD,
            "volume_z_threshold": VOLUME_Z_THRESHOLD,
            "decision_interval_hours": DECISION_INTERVAL_HOURS,
            "hold_hours": HOLD_HOURS,
            "max_active_alts": MAX_ACTIVE_ALTS,
        },
        "events": event_diagnostics(positions),
        "arms": {
            arm: {
                "return_pct": summary["return_pct"],
                "mdd_pct": summary["mdd_pct"],
                "profit_factor": summary["trades"]["profit_factor"],
                "fold_returns_pct": summary["fold_returns_pct"],
                "fold_kills": summary["fold_kills"],
                "turnover": summary["portfolio_diagnostics"][
                    "total_turnover_notional"
                ],
            }
            for arm, summary in summaries.items()
        },
        "checks": checks,
        "decision": "GO" if checks["go"] else "NO-GO",
    }
    _write_json(REPORT_PATH / "summary.json", result)
    print(
        f"reversal={reversal['return_pct']:+.2f}%/{reversal['mdd_pct']:.2f}% "
        f"PF={reversal_pf} folds={positive_folds}/5; "
        f"continuation={continuation['return_pct']:+.2f}% "
        f"events={result['events']['event_count']} decision={result['decision']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
