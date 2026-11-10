from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import RAW_DIR, REPORT_DIR
from .experts import FundingCarryExpert, SlowCrossSectionalMomentumExpert
from .run_original_top50_1y import (
    TEST_END,
    TEST_START,
    aggregate_15m_panel_to_hourly_availability,
)
from .run_trend_v21 import (
    _annualized_volatility,
    _execution_data,
    _jsonable,
    _run_fold_backtest,
    _save_combined_trade_analytics,
    _save_fold_trade_analytics,
    calculate_portfolio_diagnostics,
)
from .v2_backtest import constrain_target_weights
from .v2_config import V2BacktestConfig
from .v2_features import load_v2_feature_panel
from .v2_protocol import DEVELOPMENT_FOLDS, HISTORICAL_CORE, FoldSpec
from .walk_forward import (
    evaluate_acceptance_criteria,
    stitch_oos_equity,
    summarize_walk_forward,
)

EXPERIMENT_ID = "V2-CARRY-CSMOM-001"
CORE_LONG_COUNT = 2
CORE_SHORT_COUNT = 2
TOP50_LONG_COUNT = 5
TOP50_SHORT_COUNT = 5
TOP50_ELIGIBLE_COUNT = 37
FROZEN_TOP50_MANIFEST = (
    REPORT_DIR / "original_top50_1y_causal_from_15m" / "universe_manifest.json"
)
DEFAULT_REPORT_DIR = REPORT_DIR / "v2_carry_csmom_001"
DEFAULT_CONFIG = V2BacktestConfig()


def build_hourly_signal_panel(
    panel_15m: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build right-labeled OHLC and align funding known at each right edge."""
    hourly = aggregate_15m_panel_to_hourly_availability(dict(panel_15m))
    if "funding" not in panel_15m:
        raise KeyError("15m panel is missing funding")
    funding = panel_15m["funding"]
    if not isinstance(funding.index, pd.DatetimeIndex):
        raise TypeError("funding must use a DatetimeIndex")
    # A funding publication exactly at an hourly right edge is already known
    # for that decision; unlike OHLC, it does not belong to the prior bar.
    hourly["funding"] = funding.reindex(hourly["close"].index, method="ffill")
    return hourly


def _validate_scores(scores: pd.DataFrame) -> None:
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("scores must use a DatetimeIndex")
    if not scores.index.is_monotonic_increasing or not scores.index.is_unique:
        raise ValueError("scores index must be sorted and unique")
    if not scores.columns.is_unique:
        raise ValueError("scores columns must be unique")


def ranked_equal_side_targets(
    scores: pd.DataFrame,
    *,
    decision_frequency: str,
    long_count: int,
    short_count: int,
    config: V2BacktestConfig = DEFAULT_CONFIG,
    annual_volatility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select fixed-count tails only at frozen UTC decision timestamps."""
    _validate_scores(scores)
    if long_count <= 0 or short_count <= 0:
        raise ValueError("long_count and short_count must be positive")
    if decision_frequency not in {"8h", "daily_utc"}:
        raise ValueError("decision_frequency must be '8h' or 'daily_utc'")

    if decision_frequency == "8h":
        decision_mask = (
            (scores.index.hour % 8 == 0)
            & (scores.index.minute == 0)
            & (scores.index.second == 0)
        )
    else:
        decision_mask = (
            (scores.index.hour == 0)
            & (scores.index.minute == 0)
            & (scores.index.second == 0)
        )

    rows: list[pd.Series] = []
    side_gross = config.max_gross_exposure / 2.0
    for timestamp in scores.index[decision_mask]:
        signal = scores.loc[timestamp].replace([np.inf, -np.inf], np.nan).dropna()
        longs = list(signal[signal > 0.0].nlargest(long_count).index)
        shorts = list(signal[signal < 0.0].nsmallest(short_count).index)
        target = pd.Series(0.0, index=scores.columns, name=timestamp)
        if len(longs) == long_count:
            target.loc[longs] = side_gross / long_count
        if len(shorts) == short_count:
            target.loc[shorts] = -side_gross / short_count
        volatility = (
            None
            if annual_volatility is None
            else annual_volatility.reindex(
                index=[timestamp], columns=scores.columns
            ).iloc[0]
        )
        constrained = constrain_target_weights(
            target,
            config=config,
            annual_vol=volatility,
        )
        constrained.name = timestamp
        rows.append(constrained)
    return pd.DataFrame(rows, columns=scores.columns, dtype=float)


def build_frozen_arm_targets(
    hourly_panel: Mapping[str, pd.DataFrame],
    *,
    long_count: int,
    short_count: int,
    config: V2BacktestConfig = DEFAULT_CONFIG,
    annual_volatility: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build both primary arms plus the exact inverted-carry sanity arm."""
    carry = FundingCarryExpert().generate(hourly_panel)
    csmom = SlowCrossSectionalMomentumExpert().generate(hourly_panel)
    carry_targets = ranked_equal_side_targets(
        carry.scores,
        decision_frequency="8h",
        long_count=long_count,
        short_count=short_count,
        config=config,
        annual_volatility=annual_volatility,
    )
    return {
        "funding_carry": carry_targets,
        "slow_csmom": ranked_equal_side_targets(
            csmom.scores,
            decision_frequency="daily_utc",
            long_count=long_count,
            short_count=short_count,
            config=config,
            annual_volatility=annual_volatility,
        ),
        "inverted_carry": -carry_targets,
    }


def load_frozen_top50_manifest(
    path: Path = FROZEN_TOP50_MANIFEST,
) -> dict[str, Any]:
    """Read, never regenerate, the preregistered 37-symbol diagnostic universe."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    symbols = manifest.get("eligible_symbols")
    if not isinstance(symbols, list) or len(symbols) != TOP50_ELIGIBLE_COUNT:
        raise ValueError(
            f"frozen top50 manifest must contain {TOP50_ELIGIBLE_COUNT} symbols"
        )
    if manifest.get("test_start") != TEST_START:
        raise ValueError("frozen top50 manifest test_start changed")
    if manifest.get("test_end_requested") != TEST_END:
        raise ValueError("frozen top50 manifest test_end changed")
    return manifest


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2),
        encoding="utf-8",
    )


def _promotion_metrics(
    summary: Mapping[str, Any],
    trade_summary: Mapping[str, Any],
) -> dict[str, Any]:
    fold_returns = summary.get("fold_returns_pct", {})
    positive_folds = sum(float(value) > 0.0 for value in fold_returns.values())
    profit_factor = trade_summary.get("profit_factor")
    metrics = {
        "no_terminal_kill": int(summary.get("fold_kills", 0)) == 0,
        "positive_return_after_costs": float(summary["return_pct"]) > 0.0,
        "profit_factor": profit_factor,
        "profit_factor_at_least_1_2": (
            profit_factor is not None
            and np.isfinite(float(profit_factor))
            and float(profit_factor) >= 1.2
        ),
        "positive_core_folds": positive_folds,
        "positive_core_folds_at_least_3_of_5": positive_folds >= 3,
    }
    metrics["accepted"] = all(
        (
            metrics["no_terminal_kill"],
            metrics["positive_return_after_costs"],
            metrics["profit_factor_at_least_1_2"],
            metrics["positive_core_folds_at_least_3_of_5"],
        )
    )
    metrics["decision"] = "GO" if metrics["accepted"] else "NO-GO"
    return metrics


def _active_month_metrics(
    stitched: pd.DataFrame,
    monthly_returns: pd.Series,
) -> dict[str, Any]:
    if "gross_exposure" not in stitched:
        raise ValueError("stitched equity is missing gross_exposure")
    active_periods = set(
        stitched.index[stitched["gross_exposure"].astype(float) > 1e-12].to_period(
            "M"
        )
    )
    active = monthly_returns.loc[
        [period in active_periods for period in monthly_returns.index]
    ]
    return {
        "count": int(len(active)),
        "profitable_share": (
            float((active > 0.0).mean()) if len(active) else 0.0
        ),
        "median_return_pct": (
            float(active.median()) * 100.0 if len(active) else 0.0
        ),
        "mean_return_pct": (
            float(active.mean()) * 100.0 if len(active) else 0.0
        ),
        "returns_pct": {
            str(period): float(value) * 100.0 for period, value in active.items()
        },
    }


def _run_arm_windows(
    *,
    arm: str,
    targets: pd.DataFrame,
    annual_volatility: pd.DataFrame,
    panel_15m: Mapping[str, pd.DataFrame],
    windows: Sequence[tuple[str, str, str]],
    report_dir: Path,
    config: V2BacktestConfig,
    primary: bool,
) -> dict[str, Any]:
    arm_dir = report_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    curves: list[pd.DataFrame] = []
    stats_by_window: list[dict[str, Any]] = []
    analytics_by_window: list[dict[str, Any]] = []
    names: list[str] = []

    for name, start, end in windows:
        local_targets = targets.loc[start:end].copy()
        local_volatility = annual_volatility.reindex(local_targets.index)
        opens, funding = _execution_data(panel_15m, start, end)
        equity, weights, stats = _run_fold_backtest(
            local_targets,
            opens,
            funding,
            config=config,
            annual_volatility=local_volatility,
            fold_start=start,
        )
        prefix = arm_dir / name
        local_targets.to_csv(f"{prefix}_targets.csv")
        equity.to_csv(f"{prefix}_equity.csv")
        weights.to_csv(f"{prefix}_weights.csv")
        analytics = _save_fold_trade_analytics(
            prefix,
            name,
            equity,
            weights,
            opens,
            funding,
            config,
        )
        names.append(name)
        curves.append(equity)
        stats_by_window.append(stats)
        analytics_by_window.append(analytics)

    stitched = stitch_oos_equity(
        curves,
        names,
        expected_frequency="15min",
    )
    stitched.to_csv(arm_dir / "stitched_equity.csv")
    combined = _save_combined_trade_analytics(
        arm_dir,
        arm,
        analytics_by_window,
        curves,
        stitched,
    )
    summary = summarize_walk_forward(stitched, stats_by_window)
    monthly_returns = summary["monthly_returns"]
    summary["active_months"] = _active_month_metrics(
        stitched, monthly_returns
    )
    summary["protocol_acceptance"] = evaluate_acceptance_criteria(summary)
    summary["portfolio_diagnostics"] = calculate_portfolio_diagnostics(stitched)
    summary["trades"] = combined["overall"]
    summary["experiment_id"] = EXPERIMENT_ID
    summary["arm"] = arm
    summary["evaluation"] = "primary_core_walk_forward" if primary else "secondary_top50_diagnostic"
    summary["config"] = asdict(config)
    summary["mandatory_liquidation"] = True
    summary["end_liquidations_executed"] = sum(
        bool(item.get("liquidated_at_end")) for item in stats_by_window
    )
    if primary:
        summary["promotion"] = _promotion_metrics(summary, combined["overall"])
    _json_write(arm_dir / "summary.json", summary)
    return summary


def run_primary_core_walk_forward(
    *,
    raw_dir: Path = RAW_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR / "primary_core",
    config: V2BacktestConfig = DEFAULT_CONFIG,
    folds: Sequence[FoldSpec] = DEVELOPMENT_FOLDS,
) -> dict[str, dict[str, Any]]:
    """Run the three frozen core arms over the five existing OOS test folds."""
    panel_15m = load_v2_feature_panel(HISTORICAL_CORE, interval="15m", raw_dir=raw_dir)
    hourly = build_hourly_signal_panel(panel_15m)
    annual_volatility = _annualized_volatility(hourly["close"])
    targets = build_frozen_arm_targets(
        hourly,
        long_count=CORE_LONG_COUNT,
        short_count=CORE_SHORT_COUNT,
        config=config,
        annual_volatility=annual_volatility,
    )
    windows = [(fold.name, fold.test_start, fold.test_end) for fold in folds]
    return {
        arm: _run_arm_windows(
            arm=arm,
            targets=arm_targets,
            annual_volatility=annual_volatility,
            panel_15m=panel_15m,
            windows=windows,
            report_dir=report_dir,
            config=config,
            primary=True,
        )
        for arm, arm_targets in targets.items()
    }


def run_secondary_frozen_top50(
    *,
    raw_dir: Path = RAW_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR / "secondary_top50",
    manifest_path: Path = FROZEN_TOP50_MANIFEST,
    config: V2BacktestConfig = DEFAULT_CONFIG,
) -> dict[str, dict[str, Any]]:
    """Run only the two registered hypotheses on the frozen top-50 diagnostic."""
    manifest = load_frozen_top50_manifest(manifest_path)
    symbols = tuple(manifest["eligible_symbols"])
    panel_15m = load_v2_feature_panel(symbols, interval="15m", raw_dir=raw_dir)
    hourly = build_hourly_signal_panel(panel_15m)
    annual_volatility = _annualized_volatility(hourly["close"])
    targets = build_frozen_arm_targets(
        hourly,
        long_count=TOP50_LONG_COUNT,
        short_count=TOP50_SHORT_COUNT,
        config=config,
        annual_volatility=annual_volatility,
    )
    targets.pop("inverted_carry")
    windows = [("top50_2025_08_10_to_2026_08_09", TEST_START, TEST_END)]
    return {
        arm: _run_arm_windows(
            arm=arm,
            targets=arm_targets,
            annual_volatility=annual_volatility,
            panel_15m=panel_15m,
            windows=windows,
            report_dir=report_dir,
            config=config,
            primary=False,
        )
        for arm, arm_targets in targets.items()
    }


def run_preregistered_experiment(
    *,
    raw_dir: Path = RAW_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    """Explicit experiment entry point; importing this module performs no run."""
    report_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary": run_primary_core_walk_forward(
            raw_dir=raw_dir,
            report_dir=report_dir / "primary_core",
        ),
        "secondary": run_secondary_frozen_top50(
            raw_dir=raw_dir,
            report_dir=report_dir / "secondary_top50",
        ),
    }
    _json_write(report_dir / "summary.json", result)
    return result


def write_existing_decision_summary(
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    """Combine already-saved stages without recomputing or changing parameters."""
    paths = {
        "core_funding_carry": report_dir
        / "primary_core"
        / "funding_carry"
        / "summary.json",
        "core_slow_csmom": report_dir
        / "primary_core"
        / "slow_csmom"
        / "summary.json",
        "core_inverted_control": report_dir
        / "primary_core"
        / "inverted_carry"
        / "summary.json",
        "top50_funding_carry": report_dir
        / "secondary_top50"
        / "funding_carry"
        / "summary.json",
        "top50_slow_csmom": report_dir
        / "secondary_top50"
        / "slow_csmom"
        / "summary.json",
    }
    if not all(path.exists() for path in paths.values()):
        return {}
    loaded = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }

    def snapshot(summary: Mapping[str, Any]) -> dict[str, Any]:
        trades = summary["trades"]
        return {
            "return_pct": summary["return_pct"],
            "mdd_pct": summary["mdd_pct"],
            "profit_factor": trades["profit_factor"],
            "positive_folds": sum(
                float(value) > 0.0
                for value in summary["fold_returns_pct"].values()
            ),
            "fold_count": len(summary["fold_returns_pct"]),
            "median_monthly_return_pct": summary[
                "median_monthly_return_pct"
            ],
            "active_months": summary["active_months"],
        }

    decision = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": {name: snapshot(summary) for name, summary in loaded.items()},
        "primary_decisions": {
            "funding_carry": loaded["core_funding_carry"]["promotion"][
                "decision"
            ],
            "slow_csmom": loaded["core_slow_csmom"]["promotion"]["decision"],
        },
        "inverted_carry_is_control_only": True,
        "final_decision": "NO-GO",
        "reason": (
            "Both preregistered primary arms failed core walk-forward "
            "promotion criteria; top50 is diagnostic only."
        ),
    }
    _json_write(report_dir / "decision_summary.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Run frozen {EXPERIMENT_ID}")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--stage",
        choices=("core", "top50", "all"),
        default="all",
        help="Run primary first; run top50 only after primary outputs are saved.",
    )
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "core":
        result: dict[str, Any] = {
            "experiment_id": EXPERIMENT_ID,
            "primary": run_primary_core_walk_forward(
                raw_dir=args.raw_dir,
                report_dir=args.report_dir / "primary_core",
            ),
        }
    elif args.stage == "top50":
        result = {
            "experiment_id": EXPERIMENT_ID,
            "secondary": run_secondary_frozen_top50(
                raw_dir=args.raw_dir,
                report_dir=args.report_dir / "secondary_top50",
            ),
        }
    else:
        result = run_preregistered_experiment(
            raw_dir=args.raw_dir, report_dir=args.report_dir
        )
    _json_write(args.report_dir / f"summary_{args.stage}.json", result)
    write_existing_decision_summary(args.report_dir)


if __name__ == "__main__":
    main()
