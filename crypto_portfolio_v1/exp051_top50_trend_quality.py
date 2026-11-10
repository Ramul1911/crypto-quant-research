from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RAW_DIR, REPORT_DIR
from .exp049_multiasset_ml_allocator import (
    PORTFOLIO_CONFIG,
    _normalize_inverse_vol,
    annualized_volatility,
    build_equal_risk_targets,
    decision_index,
    exact_forward_net_labels,
    momentum_direction,
)
from .run_carry_csmom import (
    FROZEN_TOP50_MANIFEST,
    _run_arm_windows,
    build_hourly_signal_panel,
    load_frozen_top50_manifest,
)
from .run_original_top50_1y import TEST_END, TEST_START
from .run_trend_v21 import _jsonable
from .v2_features import load_v2_feature_panel

EXPERIMENT_ID = "EXP-051"
REPORT_PATH = REPORT_DIR / "exp051_top50_trend_quality"
EXP050_INDIVIDUALS = (
    REPORT_DIR / "exp050_top50_volatility_momentum" / "individual_symbol_results.csv"
)
EFFICIENCY_HORIZONS = (24, 72, 168)
QUARTILE_SHARE = 0.25


def trend_efficiency(close: pd.DataFrame) -> pd.DataFrame:
    """Equal mean causal Kaufman efficiency over frozen horizons."""
    log_close = np.log(close.where(close > 0.0))
    absolute_hourly_move = log_close.diff().abs()
    components = []
    for horizon in EFFICIENCY_HORIZONS:
        displacement = (log_close - log_close.shift(horizon)).abs()
        path = absolute_hourly_move.rolling(
            horizon,
            min_periods=horizon,
        ).sum()
        components.append(displacement.div(path.replace(0.0, np.nan)).clip(0.0, 1.0))
    return sum(components) / float(len(components))


def build_quality_bucket_targets(
    direction: pd.DataFrame,
    volatility: pd.DataFrame,
    quality: pd.DataFrame,
    bucket: str,
) -> pd.DataFrame:
    if bucket not in {"high", "low"}:
        raise ValueError("bucket must be high or low")
    rows: list[pd.Series] = []
    for timestamp in decision_index(direction.index):
        side = direction.loc[timestamp]
        vol = volatility.loc[timestamp]
        score = quality.loc[timestamp]
        usable = (
            side.notna()
            & side.ne(0.0)
            & vol.notna()
            & vol.gt(0.0)
            & score.notna()
        )
        ranked = score.loc[usable].sort_values(ascending=False)
        count = max(1, int(np.ceil(len(ranked) * QUARTILE_SHARE)))
        selected = (
            list(ranked.index[:count])
            if bucket == "high"
            else list(ranked.index[-count:])
        )
        target = _normalize_inverse_vol(side, vol, selected=selected)
        target.name = timestamp
        rows.append(target)
    return pd.DataFrame(rows, columns=direction.columns, dtype=float)


def daily_rank_ic(
    quality: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.Series:
    rows: dict[pd.Timestamp, float] = {}
    common = quality.index.intersection(labels.index)
    for timestamp in common:
        pair = pd.concat(
            [
                quality.loc[timestamp].rename("quality"),
                labels.loc[timestamp].rename("net_return"),
            ],
            axis=1,
        ).dropna()
        if len(pair) >= 5 and pair["quality"].nunique() > 1:
            rows[timestamp] = float(
                pair["quality"].corr(pair["net_return"], method="spearman")
            )
    return pd.Series(rows, dtype=float, name="rank_ic")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def main() -> None:
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    manifest = load_frozen_top50_manifest(FROZEN_TOP50_MANIFEST)
    symbols = tuple(manifest["eligible_symbols"])
    print(f"Loading frozen top-50 panel ({len(symbols)} symbols)...", flush=True)
    panel_15m = load_v2_feature_panel(symbols, interval="15m", raw_dir=RAW_DIR)
    hourly = build_hourly_signal_panel(panel_15m)
    close = hourly["close"].astype(float)
    volatility = annualized_volatility(close)
    direction = momentum_direction(close)
    quality = trend_efficiency(close)

    all_targets = build_equal_risk_targets(direction, volatility)
    high_targets = build_quality_bucket_targets(
        direction,
        volatility,
        quality,
        "high",
    )
    low_targets = build_quality_bucket_targets(
        direction,
        volatility,
        quality,
        "low",
    )
    windows = [("frozen_top50_1y", TEST_START, TEST_END)]
    summaries: dict[str, dict[str, Any]] = {}
    for arm, targets in (
        ("all_equal_risk", all_targets),
        ("high_trend_quality", high_targets),
        ("low_trend_quality", low_targets),
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
            primary=False,
        )
        summary["experiment_id"] = EXPERIMENT_ID
        summary["evaluation"] = "frozen_top50_trend_quality"
        _write_json(REPORT_PATH / arm / "summary.json", summary)
        summaries[arm] = summary

    decisions = decision_index(close.index)
    labels = exact_forward_net_labels(decisions, direction, panel_15m)
    ic = daily_rank_ic(
        quality.reindex(decisions).loc[TEST_START:TEST_END],
        labels.loc[TEST_START:TEST_END],
    )
    ic.rename_axis("timestamp").to_csv(REPORT_PATH / "daily_rank_ic.csv")

    median_quality = quality.loc[TEST_START:TEST_END].median(axis=0).rename(
        "median_trend_quality"
    )
    individuals = pd.read_csv(EXP050_INDIVIDUALS).set_index("symbol")
    individuals = individuals.join(median_quality, how="left")
    symbol_spearman = float(
        individuals["median_trend_quality"].corr(
            individuals["return_pct"],
            method="spearman",
        )
    )
    individuals.reset_index().sort_values(
        "median_trend_quality",
        ascending=False,
    ).to_csv(REPORT_PATH / "symbol_quality_vs_return.csv", index=False)

    high = summaries["high_trend_quality"]
    low = summaries["low_trend_quality"]
    all_symbols = summaries["all_equal_risk"]
    high_pf = high["trades"]["profit_factor"]
    checks = {
        "high_return_beats_all": bool(high["return_pct"] > all_symbols["return_pct"]),
        "high_return_beats_low": bool(high["return_pct"] > low["return_pct"]),
        "high_profit_factor_above_1": bool(
            high_pf is not None and float(high_pf) > 1.0
        ),
        "high_mdd_within_5pp_of_low": bool(high["mdd_pct"] >= low["mdd_pct"] - 5.0),
        "mean_daily_rank_ic_above_0_05": bool(float(ic.mean()) > 0.05),
        "symbol_quality_return_spearman_above_0_20": bool(symbol_spearman > 0.20),
    }
    checks["supported"] = bool(all(checks.values()))
    result = {
        "experiment_id": EXPERIMENT_ID,
        "window": {"start": TEST_START, "end": TEST_END},
        "eligible_count": len(symbols),
        "portfolio_results": {
            arm: {
                "return_pct": summary["return_pct"],
                "mdd_pct": summary["mdd_pct"],
                "profit_factor": summary["trades"]["profit_factor"],
                "turnover": summary["portfolio_diagnostics"]["total_turnover_notional"],
            }
            for arm, summary in summaries.items()
        },
        "daily_rank_ic": {
            "count": int(len(ic)),
            "mean": float(ic.mean()),
            "median": float(ic.median()),
            "positive_share": float((ic > 0.0).mean()),
        },
        "symbol_quality_return_spearman": symbol_spearman,
        "hypothesis_checks": checks,
    }
    _write_json(REPORT_PATH / "summary.json", result)
    print(
        f"high={high['return_pct']:+.2f}%/{high['mdd_pct']:.2f}% "
        f"all={all_symbols['return_pct']:+.2f}%/{all_symbols['mdd_pct']:.2f}% "
        f"low={low['return_pct']:+.2f}%/{low['mdd_pct']:.2f}% "
        f"IC={ic.mean():+.3f} symbol_rho={symbol_spearman:+.3f} "
        f"supported={checks['supported']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
