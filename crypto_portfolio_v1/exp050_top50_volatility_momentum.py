from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RAW_DIR, REPORT_DIR
from .exp049_multiasset_ml_allocator import (
    INDIVIDUAL_CONFIG,
    PORTFOLIO_CONFIG,
    _normalize_inverse_vol,
    annualized_volatility,
    build_equal_risk_targets,
    build_individual_targets,
    decision_index,
    momentum_direction,
)
from .run_carry_csmom import (
    FROZEN_TOP50_MANIFEST,
    _run_arm_windows,
    build_hourly_signal_panel,
    load_frozen_top50_manifest,
)
from .run_original_top50_1y import TEST_END, TEST_START
from .run_trend_v21 import _execution_data, _jsonable, _run_fold_backtest
from .v2_features import load_v2_feature_panel

EXPERIMENT_ID = "EXP-050"
REPORT_PATH = REPORT_DIR / "exp050_top50_volatility_momentum"
QUARTILE_SHARE = 0.25


def build_volatility_bucket_targets(
    direction: pd.DataFrame,
    volatility: pd.DataFrame,
    bucket: str,
) -> pd.DataFrame:
    if bucket not in {"high", "low", "middle"}:
        raise ValueError("bucket must be high, low or middle")
    rows: list[pd.Series] = []
    for timestamp in decision_index(direction.index):
        side = direction.loc[timestamp]
        vol = volatility.loc[timestamp]
        usable = side.notna() & side.ne(0.0) & vol.notna() & vol.gt(0.0)
        ranked = vol.loc[usable].sort_values(ascending=False)
        count = max(1, int(np.ceil(len(ranked) * QUARTILE_SHARE)))
        if bucket == "high":
            selected = list(ranked.index[:count])
        elif bucket == "low":
            selected = list(ranked.index[-count:])
        else:
            selected = list(ranked.index[count:-count]) if len(ranked) > 2 * count else []
        target = _normalize_inverse_vol(side, vol, selected=selected)
        target.name = timestamp
        rows.append(target)
    return pd.DataFrame(rows, columns=direction.columns, dtype=float)


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

    all_targets = build_equal_risk_targets(direction, volatility)
    high_targets = build_volatility_bucket_targets(direction, volatility, "high")
    middle_targets = build_volatility_bucket_targets(direction, volatility, "middle")
    low_targets = build_volatility_bucket_targets(direction, volatility, "low")
    windows = [("frozen_top50_1y", TEST_START, TEST_END)]

    portfolio_summaries: dict[str, dict[str, Any]] = {}
    for arm, targets in (
        ("all_equal_risk", all_targets),
        ("high_vol_quartile", high_targets),
        ("middle_vol", middle_targets),
        ("low_vol_quartile", low_targets),
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
        summary["evaluation"] = "frozen_top50_volatility_bucket"
        _write_json(REPORT_PATH / arm / "summary.json", summary)
        portfolio_summaries[arm] = summary

    individual_rows: list[dict[str, Any]] = []
    for position, symbol in enumerate(symbols, start=1):
        print(f"[{position:02d}/{len(symbols)}] {symbol}", flush=True)
        targets = build_individual_targets(direction, volatility, symbol)[[symbol]]
        one_symbol_panel = {
            "open": panel_15m["open"][[symbol]],
            "funding": panel_15m["funding"][[symbol]],
        }
        opens, funding = _execution_data(one_symbol_panel, TEST_START, TEST_END)
        local_volatility = volatility[[symbol]].reindex(targets.index)
        equity, _, stats = _run_fold_backtest(
            targets.loc[TEST_START:TEST_END],
            opens,
            funding,
            config=INDIVIDUAL_CONFIG,
            annual_volatility=local_volatility.loc[TEST_START:TEST_END],
            fold_start=TEST_START,
        )
        test_volatility = volatility.loc[TEST_START:TEST_END, symbol].dropna()
        individual_rows.append(
            {
                "symbol": symbol,
                "return_pct": float(stats["return_pct"]),
                "mdd_pct": float(stats["mdd_pct"]),
                "median_annual_volatility": float(test_volatility.median()),
                "mean_annual_volatility": float(test_volatility.mean()),
                "turnover": float(stats["total_turnover"]),
                "fees": float(stats["total_fee"]),
                "slippage": float(stats["total_slippage"]),
                "funding_pnl": float(equity["funding_pnl"].sum()),
                "killed": bool(stats["killed"]),
            }
        )

    individuals = pd.DataFrame(individual_rows)
    individuals["volatility_rank_desc"] = individuals[
        "median_annual_volatility"
    ].rank(ascending=False, method="first").astype(int)
    individuals["return_rank_desc"] = individuals["return_pct"].rank(
        ascending=False,
        method="first",
    ).astype(int)
    individuals = individuals.sort_values("median_annual_volatility", ascending=False)
    individuals.to_csv(REPORT_PATH / "individual_symbol_results.csv", index=False)

    spearman = float(
        individuals["median_annual_volatility"].corr(
            individuals["return_pct"],
            method="spearman",
        )
    )
    quintile_count = max(1, int(np.ceil(len(individuals) * 0.20)))
    top_quintile = individuals.nlargest(quintile_count, "median_annual_volatility")
    bottom_quintile = individuals.nsmallest(quintile_count, "median_annual_volatility")
    high = portfolio_summaries["high_vol_quartile"]
    low = portfolio_summaries["low_vol_quartile"]
    all_symbols = portfolio_summaries["all_equal_risk"]
    checks = {
        "high_return_beats_all": bool(high["return_pct"] > all_symbols["return_pct"]),
        "high_return_beats_low": bool(high["return_pct"] > low["return_pct"]),
        "high_mdd_within_5pp_of_low": bool(high["mdd_pct"] >= low["mdd_pct"] - 5.0),
        "spearman_at_least_0_25": bool(spearman >= 0.25),
        "top_vol_quintile_median_return_beats_bottom": bool(
            top_quintile["return_pct"].median()
            > bottom_quintile["return_pct"].median()
        ),
    }
    checks["supported"] = bool(all(checks.values()))
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "manifest_snapshot_date": manifest["snapshot_date"],
        "window": {"start": TEST_START, "end": TEST_END},
        "eligible_count": len(symbols),
        "portfolio_results": {
            arm: {
                "return_pct": result["return_pct"],
                "mdd_pct": result["mdd_pct"],
                "profit_factor": result["trades"]["profit_factor"],
                "turnover": result["portfolio_diagnostics"]["total_turnover_notional"],
            }
            for arm, result in portfolio_summaries.items()
        },
        "volatility_return_spearman": spearman,
        "top_volatility_quintile_symbols": list(top_quintile["symbol"]),
        "top_volatility_quintile_median_return_pct": float(
            top_quintile["return_pct"].median()
        ),
        "bottom_volatility_quintile_symbols": list(bottom_quintile["symbol"]),
        "bottom_volatility_quintile_median_return_pct": float(
            bottom_quintile["return_pct"].median()
        ),
        "hypothesis_checks": checks,
    }
    _write_json(REPORT_PATH / "summary.json", summary)
    print(
        f"high={high['return_pct']:+.2f}%/{high['mdd_pct']:.2f}% "
        f"all={all_symbols['return_pct']:+.2f}%/{all_symbols['mdd_pct']:.2f}% "
        f"low={low['return_pct']:+.2f}%/{low['mdd_pct']:.2f}% "
        f"spearman={spearman:+.3f} supported={checks['supported']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
