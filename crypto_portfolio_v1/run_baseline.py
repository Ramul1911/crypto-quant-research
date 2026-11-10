from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .backtest import run_backtest
from .config import REPORT_DIR, StrategyConfig
from .features import build_causal_features, describe_panel, load_market_panel

PERIODS = {
    "research_2021_2024": ("2021-01-01", "2024-12-31 23:59:59"),
    "forward_2025_latest": ("2025-01-01", None),
    "2025": ("2025-01-01", "2025-12-31 23:59:59"),
    "2026": ("2026-01-01", None),
}


def slice_features(
    features: dict[str, pd.DataFrame | pd.Series],
    start: str,
    end: str | None,
) -> dict[str, pd.DataFrame | pd.Series]:
    result: dict[str, pd.DataFrame | pd.Series] = {}
    for name, frame in features.items():
        mask = frame.index >= pd.Timestamp(start)
        if end is not None:
            mask &= frame.index <= pd.Timestamp(end)
        result[name] = frame.loc[mask]
    return result


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_market_panel()
    features = build_causal_features(panel)
    config = StrategyConfig()
    summaries: dict[str, dict] = {}

    for name, (start, end) in PERIODS.items():
        period_features = slice_features(features, start, end)
        equity, weights, stats = run_backtest(period_features, config)
        equity.to_csv(REPORT_DIR / f"baseline_equity_{name}.csv")
        weights.to_parquet(REPORT_DIR / f"baseline_weights_{name}.parquet")
        summaries[name] = stats
        print(
            f"{name}: return={stats['return_pct']:+.2f}% "
            f"MDD={stats['mdd_pct']:.2f}% "
            f"median_month={stats['median_monthly_return_pct']:+.2f}% "
            f"positive_months={stats['profitable_month_share']:.0%} "
            f"killed={stats['killed']}",
            flush=True,
        )

    report = {
        "experiment": "V1 baseline causal cross-sectional momentum",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel": describe_panel(panel),
        "lookahead": "features at close t set positions for t->t+1; no backfill",
        "periods": summaries,
    }
    path = REPORT_DIR / "baseline_summary.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
