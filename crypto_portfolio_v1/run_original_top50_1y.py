from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RAW_DIR, REPORT_DIR
from .experts import TrendExpert
from .run_hybrid import scores_to_target_weights
from .run_trend_v21 import (
    ORIGINAL_TREND_CONFIG,
    _annualized_volatility,
    _execution_data,
    _run_fold_backtest,
    _save_fold_trade_analytics,
    calculate_portfolio_diagnostics,
)
from .trade_analytics import summarize_trade_ledger
from .universe_eligibility import fetch_active_usdt_perpetuals
from .v2_features import load_v2_feature_panel

SNAPSHOT_DATE = "2025-08-10"
TEST_START = "2025-08-10 00:00:00"
TEST_END = "2026-08-09 23:59:59"

# Point-in-time CoinMarketCap top 50 from:
# https://coinmarketcap.com/historical/20250810/
# Futures symbols encode Binance contract naming where it differs from CMC.
CMC_TOP50_20250810 = (
    (1, "BTC", "BTCUSDT"),
    (2, "ETH", "ETHUSDT"),
    (3, "XRP", "XRPUSDT"),
    (4, "USDT", None),
    (5, "BNB", "BNBUSDT"),
    (6, "SOL", "SOLUSDT"),
    (7, "USDC", None),
    (8, "DOGE", "DOGEUSDT"),
    (9, "TRX", "TRXUSDT"),
    (10, "ADA", "ADAUSDT"),
    (11, "HYPE", "HYPEUSDT"),
    (12, "LINK", "LINKUSDT"),
    (13, "XLM", "XLMUSDT"),
    (14, "SUI", "SUIUSDT"),
    (15, "BCH", "BCHUSDT"),
    (16, "HBAR", "HBARUSDT"),
    (17, "USDe", None),
    (18, "AVAX", "AVAXUSDT"),
    (19, "LTC", "LTCUSDT"),
    (20, "TON", "TONUSDT"),
    (21, "LEO", None),
    (22, "SHIB", "1000SHIBUSDT"),
    (23, "UNI", "UNIUSDT"),
    (24, "DOT", "DOTUSDT"),
    (25, "DAI", None),
    (26, "CRO", "CROUSDT"),
    (27, "ENA", "ENAUSDT"),
    (28, "PEPE", "1000PEPEUSDT"),
    (29, "BGB", None),
    (30, "XMR", "XMRUSDT"),
    (31, "AAVE", "AAVEUSDT"),
    (32, "TAO", "TAOUSDT"),
    (33, "ETC", "ETCUSDT"),
    (34, "MNT", "MNTUSDT"),
    (35, "NEAR", "NEARUSDT"),
    (36, "ONDO", "ONDOUSDT"),
    (37, "APT", "APTUSDT"),
    (38, "PI", "PIUSDT"),
    (39, "ICP", "ICPUSDT"),
    (40, "OKB", "OKBUSDT"),
    (41, "PENGU", "PENGUUSDT"),
    (42, "POL", "POLUSDT"),
    (43, "KAS", "KASUSDT"),
    (44, "ARB", "ARBUSDT"),
    (45, "ALGO", "ALGOUSDT"),
    (46, "BONK", "1000BONKUSDT"),
    (47, "USD1", None),
    (48, "VET", "VETUSDT"),
    (49, "ATOM", "ATOMUSDT"),
    (50, "GT", "GTUSDT"),
)


def _utc_naive(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _data_bounds(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not path.exists():
        return None, None
    frame = pd.read_parquet(path)
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return None, None
    index = frame.index
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index.min(), index.max()


def lock_eligible_universe(
    *,
    raw_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Lock top-50 membership using only snapshot, venue and data availability."""
    active = fetch_active_usdt_perpetuals()
    start = _utc_naive(TEST_START)
    end_floor = _utc_naive(TEST_END).normalize()
    rows: list[dict[str, Any]] = []
    eligible: list[str] = []

    for rank, cmc_symbol, futures_symbol in CMC_TOP50_20250810:
        reasons: list[str] = []
        if futures_symbol is None:
            reasons.append("stable_or_no_binance_usdt_perpetual_mapping")
            one_hour_start = one_hour_end = None
            fifteen_start = fifteen_end = None
            funding_exists = False
        else:
            if futures_symbol not in active:
                reasons.append("not_active_binance_usdt_perpetual")
            one_hour_start, one_hour_end = _data_bounds(
                raw_dir / f"{futures_symbol}_1h.parquet"
            )
            fifteen_start, fifteen_end = _data_bounds(
                raw_dir / f"{futures_symbol}_15m.parquet"
            )
            funding_path = raw_dir / f"{futures_symbol}_funding.parquet"
            funding_exists = funding_path.exists()
            if one_hour_start is None:
                reasons.append("missing_1h_data")
            elif one_hour_start > start:
                reasons.append("1h_listing_after_test_start")
            if fifteen_start is None:
                reasons.append("missing_15m_data")
            elif fifteen_start > start:
                reasons.append("15m_listing_after_test_start")
            if one_hour_end is not None and one_hour_end < end_floor:
                reasons.append("1h_coverage_ends_before_test_end")
            if fifteen_end is not None and fifteen_end < end_floor:
                reasons.append("15m_coverage_ends_before_test_end")
            if not funding_exists:
                reasons.append("missing_funding_data")

        is_eligible = futures_symbol is not None and not reasons
        if is_eligible:
            eligible.append(futures_symbol)
        rows.append(
            {
                "rank": rank,
                "cmc_symbol": cmc_symbol,
                "futures_symbol": futures_symbol,
                "eligible": is_eligible,
                "reasons": reasons,
                "one_hour_start": (
                    one_hour_start.isoformat() if one_hour_start is not None else None
                ),
                "one_hour_end": (
                    one_hour_end.isoformat() if one_hour_end is not None else None
                ),
                "fifteen_minute_start": (
                    fifteen_start.isoformat() if fifteen_start is not None else None
                ),
                "fifteen_minute_end": (
                    fifteen_end.isoformat() if fifteen_end is not None else None
                ),
                "funding_exists": funding_exists,
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://coinmarketcap.com/historical/20250810/",
        "snapshot_date": SNAPSHOT_DATE,
        "selection": "CMC top 50 at test start, filtered to active Binance USDT-M perpetuals with full local coverage",
        "test_start": TEST_START,
        "test_end_requested": TEST_END,
        "top50": rows,
        "eligible_symbols": eligible,
        "eligible_count": len(eligible),
        "warning": (
            "Universe is locked before this run's PnL. Do not remove symbols "
            "after inspecting results."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, pd.Series):
        return {str(key): float(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def label_hourly_panel_at_availability(
    panel: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Relabel Binance open-time 1h rows at candle-close availability."""
    result: dict[str, pd.DataFrame] = {}
    for field, frame in panel.items():
        shifted = frame.copy()
        if not isinstance(shifted.index, pd.DatetimeIndex):
            raise TypeError(f"{field} must use a DatetimeIndex")
        shifted.index = shifted.index + pd.Timedelta(hours=1)
        result[field] = shifted
    return result


def aggregate_15m_panel_to_hourly_availability(
    panel: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build complete 1h signal bars from four closed 15m rows."""
    if "close" not in panel:
        raise KeyError("15m panel is missing close")
    counts = panel["close"].resample(
        "1h", closed="left", label="right", origin="epoch"
    ).count()
    complete = counts.eq(4)
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trades": "sum",
        "taker_buy_base": "sum",
        "taker_buy_quote": "sum",
        "funding": "last",
    }
    result: dict[str, pd.DataFrame] = {}
    for field, frame in panel.items():
        method = aggregations.get(field, "last")
        grouped = frame.resample(
            "1h", closed="left", label="right", origin="epoch"
        )
        hourly = getattr(grouped, method)()
        result[field] = hourly.where(complete.reindex_like(hourly).fillna(False))
    return result


def run_original_top50_1y(
    *,
    raw_dir: Path = RAW_DIR,
    report_dir: Path = REPORT_DIR / "original_top50_1y_causal_from_15m",
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = lock_eligible_universe(
        raw_dir=raw_dir,
        output_path=report_dir / "universe_manifest.json",
    )
    symbols = tuple(manifest["eligible_symbols"])
    if not symbols:
        raise ValueError("no eligible top-50 symbols")

    panel_15m = load_v2_feature_panel(symbols, interval="15m", raw_dir=raw_dir)
    panel_1h = aggregate_15m_panel_to_hourly_availability(panel_15m)
    close = panel_1h["close"].astype(float)
    output = TrendExpert().generate(panel_1h)
    scores = output.scores.loc[TEST_START:TEST_END]
    confidence = output.confidence.reindex(scores.index)
    annual_volatility = _annualized_volatility(close).reindex(scores.index)
    targets = scores_to_target_weights(
        scores,
        confidence,
        annual_volatility,
        config=ORIGINAL_TREND_CONFIG,
    )
    opens, funding = _execution_data(panel_15m, TEST_START, TEST_END)
    equity, weights, stats = _run_fold_backtest(
        targets,
        opens,
        funding,
        config=ORIGINAL_TREND_CONFIG,
        annual_volatility=annual_volatility,
        fold_start=TEST_START,
    )

    prefix = report_dir / "original_top50_1y_causal_from_15m"
    targets.to_csv(f"{prefix}_targets.csv")
    equity.to_csv(f"{prefix}_equity.csv")
    weights.to_csv(f"{prefix}_weights.csv")
    analytics = _save_fold_trade_analytics(
        prefix,
        "top50_1y_causal_from_15m",
        equity,
        weights,
        opens,
        funding,
        ORIGINAL_TREND_CONFIG,
    )
    trade_summary = summarize_trade_ledger(analytics["trades"])
    portfolio = calculate_portfolio_diagnostics(equity)
    monthly = stats["calendar_monthly_returns"].astype(float)
    boundary_periods = {
        pd.Timestamp(TEST_START).to_period("M"),
        pd.Timestamp(equity.index.max()).to_period("M"),
    }
    complete_monthly = monthly.loc[
        [period not in boundary_periods for period in monthly.index]
    ]

    summary = {
        "experiment": (
            "Causal Original Trend on 15m-built 1h bars, point-in-time "
            "CMC top-50, one year"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": manifest["source"],
        "snapshot_date": SNAPSHOT_DATE,
        "test_start": TEST_START,
        "test_end_requested": TEST_END,
        "test_end_actual": str(equity.index.max()),
        "eligible_symbols": list(symbols),
        "eligible_count": len(symbols),
        "portfolio": {
            "return_pct": float(stats["return_pct"]),
            "mdd_pct": float(stats["mdd_pct"]),
            "profitable_month_share": float((monthly > 0.0).mean()),
            "median_monthly_return_pct": float(monthly.median()) * 100.0,
            "mean_monthly_return_pct": float(monthly.mean()) * 100.0,
            "complete_month_count": int(len(complete_monthly)),
            "complete_month_profitable_share": float(
                (complete_monthly > 0.0).mean()
            ),
            "complete_month_median_return_pct": (
                float(complete_monthly.median()) * 100.0
            ),
            "complete_month_mean_return_pct": (
                float(complete_monthly.mean()) * 100.0
            ),
            "monthly_returns_pct": {
                str(period): float(value) * 100.0
                for period, value in monthly.items()
            },
            **portfolio,
        },
        "trades": trade_summary["overall"],
        "universe_manifest": "universe_manifest.json",
        "methodology": {
            "strategy": "Original deterministic TrendExpert",
            "signal_timestamp": (
                "Four completed Binance 15m rows aggregated with right-edge "
                "1h availability timestamp"
            ),
            "execution": "next available 15m open",
            "costs": {
                "fee_rate": ORIGINAL_TREND_CONFIG.fee_rate,
                "slippage_rate": ORIGINAL_TREND_CONFIG.slippage_rate,
                "funding": True,
            },
            "liquidate_at_end": True,
            "post_hoc_symbol_removal": False,
        },
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    trade_summary["by_symbol"].to_csv(report_dir / "trade_summary_by_symbol.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run causal Original Trend on point-in-time CMC top-50 for one year."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR / "original_top50_1y_causal_from_15m",
    )
    args = parser.parse_args()
    summary = run_original_top50_1y(
        raw_dir=args.raw_dir,
        report_dir=args.report_dir,
    )
    portfolio = summary["portfolio"]
    print(
        f"symbols={summary['eligible_count']} "
        f"return={portfolio['return_pct']:+.2f}% "
        f"MDD={portfolio['mdd_pct']:.2f}% "
        f"median_month={portfolio['median_monthly_return_pct']:+.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
