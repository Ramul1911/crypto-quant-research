from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from .config import RAW_DIR, REPORT_DIR
from .data_quality import check_ohlcv_quality
from .universe import FAPI
from .v2_protocol import (
    DEVELOPMENT_END,
    HISTORICAL_CORE,
    HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE,
    HISTORICAL_EXPANSION_CANDIDATES_ALL,
    HISTORICAL_EXPANSION_CANDIDATES_WAVE1,
    HISTORICAL_EXPANSION_CANDIDATES_WAVE2,
    MAX_ELIGIBILITY_GAP_RATE,
    RESEARCH_START,
)

# Wave-1 lock file kept immutable for the prior 16-set experiment.
WAVE1_ELIGIBILITY_PATH = REPORT_DIR / "universe_expansion_eligibility.json"
# Default for combined wave-1+wave-2 historical expansion claims.
DEFAULT_ELIGIBILITY_PATH = REPORT_DIR / "universe_expansion_eligibility_wave2.json"
WAVE2_ELIGIBILITY_PATH = DEFAULT_ELIGIBILITY_PATH


def _utc_naive(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def fetch_active_usdt_perpetuals(
    session: requests.Session | None = None,
) -> set[str]:
    """Return currently TRADING USDT-M perpetual symbols from Binance."""
    client = session or requests.Session()
    response = client.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=30)
    response.raise_for_status()
    payload = response.json()
    return {
        item["symbol"]
        for item in payload.get("symbols", [])
        if item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
    }


def evaluate_symbol_eligibility(
    symbol: str,
    bars: pd.DataFrame | None,
    funding: pd.DataFrame | None,
    *,
    active_symbols: set[str] | None,
    research_start: str = RESEARCH_START,
    development_end: str = DEVELOPMENT_END,
    max_gap_rate: float = MAX_ELIGIBILITY_GAP_RATE,
    interval: str = "1h",
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Score one expansion candidate using data gates only (no returns)."""
    reasons: list[str] = []
    report: dict[str, Any] = {
        "symbol": symbol,
        "eligible": False,
        "reasons": reasons,
        "bars_path_exists": bars is not None,
        "funding_path_exists": funding is not None,
        "active_trading": None,
        "listing_start": None,
        "last_bar_start": None,
        "gap_count": None,
        "gap_rate": None,
        "rows": 0,
        "funding_rows": 0 if funding is None else int(len(funding)),
        "quality_ok_ignoring_gaps": None,
    }

    if active_symbols is not None:
        report["active_trading"] = symbol in active_symbols
        if not report["active_trading"]:
            reasons.append("not_active_usdt_perpetual")

    if bars is None:
        reasons.append("missing_1h_bars")
        return report
    if funding is None:
        reasons.append("missing_funding")

    # Allow the final development-hour bar to finish after DEVELOPMENT_END.
    quality_as_of = as_of or (
        _utc_naive(development_end) + pd.Timedelta(hours=1)
    )
    quality = check_ohlcv_quality(
        bars,
        interval,
        as_of=quality_as_of,
    )
    report["listing_start"] = quality["listing_start"]
    report["last_bar_start"] = quality["last_bar_start"]
    report["gap_count"] = quality["gap_count"]
    report["rows"] = quality["rows"]
    report["quality_ok_ignoring_gaps"] = not (
        quality["duplicate_count"]
        or quality["unfinished_bar_count"]
        or quality["ohlc_violation_count"]
    )

    if quality["rows"] == 0 or quality["listing_start"] is None:
        reasons.append("empty_bars")
        return report

    listing = _utc_naive(quality["listing_start"])
    if listing > _utc_naive(research_start):
        reasons.append("listing_after_research_start")

    last_bar = _utc_naive(quality["last_bar_start"])
    coverage_floor = _utc_naive(development_end).normalize() - pd.Timedelta(days=1)
    if last_bar < coverage_floor:
        reasons.append("coverage_ends_before_development_end")

    expected_span = pd.date_range(
        max(listing, _utc_naive(research_start)),
        min(last_bar, _utc_naive(development_end)),
        freq=pd.Timedelta(hours=1),
    )
    expected_count = max(len(expected_span), 1)
    gap_rate = float(quality["gap_count"]) / float(expected_count)
    report["gap_rate"] = gap_rate
    if gap_rate > max_gap_rate:
        reasons.append("gap_rate_above_limit")

    if quality["duplicate_count"]:
        reasons.append("duplicate_timestamps")
    if quality["unfinished_bar_count"]:
        reasons.append("unfinished_bars")
    if quality["ohlc_violation_count"]:
        reasons.append("ohlc_violations")
    if funding is not None and len(funding) == 0:
        reasons.append("empty_funding")

    report["eligible"] = len(reasons) == 0
    return report


def audit_expansion_candidates(
    *,
    raw_dir: Path = RAW_DIR,
    candidates: Sequence[str] = HISTORICAL_EXPANSION_CANDIDATES_ALL,
    active_symbols: set[str] | None = None,
    fetch_active: bool = True,
    research_start: str = RESEARCH_START,
    development_end: str = DEVELOPMENT_END,
    max_gap_rate: float = MAX_ELIGIBILITY_GAP_RATE,
    interval: str = "1h",
    as_of: str | pd.Timestamp | None = None,
    output_path: Path | None = DEFAULT_ELIGIBILITY_PATH,
    policy: str = "V2-UNIVERSE-004",
) -> dict[str, Any]:
    """Run the mechanical eligibility audit and optionally persist the lock file."""
    if active_symbols is None and fetch_active:
        active_symbols = fetch_active_usdt_perpetuals()

    symbol_reports: list[dict[str, Any]] = []
    eligible: list[str] = []
    rejected: list[dict[str, Any]] = []

    for symbol in candidates:
        bars_path = raw_dir / f"{symbol}_{interval}.parquet"
        funding_path = raw_dir / f"{symbol}_funding.parquet"
        bars = pd.read_parquet(bars_path) if bars_path.exists() else None
        funding = pd.read_parquet(funding_path) if funding_path.exists() else None
        report = evaluate_symbol_eligibility(
            symbol,
            bars,
            funding,
            active_symbols=active_symbols,
            research_start=research_start,
            development_end=development_end,
            max_gap_rate=max_gap_rate,
            interval=interval,
            as_of=as_of,
        )
        symbol_reports.append(report)
        if report["eligible"]:
            eligible.append(symbol)
        else:
            rejected.append(
                {"symbol": symbol, "reasons": list(report["reasons"])}
            )

    expanded_symbols = list(HISTORICAL_CORE) + eligible
    wave1_eligible = [
        symbol for symbol in eligible if symbol in HISTORICAL_EXPANSION_CANDIDATES_WAVE1
    ]
    wave2_eligible = [
        symbol for symbol in eligible if symbol in HISTORICAL_EXPANSION_CANDIDATES_WAVE2
    ]
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": policy,
        "selection_type": "v2_historical_expanded_eligible",
        "research_start": research_start,
        "development_end": development_end,
        "max_gap_rate": max_gap_rate,
        "interval": interval,
        "historical_core": list(HISTORICAL_CORE),
        "expansion_candidates_wave1": list(HISTORICAL_EXPANSION_CANDIDATES_WAVE1),
        "expansion_candidates_wave2": list(HISTORICAL_EXPANSION_CANDIDATES_WAVE2),
        "expansion_candidates": list(candidates),
        "eligible_expansion_symbols": eligible,
        "eligible_wave1_symbols": wave1_eligible,
        "eligible_wave2_symbols": wave2_eligible,
        "rejected_expansion_symbols": rejected,
        "expanded_symbols": expanded_symbols,
        "candidate_universe_before_audit": list(
            HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE
        ),
        "symbol_reports": symbol_reports,
        "warning": (
            "Eligible set is locked by data gates only. Do not add or remove "
            "symbols after inspecting OOS returns for this experiment."
        ),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        manifest["output_path"] = str(output_path)
    return manifest


def load_eligibility_manifest(
    path: Path | None = DEFAULT_ELIGIBILITY_PATH,
) -> dict[str, Any]:
    destination = path or DEFAULT_ELIGIBILITY_PATH
    if not destination.exists():
        raise FileNotFoundError(
            f"Missing eligibility manifest at {destination}; "
            "run python -m crypto_portfolio_v1.universe_eligibility"
        )
    return json.loads(destination.read_text(encoding="utf-8"))


def resolve_universe_symbols(
    universe: str,
    *,
    eligibility_path: Path | None = DEFAULT_ELIGIBILITY_PATH,
    exclude: Sequence[str] = (),
) -> tuple[str, ...]:
    """Resolve CLI universe presets to an ordered symbol tuple."""
    excluded = set(exclude)
    if universe == "historical_core":
        symbols = HISTORICAL_CORE
    elif universe == "historical_expanded_candidates":
        symbols = HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE
    elif universe == "historical_expanded":
        manifest = load_eligibility_manifest(eligibility_path)
        symbols = tuple(manifest["expanded_symbols"])
    else:
        raise ValueError(
            "universe must be historical_core, historical_expanded, "
            "or historical_expanded_candidates"
        )
    return tuple(symbol for symbol in symbols if symbol not in excluded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mechanical eligibility audit for historical expansion candidates."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ELIGIBILITY_PATH,
    )
    parser.add_argument(
        "--wave",
        choices=("all", "wave1", "wave2"),
        default="all",
        help="Which pre-declared expansion whitelist to audit.",
    )
    parser.add_argument(
        "--skip-active-check",
        action="store_true",
        help="Treat all candidates as active (offline / unit tests).",
    )
    args = parser.parse_args()
    if args.wave == "wave1":
        candidates = HISTORICAL_EXPANSION_CANDIDATES_WAVE1
        policy = "V2-UNIVERSE-001"
    elif args.wave == "wave2":
        candidates = HISTORICAL_EXPANSION_CANDIDATES_WAVE2
        policy = "V2-UNIVERSE-004"
    else:
        candidates = HISTORICAL_EXPANSION_CANDIDATES_ALL
        policy = "V2-UNIVERSE-004"
    active_symbols = set(candidates) if args.skip_active_check else None
    manifest = audit_expansion_candidates(
        raw_dir=args.raw_dir,
        candidates=candidates,
        fetch_active=not args.skip_active_check,
        active_symbols=active_symbols,
        output_path=args.output,
        policy=policy,
    )
    print(
        "eligible="
        + (",".join(manifest["eligible_expansion_symbols"]) or "none"),
        flush=True,
    )
    print(
        "rejected="
        + (
            ",".join(
                item["symbol"] for item in manifest["rejected_expansion_symbols"]
            )
            or "none"
        ),
        flush=True,
    )
    print(f"saved -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
