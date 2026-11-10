from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import DATA_DIR, RAW_DIR, DataConfig
from .universe import FAPI, select_live_universe
from .v2_protocol import (
    HISTORICAL_CORE,
    HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE,
)

INTERVAL_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _request(
    session: requests.Session,
    path: str,
    params: dict[str, Any],
    retries: int = 6,
) -> Any:
    delay = 0.5
    for attempt in range(retries):
        response = session.get(
            f"{FAPI}{path}",
            params=params,
            timeout=30,
        )
        if response.status_code not in (418, 429) and response.status_code < 500:
            response.raise_for_status()
            return response.json()
        if attempt == retries - 1:
            response.raise_for_status()
        time.sleep(delay)
        delay = min(delay * 2.0, 10.0)
    raise RuntimeError(f"Unable to fetch {path}")


def _utc_ms(value: str | pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def _utc_naive(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def keep_complete_bars(
    bars: pd.DataFrame,
    interval: str,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return only bars whose full interval was known by ``as_of``."""
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must use a DatetimeIndex of candle open times")
    cutoff = _utc_naive(
        pd.Timestamp.now(tz="UTC") if as_of is None else as_of
    )
    duration = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    index = bars.index
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    complete = index + duration <= cutoff
    result = bars.loc[complete].copy()
    result.index = index[complete]
    result.index.name = bars.index.name
    return result


def aggregate_1h_to_4h(
    bars_1h: pd.DataFrame,
    *,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Aggregate closed 1h bars into causal 4h bars timestamped at availability.

    A row stamped 04:00 contains the 00:00, 01:00, 02:00 and 03:00
    one-hour candles.  This close-time labeling prevents a 00:00 decision from
    observing data that is only known four hours later.
    """
    missing = [column for column in OHLCV_COLUMNS if column not in bars_1h]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    if not isinstance(bars_1h.index, pd.DatetimeIndex):
        raise TypeError("bars_1h must use a DatetimeIndex")
    if bars_1h.index.has_duplicates:
        raise ValueError("bars_1h contains duplicate timestamps")

    ordered = bars_1h.sort_index()
    aggregations: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for column in ordered.columns:
        if column not in aggregations:
            aggregations[column] = "sum" if column in {
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
            } else "last"

    grouped = ordered.resample(
        "4h",
        closed="left",
        label="right",
        origin="epoch",
    )
    result = grouped.agg(aggregations)
    counts = grouped["close"].count()
    if drop_incomplete:
        result = result.loc[counts == 4]
    return result.dropna(subset=list(OHLCV_COLUMNS))


causal_4h_from_1h = aggregate_1h_to_4h


def download_futures_klines(
    symbol: str,
    config: DataConfig,
    force: bool = False,
) -> pd.DataFrame:
    if config.interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {config.interval}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{symbol}_{config.interval}.parquet"
    existing: pd.DataFrame | None = None
    if path.exists() and not force:
        existing = pd.read_parquet(path).sort_index()

    step_ms = INTERVAL_MS[config.interval]
    start_ms = _utc_ms(config.start)
    if existing is not None and not existing.empty:
        start_ms = int(existing.index[-1].timestamp() * 1000) + step_ms

    now = pd.Timestamp.now(tz="UTC")
    now_ms = int(now.timestamp() * 1000)
    session = requests.Session()
    rows: list[list[Any]] = []
    cursor = start_ms
    while cursor < now_ms:
        payload = _request(
            session,
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": config.interval,
                "startTime": cursor,
                "limit": 1500,
            },
        )
        if not payload:
            break
        rows.extend(payload)
        next_cursor = int(payload[-1][0]) + step_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(config.request_pause_seconds)

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    if rows:
        fresh = pd.DataFrame(rows, columns=columns)
        fresh["timestamp"] = pd.to_datetime(
            fresh["open_time"].astype("int64"),
            unit="ms",
            utc=True,
        ).dt.tz_convert(None)
        numeric = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        fresh[numeric] = fresh[numeric].astype(float)
        # A signal at timestamp t may only use a candle after its close.
        fresh = fresh[fresh["close_time"].astype("int64") < now_ms]
        fresh = fresh.set_index("timestamp")[numeric]
        combined = (
            pd.concat([existing, fresh])
            if existing is not None
            else fresh
        )
    elif existing is not None:
        combined = existing
    else:
        raise RuntimeError(f"No kline data returned for {symbol}")

    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined = keep_complete_bars(combined, config.interval, as_of=now)
    combined.to_parquet(path)
    return combined


def download_funding_history(
    symbol: str,
    start: str,
    force: bool = False,
) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{symbol}_funding.parquet"
    existing: pd.DataFrame | None = None
    if path.exists() and not force:
        existing = pd.read_parquet(path).sort_index()

    cursor = _utc_ms(start)
    if existing is not None and not existing.empty:
        cursor = int(existing.index[-1].timestamp() * 1000) + 1
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    session = requests.Session()
    records: list[dict[str, Any]] = []
    while cursor < now_ms:
        payload = _request(
            session,
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": now_ms,
                "limit": 1000,
            },
        )
        if not payload:
            break
        records.extend(payload)
        next_cursor = int(payload[-1]["fundingTime"]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.08)

    if records:
        fresh = pd.DataFrame(records)
        fresh["timestamp"] = pd.to_datetime(
            fresh["fundingTime"].astype("int64"),
            unit="ms",
            utc=True,
        ).dt.tz_convert(None)
        fresh["funding_rate"] = fresh["fundingRate"].astype(float)
        fresh = fresh.set_index("timestamp")[["funding_rate"]]
        combined = (
            pd.concat([existing, fresh])
            if existing is not None
            else fresh
        )
    elif existing is not None:
        combined = existing
    else:
        combined = pd.DataFrame(
            {"funding_rate": pd.Series(dtype=float)},
            index=pd.DatetimeIndex([], name="timestamp"),
        )
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(path)
    return combined


def load_or_select_universe() -> dict[str, Any]:
    path = DATA_DIR / "universe_latest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return select_live_universe()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--interval", default="1h", choices=sorted(INTERVAL_MS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument(
        "--universe",
        choices=(
            "live",
            "historical_core",
            "historical_expanded_candidates",
            "historical_expanded",
        ),
        default=None,
        help=(
            "Preset symbol universe. historical_expanded_candidates downloads "
            "the locked mid-cap whitelist before eligibility audit."
        ),
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif args.universe == "historical_core":
        symbols = list(HISTORICAL_CORE)
    elif args.universe == "historical_expanded_candidates":
        symbols = list(HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE)
    elif args.universe == "historical_expanded":
        from .universe_eligibility import resolve_universe_symbols

        symbols = list(resolve_universe_symbols("historical_expanded"))
    elif args.universe == "live":
        symbols = load_or_select_universe()["symbols"]
    else:
        symbols = load_or_select_universe()["symbols"]
    config = DataConfig(interval=args.interval, start=args.start)
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "start": config.start,
            "interval": config.interval,
        },
        "universe_manifest": str(DATA_DIR / "universe_latest.json"),
        "symbols": {},
    }
    for position, symbol in enumerate(symbols, start=1):
        print(f"[{position}/{len(symbols)}] {symbol} klines ...", flush=True)
        bars = download_futures_klines(symbol, config, force=args.force)
        print(f"[{position}/{len(symbols)}] {symbol} funding ...", flush=True)
        funding = download_funding_history(symbol, config.start, force=args.force)
        manifest["symbols"][symbol] = {
            "bars": len(bars),
            "bar_start": str(bars.index.min()),
            "bar_end": str(bars.index.max()),
            "funding_rows": len(funding),
            "funding_start": str(funding.index.min()) if len(funding) else None,
            "funding_end": str(funding.index.max()) if len(funding) else None,
        }
        print(
            f"    bars={len(bars):,} funding={len(funding):,} "
            f"{bars.index.min()}..{bars.index.max()}",
            flush=True,
        )

    path = DATA_DIR / f"download_manifest_{config.interval}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
