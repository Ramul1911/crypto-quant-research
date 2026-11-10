from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, RAW_DIR


def load_symbols() -> list[str]:
    manifest = json.loads(
        (DATA_DIR / "universe_latest.json").read_text(encoding="utf-8")
    )
    return list(manifest["symbols"])


def load_market_panel(
    symbols: list[str] | None = None,
    interval: str = "1h",
) -> dict[str, pd.DataFrame]:
    symbols = symbols or load_symbols()
    closes: dict[str, pd.Series] = {}
    quote_volume: dict[str, pd.Series] = {}
    funding: dict[str, pd.Series] = {}
    for symbol in symbols:
        bars_path = RAW_DIR / f"{symbol}_{interval}.parquet"
        if not bars_path.exists():
            raise FileNotFoundError(f"Missing {bars_path}; run the data downloader")
        bars = pd.read_parquet(bars_path).sort_index()
        closes[symbol] = bars["close"].rename(symbol)
        quote_volume[symbol] = bars["quote_volume"].rename(symbol)

        funding_path = RAW_DIR / f"{symbol}_funding.parquet"
        if funding_path.exists():
            rates = pd.read_parquet(funding_path).sort_index()
            funding[symbol] = rates["funding_rate"].rename(symbol)

    close_frame = pd.concat(closes.values(), axis=1).sort_index()
    volume_frame = pd.concat(quote_volume.values(), axis=1).reindex(
        close_frame.index
    )
    funding_frame = pd.concat(funding.values(), axis=1).reindex(
        close_frame.index
    )
    funding_frame = funding_frame.fillna(0.0)
    return {
        "close": close_frame,
        "quote_volume": volume_frame,
        "funding": funding_frame,
    }


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def build_causal_features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = panel["close"]
    quote_volume = panel["quote_volume"]
    returns = close.pct_change()
    momentum_24h = close.pct_change(24)
    momentum_72h = close.pct_change(72)
    momentum_168h = close.pct_change(168)
    annual_vol_72h = returns.rolling(72, min_periods=48).std() * np.sqrt(24 * 365)
    ema_168h = close.ewm(span=168, adjust=False, min_periods=168).mean()
    dollar_volume_24h = quote_volume.rolling(24, min_periods=12).sum()

    score = (
        0.35 * cross_sectional_zscore(momentum_24h)
        + 0.20 * cross_sectional_zscore(momentum_72h)
        + 0.45 * cross_sectional_zscore(momentum_168h)
    )
    trend = np.sign(close / ema_168h - 1.0)
    # A score is tradable only when relative and absolute trend agree.
    score = score.where(np.sign(score) == trend)

    return {
        "returns": returns,
        "score": score,
        "annual_vol": annual_vol_72h,
        "dollar_volume_24h": dollar_volume_24h,
        "funding": panel["funding"],
        "market_trend": trend["BTCUSDT"].fillna(0.0),
    }


def describe_panel(panel: dict[str, pd.DataFrame]) -> dict[str, Any]:
    close = panel["close"]
    return {
        "rows": len(close),
        "symbols": list(close.columns),
        "start": str(close.index.min()),
        "end": str(close.index.max()),
        "available_rows": {
            symbol: int(close[symbol].notna().sum())
            for symbol in close.columns
        },
    }

