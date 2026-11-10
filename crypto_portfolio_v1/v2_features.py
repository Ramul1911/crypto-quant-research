from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .config import RAW_DIR
from .v2_protocol import HISTORICAL_CORE

V2_BAR_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
)


def _clean_index(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex")
    result = frame.copy()
    if result.index.tz is not None:
        result.index = result.index.tz_convert("UTC").tz_localize(None)
    return result[~result.index.duplicated(keep="last")].sort_index()


def build_v2_feature_panel(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    funding_by_symbol: Mapping[str, pd.DataFrame | pd.Series] | None = None,
    *,
    symbols: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Align complete hourly bars and causal last-known funding.

    The output has an identical index and column order for every field. Funding
    is forward-filled from publication time but never backfilled into history.
    """

    selected = tuple(symbols) if symbols is not None else tuple(bars_by_symbol)
    if not selected:
        raise ValueError("at least one symbol is required")
    missing_symbols = [symbol for symbol in selected if symbol not in bars_by_symbol]
    if missing_symbols:
        raise KeyError(f"missing bar data for symbols: {missing_symbols}")

    cleaned_bars: dict[str, pd.DataFrame] = {}
    for symbol in selected:
        bars = _clean_index(bars_by_symbol[symbol], f"bars[{symbol!r}]")
        missing_fields = [field for field in V2_BAR_FIELDS if field not in bars]
        if missing_fields:
            raise KeyError(f"{symbol} bars are missing fields: {missing_fields}")
        cleaned_bars[symbol] = bars

    master_index = cleaned_bars[selected[0]].index
    for symbol in selected[1:]:
        master_index = master_index.union(cleaned_bars[symbol].index)
    master_index = master_index.sort_values()

    panel: dict[str, pd.DataFrame] = {}
    for field in V2_BAR_FIELDS:
        panel[field] = pd.concat(
            [
                cleaned_bars[symbol][field]
                .astype(float)
                .reindex(master_index)
                .rename(symbol)
                for symbol in selected
            ],
            axis=1,
        ).reindex(columns=selected)

    funding_by_symbol = funding_by_symbol or {}
    aligned_funding: dict[str, pd.Series] = {}
    for symbol in selected:
        raw = funding_by_symbol.get(symbol)
        if raw is None:
            aligned_funding[symbol] = pd.Series(
                index=master_index,
                dtype=float,
                name=symbol,
            )
            continue
        if isinstance(raw, pd.DataFrame):
            if "funding_rate" not in raw:
                raise KeyError(f"{symbol} funding data is missing 'funding_rate'")
            series = raw["funding_rate"]
        else:
            series = raw
        series = _clean_index(series.to_frame("funding_rate"), f"funding[{symbol!r}]")[
            "funding_rate"
        ].astype(float)
        # Reindex with ffill handles observations between bar timestamps and
        # leaves bars before the first publication as NaN.
        aligned_funding[symbol] = series.reindex(master_index, method="ffill").rename(
            symbol
        )
    panel["funding"] = pd.concat(aligned_funding.values(), axis=1).reindex(
        index=master_index,
        columns=selected,
    )
    return panel


def load_v2_feature_panel(
    symbols: Sequence[str] = HISTORICAL_CORE,
    *,
    interval: str = "1h",
    raw_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the fixed historical core with all fields needed by V2 experts."""

    root = Path(raw_dir) if raw_dir is not None else RAW_DIR
    selected = tuple(symbols)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    funding_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in selected:
        bars_path = root / f"{symbol}_{interval}.parquet"
        if not bars_path.exists():
            raise FileNotFoundError(f"Missing {bars_path}; run the data downloader")
        bars_by_symbol[symbol] = pd.read_parquet(bars_path)

        funding_path = root / f"{symbol}_funding.parquet"
        if funding_path.exists():
            funding_by_symbol[symbol] = pd.read_parquet(funding_path)

    return build_v2_feature_panel(
        bars_by_symbol,
        funding_by_symbol,
        symbols=selected,
    )


# Explicit aliases make the V2 entry point easy to discover without changing
# the original V1 feature module.
load_market_panel_v2 = load_v2_feature_panel
build_v2_features = build_v2_feature_panel
