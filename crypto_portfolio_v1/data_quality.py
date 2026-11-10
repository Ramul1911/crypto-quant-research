from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .data import INTERVAL_MS, OHLCV_COLUMNS


def _normalized_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a DatetimeIndex of candle open times")
    index = frame.index
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index


def _utc_naive(value: str | pd.Timestamp | None) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def duplicate_timestamps(frame: pd.DataFrame) -> list[str]:
    index = _normalized_index(frame)
    duplicates = index[index.duplicated()].unique()
    return [timestamp.isoformat() for timestamp in duplicates]


def missing_bar_timestamps(frame: pd.DataFrame, interval: str) -> list[str]:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    index = _normalized_index(frame).drop_duplicates().sort_values()
    if len(index) < 2:
        return []
    expected = pd.date_range(
        index[0],
        index[-1],
        freq=pd.Timedelta(milliseconds=INTERVAL_MS[interval]),
    )
    missing = expected.difference(index)
    return [timestamp.isoformat() for timestamp in missing]


def unfinished_bar_timestamps(
    frame: pd.DataFrame,
    interval: str,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> list[str]:
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    index = _normalized_index(frame)
    duration = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    unfinished = index + duration > _utc_naive(as_of)
    return [timestamp.isoformat() for timestamp in index[unfinished]]


def ohlc_invariant_violations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    missing = [column for column in OHLCV_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    index = _normalized_index(frame)
    numeric = frame.loc[:, list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    invalid = (
        numeric.isna().any(axis=1)
        | (numeric["high"] < numeric[["open", "close"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close"]].min(axis=1))
        | (numeric["high"] < numeric["low"])
        | (numeric["volume"] < 0)
    )
    violations: list[dict[str, Any]] = []
    for position in range(len(frame)):
        if bool(invalid.iloc[position]):
            violations.append(
                {
                    "timestamp": index[position].isoformat(),
                    "reason": "invalid_ohlcv",
                }
            )
    return violations


def listing_start(frame: pd.DataFrame) -> str | None:
    index = _normalized_index(frame)
    return None if len(index) == 0 else index.min().isoformat()


def check_ohlcv_quality(
    frame: pd.DataFrame,
    interval: str,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    duplicates = duplicate_timestamps(frame)
    gaps = missing_bar_timestamps(frame, interval)
    unfinished = unfinished_bar_timestamps(
        frame,
        interval,
        as_of=as_of,
    )
    violations = ohlc_invariant_violations(frame)
    return {
        "interval": interval,
        "rows": len(frame),
        "listing_start": listing_start(frame),
        "last_bar_start": (
            None
            if frame.empty
            else _normalized_index(frame).max().isoformat()
        ),
        "duplicate_count": len(duplicates),
        "duplicate_timestamps": duplicates,
        "gap_count": len(gaps),
        "missing_timestamps": gaps,
        "unfinished_bar_count": len(unfinished),
        "unfinished_bar_timestamps": unfinished,
        "ohlc_violation_count": len(violations),
        "ohlc_violations": violations,
        "ok": not (duplicates or gaps or unfinished or violations),
    }


def generate_data_manifest(
    datasets: Mapping[str, pd.DataFrame],
    interval: str,
    *,
    as_of: str | pd.Timestamp | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Create an auditable quality manifest for a symbol/frame mapping."""
    reports = {
        symbol: check_ohlcv_quality(frame, interval, as_of=as_of)
        for symbol, frame in sorted(datasets.items())
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_utc": _utc_naive(as_of).isoformat(),
        "interval": interval,
        "symbols": reports,
        "all_ok": all(report["ok"] for report in reports.values()),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
