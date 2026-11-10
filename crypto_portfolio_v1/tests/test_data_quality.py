from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from crypto_portfolio_v1.collectors.liquidations import (
    parse_and_persist_liquidation_event,
)
from crypto_portfolio_v1.collectors.open_interest import (
    parse_open_interest_snapshot,
    persist_open_interest_snapshot,
)
from crypto_portfolio_v1.data import aggregate_1h_to_4h, keep_complete_bars
from crypto_portfolio_v1.data_quality import (
    check_ohlcv_quality,
    generate_data_manifest,
)
from crypto_portfolio_v1.universe import (
    archive_universe_snapshot,
    historical_test_universe,
    load_archived_universe,
)
from crypto_portfolio_v1.v2_protocol import HISTORICAL_CORE


def _bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0 + i for i in range(len(index))],
            "high": [11.0 + i for i in range(len(index))],
            "low": [9.0 + i for i in range(len(index))],
            "close": [10.5 + i for i in range(len(index))],
            "volume": [100.0 + i for i in range(len(index))],
        },
        index=index,
    )


def test_quality_detects_duplicates_gaps_unfinished_and_bad_ohlc() -> None:
    index = pd.DatetimeIndex(
        [
            "2026-01-01 00:00",
            "2026-01-01 01:00",
            "2026-01-01 01:00",
            "2026-01-01 03:00",
            "2026-01-01 04:00",
        ]
    )
    frame = _bars(index)
    frame.iloc[3, frame.columns.get_loc("high")] = 1.0

    report = check_ohlcv_quality(
        frame,
        "1h",
        as_of="2026-01-01 04:30",
    )

    assert report["duplicate_count"] == 1
    assert report["gap_count"] == 1
    assert report["missing_timestamps"] == ["2026-01-01T02:00:00"]
    assert report["unfinished_bar_count"] == 1
    assert report["ohlc_violation_count"] == 1
    assert report["listing_start"] == "2026-01-01T00:00:00"
    assert not report["ok"]


def test_quality_manifest_is_json_serializable(tmp_path) -> None:
    frame = _bars(pd.date_range("2026-01-01", periods=3, freq="1h"))
    path = tmp_path / "manifest.json"

    manifest = generate_data_manifest(
        {"BTCUSDT": frame},
        "1h",
        as_of="2026-01-01 03:00",
        output_path=path,
    )

    assert manifest["all_ok"]
    assert json.loads(path.read_text(encoding="utf-8"))["symbols"]["BTCUSDT"]["rows"] == 3


def test_complete_bar_filter_and_causal_4h_aggregation() -> None:
    index = pd.date_range("2026-01-01", periods=9, freq="1h")
    frame = _bars(index)

    complete = keep_complete_bars(frame, "1h", "2026-01-01 08:30")
    result = aggregate_1h_to_4h(complete)

    assert list(result.index) == [
        pd.Timestamp("2026-01-01 04:00"),
        pd.Timestamp("2026-01-01 08:00"),
    ]
    assert result.iloc[0]["open"] == frame.iloc[0]["open"]
    assert result.iloc[0]["close"] == frame.iloc[3]["close"]
    assert result.iloc[0]["high"] == frame.iloc[:4]["high"].max()
    assert result.iloc[0]["volume"] == frame.iloc[:4]["volume"].sum()


def test_point_in_time_universe_archive_and_historical_policy(tmp_path) -> None:
    manifest = {
        "selected_at_utc": "2026-08-09T12:00:00+00:00",
        "core": list(HISTORICAL_CORE),
        "selected_satellites": ["SUIUSDT"],
        "symbols": list(HISTORICAL_CORE) + ["SUIUSDT"],
        "data_availability": "forward_only",
    }
    path = archive_universe_snapshot(manifest, tmp_path)

    assert path.name == "2026-08-09.json"
    assert load_archived_universe("2026-08-09", tmp_path)["symbols"][-1] == "SUIUSDT"
    historical = historical_test_universe()
    assert historical["symbols"] == list(HISTORICAL_CORE)
    assert historical["selected_satellites"] == []


def test_forward_only_oi_and_liquidation_persistence(tmp_path) -> None:
    observed_at = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)
    oi = parse_open_interest_snapshot(
        {"symbol": "BTCUSDT", "openInterest": "123.5"},
        collected_at=observed_at,
    )
    oi_path = persist_open_interest_snapshot(oi, tmp_path / "oi")

    message = {
        "e": "forceOrder",
        "E": 1786276800000,
        "o": {
            "s": "BTCUSDT",
            "S": "SELL",
            "o": "LIMIT",
            "f": "IOC",
            "q": "2.5",
            "p": "60000",
            "ap": "59990",
            "X": "FILLED",
            "T": 1786276800000,
            "z": "2.5",
        },
    }
    liquidation, liquidation_path = parse_and_persist_liquidation_event(
        message,
        output_dir=tmp_path / "liquidations",
        received_at=observed_at,
    )

    assert oi["availability"] == "forward_only"
    assert oi_path.read_text(encoding="utf-8").count("\n") == 1
    assert liquidation["availability"] == "forward_only"
    assert liquidation["average_price"] == 59990.0
    assert liquidation_path.exists()
