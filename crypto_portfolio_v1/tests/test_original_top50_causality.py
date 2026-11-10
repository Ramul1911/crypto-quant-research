from __future__ import annotations

import pandas as pd

from crypto_portfolio_v1.run_original_top50_1y import (
    aggregate_15m_panel_to_hourly_availability,
    label_hourly_panel_at_availability,
)


def test_hourly_panel_is_labeled_when_candle_becomes_available() -> None:
    index = pd.date_range("2026-01-01 00:00", periods=2, freq="h")
    panel = {
        "close": pd.DataFrame({"BTCUSDT": [100.0, 101.0]}, index=index),
        "volume": pd.DataFrame({"BTCUSDT": [1.0, 2.0]}, index=index),
    }

    shifted = label_hourly_panel_at_availability(panel)

    expected = pd.date_range("2026-01-01 01:00", periods=2, freq="h")
    assert shifted["close"].index.equals(expected)
    assert shifted["close"].iloc[0, 0] == 100.0
    assert panel["close"].index.equals(index)


def test_four_15m_rows_build_one_right_labeled_hour() -> None:
    index = pd.date_range("2026-01-01 11:00", periods=5, freq="15min")
    panel = {
        "open": pd.DataFrame({"BTCUSDT": [10, 11, 12, 13, 20]}, index=index),
        "high": pd.DataFrame({"BTCUSDT": [11, 13, 14, 15, 21]}, index=index),
        "low": pd.DataFrame({"BTCUSDT": [9, 10, 11, 12, 19]}, index=index),
        "close": pd.DataFrame({"BTCUSDT": [11, 12, 13, 14, 20]}, index=index),
        "volume": pd.DataFrame({"BTCUSDT": [1, 2, 3, 4, 5]}, index=index),
    }

    hourly = aggregate_15m_panel_to_hourly_availability(panel)

    row = pd.Timestamp("2026-01-01 12:00")
    assert hourly["open"].loc[row, "BTCUSDT"] == 10
    assert hourly["high"].loc[row, "BTCUSDT"] == 15
    assert hourly["low"].loc[row, "BTCUSDT"] == 9
    assert hourly["close"].loc[row, "BTCUSDT"] == 14
    assert hourly["volume"].loc[row, "BTCUSDT"] == 10
    assert pd.isna(hourly["close"].loc["2026-01-01 13:00", "BTCUSDT"])

