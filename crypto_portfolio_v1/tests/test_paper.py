from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from crypto_portfolio_v1.paper import (
    collect_paper_snapshot,
    evaluate_paper_equity,
    initialize_paper_state,
)
from crypto_portfolio_v1.v2_protocol import PAPER_HOLDOUT_START


def test_initialize_paper_state_schedules_without_fake_run(tmp_path) -> None:
    result = initialize_paper_state(
        output_dir=tmp_path,
        initialized_at="2026-08-09T16:47:00+00:00",
    )
    assert result["status"] == "scheduled"
    assert result["runs"] == []
    assert result["paper_holdout_start"] == PAPER_HOLDOUT_START
    with pytest.raises(FileExistsError):
        initialize_paper_state(
            output_dir=tmp_path,
            initialized_at="2026-08-09T16:48:00+00:00",
        )


def test_collection_is_forward_only_immutable_and_offline(tmp_path) -> None:
    calls: list[tuple[str, str]] = []
    observed_at = datetime(2026, 8, 10, 1, 2, 3, tzinfo=timezone.utc)
    liquidation_path = tmp_path / "persisted-liquidations.jsonl"
    liquidation_path.write_text(
        json.dumps(
            {
                "event_time_utc": observed_at.isoformat(),
                "symbol": "BTCUSDT",
                "availability": "forward_only",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def universe_provider(at: datetime) -> dict:
        assert at == observed_at
        return {
            "selected_at_utc": at.isoformat(),
            "symbols": ["btcusdt", "ethusdt"],
            "availability": "forward_only",
        }

    def refresh(symbol: str, interval: str) -> pd.DataFrame:
        calls.append((symbol, interval))
        return pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.date_range("2026-08-09", periods=2, freq="h"),
        )

    def collect_oi(symbols, output_dir, at):
        assert list(symbols) == ["BTCUSDT", "ETHUSDT"]
        assert output_dir == tmp_path / "paper" / "open_interest"
        assert at == observed_at
        return [{"symbol": symbol, "open_interest": 10.0} for symbol in symbols]

    result = collect_paper_snapshot(
        output_dir=tmp_path / "paper",
        collected_at=observed_at,
        liquidation_paths=[liquidation_path],
        universe_provider=universe_provider,
        data_refresher=refresh,
        open_interest_collector=collect_oi,
    )

    assert calls == [
        ("BTCUSDT", "1h"),
        ("BTCUSDT", "15m"),
        ("ETHUSDT", "1h"),
        ("ETHUSDT", "15m"),
    ]
    assert result["live_orders"] is False
    assert result["paper_holdout_start"] == PAPER_HOLDOUT_START
    assert result["liquidations"]["event_count"] == 1
    snapshot = tmp_path / "paper" / "snapshots" / "2026-08-10.json"
    universe = tmp_path / "paper" / "universe_history" / "2026-08-10.json"
    state = json.loads(
        (tmp_path / "paper" / "paper_state.json").read_text(encoding="utf-8")
    )
    assert snapshot.exists() and universe.exists()
    assert len(state["runs"]) == 1
    assert not list((tmp_path / "paper").rglob("*.tmp"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collect_paper_snapshot(
            output_dir=tmp_path / "paper",
            collected_at=observed_at,
            universe_provider=lambda _: pytest.fail("must fail before collection"),
            data_refresher=refresh,
            open_interest_collector=collect_oi,
        )
    assert len(state["runs"]) == 1


def test_collection_refuses_pre_holdout_before_dependencies(tmp_path) -> None:
    with pytest.raises(ValueError, match="Paper collection begins"):
        collect_paper_snapshot(
            output_dir=tmp_path,
            collected_at="2026-08-09T23:59:59+00:00",
            universe_provider=lambda _: pytest.fail("no network before validation"),
            data_refresher=lambda *_: pytest.fail("no downloads before validation"),
            open_interest_collector=lambda *_: pytest.fail("no OI before validation"),
        )


def test_evaluate_paper_equity_reports_maturity_drawdown_and_months() -> None:
    report = evaluate_paper_equity(
        [
            {"timestamp": "2026-08-10T00:00:00Z", "equity": 100.0},
            {"timestamp": "2026-08-31T00:00:00Z", "equity": 120.0},
            {"timestamp": "2026-09-30T00:00:00Z", "equity": 90.0},
            {"timestamp": "2026-11-09T00:00:00Z", "equity": 110.0},
        ]
    )

    assert report["elapsed_days"] == 91.0
    assert report["return_pct"] == pytest.approx(10.0)
    assert report["max_drawdown_pct"] == pytest.approx(-25.0)
    assert report["mature_90_days"] is True
    assert [row["month"] for row in report["monthly"]] == [
        "2026-08",
        "2026-09",
        "2026-11",
    ]
    assert report["monthly"][0]["return_pct"] == pytest.approx(20.0)
    assert report["monthly"][1]["return_pct"] == pytest.approx(-25.0)
    assert report["monthly"][2]["return_pct"] == pytest.approx(22.2222222)


def test_evaluate_paper_equity_reads_jsonl_and_rejects_duplicates(tmp_path) -> None:
    equity_path = tmp_path / "equity.jsonl"
    equity_path.write_text(
        '{"timestamp":"2026-08-10T00:00:00Z","equity":100}\n'
        '{"timestamp":"2026-08-11T00:00:00Z","equity":101}\n',
        encoding="utf-8",
    )
    assert evaluate_paper_equity(equity_path)["elapsed_days"] == 1.0

    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_paper_equity(
            [
                {"timestamp": "2026-08-10", "equity": 100},
                {"timestamp": "2026-08-10", "equity": 101},
            ]
        )
