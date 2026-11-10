from __future__ import annotations

import json

import pandas as pd

from crypto_portfolio_v1.universe import (
    historical_expanded_candidate_universe,
    historical_test_universe,
)
from crypto_portfolio_v1.universe_eligibility import (
    audit_expansion_candidates,
    evaluate_symbol_eligibility,
    resolve_universe_symbols,
)
from crypto_portfolio_v1.v2_protocol import (
    HISTORICAL_CORE,
    HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE,
    HISTORICAL_EXPANSION_CANDIDATES,
    HISTORICAL_EXPANSION_CANDIDATES_ALL,
    HISTORICAL_EXPANSION_CANDIDATES_WAVE2,
)


def _bars(start: str, periods: int = 48) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="h")
    return pd.DataFrame(
        {
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 100.0,
        },
        index=index,
    )


def _funding(start: str, periods: int = 6) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="8h")
    return pd.DataFrame({"funding_rate": 0.0001}, index=index)


def test_protocol_expansion_whitelist_is_frozen() -> None:
    assert HISTORICAL_EXPANSION_CANDIDATES == (
        "LTCUSDT",
        "BCHUSDT",
        "ATOMUSDT",
        "DOTUSDT",
        "FILUSDT",
        "AVAXUSDT",
        "UNIUSDT",
        "AAVEUSDT",
    )
    assert HISTORICAL_EXPANSION_CANDIDATES_WAVE2 == (
        "TRXUSDT",
        "XLMUSDT",
        "ETCUSDT",
        "ALGOUSDT",
        "VETUSDT",
        "DASHUSDT",
        "ZECUSDT",
        "THETAUSDT",
        "XTZUSDT",
        "EOSUSDT",
        "IOTAUSDT",
        "NEOUSDT",
    )
    assert HISTORICAL_EXPANSION_CANDIDATES_ALL == (
        HISTORICAL_EXPANSION_CANDIDATES + HISTORICAL_EXPANSION_CANDIDATES_WAVE2
    )
    assert HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE[:8] == HISTORICAL_CORE
    assert len(HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE) == 28
    assert historical_test_universe()["symbols"] == list(HISTORICAL_CORE)
    assert historical_expanded_candidate_universe()["symbols"] == list(
        HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE
    )


def test_late_listing_is_rejected() -> None:
    report = evaluate_symbol_eligibility(
        "AVAXUSDT",
        _bars("2021-06-01", periods=100),
        _funding("2021-06-01"),
        active_symbols={"AVAXUSDT"},
        research_start="2021-01-01",
        development_end="2021-06-10 23:59:59",
        as_of="2021-06-10 23:59:59",
    )
    assert not report["eligible"]
    assert "listing_after_research_start" in report["reasons"]


def test_eligible_symbol_passes_data_gates(tmp_path) -> None:
    symbol = "LTCUSDT"
    bars = _bars("2020-12-01", periods=24 * 40)
    funding = _funding("2020-12-01", periods=120)
    bars.to_parquet(tmp_path / f"{symbol}_1h.parquet")
    funding.to_parquet(tmp_path / f"{symbol}_funding.parquet")

    manifest = audit_expansion_candidates(
        raw_dir=tmp_path,
        candidates=(symbol,),
        active_symbols={symbol},
        fetch_active=False,
        research_start="2021-01-01",
        development_end="2021-01-10 23:59:59",
        as_of="2021-01-10 23:59:59",
        output_path=tmp_path / "eligibility.json",
    )

    assert manifest["eligible_expansion_symbols"] == [symbol]
    assert manifest["expanded_symbols"] == list(HISTORICAL_CORE) + [symbol]
    locked = json.loads((tmp_path / "eligibility.json").read_text(encoding="utf-8"))
    assert locked["eligible_expansion_symbols"] == [symbol]


def test_resolve_universe_presets(tmp_path) -> None:
    path = tmp_path / "eligibility.json"
    path.write_text(
        json.dumps(
            {
                "expanded_symbols": list(HISTORICAL_CORE) + ["LTCUSDT"],
                "eligible_expansion_symbols": ["LTCUSDT"],
            }
        ),
        encoding="utf-8",
    )
    assert resolve_universe_symbols("historical_core") == HISTORICAL_CORE
    assert resolve_universe_symbols("historical_expanded_candidates") == (
        HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE
    )
    assert resolve_universe_symbols(
        "historical_expanded",
        eligibility_path=path,
    )[-1] == "LTCUSDT"
