from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .config import (
    DATA_DIR,
    REPORT_DIR,
    SATELLITE_CANDIDATES,
    UniverseConfig,
)
from .v2_protocol import (
    HISTORICAL_CORE,
    HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE,
    HISTORICAL_EXPANSION_CANDIDATES_ALL,
)

CORE_SYMBOLS = HISTORICAL_CORE

FAPI = "https://fapi.binance.com"


def archive_universe_snapshot(
    manifest: dict[str, Any],
    history_dir: Path | None = None,
) -> Path:
    """Persist one immutable point-in-time universe file per UTC day."""
    selected_at = pd_timestamp_utc(manifest["selected_at_utc"])
    destination = (
        history_dir or DATA_DIR / "universe_history"
    ) / f"{selected_at.date().isoformat()}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def pd_timestamp_utc(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def historical_test_universe() -> dict[str, Any]:
    """Return the protocol-fixed historical universe without live satellites."""
    core = list(HISTORICAL_CORE)
    return {
        "selection_type": "v2_historical_fixed_core",
        "core": core,
        "selected_satellites": [],
        "symbols": core,
        "satellite_policy": "forward_only",
        "warning": (
            "Satellite membership is not backfilled. Archived satellite "
            "snapshots may only be used from their selection time forward."
        ),
    }


def historical_expanded_candidate_universe() -> dict[str, Any]:
    """Return core plus the pre-declared mid-cap candidates (pre-audit)."""
    return {
        "selection_type": "v2_historical_expanded_candidates",
        "core": list(HISTORICAL_CORE),
        "expansion_candidates": list(HISTORICAL_EXPANSION_CANDIDATES_ALL),
        "selected_satellites": [],
        "symbols": list(HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE),
        "satellite_policy": "forward_only",
        "warning": (
            "Candidate universe is pre-audit only. Run universe_eligibility "
            "before any historical_expanded performance claim."
        ),
    }


def historical_expanded_universe(
    eligibility_path: Path | None = None,
) -> dict[str, Any]:
    """Return core plus mechanically eligible midcaps from the locked audit."""
    from .universe_eligibility import (
        DEFAULT_ELIGIBILITY_PATH,
        load_eligibility_manifest,
    )

    path = eligibility_path or DEFAULT_ELIGIBILITY_PATH
    manifest = load_eligibility_manifest(path)
    eligible = list(manifest["eligible_expansion_symbols"])
    return {
        "selection_type": "v2_historical_expanded_eligible",
        "core": list(HISTORICAL_CORE),
        "expansion_candidates": list(HISTORICAL_EXPANSION_CANDIDATES_ALL),
        "eligible_expansion_symbols": eligible,
        "rejected_expansion_symbols": list(
            manifest.get("rejected_expansion_symbols", [])
        ),
        "selected_satellites": [],
        "symbols": list(HISTORICAL_CORE) + eligible,
        "eligibility_path": str(path),
        "satellite_policy": "forward_only",
        "warning": manifest.get(
            "warning",
            "Eligible expansion is locked; do not edit after seeing OOS returns.",
        ),
    }


def load_archived_universe(
    date: str,
    history_dir: Path | None = None,
) -> dict[str, Any]:
    """Load an exact UTC-day snapshot; never substitute a future snapshot."""
    destination = (history_dir or DATA_DIR / "universe_history") / f"{date}.json"
    if not destination.exists():
        raise FileNotFoundError(
            f"No point-in-time universe snapshot for {date}; "
            "satellites are forward-only"
        )
    return json.loads(destination.read_text(encoding="utf-8"))


def _get_json(
    session: requests.Session,
    path: str,
    params: dict[str, Any] | None = None,
    retries: int = 5,
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
        delay = min(delay * 2.0, 8.0)
    raise RuntimeError(f"Unable to fetch {path}")


def select_live_universe(
    config: UniverseConfig = UniverseConfig(),
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Select active core contracts and four liquid satellites using live data."""
    session = requests.Session()
    exchange_info = _get_json(session, "/fapi/v1/exchangeInfo")
    tickers = _get_json(session, "/fapi/v1/ticker/24hr")
    marks = _get_json(session, "/fapi/v1/premiumIndex")

    active = {
        item["symbol"]: item
        for item in exchange_info["symbols"]
        if item.get("status") == "TRADING"
        and item.get("contractType") == "PERPETUAL"
        and item.get("quoteAsset") == "USDT"
    }
    ticker_by_symbol = {item["symbol"]: item for item in tickers}
    mark_by_symbol = {
        item["symbol"]: float(item["markPrice"])
        for item in marks
        if "symbol" in item and "markPrice" in item
    }

    core = [symbol for symbol in CORE_SYMBOLS if symbol in active]
    missing_core = [symbol for symbol in CORE_SYMBOLS if symbol not in active]
    candidates: list[dict[str, Any]] = []
    for symbol in SATELLITE_CANDIDATES:
        if symbol not in active or symbol not in ticker_by_symbol:
            continue
        quote_volume = float(ticker_by_symbol[symbol].get("quoteVolume", 0.0))
        try:
            oi_payload = _get_json(
                session,
                "/fapi/v1/openInterest",
                {"symbol": symbol},
            )
            open_interest_usdt = (
                float(oi_payload["openInterest"]) * mark_by_symbol.get(symbol, 0.0)
            )
        except (requests.RequestException, KeyError, TypeError, ValueError):
            open_interest_usdt = 0.0

        eligible = (
            quote_volume >= config.min_quote_volume_24h
            and open_interest_usdt >= config.min_open_interest_usdt
        )
        liquidity_score = (
            math.sqrt(quote_volume * open_interest_usdt)
            if quote_volume > 0 and open_interest_usdt > 0
            else 0.0
        )
        candidates.append(
            {
                "symbol": symbol,
                "quote_volume_24h": quote_volume,
                "open_interest_usdt": open_interest_usdt,
                "liquidity_score": liquidity_score,
                "eligible": eligible,
            }
        )
        time.sleep(0.03)

    candidates.sort(key=lambda item: item["liquidity_score"], reverse=True)
    eligible = [item for item in candidates if item["eligible"]]
    selected_satellites = [
        item["symbol"] for item in eligible[: config.n_satellites]
    ]
    if len(selected_satellites) < config.n_satellites:
        fallback = [
            item["symbol"]
            for item in candidates
            if item["symbol"] not in selected_satellites
        ]
        selected_satellites.extend(
            fallback[: config.n_satellites - len(selected_satellites)]
        )

    manifest = {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_type": "live_research_universe",
        "data_availability": "forward_only",
        "config": asdict(config),
        "core": core,
        "missing_core": missing_core,
        "selected_satellites": selected_satellites,
        "symbols": core + selected_satellites,
        "satellite_ranking": candidates,
        "warning": (
            "Live selection is valid for forward runs. Using it retrospectively "
            "introduces survivorship bias; historical claims require archived universes."
        ),
    }
    destination = output_path or DATA_DIR / "universe_latest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    archive_path = archive_universe_snapshot(manifest)
    manifest["archive_path"] = str(archive_path)
    return manifest


def main() -> None:
    manifest = select_live_universe()
    print("Core:", ", ".join(manifest["core"]))
    print("Satellites:", ", ".join(manifest["selected_satellites"]))
    print("Universe:", ", ".join(manifest["symbols"]))
    if manifest["missing_core"]:
        print("Missing core:", ", ".join(manifest["missing_core"]))
    print(f"Saved -> {DATA_DIR / 'universe_latest.json'}")


if __name__ == "__main__":
    main()
