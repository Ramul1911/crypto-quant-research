from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from ..config import DATA_DIR
from ..universe import FAPI


def _iso_utc(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat()


def parse_open_interest_snapshot(
    payload: dict[str, Any],
    *,
    symbol: str | None = None,
    collected_at: datetime | None = None,
) -> dict[str, Any]:
    """Normalize Binance's current-OI response with provenance metadata."""
    resolved_symbol = str(payload.get("symbol") or symbol or "").upper()
    if not resolved_symbol:
        raise ValueError("Open-interest payload has no symbol")
    try:
        open_interest = float(payload["openInterest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid openInterest payload") from exc
    if open_interest < 0:
        raise ValueError("openInterest cannot be negative")
    return {
        "collected_at_utc": _iso_utc(collected_at),
        "symbol": resolved_symbol,
        "open_interest": open_interest,
        "source": "binance_futures_public_current_open_interest",
        "availability": "forward_only",
    }


def persist_open_interest_snapshot(
    record: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    """Append a normalized snapshot to a UTC-day JSONL file."""
    timestamp = datetime.fromisoformat(
        str(record["collected_at_utc"]).replace("Z", "+00:00")
    )
    destination = (
        output_dir or DATA_DIR / "forward" / "open_interest"
    ) / f"{timestamp.astimezone(timezone.utc).date().isoformat()}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    return destination


def collect_open_interest_snapshot(
    symbols: Iterable[str],
    *,
    output_dir: Path | None = None,
    session: requests.Session | None = None,
    collected_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Collect and persist one current public OI observation per symbol."""
    client = session or requests.Session()
    observed_at = collected_at or datetime.now(timezone.utc)
    records: list[dict[str, Any]] = []
    for symbol in symbols:
        response = client.get(
            f"{FAPI}/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=30,
        )
        response.raise_for_status()
        record = parse_open_interest_snapshot(
            response.json(),
            symbol=symbol,
            collected_at=observed_at,
        )
        persist_open_interest_snapshot(record, output_dir)
        records.append(record)
    return records
