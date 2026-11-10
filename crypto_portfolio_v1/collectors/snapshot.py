from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from ..config import DATA_DIR
from .open_interest import collect_open_interest_snapshot


def collect_forward_snapshot(
    symbols: Iterable[str],
    *,
    output_dir: Path | None = None,
    session: requests.Session | None = None,
    collected_at: datetime | None = None,
) -> dict[str, Any]:
    """Run a one-shot forward-only collection and write its manifest."""
    destination = output_dir or DATA_DIR / "forward"
    observed_at = collected_at or datetime.now(timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    normalized_symbols = list(dict.fromkeys(symbol.upper() for symbol in symbols))
    oi_records = collect_open_interest_snapshot(
        normalized_symbols,
        output_dir=destination / "open_interest",
        session=session,
        collected_at=observed_at,
    )
    manifest = {
        "collected_at_utc": observed_at.astimezone(timezone.utc).isoformat(),
        "availability": "forward_only",
        "free_data_notice": (
            "Public current snapshots and websocket events have no historical "
            "point-in-time backfill and must not be used before collection."
        ),
        "symbols": normalized_symbols,
        "open_interest_records": len(oi_records),
        "liquidation_stream": "binance !forceOrder@arr websocket; persistence ready",
    }
    manifest_dir = destination / "snapshots"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    filename = observed_at.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%S.%fZ.json"
    )
    path = manifest_dir / filename
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(path)
    return manifest
