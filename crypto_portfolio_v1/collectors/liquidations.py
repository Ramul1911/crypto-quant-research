from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import DATA_DIR


def _event_payload(message: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    payload = json.loads(message) if isinstance(message, str) else message
    if not isinstance(payload, dict):
        raise ValueError("Liquidation event must be a JSON object")
    if "data" in payload and isinstance(payload["data"], dict):
        payload = payload["data"]
    return payload


def parse_liquidation_event(
    message: str | bytes | dict[str, Any],
    *,
    received_at: datetime | None = None,
) -> dict[str, Any]:
    """Normalize a Binance ``forceOrder`` websocket message."""
    payload = _event_payload(message)
    if payload.get("e") != "forceOrder" or not isinstance(payload.get("o"), dict):
        raise ValueError("Not a Binance forceOrder event")
    order = payload["o"]
    timestamp = received_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    try:
        event_time_ms = int(payload["E"])
        trade_time_ms = int(order.get("T", event_time_ms))
        price = float(order["p"])
        average_price = float(order.get("ap", price))
        original_quantity = float(order["q"])
        filled_quantity = float(order.get("z", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Malformed Binance forceOrder event") from exc
    return {
        "received_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "event_time_utc": datetime.fromtimestamp(
            event_time_ms / 1000, tz=timezone.utc
        ).isoformat(),
        "trade_time_utc": datetime.fromtimestamp(
            trade_time_ms / 1000, tz=timezone.utc
        ).isoformat(),
        "symbol": str(order["s"]).upper(),
        "side": str(order["S"]).upper(),
        "order_type": order.get("o"),
        "time_in_force": order.get("f"),
        "price": price,
        "average_price": average_price,
        "original_quantity": original_quantity,
        "filled_quantity": filled_quantity,
        "status": order.get("X"),
        "source": "binance_futures_force_order_websocket",
        "availability": "forward_only",
    }


def persist_liquidation_event(
    event: dict[str, Any],
    output_dir: Path | None = None,
) -> Path:
    """Append one parsed event to a UTC-day JSONL stream."""
    timestamp = datetime.fromisoformat(
        str(event["event_time_utc"]).replace("Z", "+00:00")
    )
    destination = (
        output_dir or DATA_DIR / "forward" / "liquidations"
    ) / f"{timestamp.astimezone(timezone.utc).date().isoformat()}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    return destination


def parse_and_persist_liquidation_event(
    message: str | bytes | dict[str, Any],
    *,
    output_dir: Path | None = None,
    received_at: datetime | None = None,
) -> tuple[dict[str, Any], Path]:
    """Websocket-callback-compatible parse and persistence helper."""
    event = parse_liquidation_event(message, received_at=received_at)
    return event, persist_liquidation_event(event, output_dir)
