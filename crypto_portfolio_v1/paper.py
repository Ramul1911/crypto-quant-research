from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import DATA_DIR, DataConfig
from .data import download_futures_klines
from .collectors.open_interest import collect_open_interest_snapshot
from .universe import select_live_universe
from .v2_protocol import PAPER_HOLDOUT_START

UniverseProvider = Callable[[datetime], Mapping[str, Any]]
DataRefresher = Callable[[str, str], Any]
OpenInterestCollector = Callable[
    [Sequence[str], Path, datetime], Sequence[Mapping[str, Any]]
]


def _utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        timestamp = value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    overwrite: bool = True,
) -> None:
    """Write complete JSON atomically, optionally requiring a new destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(f"Refusing to overwrite snapshot: {path}") from exc
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_universe_provider(observed_at: datetime) -> Mapping[str, Any]:
    del observed_at
    return select_live_universe()


def _default_data_refresher(symbol: str, interval: str) -> Any:
    return download_futures_klines(
        symbol,
        DataConfig(interval=interval, start="2021-01-01"),
        force=False,
    )


def _default_oi_collector(
    symbols: Sequence[str],
    output_dir: Path,
    observed_at: datetime,
) -> Sequence[Mapping[str, Any]]:
    return collect_open_interest_snapshot(
        symbols,
        output_dir=output_dir,
        collected_at=observed_at,
    )


def _frame_summary(value: Any) -> dict[str, Any]:
    count = len(value)
    index = getattr(value, "index", None)
    return {
        "rows": count,
        "start": str(index.min()) if count and index is not None else None,
        "end": str(index.max()) if count and index is not None else None,
    }


def _liquidation_events(
    paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        raw = path.read_bytes()
        parsed: list[Any]
        if path.suffix.lower() == ".jsonl":
            parsed = [
                json.loads(line)
                for line in raw.decode("utf-8").splitlines()
                if line.strip()
            ]
        else:
            value = json.loads(raw)
            parsed = value if isinstance(value, list) else [value]
        if not all(isinstance(item, dict) for item in parsed):
            raise ValueError(f"Liquidation source must contain JSON objects: {path}")
        events.extend(parsed)
        sources.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "events": len(parsed),
            }
        )
    return events, sources


def _append_state(path: Path, run: Mapping[str, Any]) -> None:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("runs"), list):
            raise ValueError(f"Invalid paper state: {path}")
    else:
        state = {
            "protocol": "crypto_portfolio_v2_forward_paper",
            "paper_holdout_start": PAPER_HOLDOUT_START,
            "runs": [],
        }
    snapshot = str(run["snapshot"])
    if any(item.get("snapshot") == snapshot for item in state["runs"]):
        raise FileExistsError(f"Paper state already contains snapshot: {snapshot}")
    state["runs"].append(dict(run))
    atomic_write_json(path, state)


def initialize_paper_state(
    *,
    output_dir: Path | None = None,
    initialized_at: datetime | str | None = None,
) -> dict[str, Any]:
    """Initialize the immutable holdout ledger without fabricating a snapshot."""
    observed_at = _utc(initialized_at or datetime.now(timezone.utc))
    holdout_start = _utc(PAPER_HOLDOUT_START)
    destination = Path(output_dir or DATA_DIR / "paper")
    path = destination / "paper_state.json"
    state = {
        "protocol": "crypto_portfolio_v2_forward_paper",
        "paper_holdout_start": PAPER_HOLDOUT_START,
        "initialized_at_utc": observed_at.isoformat(),
        "status": "active" if observed_at >= holdout_start else "scheduled",
        "runs": [],
    }
    atomic_write_json(path, state, overwrite=False)
    return {**state, "state_path": str(path)}


def collect_paper_snapshot(
    *,
    output_dir: Path | None = None,
    collected_at: datetime | str | None = None,
    liquidation_paths: Iterable[Path] = (),
    universe_provider: UniverseProvider | None = None,
    data_refresher: DataRefresher | None = None,
    open_interest_collector: OpenInterestCollector | None = None,
) -> dict[str, Any]:
    """Collect one daily, forward-only paper snapshot without placing orders."""
    observed_at = _utc(collected_at or datetime.now(timezone.utc))
    holdout_start = _utc(PAPER_HOLDOUT_START)
    if observed_at < holdout_start:
        raise ValueError(
            f"Paper collection begins at {holdout_start.isoformat()}; "
            f"received {observed_at.isoformat()}"
        )

    destination = Path(output_dir or DATA_DIR / "paper")
    day = observed_at.date().isoformat()
    snapshot_path = destination / "snapshots" / f"{day}.json"
    universe_path = destination / "universe_history" / f"{day}.json"
    if snapshot_path.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {snapshot_path}")
    if universe_path.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {universe_path}")

    universe = dict((universe_provider or _default_universe_provider)(observed_at))
    symbols = list(dict.fromkeys(str(item).upper() for item in universe["symbols"]))
    if not symbols:
        raise ValueError("Paper universe is empty")
    archived_universe = dict(universe)
    archived_universe.setdefault("selected_at_utc", observed_at.isoformat())
    archived_universe["paper_snapshot_date"] = day
    atomic_write_json(universe_path, archived_universe, overwrite=False)

    refresh = data_refresher or _default_data_refresher
    bars: dict[str, dict[str, dict[str, Any]]] = {}
    for symbol in symbols:
        bars[symbol] = {
            interval: _frame_summary(refresh(symbol, interval))
            for interval in ("1h", "15m")
        }

    collect_oi = open_interest_collector or _default_oi_collector
    oi_records = list(
        collect_oi(symbols, destination / "open_interest", observed_at)
    )
    liquidation_events, liquidation_sources = _liquidation_events(liquidation_paths)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "paper_collection_only",
        "live_orders": False,
        "availability": "forward_only",
        "paper_holdout_start": PAPER_HOLDOUT_START,
        "collected_at_utc": observed_at.isoformat(),
        "snapshot_date": day,
        "universe_archive": str(universe_path),
        "symbols": symbols,
        "bars": bars,
        "open_interest": {
            "records": len(oi_records),
            "symbols": [str(item.get("symbol", "")).upper() for item in oi_records],
        },
        "liquidations": {
            "events": liquidation_events,
            "event_count": len(liquidation_events),
            "sources": liquidation_sources,
        },
    }
    atomic_write_json(snapshot_path, manifest, overwrite=False)
    _append_state(
        destination / "paper_state.json",
        {
            "snapshot": str(snapshot_path),
            "collected_at_utc": observed_at.isoformat(),
            "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "symbols": symbols,
            "liquidation_events": len(liquidation_events),
        },
    )
    manifest["snapshot_path"] = str(snapshot_path)
    manifest["state_path"] = str(destination / "paper_state.json")
    return manifest


def _read_equity_records(source: Path | Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not isinstance(source, (str, Path)):
        return list(source)
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("equity", payload.get("records"))
    if not isinstance(payload, list):
        raise ValueError("Equity JSON must be a list or contain 'equity'/'records'")
    return payload


def evaluate_paper_equity(
    source: Path | Iterable[Mapping[str, Any]],
    *,
    timestamp_key: str = "timestamp",
    equity_key: str = "equity",
) -> dict[str, Any]:
    """Evaluate accumulated point-in-time equity observations."""
    points = sorted(
        (
            _utc(str(record[timestamp_key])),
            float(record[equity_key]),
        )
        for record in _read_equity_records(source)
    )
    if not points:
        raise ValueError("No paper equity observations")
    if any(equity <= 0 for _, equity in points):
        raise ValueError("Paper equity must be positive")
    if len({timestamp for timestamp, _ in points}) != len(points):
        raise ValueError("Duplicate paper equity timestamps")

    start_time, start_equity = points[0]
    end_time, end_equity = points[-1]
    peak = start_equity
    max_drawdown = 0.0
    for _, equity in points:
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)

    by_month: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for timestamp, equity in points:
        by_month[timestamp.strftime("%Y-%m")].append((timestamp, equity))
    monthly: list[dict[str, Any]] = []
    prior_close: float | None = None
    for month in sorted(by_month):
        month_points = by_month[month]
        baseline = prior_close if prior_close is not None else month_points[0][1]
        close = month_points[-1][1]
        monthly.append(
            {
                "month": month,
                "start_equity": baseline,
                "end_equity": close,
                "return_pct": (close / baseline - 1.0) * 100.0,
                "observations": len(month_points),
            }
        )
        prior_close = close

    elapsed_days = (end_time - start_time).total_seconds() / 86400.0
    return {
        "start_utc": start_time.isoformat(),
        "end_utc": end_time.isoformat(),
        "elapsed_days": elapsed_days,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "return_pct": (end_equity / start_equity - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "monthly": monthly,
        "mature_90_days": elapsed_days >= 90.0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Forward-only paper collection/evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output-dir", type=Path, default=None)
    collect_parser.add_argument("--at", default=None, help="UTC ISO-8601 collection time")
    collect_parser.add_argument("--liquidations", type=Path, nargs="*", default=[])
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("equity", type=Path)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--output-dir", type=Path, default=None)
    init_parser.add_argument("--at", default=None, help="UTC ISO-8601 initialization time")
    args = parser.parse_args(argv)

    if args.command == "collect":
        result = collect_paper_snapshot(
            output_dir=args.output_dir,
            collected_at=args.at,
            liquidation_paths=args.liquidations,
        )
    elif args.command == "evaluate":
        result = evaluate_paper_equity(args.equity)
    else:
        result = initialize_paper_state(
            output_dir=args.output_dir,
            initialized_at=args.at,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
