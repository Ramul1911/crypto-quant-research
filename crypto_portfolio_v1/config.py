from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORT_DIR = ROOT / "reports"

CORE_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "ADAUSDT",
)

SATELLITE_CANDIDATES = (
    "SUIUSDT",
    "AVAXUSDT",
    "HYPEUSDT",
    "PEPEUSDT",
    "NEARUSDT",
    "APTUSDT",
    "INJUSDT",
    "WIFUSDT",
    "TRXUSDT",
)


@dataclass(frozen=True)
class UniverseConfig:
    n_satellites: int = 4
    min_quote_volume_24h: float = 50_000_000.0
    min_open_interest_usdt: float = 20_000_000.0


@dataclass(frozen=True)
class DataConfig:
    interval: str = "1h"
    start: str = "2021-01-01"
    request_pause_seconds: float = 0.08


@dataclass(frozen=True)
class StrategyConfig:
    decision_every_hours: int = 4
    rebalance_every_hours: int = 12
    min_history_hours: int = 0
    long_count: int = 2
    short_count: int = 2
    gross_leverage: float = 1.5
    max_symbol_weight: float = 0.30
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    annual_vol_target: float = 0.45
    max_drawdown_soft_1: float = 0.10
    max_drawdown_soft_2: float = 0.15
    max_drawdown_soft_3: float = 0.20
    max_drawdown_flat: float = 0.25

