from __future__ import annotations

from dataclasses import dataclass

HISTORICAL_CORE = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "ADAUSDT",
)

# Pre-declared mid-cap expansion whitelist (V2-UNIVERSE-001). Locked before any
# OOS return inspection. Late listings / memes remain satellite / forward-only.
HISTORICAL_EXPANSION_CANDIDATES_WAVE1 = (
    "LTCUSDT",
    "BCHUSDT",
    "ATOMUSDT",
    "DOTUSDT",
    "FILUSDT",
    "AVAXUSDT",
    "UNIUSDT",
    "AAVEUSDT",
)

# Alias kept for existing imports / wave-1 reports.
HISTORICAL_EXPANSION_CANDIDATES = HISTORICAL_EXPANSION_CANDIDATES_WAVE1

# CMC-filtered wave-2 (V2-UNIVERSE-004). Locked before wave-2 PnL inspection.
# Today's CMC late listings / stables / memes are not included here.
HISTORICAL_EXPANSION_CANDIDATES_WAVE2 = (
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

HISTORICAL_EXPANSION_CANDIDATES_ALL = (
    HISTORICAL_EXPANSION_CANDIDATES_WAVE1 + HISTORICAL_EXPANSION_CANDIDATES_WAVE2
)

HISTORICAL_EXPANDED_CANDIDATE_UNIVERSE = (
    HISTORICAL_CORE + HISTORICAL_EXPANSION_CANDIDATES_ALL
)

# Not eligible for historical claims; live satellites only after archive time.
FORWARD_ONLY_SATELLITE_EXAMPLES = (
    "SUIUSDT",
    "HYPEUSDT",
    "APTUSDT",
    "PEPEUSDT",
    "WIFUSDT",
    "NEARUSDT",
    "INJUSDT",
    "TAOUSDT",
    "ONDOUSDT",
)

DEVELOPMENT_END = "2026-08-09 23:59:59"
PAPER_HOLDOUT_START = "2026-08-10 00:00:00"
MAX_LABEL_HORIZON_HOURS = 72
RESEARCH_START = "2021-01-01"
MAX_ELIGIBILITY_GAP_RATE = 0.01


@dataclass(frozen=True)
class AcceptanceCriteria:
    max_mdd_pct: float = -28.0
    fold_kill_pct: float = -25.0
    min_profitable_month_share: float = 0.65
    min_median_monthly_return_pct: float = 8.0
    min_mean_monthly_return_pct: float = 10.0
    max_mean_monthly_return_pct: float = 30.0


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


DEVELOPMENT_FOLDS = (
    FoldSpec(
        "2024H2",
        "2021-01-01",
        "2023-12-31 23:59:59",
        "2024-01-01",
        "2024-06-30 23:59:59",
        "2024-07-01",
        "2024-12-31 23:59:59",
    ),
    FoldSpec(
        "2025H1",
        "2021-01-01",
        "2024-06-30 23:59:59",
        "2024-07-01",
        "2024-12-31 23:59:59",
        "2025-01-01",
        "2025-06-30 23:59:59",
    ),
    FoldSpec(
        "2025H2",
        "2021-01-01",
        "2024-12-31 23:59:59",
        "2025-01-01",
        "2025-06-30 23:59:59",
        "2025-07-01",
        "2025-12-31 23:59:59",
    ),
    FoldSpec(
        "2026H1",
        "2021-01-01",
        "2025-06-30 23:59:59",
        "2025-07-01",
        "2025-12-31 23:59:59",
        "2026-01-01",
        "2026-06-30 23:59:59",
    ),
    FoldSpec(
        "2026H2_YTD",
        "2021-01-01",
        "2025-12-31 23:59:59",
        "2026-01-01",
        "2026-06-30 23:59:59",
        "2026-07-01",
        DEVELOPMENT_END,
    ),
)

