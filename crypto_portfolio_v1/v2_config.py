from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class V2BacktestConfig:
    """Execution and portfolio-risk settings for the V2 simulator."""

    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    annual_vol_target: float = 0.45
    periods_per_year: int = 4 * 24 * 365
    max_symbol_weight: float = 0.30
    max_gross_exposure: float = 1.50
    max_abs_net_exposure: float = 0.30
    max_cluster_exposure: float = 0.60
    correlation_clusters: Mapping[str, str] = field(default_factory=dict)
    drawdown_level_1: float = 0.10
    drawdown_level_2: float = 0.15
    drawdown_level_3: float = 0.20
    terminal_drawdown: float = 0.25
    drawdown_multiplier_1: float = 0.70
    drawdown_multiplier_2: float = 0.50
    drawdown_multiplier_3: float = 0.25
    accounting_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        nonnegative = (
            "fee_rate",
            "slippage_rate",
            "annual_vol_target",
            "max_symbol_weight",
            "max_gross_exposure",
            "max_abs_net_exposure",
            "max_cluster_exposure",
        )
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        levels = (
            self.drawdown_level_1,
            self.drawdown_level_2,
            self.drawdown_level_3,
            self.terminal_drawdown,
        )
        if not (0 < levels[0] < levels[1] < levels[2] < levels[3] < 1):
            raise ValueError("drawdown levels must be strictly increasing and below one")
        multipliers = (
            self.drawdown_multiplier_1,
            self.drawdown_multiplier_2,
            self.drawdown_multiplier_3,
        )
        if not (1 >= multipliers[0] >= multipliers[1] >= multipliers[2] >= 0):
            raise ValueError("drawdown multipliers must be decreasing values in [0, 1]")

    @property
    def transaction_cost_rate(self) -> float:
        return self.fee_rate + self.slippage_rate


# Short name for callers that keep one config per V2 experiment.
V2Config = V2BacktestConfig
