from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REGIMES: Final[tuple[str, ...]] = (
    "trend_up",
    "trend_down",
    "chop",
    "shock",
)
DEFAULT_HORIZONS: Final[tuple[int, ...]] = (6, 24, 72)
DEFAULT_QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class TemporalGateConfig:
    asset_feature_count: int
    global_feature_count: int
    hidden_size: int = 96
    levels: int = 5
    kernel_size: int = 3
    dropout: float = 0.10
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES

    def __post_init__(self) -> None:
        if self.asset_feature_count <= 0 or self.global_feature_count <= 0:
            raise ValueError("feature counts must be positive")
        if self.hidden_size <= 0 or self.levels <= 0:
            raise ValueError("hidden_size and levels must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least two")
        if tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("quantiles must be increasing")


class _CausalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = self.conv(F.pad(value, (self.left_padding, 0)))
        value, gate = value.chunk(2, dim=1)
        value = torch.tanh(value) * torch.sigmoid(gate)
        return self.norm(residual + self.dropout(value))


class TemporalGate(nn.Module):
    """Shared causal temporal encoder and multi-task portfolio gate.

    ``asset_features`` has shape ``[batch, time, assets, features]`` and
    ``global_features`` has shape ``[batch, time, features]``. The same TCN is
    applied to every asset, which keeps the parameter count independent of the
    portfolio size and maps efficiently to BF16 tensor cores.
    """

    def __init__(self, config: TemporalGateConfig) -> None:
        super().__init__()
        self.config = config
        input_size = config.asset_feature_count + config.global_feature_count
        self.input_projection = nn.Conv1d(input_size, config.hidden_size, 1)
        self.blocks = nn.Sequential(
            *[
                _CausalBlock(
                    config.hidden_size,
                    config.kernel_size,
                    dilation=2**level,
                    dropout=config.dropout,
                )
                for level in range(config.levels)
            ]
        )
        self.regime_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, len(REGIMES)),
        )
        task_count = len(config.horizons)
        quantile_count = len(config.quantiles)
        self.quantile_head = nn.Linear(
            config.hidden_size, task_count * quantile_count
        )
        self.risk_head = nn.Linear(config.hidden_size, task_count)
        self.rank_head = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        asset_features: Tensor,
        global_features: Tensor,
    ) -> dict[str, Tensor]:
        if asset_features.ndim != 4 or global_features.ndim != 3:
            raise ValueError(
                "expected asset [B,T,A,F] and global [B,T,G] features"
            )
        batch, steps, assets, asset_feature_count = asset_features.shape
        if global_features.shape[:2] != (batch, steps):
            raise ValueError("asset and global batch/time dimensions must match")
        if asset_feature_count != self.config.asset_feature_count:
            raise ValueError("unexpected asset feature count")
        if global_features.shape[-1] != self.config.global_feature_count:
            raise ValueError("unexpected global feature count")

        global_by_asset = global_features.unsqueeze(2).expand(
            batch, steps, assets, -1
        )
        value = torch.cat((asset_features, global_by_asset), dim=-1)
        value = value.permute(0, 2, 3, 1).reshape(
            batch * assets, -1, steps
        )
        encoded = self.blocks(self.input_projection(value))[:, :, -1]
        encoded = encoded.reshape(batch, assets, -1)

        regime_logits = self.regime_head(encoded.mean(dim=1))
        raw_quantiles = self.quantile_head(encoded).reshape(
            batch,
            assets,
            len(self.config.horizons),
            len(self.config.quantiles),
        )
        # Sorting is differentiable almost everywhere and prevents quantile
        # crossing without imposing an unrealistic minimum spread.
        return_quantiles = torch.sort(raw_quantiles, dim=-1).values

        return {
            "regime_logits": regime_logits,
            "return_quantiles": return_quantiles,
            "adverse_excursion": torch.sigmoid(self.risk_head(encoded)),
            "rank_scores": self.rank_head(encoded).squeeze(-1),
        }


# Descriptive alias for callers that prefer the architecture name.
TemporalGateTCN = TemporalGate

