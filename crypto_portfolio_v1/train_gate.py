from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .models.temporal_gate import (
    DEFAULT_HORIZONS,
    DEFAULT_QUANTILES,
    REGIMES,
    TemporalGate,
    TemporalGateConfig,
)

SEQUENCE_LENGTH = 168
PURGE_HOURS = 72


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validate_panel(
    panel: Mapping[str, pd.DataFrame],
) -> tuple[pd.DatetimeIndex, tuple[str, ...]]:
    if "close" not in panel:
        raise KeyError("panel is missing required field 'close'")
    close = panel["close"]
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("panel must use a DatetimeIndex")
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError("panel index must be unique and increasing")
    for name, frame in panel.items():
        if not frame.index.equals(close.index) or not frame.columns.equals(
            close.columns
        ):
            raise ValueError(f"panel field {name!r} is not aligned")
    return close.index, tuple(str(value) for value in close.columns)


def build_temporal_features(
    panel: Mapping[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Create fixed-scale causal per-asset and market-wide features."""

    index, _ = _validate_panel(panel)
    close = panel["close"].astype(float)
    open_ = panel.get("open", close).astype(float)
    high = panel.get("high", close).astype(float)
    low = panel.get("low", close).astype(float)
    quote_volume = panel.get(
        "quote_volume", pd.DataFrame(0.0, index=index, columns=close.columns)
    ).astype(float)
    trades = panel.get(
        "trades", pd.DataFrame(0.0, index=index, columns=close.columns)
    ).astype(float)
    taker = panel.get(
        "taker_buy_quote",
        pd.DataFrame(0.0, index=index, columns=close.columns),
    ).astype(float)
    funding = panel.get(
        "funding", pd.DataFrame(0.0, index=index, columns=close.columns)
    ).astype(float).ffill()

    log_close = np.log(close.where(close > 0.0))
    returns = log_close.diff()
    epsilon = np.finfo(float).eps
    per_asset: dict[str, pd.DataFrame] = {
        f"log_return_{h}h": log_close.diff(h).clip(-1.0, 1.0)
        for h in (1, 6, 24, 72, 168)
    }
    per_asset.update(
        {
            "realized_volatility_24h": returns.rolling(
                24, min_periods=2
            ).std(ddof=0).clip(0.0, 0.25),
            "candle_return": close.div(
                open_.abs().where(open_.abs() > epsilon)
            ).sub(1.0).clip(-0.5, 0.5),
            "candle_range": high.sub(low).div(
                open_.abs().where(open_.abs() > epsilon)
            ).clip(0.0, 1.0),
            "quote_volume_change_24h": np.log1p(
                quote_volume.clip(lower=0.0)
            ).diff(24).clip(-5.0, 5.0),
            "trade_change_24h": np.log1p(
                trades.clip(lower=0.0)
            ).diff(24).clip(-5.0, 5.0),
            "taker_imbalance": (
                2.0
                * taker.div(quote_volume.where(quote_volume.abs() > epsilon))
                - 1.0
            ).clip(-1.0, 1.0),
            "funding_rate": funding.clip(-0.02, 0.02) * 50.0,
        }
    )

    one_hour = per_asset["log_return_1h"]
    ema_168 = close.ewm(span=168, adjust=False, min_periods=2).mean()
    hour = index.hour.to_numpy()
    day = index.dayofweek.to_numpy()
    global_frame = pd.DataFrame(
        {
            "market_return": one_hour.mean(axis=1),
            "return_dispersion": one_hour.std(axis=1, ddof=0),
            "market_volatility_24h": one_hour.mean(axis=1)
            .rolling(24, min_periods=2)
            .std(ddof=0),
            "breadth_ema_168h": (close > ema_168).mean(axis=1) - 0.5,
            "mean_funding": funding.mean(axis=1).clip(-0.02, 0.02) * 50.0,
            "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            "day_sin": np.sin(2.0 * np.pi * day / 7.0),
            "day_cos": np.cos(2.0 * np.pi * day / 7.0),
        },
        index=index,
    )

    asset_names = tuple(per_asset)
    global_names = tuple(str(value) for value in global_frame.columns)
    asset_values = np.stack(
        [frame.to_numpy(dtype=np.float32) for frame in per_asset.values()],
        axis=-1,
    )
    global_values = global_frame.to_numpy(dtype=np.float32)
    asset_values = np.nan_to_num(asset_values, nan=0.0, posinf=0.0, neginf=0.0)
    global_values = np.nan_to_num(
        global_values, nan=0.0, posinf=0.0, neginf=0.0
    )
    return asset_values, global_values, asset_names, global_names


class TemporalSequenceDataset(Dataset[dict[str, Tensor]]):
    """Window views over causal features; labels only inspect future rows."""

    def __init__(
        self,
        asset_features: np.ndarray,
        global_features: np.ndarray,
        close: np.ndarray,
        index: pd.DatetimeIndex,
        endpoints: np.ndarray,
        *,
        sequence_length: int,
        horizons: tuple[int, ...],
        with_labels: bool,
        symbols: tuple[str, ...],
    ) -> None:
        self.asset_features = torch.from_numpy(asset_features)
        self.global_features = torch.from_numpy(global_features)
        self.close = np.asarray(close, dtype=np.float64)
        self.index = index
        self.endpoints = np.asarray(endpoints, dtype=np.int64)
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.with_labels = with_labels
        self.symbols = symbols
        self.timestamps = index[self.endpoints]

    def __len__(self) -> int:
        return len(self.endpoints)

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        endpoint = int(self.endpoints[item])
        start = endpoint - self.sequence_length + 1
        result = {
            "asset_features": self.asset_features[start : endpoint + 1],
            "global_features": self.global_features[start : endpoint + 1],
        }
        if not self.with_labels:
            return result

        current = self.close[endpoint]
        future_returns = []
        adverse = []
        for horizon in self.horizons:
            path = self.close[endpoint + 1 : endpoint + horizon + 1]
            future_returns.append(self.close[endpoint + horizon] / current - 1.0)
            worst = np.nanmin(path / current - 1.0, axis=0)
            adverse.append(np.clip(-worst / 0.20, 0.0, 1.0))
        returns = np.stack(future_returns, axis=-1)
        adverse_targets = np.stack(adverse, axis=-1)

        rank_source = returns[:, min(1, returns.shape[-1] - 1)]
        order = pd.Series(rank_source).rank(method="average", pct=True).to_numpy()
        rank_targets = 2.0 * order - 1.0
        market_return = float(np.nanmean(rank_source))
        market_path = np.nanmean(
            self.close[endpoint + 1 : endpoint + self.horizons[-1] + 1]
            / current
            - 1.0,
            axis=1,
        )
        if float(np.nanmin(market_path)) < -0.06:
            regime = 3
        elif market_return > 0.005:
            regime = 0
        elif market_return < -0.005:
            regime = 1
        else:
            regime = 2

        result.update(
            {
                "regime": torch.tensor(regime, dtype=torch.long),
                "future_returns": torch.tensor(returns, dtype=torch.float32),
                "adverse_excursion": torch.tensor(
                    adverse_targets, dtype=torch.float32
                ),
                "rank_targets": torch.tensor(
                    rank_targets, dtype=torch.float32
                ),
            }
        )
        return result


def build_temporal_sequences(
    panel: Mapping[str, pd.DataFrame],
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    with_labels: bool = True,
) -> TemporalSequenceDataset:
    """Build windows whose final observation is the decision timestamp.

    When labels are requested, every complete future horizon must also fall at
    or before ``end``. Thus a sample can neither consume features nor labels
    from the next split.
    """

    index, symbols = _validate_panel(panel)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    horizon_tuple = tuple(int(value) for value in horizons)
    if not horizon_tuple or min(horizon_tuple) <= 0:
        raise ValueError("horizons must be positive")
    asset, global_, _, _ = build_temporal_features(panel)

    first = sequence_length - 1
    last = len(index) - 1 - (max(horizon_tuple) if with_labels else 0)
    endpoints = np.arange(first, max(first, last + 1), dtype=np.int64)
    if start is not None:
        endpoints = endpoints[index[endpoints] >= pd.Timestamp(start)]
    if end is not None:
        end_time = pd.Timestamp(end)
        endpoints = endpoints[index[endpoints] <= end_time]
        if with_labels:
            endpoints = endpoints[index[endpoints + max(horizon_tuple)] <= end_time]
    if with_labels and len(endpoints):
        close_values = panel["close"].to_numpy(dtype=np.float64)
        valid = np.isfinite(close_values) & (close_values > 0.0)
        max_horizon = max(horizon_tuple)
        endpoints = np.asarray(
            [
                endpoint
                for endpoint in endpoints
                if valid[endpoint : endpoint + max_horizon + 1].all()
            ],
            dtype=np.int64,
        )
    return TemporalSequenceDataset(
        asset,
        global_,
        panel["close"].to_numpy(dtype=np.float64),
        index,
        endpoints,
        sequence_length=sequence_length,
        horizons=horizon_tuple,
        with_labels=with_labels,
        symbols=symbols,
    )


def _quantile_loss(prediction: Tensor, target: Tensor, levels: Tensor) -> Tensor:
    error = target.unsqueeze(-1) - prediction
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


def temporal_gate_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> Tensor:
    levels = torch.as_tensor(
        quantiles,
        device=output["return_quantiles"].device,
        dtype=output["return_quantiles"].dtype,
    )
    return (
        F.cross_entropy(output["regime_logits"], batch["regime"])
        + 2.0
        * _quantile_loss(
            output["return_quantiles"], batch["future_returns"], levels
        )
        + 0.5
        * F.mse_loss(output["adverse_excursion"], batch["adverse_excursion"])
        + 0.25
        * F.mse_loss(
            torch.tanh(output["rank_scores"]), batch["rank_targets"]
        )
    )


@dataclass(frozen=True)
class TrainingResult:
    model: TemporalGate
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    checkpoint_path: Path | None


def train_temporal_gate(
    panel: Mapping[str, pd.DataFrame],
    *,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    validation_start: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    checkpoint_path: str | Path | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    purge_hours: int = PURGE_HOURS,
    hidden_size: int = 96,
    batch_size: int = 128,
    epochs: int = 50,
    learning_rate: float = 3e-4,
    patience: int = 7,
    seed: int = 2602,
    device: str | torch.device | None = None,
) -> TrainingResult:
    """Train with a purged chronological split and BF16 CUDA autocast."""

    set_deterministic_seed(seed)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if purge_hours < max(DEFAULT_HORIZONS):
        raise ValueError("purge_hours must cover the maximum 72h label horizon")
    # ``end`` is a label boundary: the sequence builder moves the final
    # decision back by the full horizon, creating the requested purge.
    purged_train_end = min(pd.Timestamp(train_end), pd.Timestamp(validation_start))
    train_data = build_temporal_sequences(
        panel,
        sequence_length=sequence_length,
        start=train_start,
        end=purged_train_end,
        with_labels=True,
    )
    validation_data = build_temporal_sequences(
        panel,
        sequence_length=sequence_length,
        start=validation_start,
        end=validation_end,
        with_labels=True,
    )
    if not train_data or not validation_data:
        raise ValueError("training and validation splits must contain samples")

    config = TemporalGateConfig(
        asset_feature_count=train_data.asset_features.shape[-1],
        global_feature_count=train_data.global_features.shape[-1],
        hidden_size=hidden_size,
    )
    model = TemporalGate(config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )
    bf16 = selected_device.type == "cuda" and torch.cuda.is_bf16_supported()

    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {key: value.to(selected_device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=bf16,
            ):
                output = model(
                    batch["asset_features"], batch["global_features"]
                )
                loss = temporal_gate_loss(output, batch, config.quantiles)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch["regime"])
        train_losses.append(running / len(train_data))

        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                batch = {
                    key: value.to(selected_device) for key, value in batch.items()
                }
                with torch.autocast(
                    device_type=selected_device.type,
                    dtype=torch.bfloat16,
                    enabled=bf16,
                ):
                    output = model(
                        batch["asset_features"], batch["global_features"]
                    )
                    loss = temporal_gate_loss(output, batch, config.quantiles)
                running += float(loss) * len(batch["regime"])
        validation_loss = running / len(validation_data)
        validation_losses.append(validation_loss)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    saved_path = Path(checkpoint_path) if checkpoint_path is not None else None
    if saved_path is not None:
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "config": asdict(config),
                "symbols": train_data.symbols,
                "sequence_length": sequence_length,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
            },
            saved_path,
        )
    return TrainingResult(
        model=model,
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        best_epoch=best_epoch,
        checkpoint_path=saved_path,
    )


@dataclass(frozen=True)
class GatePrediction:
    regime_probabilities: pd.DataFrame
    rank_scores: pd.DataFrame
    return_quantiles: Mapping[int, Mapping[float, pd.DataFrame]]
    adverse_excursion: Mapping[int, pd.DataFrame]

    @property
    def diagnostics(self) -> dict[str, pd.DataFrame]:
        result = {"rank_score": self.rank_scores}
        for horizon, quantile_frames in self.return_quantiles.items():
            for quantile, frame in quantile_frames.items():
                result[f"return_{horizon}h_q{quantile:g}"] = frame
        for horizon, frame in self.adverse_excursion.items():
            result[f"adverse_excursion_{horizon}h"] = frame
        return result


def predict_temporal_gate(
    model: TemporalGate,
    panel: Mapping[str, pd.DataFrame],
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = 256,
    device: str | torch.device | None = None,
) -> GatePrediction:
    dataset = build_temporal_sequences(
        panel, sequence_length=sequence_length, with_labels=False
    )
    if not dataset:
        raise ValueError(
            f"panel needs at least {sequence_length} rows for prediction"
        )
    selected_device = torch.device(
        device
        or next(model.parameters()).device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(selected_device)
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {
        "regime_logits": [],
        "return_quantiles": [],
        "adverse_excursion": [],
        "rank_scores": [],
    }
    bf16 = selected_device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            asset = batch["asset_features"].to(selected_device)
            global_ = batch["global_features"].to(selected_device)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.bfloat16,
                enabled=bf16,
            ):
                output = model(asset, global_)
            for name, value in output.items():
                chunks[name].append(value.float().cpu().numpy())

    values = {name: np.concatenate(parts) for name, parts in chunks.items()}
    regime_values = torch.softmax(
        torch.from_numpy(values["regime_logits"]), dim=-1
    ).numpy()
    timestamps = dataset.timestamps
    symbols = list(dataset.symbols)
    quantile_frames: dict[int, dict[float, pd.DataFrame]] = {}
    for horizon_index, horizon in enumerate(model.config.horizons):
        quantile_frames[horizon] = {}
        for quantile_index, quantile in enumerate(model.config.quantiles):
            quantile_frames[horizon][quantile] = pd.DataFrame(
                values["return_quantiles"][
                    :, :, horizon_index, quantile_index
                ],
                index=timestamps,
                columns=symbols,
            )
    adverse_frames = {
        horizon: pd.DataFrame(
            values["adverse_excursion"][:, :, horizon_index],
            index=timestamps,
            columns=symbols,
        )
        for horizon_index, horizon in enumerate(model.config.horizons)
    }
    return GatePrediction(
        regime_probabilities=pd.DataFrame(
            regime_values, index=timestamps, columns=REGIMES
        ),
        rank_scores=pd.DataFrame(
            values["rank_scores"], index=timestamps, columns=symbols
        ),
        return_quantiles=quantile_frames,
        adverse_excursion=adverse_frames,
    )


# Short aliases for training scripts and notebooks.
train_gate = train_temporal_gate
predict_gate = predict_temporal_gate

