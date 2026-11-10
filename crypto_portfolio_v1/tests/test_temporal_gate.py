from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from crypto_portfolio_v1.models import TemporalGate, TemporalGateConfig
from crypto_portfolio_v1.train_gate import (
    build_temporal_sequences,
    predict_temporal_gate,
)


def _panel(rows: int = 300) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=rows, freq="h")
    columns = ["BTC", "ETH", "SOL"]
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.01, size=(rows, len(columns)))
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=index,
        columns=columns,
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = pd.DataFrame(
        rng.lognormal(12.0, 0.2, size=close.shape),
        index=index,
        columns=columns,
    )
    trades = pd.DataFrame(
        rng.lognormal(8.0, 0.1, size=close.shape),
        index=index,
        columns=columns,
    )
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume / close,
        "quote_volume": volume,
        "trades": trades,
        "taker_buy_base": 0.5 * volume / close,
        "taker_buy_quote": 0.5 * volume,
        "funding": pd.DataFrame(0.0, index=index, columns=columns),
    }


def test_multitask_shapes_and_monotonic_quantiles() -> None:
    model = TemporalGate(
        TemporalGateConfig(
            asset_feature_count=12,
            global_feature_count=9,
            hidden_size=24,
            levels=3,
            dropout=0.0,
        )
    )
    output = model(torch.randn(2, 168, 3, 12), torch.randn(2, 168, 9))

    assert output["regime_logits"].shape == (2, 4)
    assert output["return_quantiles"].shape == (2, 3, 3, 3)
    assert output["adverse_excursion"].shape == (2, 3, 3)
    assert output["rank_scores"].shape == (2, 3)
    quantiles = output["return_quantiles"]
    assert torch.all(quantiles[..., 1:] >= quantiles[..., :-1])


def test_sequences_are_causal_under_future_mutation() -> None:
    panel = _panel()
    cutoff = panel["close"].index[225]
    original = build_temporal_sequences(panel, with_labels=False)

    changed_panel = {name: frame.copy() for name, frame in panel.items()}
    for frame in changed_panel.values():
        frame.loc[frame.index > cutoff] *= 17.0
    changed = build_temporal_sequences(changed_panel, with_labels=False)

    original_position = int(np.flatnonzero(original.timestamps == cutoff)[0])
    changed_position = int(np.flatnonzero(changed.timestamps == cutoff)[0])
    assert torch.equal(
        original[original_position]["asset_features"],
        changed[changed_position]["asset_features"],
    )
    assert torch.equal(
        original[original_position]["global_features"],
        changed[changed_position]["global_features"],
    )


def test_labeled_sequences_keep_full_future_inside_boundary() -> None:
    panel = _panel()
    boundary = panel["close"].index[250]
    dataset = build_temporal_sequences(panel, end=boundary, with_labels=True)

    assert dataset.timestamps[-1] == panel["close"].index[250 - 72]
    sample = dataset[-1]
    assert sample["future_returns"].shape == (3, 3)
    assert sample["adverse_excursion"].shape == (3, 3)


def test_prediction_returns_labeled_dataframes() -> None:
    panel = _panel(180)
    model = TemporalGate(
        TemporalGateConfig(12, 9, hidden_size=16, levels=2, dropout=0.0)
    )
    prediction = predict_temporal_gate(model, panel, batch_size=8)

    assert list(prediction.regime_probabilities.columns) == [
        "trend_up",
        "trend_down",
        "chop",
        "shock",
    ]
    assert prediction.rank_scores.shape == (13, 3)
    assert prediction.return_quantiles[24][0.5].shape == (13, 3)
    assert prediction.adverse_excursion[72].shape == (13, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_autocast_smoke() -> None:
    model = TemporalGate(
        TemporalGateConfig(12, 9, hidden_size=32, levels=2, dropout=0.0)
    ).cuda()
    asset = torch.randn(1, 168, 3, 12, device="cuda")
    global_ = torch.randn(1, 168, 9, device="cuda")
    enabled = torch.cuda.is_bf16_supported()
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=enabled
    ):
        output = model(asset, global_)
    assert output["regime_logits"].is_cuda

