from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crypto_portfolio_v1.run_hybrid import (
    consensus_gate_predictions,
    scores_to_target_weights,
)
from crypto_portfolio_v1.train_gate import GatePrediction
from crypto_portfolio_v1.v2_config import V2BacktestConfig


def _prediction(value: float) -> GatePrediction:
    index = pd.date_range("2026-01-01", periods=2, freq="h")
    symbols = ["BTC", "ETH"]
    regimes = pd.DataFrame(
        [
            [value, 1.0 - value, 0.0, 0.0],
            [0.0, value, 1.0 - value, 0.0],
        ],
        index=index,
        columns=["trend_up", "trend_down", "chop", "shock"],
    )
    asset_frame = pd.DataFrame(value, index=index, columns=symbols)
    return GatePrediction(
        regime_probabilities=regimes,
        rank_scores=asset_frame,
        return_quantiles={
            6: {0.1: asset_frame, 0.5: asset_frame, 0.9: asset_frame},
            24: {0.1: asset_frame, 0.5: asset_frame, 0.9: asset_frame},
            72: {0.1: asset_frame, 0.5: asset_frame, 0.9: asset_frame},
        },
        adverse_excursion={
            6: asset_frame,
            24: asset_frame,
            72: asset_frame,
        },
    )


def test_seed_consensus_averages_every_gate_head() -> None:
    predictions = [_prediction(0.2), _prediction(0.5), _prediction(0.8)]
    consensus = consensus_gate_predictions(predictions)

    assert consensus.regime_probabilities.loc[
        pd.Timestamp("2026-01-01 00:00"), "trend_up"
    ] == pytest.approx(0.5)
    assert consensus.regime_probabilities.sum(axis=1).to_numpy() == pytest.approx(
        np.ones(2)
    )
    assert consensus.rank_scores.to_numpy() == pytest.approx(
        np.full((2, 2), 0.5)
    )
    assert consensus.return_quantiles[24][0.5].to_numpy() == pytest.approx(
        np.full((2, 2), 0.5)
    )
    assert consensus.adverse_excursion[24].to_numpy() == pytest.approx(
        np.full((2, 2), 0.5)
    )


def test_score_to_weight_selects_two_per_side_and_applies_v2_limits() -> None:
    index = pd.date_range("2026-01-01 00:00", periods=5, freq="h")
    symbols = ["A", "B", "C", "D", "E", "F"]
    scores = pd.DataFrame(
        [[0.9, 0.8, 0.1, -0.9, -0.8, -0.1]] * len(index),
        index=index,
        columns=symbols,
    )
    confidence = pd.DataFrame(1.0, index=index, columns=symbols)
    annual_volatility = pd.DataFrame(
        [[0.20, 0.40, 0.30, 0.20, 0.40, 0.30]] * len(index),
        index=index,
        columns=symbols,
    )
    config = V2BacktestConfig(
        annual_vol_target=0.25,
        max_symbol_weight=0.25,
        max_gross_exposure=0.80,
        max_abs_net_exposure=0.05,
        max_cluster_exposure=1.0,
    )

    weights = scores_to_target_weights(
        scores,
        confidence,
        annual_volatility,
        config=config,
    )

    assert list(weights.index) == [index[0], index[4]]
    assert set(weights.columns[weights.iloc[0] > 0.0]) == {"A", "B"}
    assert set(weights.columns[weights.iloc[0] < 0.0]) == {"D", "E"}
    assert weights.loc[index[0], ["C", "F"]].eq(0.0).all()
    assert weights.abs().max().max() <= config.max_symbol_weight + 1e-12
    assert weights.abs().sum(axis=1).max() <= config.max_gross_exposure + 1e-12
    assert weights.sum(axis=1).abs().max() <= config.max_abs_net_exposure + 1e-12
    projected = np.sqrt(
        (
            weights.to_numpy() ** 2
            * annual_volatility.loc[weights.index].to_numpy() ** 2
        ).sum(axis=1)
    )
    assert projected.max() <= config.annual_vol_target + 1e-12


def test_score_to_weight_uses_confidence_before_ranking() -> None:
    index = pd.DatetimeIndex(["2026-01-01 00:00"])
    scores = pd.DataFrame(
        [[1.0, 0.8, -1.0, -0.8]],
        index=index,
        columns=["A", "B", "C", "D"],
    )
    confidence = pd.DataFrame(
        [[0.0, 1.0, 0.0, 1.0]],
        index=index,
        columns=scores.columns,
    )
    volatility = pd.DataFrame(0.5, index=index, columns=scores.columns)

    weights = scores_to_target_weights(scores, confidence, volatility)

    assert weights.loc[index[0], "A"] == 0.0
    assert weights.loc[index[0], "C"] == 0.0
    assert weights.loc[index[0], "B"] > 0.0
    assert weights.loc[index[0], "D"] < 0.0
