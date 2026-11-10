from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from crypto_portfolio_v1.experts.base import ExpertOutput
from crypto_portfolio_v1.gating import combine_expert_outputs


def _output(score: float, confidence: float = 1.0) -> ExpertOutput:
    index = pd.date_range("2026-01-01", periods=2, freq="h")
    columns = ["BTC", "ETH"]
    return ExpertOutput(
        scores=pd.DataFrame(score, index=index, columns=columns),
        confidence=pd.DataFrame(confidence, index=index, columns=columns),
    )


def _probabilities(**values: float) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=2, freq="h")
    defaults = {
        "trend_up": 0.0,
        "trend_down": 0.0,
        "chop": 0.0,
        "shock": 0.0,
    }
    defaults.update(values)
    return pd.DataFrame(defaults, index=index)


def test_one_hot_regime_routes_matching_expert_deterministically() -> None:
    outputs = {
        "trend": _output(0.8),
        "funding": _output(-0.6, confidence=0.0),
        "shock": _output(-1.0, confidence=0.0),
    }
    probabilities = _probabilities(trend_up=1.0)

    first = combine_expert_outputs(outputs, probabilities)
    second = combine_expert_outputs(outputs, probabilities)

    assert_frame_equal(first.scores, second.scores)
    assert_frame_equal(first.confidence, second.confidence)
    assert (first.scores == 0.8).all().all()
    assert (first.confidence == 1.0).all().all()


def test_conflicting_high_confidence_experts_abstain() -> None:
    outputs = {
        "TrendExpert": _output(1.0),
        "FundingExpert": _output(0.0, confidence=0.0),
        "ShockExpert": _output(-1.0),
    }
    probabilities = _probabilities(trend_up=0.5, shock=0.5)
    result = combine_expert_outputs(outputs, probabilities)

    assert (result.scores == 0.0).all().all()
    assert (result.confidence == 0.0).all().all()
    assert (result.diagnostics["abstained"] == 1.0).all().all()


def test_missing_regime_history_abstains_without_backfill() -> None:
    outputs = {
        "trend": _output(0.8),
        "funding": _output(0.2),
        "shock": _output(-0.5),
    }
    probabilities = _probabilities(chop=1.0).iloc[1:]
    result = combine_expert_outputs(outputs, probabilities)

    assert (result.scores.iloc[0] == 0.0).all()
    assert (result.confidence.iloc[0] == 0.0).all()
    assert (result.diagnostics["abstained"].iloc[0] == 1.0).all()

