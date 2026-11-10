from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .experts.base import ExpertOutput
from .models.temporal_gate import REGIMES


def _expert_kind(name: str) -> str:
    normalized = name.lower()
    for kind in ("trend", "funding", "shock"):
        if kind in normalized:
            return kind
    raise KeyError(
        f"cannot route expert {name!r}; name must contain trend, funding, or shock"
    )


def combine_expert_outputs(
    outputs: Mapping[str, ExpertOutput],
    regime_probabilities: pd.DataFrame,
    *,
    uncertainty_threshold: float = 0.72,
    disagreement_threshold: float = 0.55,
    min_support: float = 0.05,
) -> ExpertOutput:
    """Regime-route V2 experts and abstain on ambiguous recommendations.

    The output retains the expert panel's full shape. Regime rows that are not
    available yet are not backfilled; those timestamps deterministically
    abstain with zero score and confidence.
    """

    if not outputs:
        raise ValueError("at least one expert output is required")
    if not 0.0 <= uncertainty_threshold <= 1.0:
        raise ValueError("uncertainty_threshold must be in [0, 1]")
    if not 0.0 <= disagreement_threshold <= 1.0:
        raise ValueError("disagreement_threshold must be in [0, 1]")
    missing_regimes = [name for name in REGIMES if name not in regime_probabilities]
    if missing_regimes:
        raise KeyError(f"regime probabilities are missing: {missing_regimes}")

    first = next(iter(outputs.values()))
    index, columns = first.scores.index, first.scores.columns
    routed: dict[str, ExpertOutput] = {}
    for name, output in outputs.items():
        kind = _expert_kind(name)
        if kind in routed:
            raise ValueError(f"multiple {kind!r} expert outputs were supplied")
        if not output.scores.index.equals(index) or not output.scores.columns.equals(
            columns
        ):
            raise ValueError("all expert outputs must have the same shape and labels")
        routed[kind] = output
    missing_experts = [
        name for name in ("trend", "funding", "shock") if name not in routed
    ]
    if missing_experts:
        raise KeyError(f"missing expert outputs: {missing_experts}")

    probabilities = regime_probabilities.loc[:, REGIMES].astype(float).clip(
        lower=0.0
    )
    row_sum = probabilities.sum(axis=1).replace(0.0, np.nan)
    probabilities = probabilities.div(row_sum, axis=0)
    probabilities = probabilities.reindex(index)
    expanded = {
        name: pd.DataFrame(
            np.repeat(probabilities[[name]].to_numpy(), len(columns), axis=1),
            index=index,
            columns=columns,
        )
        for name in REGIMES
    }

    trend = routed["trend"]
    trend_regime = expanded["trend_up"].where(
        trend.scores >= 0.0, expanded["trend_down"]
    )
    regime_weights = {
        "trend": trend_regime,
        "funding": expanded["chop"] + 0.20 * (
            expanded["trend_up"] + expanded["trend_down"]
        ),
        "shock": expanded["shock"],
    }

    effective_weights: dict[str, pd.DataFrame] = {}
    for kind, output in routed.items():
        effective_weights[kind] = (
            regime_weights[kind] * output.confidence.fillna(0.0).clip(0.0, 1.0)
        )
    support = sum(effective_weights.values())
    numerator = sum(
        effective_weights[kind] * routed[kind].scores.fillna(0.0)
        for kind in routed
    )
    safe_support = support.where(support > np.finfo(float).eps)
    combined = numerator.div(safe_support).fillna(0.0).clip(-1.0, 1.0)

    weighted_variance = sum(
        effective_weights[kind] * routed[kind].scores.fillna(0.0).sub(combined) ** 2
        for kind in routed
    ).div(safe_support)
    disagreement = np.sqrt(weighted_variance.clip(lower=0.0)).fillna(0.0).clip(
        0.0, 1.0
    )

    probability_values = probabilities.to_numpy(dtype=float)
    log_probabilities = np.zeros_like(probability_values)
    positive = probability_values > 0.0
    np.log(probability_values, out=log_probabilities, where=positive)
    entropy = -np.nansum(
        probability_values * log_probabilities, axis=1
    ) / np.log(len(REGIMES))
    invalid_probability = probabilities.isna().any(axis=1).to_numpy()
    entropy[invalid_probability] = 1.0
    uncertainty = pd.DataFrame(
        np.repeat(entropy[:, None], len(columns), axis=1),
        index=index,
        columns=columns,
    ).clip(0.0, 1.0)

    abstain = (
        (support < min_support)
        | (uncertainty > uncertainty_threshold)
        | (disagreement > disagreement_threshold)
    )
    scores = combined.mask(abstain, 0.0)
    confidence = (
        support.clip(0.0, 1.0)
        * (1.0 - uncertainty)
        * (1.0 - disagreement)
    ).clip(0.0, 1.0)
    confidence = confidence.mask(abstain, 0.0)

    return ExpertOutput(
        scores=scores,
        confidence=confidence,
        diagnostics={
            "regime_uncertainty": uncertainty,
            "expert_disagreement": disagreement,
            "effective_support": support,
            "abstained": abstain.astype(float),
        },
    )

