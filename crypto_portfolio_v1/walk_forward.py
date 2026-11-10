from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v2_protocol import (
    DEVELOPMENT_FOLDS,
    MAX_LABEL_HORIZON_HOURS,
    AcceptanceCriteria,
    FoldSpec,
)


@dataclass(frozen=True)
class FoldMasks:
    """Boolean membership masks for one purged walk-forward fold."""

    name: str
    train: pd.Series
    validation: pd.Series
    test: pd.Series

    def as_dict(self) -> dict[str, pd.Series]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }

    def __getitem__(self, split: str) -> pd.Series:
        try:
            return self.as_dict()[split]
        except KeyError as error:
            raise KeyError(f"unknown fold split: {split}") from error


def _validated_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("walk-forward data must use a DatetimeIndex")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("walk-forward index must be sorted and unique")
    return index


def _timestamp(value: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if index.tz is not None:
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(index.tz)
        else:
            timestamp = timestamp.tz_convert(index.tz)
    elif timestamp.tzinfo is not None:
        raise ValueError("fold timestamps and data index must have matching timezones")
    return timestamp


def build_fold_masks(
    index: pd.DatetimeIndex,
    fold: FoldSpec,
    purge_hours: int = MAX_LABEL_HORIZON_HOURS,
) -> FoldMasks:
    """Build strict train/validation/test masks with causal boundary purges.

    Observations whose maximum label horizon can touch the next split are
    excluded.  Thus a timestamp ``t`` is retained only when
    ``t + purge < next_split_start``.  The resulting empty interval is the
    purge/embargo between adjacent samples.
    """
    index = _validated_index(index)
    if purge_hours < 0:
        raise ValueError("purge_hours cannot be negative")
    purge = pd.Timedelta(hours=purge_hours)

    train_start = _timestamp(fold.train_start, index)
    train_end = _timestamp(fold.train_end, index)
    validation_start = _timestamp(fold.validation_start, index)
    validation_end = _timestamp(fold.validation_end, index)
    test_start = _timestamp(fold.test_start, index)
    test_end = _timestamp(fold.test_end, index)
    if not (
        train_start <= train_end < validation_start <= validation_end
        < test_start <= test_end
    ):
        raise ValueError(f"fold {fold.name!r} has invalid or overlapping boundaries")

    train = (index >= train_start) & (index <= train_end)
    validation = (index >= validation_start) & (index <= validation_end)
    test = (index >= test_start) & (index <= test_end)
    if purge:
        train &= index + purge < validation_start
        validation &= index + purge < test_start

    masks = FoldMasks(
        fold.name,
        pd.Series(train, index=index, dtype=bool, name="train"),
        pd.Series(validation, index=index, dtype=bool, name="validation"),
        pd.Series(test, index=index, dtype=bool, name="test"),
    )
    assert_no_mask_overlap(masks)
    return masks


def build_development_masks(
    index: pd.DatetimeIndex,
    folds: Sequence[FoldSpec] = DEVELOPMENT_FOLDS,
    purge_hours: int = MAX_LABEL_HORIZON_HOURS,
) -> dict[str, FoldMasks]:
    """Build masks for every V2 development fold."""
    result = {
        fold.name: build_fold_masks(index, fold, purge_hours) for fold in folds
    }
    if len(result) != len(folds):
        raise ValueError("fold names must be unique")
    return result


def assert_no_mask_overlap(masks: FoldMasks | Mapping[str, pd.Series]) -> None:
    """Assert that train, validation and test membership is mutually exclusive."""
    parts = masks.as_dict() if isinstance(masks, FoldMasks) else dict(masks)
    required = {"train", "validation", "test"}
    if set(parts) != required:
        raise ValueError(f"masks must contain exactly {sorted(required)}")
    indexes = [mask.index for mask in parts.values()]
    if not all(indexes[0].equals(index) for index in indexes[1:]):
        raise ValueError("all masks must have the same index")
    membership = sum(mask.astype(np.int8) for mask in parts.values())
    if bool((membership > 1).any()):
        raise AssertionError("train, validation and test masks overlap")


def slice_aligned(
    values: Mapping[str, pd.DataFrame | pd.Series],
    selector: pd.Series | np.ndarray,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Slice an aligned mapping without silently reindexing any member."""
    if not values:
        return {}
    first = next(iter(values.values()))
    index = _validated_index(first.index)
    for name, value in values.items():
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            raise TypeError(f"{name} must be a DataFrame or Series")
        if not value.index.equals(index):
            raise ValueError(f"{name} is not aligned to the common index")

    if isinstance(selector, pd.Series):
        if not selector.index.equals(index):
            raise ValueError("selector is not aligned to the common index")
        mask = selector.astype(bool).to_numpy()
    else:
        mask = np.asarray(selector)
        if mask.ndim != 1 or len(mask) != len(index):
            raise ValueError("selector length must match the common index")
        if mask.dtype != bool:
            raise TypeError("selector must be boolean")
    return {name: value.loc[index[mask]].copy() for name, value in values.items()}


def slice_fold(
    values: Mapping[str, pd.DataFrame | pd.Series],
    masks: FoldMasks,
) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    """Return aligned train, validation and test mappings."""
    return {
        split: slice_aligned(values, mask)
        for split, mask in masks.as_dict().items()
    }


def _curve_frequency(curve: pd.DataFrame) -> pd.Timedelta | None:
    differences = curve.index.to_series().diff().dropna()
    if differences.empty:
        return None
    if (differences <= pd.Timedelta(0)).any():
        raise ValueError("curve index must be strictly increasing")
    modes = differences.mode()
    return modes.iloc[0] if len(modes) == 1 else differences.median()


def _assert_curve_cadence(
    curve: pd.DataFrame, frequency: pd.Timedelta | None
) -> None:
    if frequency is None or len(curve) < 2:
        return
    differences = curve.index.to_series().diff().dropna()
    if not bool((differences == frequency).all()):
        raise ValueError("gap or inconsistent cadence inside an OOS fold curve")


def stitch_oos_equity(
    curves: Sequence[pd.DataFrame],
    fold_names: Sequence[str] | None = None,
    *,
    initial_equities: float | Sequence[float] = 1.0,
    expected_frequency: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Continuously compound sequential independent OOS equity curves.

    Every curve is interpreted relative to its own initial equity.  Boundary
    timestamps must be exactly one observation apart, preventing accidental
    overlap or missing OOS intervals.
    """
    if fold_names is not None and len(fold_names) != len(curves):
        raise ValueError("fold_names length must match curves")
    if np.isscalar(initial_equities):
        starts = [float(initial_equities)] * len(curves)
    else:
        starts = [float(value) for value in initial_equities]
        if len(starts) != len(curves):
            raise ValueError("initial_equities length must match curves")
    if any(value <= 0 or not np.isfinite(value) for value in starts):
        raise ValueError("initial equities must be finite and positive")

    frequency = (
        None if expected_frequency is None else pd.Timedelta(expected_frequency)
    )
    if frequency is not None and frequency <= pd.Timedelta(0):
        raise ValueError("expected_frequency must be positive")
    nonempty_frequencies = {
        item for curve in curves if not curve.empty
        for item in [_curve_frequency(curve)] if item is not None
    }
    if frequency is None and len(nonempty_frequencies) == 1:
        frequency = next(iter(nonempty_frequencies))
    elif frequency is None and len(nonempty_frequencies) > 1:
        raise ValueError("fold curves do not have a consistent frequency")
    elif frequency is not None and any(
        item != frequency for item in nonempty_frequencies
    ):
        raise ValueError("a fold curve does not match expected_frequency")

    stitched: list[pd.DataFrame] = []
    previous_end: float | None = None
    last_timestamp: pd.Timestamp | None = None
    for position, (curve, initial) in enumerate(zip(curves, starts)):
        if curve.empty:
            raise ValueError("OOS fold curves cannot be empty")
        if "equity" not in curve.columns:
            raise ValueError("every curve must contain an equity column")
        part = curve.copy()
        _validated_index(part.index)
        _assert_curve_cadence(part, frequency)
        equity = pd.to_numeric(part["equity"], errors="coerce")
        if not np.isfinite(equity.to_numpy()).all() or (equity <= 0).any():
            raise ValueError("fold equity must be finite and positive")
        if last_timestamp is not None:
            if part.index[0] <= last_timestamp:
                raise ValueError("OOS fold curves overlap or are out of order")
            if frequency is None:
                raise ValueError(
                    "expected_frequency is required when cadence cannot be inferred"
                )
            if part.index[0] - last_timestamp != frequency:
                raise ValueError("gap between sequential OOS fold curves")

        scale = 1.0 if previous_end is None else previous_end / initial
        monetary_columns = (
            "equity",
            "market_pnl",
            "funding_pnl",
            "fee",
            "slippage",
            "trading_cost",
            "turnover_notional",
            "accounting_error",
        )
        for column in monetary_columns:
            if column in part:
                part[column] = pd.to_numeric(part[column], errors="raise") * scale
        part["fold"] = (
            fold_names[position] if fold_names is not None else str(position)
        )
        stitched.append(part)
        previous_end = float(part["equity"].iloc[-1])
        last_timestamp = part.index[-1]

    if not stitched:
        return pd.DataFrame()
    result = pd.concat(stitched)
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1.0
    return result


def _monthly_returns(equity: pd.Series, initial_equity: float) -> pd.Series:
    month_ends = equity.groupby(equity.index.to_period("M")).last()
    prior = month_ends.shift(1)
    prior.iloc[0] = initial_equity
    return month_ends / prior - 1.0


def aggregate_walk_forward_stats(
    stitched_equity: pd.DataFrame,
    fold_stats: Sequence[Mapping[str, Any]] | None = None,
    *,
    initial_equity: float = 1.0,
    baseline_equity: pd.DataFrame | pd.Series | None = None,
) -> dict[str, Any]:
    """Aggregate OOS fold/month performance and baseline comparison."""
    if stitched_equity.empty:
        raise ValueError("stitched_equity cannot be empty")
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if "equity" not in stitched_equity:
        raise ValueError("stitched_equity must contain an equity column")
    _validated_index(stitched_equity.index)
    equity = stitched_equity["equity"].astype(float)
    if not np.isfinite(equity.to_numpy()).all() or (equity <= 0).any():
        raise ValueError("equity must be finite and positive")

    monthly = _monthly_returns(equity, initial_equity)
    fold_returns: dict[str, float] = {}
    if "fold" in stitched_equity:
        prior = initial_equity
        for name, group in stitched_equity.groupby("fold", sort=False):
            end = float(group["equity"].iloc[-1])
            fold_returns[str(name)] = (end / prior - 1.0) * 100.0
            prior = end

    supplied = list(fold_stats or ())
    explicit_kills = sum(bool(item.get("killed", False)) for item in supplied)
    baseline_return_pct: float | None = None
    if baseline_equity is not None:
        baseline = (
            baseline_equity["equity"]
            if isinstance(baseline_equity, pd.DataFrame)
            else baseline_equity
        )
        if baseline.empty:
            raise ValueError("baseline_equity cannot be empty")
        baseline_return_pct = (
            float(baseline.iloc[-1]) / initial_equity - 1.0
        ) * 100.0

    strategy_return_pct = (float(equity.iloc[-1]) / initial_equity - 1.0) * 100.0
    drawdown = equity / equity.cummax() - 1.0
    result: dict[str, Any] = {
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
        "initial_equity": float(initial_equity),
        "final_equity": float(equity.iloc[-1]),
        "return_pct": strategy_return_pct,
        "mdd_pct": float(drawdown.min()) * 100.0,
        "monthly_returns": monthly,
        "monthly_returns_pct": {
            str(period): float(value) * 100.0 for period, value in monthly.items()
        },
        "month_count": int(len(monthly)),
        "profitable_months": int((monthly > 0).sum()),
        "profitable_month_share": float((monthly > 0).mean()),
        "median_monthly_return_pct": float(monthly.median()) * 100.0,
        "mean_monthly_return_pct": float(monthly.mean()) * 100.0,
        "fold_returns_pct": fold_returns,
        "fold_kills": explicit_kills,
        "baseline_return_pct": baseline_return_pct,
        "baseline_excess_return_pct": (
            None
            if baseline_return_pct is None
            else strategy_return_pct - baseline_return_pct
        ),
    }
    return result


def evaluate_acceptance_criteria(
    stats: Mapping[str, Any],
    criteria: AcceptanceCriteria = AcceptanceCriteria(),
) -> dict[str, Any]:
    """Evaluate every V2 go/no-go criterion and return an auditable report."""
    required = (
        "mdd_pct",
        "profitable_month_share",
        "median_monthly_return_pct",
        "mean_monthly_return_pct",
        "fold_returns_pct",
    )
    missing = [name for name in required if name not in stats]
    if missing:
        raise ValueError(f"stats missing required fields: {missing}")
    fold_returns = stats["fold_returns_pct"]
    if not isinstance(fold_returns, Mapping):
        raise TypeError("fold_returns_pct must be a mapping")
    killed_folds = [
        str(name)
        for name, value in fold_returns.items()
        if float(value) <= criteria.fold_kill_pct
    ]
    explicit_kills = int(stats.get("fold_kills", 0))
    checks = {
        "max_mdd": float(stats["mdd_pct"]) >= criteria.max_mdd_pct,
        "no_fold_kills": not killed_folds and explicit_kills == 0,
        "profitable_month_share": (
            float(stats["profitable_month_share"])
            >= criteria.min_profitable_month_share
        ),
        "median_monthly_return": (
            float(stats["median_monthly_return_pct"])
            >= criteria.min_median_monthly_return_pct
        ),
        "mean_monthly_return_min": (
            float(stats["mean_monthly_return_pct"])
            >= criteria.min_mean_monthly_return_pct
        ),
        "mean_monthly_return_max": (
            float(stats["mean_monthly_return_pct"])
            <= criteria.max_mean_monthly_return_pct
        ),
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "accepted": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "killed_folds": killed_folds,
        "explicit_fold_kills": explicit_kills,
        "criteria": asdict(criteria),
        "profitable_months": stats.get("profitable_months"),
        "month_count": stats.get("month_count"),
        "profitable_month_share": float(stats["profitable_month_share"]),
        "median_monthly_return_pct": float(stats["median_monthly_return_pct"]),
        "mean_monthly_return_pct": float(stats["mean_monthly_return_pct"]),
        "mdd_pct": float(stats["mdd_pct"]),
        "baseline_return_pct": stats.get("baseline_return_pct"),
        "baseline_excess_return_pct": stats.get("baseline_excess_return_pct"),
    }


def summarize_walk_forward(
    stitched_equity: pd.DataFrame,
    fold_stats: Sequence[Mapping[str, Any]] | None = None,
    *,
    initial_equity: float = 1.0,
    baseline_equity: pd.DataFrame | pd.Series | None = None,
    criteria: AcceptanceCriteria = AcceptanceCriteria(),
) -> dict[str, Any]:
    """Aggregate a walk-forward run and attach its go/no-go evaluation."""
    stats = aggregate_walk_forward_stats(
        stitched_equity,
        fold_stats,
        initial_equity=initial_equity,
        baseline_equity=baseline_equity,
    )
    stats["acceptance"] = evaluate_acceptance_criteria(stats, criteria)
    return stats


# Explicit aliases keep call sites readable while using one implementation.
make_fold_masks = build_fold_masks
stitch_equity_curves = stitch_oos_equity
evaluate_acceptance = evaluate_acceptance_criteria
