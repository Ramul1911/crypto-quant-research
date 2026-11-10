from __future__ import annotations

import pandas as pd
import pytest

from crypto_portfolio_v1.v2_protocol import AcceptanceCriteria, FoldSpec
from crypto_portfolio_v1.walk_forward import (
    aggregate_walk_forward_stats,
    assert_no_mask_overlap,
    build_fold_masks,
    evaluate_acceptance_criteria,
    slice_aligned,
    stitch_oos_equity,
)


def _fold() -> FoldSpec:
    return FoldSpec(
        name="example",
        train_start="2025-01-01",
        train_end="2025-01-31 23:59:59",
        validation_start="2025-02-01",
        validation_end="2025-02-28 23:59:59",
        test_start="2025-03-01",
        test_end="2025-03-31 23:59:59",
    )


def test_purge_boundaries_exclude_full_label_horizon() -> None:
    index = pd.date_range("2025-01-28", "2025-03-02", freq="h")
    masks = build_fold_masks(index, _fold(), purge_hours=72)

    assert masks.train.loc["2025-01-28 23:00"]
    assert not masks.train.loc["2025-01-29 00:00"]
    assert masks.validation.loc["2025-02-25 23:00"]
    assert not masks.validation.loc["2025-02-26 00:00"]
    assert masks.test.loc["2025-03-01 00:00"]

    last_train = masks.train.index[masks.train][-1]
    last_validation = masks.validation.index[masks.validation][-1]
    assert last_train + pd.Timedelta(hours=72) < pd.Timestamp("2025-02-01")
    assert last_validation + pd.Timedelta(hours=72) < pd.Timestamp("2025-03-01")


def test_masks_never_overlap_and_aligned_slicing_is_strict() -> None:
    index = pd.date_range("2025-01-01", "2025-03-31 23:00", freq="h")
    masks = build_fold_masks(index, _fold())
    assert_no_mask_overlap(masks)
    membership = (
        masks.train.astype(int)
        + masks.validation.astype(int)
        + masks.test.astype(int)
    )
    assert membership.max() == 1

    frame = pd.DataFrame({"x": range(len(index))}, index=index)
    series = frame["x"] * 2
    sliced = slice_aligned({"frame": frame, "series": series}, masks.test)
    assert sliced["frame"].index.equals(sliced["series"].index)
    assert sliced["frame"].index.min() == pd.Timestamp("2025-03-01")

    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="not aligned"):
        slice_aligned({"frame": frame, "shifted": shifted}, masks.test)


def test_stitched_equity_is_continuous_and_preserves_fold_returns() -> None:
    first = pd.DataFrame(
        {"equity": [1.10, 1.21]},
        index=pd.date_range("2025-01-01", periods=2, freq="D"),
    )
    second = pd.DataFrame(
        {"equity": [1.20, 1.32]},
        index=pd.date_range("2025-01-03", periods=2, freq="D"),
    )
    stitched = stitch_oos_equity(
        [first, second], ["one", "two"], initial_equities=[1.0, 1.0]
    )

    assert stitched.loc["2025-01-03", "equity"] == pytest.approx(1.21 * 1.20)
    assert stitched.loc["2025-01-04", "equity"] == pytest.approx(1.21 * 1.32)
    original_second_returns = second["equity"].pct_change().dropna()
    stitched_second_returns = (
        stitched.loc[stitched["fold"] == "two", "equity"].pct_change().dropna()
    )
    pd.testing.assert_series_equal(
        stitched_second_returns,
        original_second_returns,
        check_names=False,
    )
    assert stitched.index.is_unique
    assert stitched.index.is_monotonic_increasing

    overlap = second.copy()
    overlap.index = pd.date_range("2025-01-02", periods=2, freq="D")
    with pytest.raises(ValueError, match="overlap"):
        stitch_oos_equity([first, overlap])

    gap = second.copy()
    gap.index = pd.date_range("2025-01-04", periods=2, freq="D")
    with pytest.raises(ValueError, match="gap"):
        stitch_oos_equity([first, gap])


def test_criteria_behavior_and_baseline_reporting() -> None:
    index = pd.to_datetime(
        ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"]
    )
    stitched = pd.DataFrame(
        {
            "equity": [1.12, 1.232, 1.380, 1.518],
            "fold": ["one", "one", "two", "two"],
        },
        index=index,
    )
    baseline = pd.Series([1.02, 1.08], index=index[[0, -1]])
    stats = aggregate_walk_forward_stats(
        stitched,
        [{"killed": False}, {"killed": False}],
        baseline_equity=baseline,
    )
    report = evaluate_acceptance_criteria(stats)

    assert report["accepted"]
    assert report["decision"] == "GO"
    assert stats["profitable_months"] == 4
    assert stats["median_monthly_return_pct"] >= 8.0
    assert stats["mean_monthly_return_pct"] >= 10.0
    assert stats["baseline_excess_return_pct"] == pytest.approx(
        stats["return_pct"] - 8.0
    )

    failed = dict(stats)
    failed["fold_returns_pct"] = {"one": -25.0, "two": 12.0}
    failed_report = evaluate_acceptance_criteria(
        failed, AcceptanceCriteria(fold_kill_pct=-25.0)
    )
    assert not failed_report["accepted"]
    assert failed_report["decision"] == "NO-GO"
    assert failed_report["killed_folds"] == ["one"]
    assert "no_fold_kills" in failed_report["failed_checks"]
