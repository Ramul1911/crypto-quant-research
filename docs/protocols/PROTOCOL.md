# Causal FLIP Transformer protocol

Frozen before training on 2026-08-22.

## Objective

Test whether the existing FLIP Transformer architecture can learn a stable
directional signal when every feature, label, split and execution timestamp is
strictly causal.

Historical results are development walk-forward diagnostics because these
periods have already been inspected. The first untouched forward period starts
after the final production candidate is frozen.

## Data and timestamps

- Signal features: DOGEUSDT, BTCUSDT and ETHUSDT completed 1h candles.
- A feature row stamped at hour `t` may use that completed hourly candle.
- Completed 4h bars are stamped at their fourth constituent hour.
- Daily features use the previous completed UTC day.
- Macro and daily sentiment inputs are delayed by one day.
- Missing pre-history is never backfilled from the future.
- Execution: DOGEUSDC perpetual at the next hourly open after the signal.
- Funding, fee `4 bps/side` and slippage `2 bps/side` are included.

## Target and model

- Fixed horizon: `6h`.
- Input window: 80 rows ending at and including decision row `t`.
- Direction target: sign of DOGEUSDT `close[t+6] / open[t+1] - 1`.
- Auxiliary regression target: the same future return in percentage points.
- Architecture and optimizer hyperparameters are frozen from the old FLIP
  bundle; no Optuna search is allowed in this experiment.
- Fixed seeds: `11, 29, 47, 71, 89`.

## Walk-forward folds

| Fold | Train | Validation | Test |
|---|---|---|---|
| wf1 | through 2023-12-31 | 2024H1 | 2024H2 |
| wf2 | through 2024-06-30 | 2024H2 | 2025H1 |
| wf3 | through 2024-12-31 | 2025H1 | 2025H2 |
| wf4 | through 2025-06-30 | 2025H2 | 2026 YTD |

Targets crossing a split boundary are purged. Feature normalization is fitted
on the training rows of each fold only.

## Selection

For every seed:

1. Early stopping uses validation loss.
2. A symmetric flip confidence threshold is selected on validation only from
   `0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65`.
3. Validation score is net return minus a penalty of two percentage points for
   each MDD percentage point beyond `-35%`.
4. Configurations with fewer than 20 validation trades are ineligible.

The seed with the highest validation score is the fold winner. Its threshold
and checkpoint are then evaluated once on the next test window. Test metrics
for all five seeds are retained only as a stability diagnostic and are never
used for selection.

## Decision rule

- While flat, enter only when one class probability reaches the selected
  threshold.
- While long, hold until short probability reaches the threshold.
- While short, hold until long probability reaches the threshold.
- A reversal closes the old position and opens the new one at the same next
  hourly open.
- Every fold is liquidated at its final available close.

## Acceptance criteria

The architecture is a research GO only if:

- selected-seed stitched net return is positive;
- at least 3 of 4 selected-seed test folds are profitable;
- stitched profit factor is at least `1.20`;
- stitched maximum drawdown is no worse than `-35%`;
- median test return across all seeds is positive in at least 3 of 4 folds.

Failure of any criterion is NO-GO. Thresholds, seeds, folds and acceptance
criteria must not be changed after seeing results.
