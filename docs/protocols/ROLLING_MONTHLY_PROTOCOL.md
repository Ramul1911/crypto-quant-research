# Rolling monthly FLIP protocol

Frozen on 2026-08-23 before this pilot.

## Purpose

Test whether the fixed Optuna trial `51` architecture can adapt causally as
new market regimes, including 2026, enter its training window.

## Folds

- Test months: `2025-01` through `2026-08`.
- For each test month:
  - train on the 36 complete calendar months ending immediately before the
    validation month;
  - validate on the complete calendar month immediately before test;
  - test on the next calendar month;
  - shift all boundaries forward by one month and retrain from scratch.
- Purge 24-hour targets crossing train or validation boundaries.
- A position is forcibly closed at every month boundary before the next
  checkpoint is installed.

Example:

- train `2021-12-01 → 2024-11-30`;
- validation `2024-12-01 → 2024-12-31`;
- test `2025-01-01 → 2025-01-31`.

## Model and selection

- Architecture: fixed Optuna trial `51`.
- Seeds per fold: `0, 5, 8, 11, 12, 22, 29, 37, 47, 89`.
- Full temperature/LTH/STH grid on the validation month only.
- Require at least 5 closed validation trades.
- Select one seed × correction by validation net return with the standard
  drawdown penalty beyond 35%.
- Apply the selected checkpoint and correction unchanged to the next month.

This is a 200-training pilot. The seed set may only be expanded in a later,
separately labelled experiment after the pilot is complete.

## Accounting

- DOGEUSDC execution at next-hour open.
- 93.1% account allocation.
- Fees, 2 bps slippage per side, and funding included.
- Compound the 20 one-month test equity curves into one stitched account.
- Report fold-level validation and test metrics, monthly returns, stitched
  return, MDD, profit factor, and seed-selection frequency.

Historical 2026 has already been inspected in prior experiments, so this is
a development diagnostic. Only future data after the final freeze is truly
untouched.
