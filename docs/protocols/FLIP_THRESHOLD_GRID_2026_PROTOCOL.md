# Honest FLIP threshold grid on 2026 causal data

Frozen on 2026-09-10 before running the grid or viewing any per-cell results.

## Motivation

`prod_doge_1h_v1/model_card.json` shows the production thresholds `0.48/0.52`
were chosen by a 529-cell grid search (`long, short in [0.48, 0.70]`, step
`0.01`) scored on `2025-01-01 -> 2026-02-28`. That period therefore cannot be
used again to judge those thresholds. This experiment re-runs the identical
grid, unchanged, but scores every cell purely on data the original search
never touched, to see whether `0.48/0.52` sits in a genuinely robust region or
was a lucky cell.

## Frozen setup

- Model: frozen `prod_doge_1h_v1/model.pt` (seed 72), no retraining.
- Features: deterministic causal pipeline from `run_flip_causal_2025_2026.py`
  (completed 1h/4h/1d only, no partial/mutable higher-timeframe context).
- Calibration: same dynamic-temperature scaling as production, computed once
  and shared across the whole grid (temperature does not depend on threshold).
- Execution: hourly OHLC DOGEUSDC perpetual, decision on bar close, entry on
  next bar open, matching `model_card.json`'s stated production execution
  convention (not the separate 1-minute parity check).
- Costs: fee `4 bps/side`, slippage `2 bps/side`, real funding, position
  fraction `93.1%`.
- Grid: `long_threshold, short_threshold in [0.48, 0.70]`, step `0.01`
  (`23 x 23 = 529` cells), identical to the original search.

## Windows

- `2025_selection_period` (2025-01-01 -> 2025-12-31): reference only, already
  used for champion/threshold selection, not evidence.
- `2026_touched_jan_feb` (2026-01-01 -> 2026-02-28): also inside the original
  grid search window, reference only.
- `2026_untouched_mar_onward` (2026-03-01 -> latest available): the only
  window this experiment treats as evidence. No parameter in this repository
  has ever been chosen by looking at it.
- `full`: entire 2025 -> present span, reported for context only.

## Evaluation

- Primary metric: `2026_untouched_mar_onward` return and MDD per cell.
- Cells with fewer than 10 trades in the untouched window are marked
  unreliable and excluded from the "robust region" claim.
- The production cell `(0.48, 0.52)` is reported explicitly with its rank
  among all 529 cells on the untouched window.
- Robustness check: compare the production cell's untouched return against
  the neighborhood median (its `3x3` block of adjacent cells) and against the
  full-grid median and interquartile range.
- No threshold is adopted or changed as a result of this run. This is a
  diagnostic, not a re-optimization.
