# Simple three-model 6h edge protocol

Frozen on 2026-08-23 before training.

## Goal

Check whether a simple six-hour directional/flat signal has raw transferable
edge before any Optuna, seed search, temperature correction, or threshold grid.

## Fixed setup

- Three independent binary Transformers: `UP`, `DOWN`, `FLAT`.
- Seed `0` for every model.
- Sequence length `60` completed hourly candles.
- Forecast horizon: entry at DOGEUSDC `open[t+1]`, comparison price at
  DOGEUSDC `open[t+7]` (six elapsed hours).
- Flat band: `±0.20%` raw price return.
- Labels:
  - `UP=1` when return is greater than `+0.20%`;
  - `DOWN=1` when return is less than `-0.20%`;
  - `FLAT=1` otherwise.
- Architecture shape: d_model `256`, 8 heads, 3 layers, FF multiplier `2`,
  dropout `0.25`.
- Class weights are calculated from train labels only.

## Time split

- Train: all eligible samples through `2024-12-31`.
- Validation: calendar year `2025`.
- Frozen diagnostic: `2026-01-01 → latest`.
- Six-hour labels crossing a split boundary are purged.
- Feature scaling statistics use train rows only.

## Controller and execution

- At every completed hour, choose the class with the largest independent
  positive-class probability.
- `UP → LONG`, `DOWN → SHORT`, `FLAT → no position`.
- The same side means HOLD; the opposite side means immediate FLIP; FLAT
  closes an existing position.
- Execute at next-hour DOGEUSDC open.
- Include 93.1% allocation, 4 bps fee and 2 bps slippage per side, plus
  funding.
- No confidence thresholds, temperature scaling, hysteresis, or grid.

## Edge decision

Continue to Optuna only if:

- mean directional (`UP`, `DOWN`) ROC-AUC is above `0.52` separately in
  2025 and 2026;
- net profit factor is above `1.0` separately in 2025 and 2026;
- the strategy beats fixed LONG and fixed SHORT baselines;
- turnover does not erase an otherwise positive gross edge.

Historical 2026 has already been inspected, so it remains a development
diagnostic rather than untouched OOS.
