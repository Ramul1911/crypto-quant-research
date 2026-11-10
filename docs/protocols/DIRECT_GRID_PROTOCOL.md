# Direct 2025–2026 FLIP correction-grid protocol

Frozen before the run on 2026-08-22.

## Split

- Train: causal history through `2024-06-30 23:59:59`.
- Validation/selection: `2024-07-01 → 2024-12-31`.
- Test: `2025-01-01 → 2026-08-22 latest completed hour`.
- Test is evaluated once for the single seed and correction selected on
  validation.

## Training

- Existing causal 129-feature schema and H6 target.
- Fixed architecture/hyperparameters from the FLIP bundle.
- Fixed seeds: `11, 29, 47, 71, 89`.
- Scaler fitted on train only.
- Early stopping uses validation loss.

## Correction grid

Exact FLIP signal rule is restored: independent `LTH` and `STH`; if both pass,
the larger probability wins; if neither passes, signal is FLAT.

- LTH: `0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65`.
- STH: `0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70`.
- Fixed temperature: `0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00`.
- Dynamic factors: `body`, `volume`, `efficiency`, `body+volume`,
  `body+efficiency`.
- Dynamic alpha: `1.0, 2.0, 3.0`.
- Dynamic Tmin: `0.50, 0.60, 0.70`.
- Dynamic Tmax: `1.10, 1.20, 1.40`.

Dynamic temperature uses only rolling 100-hour candle information available
at the decision timestamp. Every seed receives the full joint correction grid
on validation.

## Selection and execution

- Minimum 20 validation trades.
- Score = validation net return minus `2 ×` each MDD percentage point beyond
  `-35%`.
- Highest-scoring `seed × correction × LTH × STH` is frozen.
- DOGEUSDC next-hour open execution, 93.1% position fraction,
  `4 bps/side` fee, `2 bps/side` slippage and actual funding.

The selected configuration is then evaluated once on the full 2025–2026 test.
No post-test parameter or seed change is allowed.
