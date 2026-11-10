# Fixed-architecture 50-seed protocol

Frozen on 2026-08-23 before this run.

## Fixed model

Use Optuna trial `51` without further architecture search:

- horizon `24h`;
- sequence length `40h`;
- d_model `256`, 8 heads, 3 layers, FF multiplier `2`;
- dropout `0.25`;
- learning rate `0.0002714918929018466`;
- weight decay `0.0005195621155056025`;
- batch size `256`;
- direction-loss weight `0.9`.

## Train and selection

- Train data: through `2024-12-31`.
- Seeds: `0…49`.
- Early stopping and champion selection: full calendar year `2025`.
- Run the full 144-temperature × 49-LTH/STH correction grid separately for
  each seed.
- A correction is eligible only when its net return is positive in both
  `2025H1` and `2025H2`, with at least 20 closed trades in each half.
- Among eligible corrections, select the largest full-2025 net-return score,
  retaining the existing drawdown penalty beyond 35%.
- The same correction must be used in both halves; no half-specific grids.
- If no correction is eligible, report NO-GO and do not manufacture a
  champion by relaxing the frozen rule.

This half-year eligibility rule prevents a permanently short model from
winning solely because of the falling market in `2025H2`.

## Diagnostic

- Freeze exactly one model × seed × correction using 2025 only.
- Apply it unchanged to `2026-01-01 → latest`.
- Execution remains DOGEUSDC next-hour open with 93.1% position, fees,
  2 bps slippage per side, and funding.
- Historical 2026 has already been inspected in earlier experiments.
  Therefore it is a post-hoc diagnostic, not untouched OOS.
- Genuine validation starts with forward data after this checkpoint freeze.
