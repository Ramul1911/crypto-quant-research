# Optuna best-seed champion protocol

Frozen on 2026-08-22 before the search.

This is an intentionally champion-oriented development search. It uses the
best seed rather than the median seed, as requested. Therefore historical
results do not establish reproducibility; only forward performance after the
freeze can validate the champion.

## Phase 1 — architecture Optuna

- 60 trials.
- Internal train: through `2024-12-31`.
- Internal validation/objective: `2025-01-01 → 2025-06-30`.
- Two fixed search seeds per trial: `11, 29`.
- Trial score: best validation correction score across the two seeds.
- Compact exact-FLIP correction grid on internal validation.
- Keep the top 3 architecture trials.

Search space:

- horizon: `3, 6, 12, 24`;
- sequence length: `40, 60, 80, 120, 168`;
- model width: `64, 128, 256, 384`;
- attention heads: `4, 8`;
- layers: `2…5`;
- FF multiplier: `2, 4`;
- dropout: `0.05…0.35`;
- learning rate: `1e-5…5e-4`;
- weight decay: `1e-6…1e-3`;
- batch size: `256, 512`;
- direction-loss weight: `0.60…1.00`.

## Phase 2 — 20-seed champion

- Final train includes 2025H1: through `2025-06-30`.
- Champion selection: `2025-07-01 → 2025-12-31`.
- Seeds: `0…19`.
- Train all 20 seeds for each of the top 3 Optuna architectures.
- Full exact-FLIP correction grid per checkpoint:
  144 fixed/dynamic temperature corrections × 49 LTH/STH pairs.
- Select the single best architecture × seed × correction by validation score.

## Phase 3 — diagnostic

- Freeze the selected checkpoint and correction.
- Evaluate once on `2026-01-01 → 2026-08-22`.
- DOGEUSDC next-hour open, 93.1% position, fee, 2 bps slippage and funding.
- Do not change the champion after seeing 2026.
- True untouched OOS starts after the final champion freeze.
