# Six-step future-close trajectory protocol

Frozen on 2026-08-23 before training.

## Model target

One regression Transformer predicts six cumulative DOGEUSDC returns:

`close[t+k] / open[t+1] - 1`, for `k=1…6`.

Returns are expressed in percent during training. Absolute prices are not
predicted.

## Fixed setup

- Core-30 feature set.
- Sequence `60`, seed `0`.
- d_model `256`, 8 heads, 3 layers, FF multiplier `2`, dropout `0.25`.
- Equal-weight Huber loss over all six future closes.
- Train through 2024, validation 2025, frozen diagnostic 2026.
- Purge samples whose sixth target crosses a split boundary.
- Maximum 300 epochs, patience 25, min delta `1e-5`.
- Reduce learning rate by `0.5` after 7 stale validation epochs, floor `1e-6`.
- Always restore the checkpoint with minimum validation Huber loss.
- No Optuna, seed search, target weights, confidence grid or temperature.

## Fixed path-to-position controller

- Later horizons receive weights `1…6`, normalised to sum to one.
- LONG when at least 4 of 6 predicted returns are positive and weighted
  expected return is at least `+0.20%`.
- SHORT when at least 4 of 6 are negative and weighted expected return is at
  most `-0.20%`.
- Otherwise FLAT.
- Same side means HOLD; opposite side means immediate FLIP; FLAT closes.
- Execute at next-hour DOGEUSDC open with existing costs and funding.

## Evaluation

- Per-horizon MAE, zero-return baseline MAE, sign accuracy and Spearman rank
  correlation.
- Net/gross return, MDD, PF, trades and monthly returns for 2025 and 2026.
- Continue only if regression ranking and net PF exceed their baselines in
  both periods.
