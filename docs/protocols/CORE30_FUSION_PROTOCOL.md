# Core-30 frozen-score fusion protocol

Frozen on 2026-08-23 before fitting the fusion layer.

## Inputs

- Frozen UP, DOWN and FLAT checkpoints from the core-30 experiment.
- Their hourly positive-class probabilities only.
- Transformer weights, features, sequence, horizon, seed and flat band remain
  unchanged.

## Fusion

- Convert each binary probability to a clipped logit.
- Fit one unweighted multinomial logistic regression on calendar 2025.
- Target classes are mutually exclusive `SHORT=-1`, `FLAT=0`, `LONG=1`.
- Fixed settings: `C=1.0`, L2 regularisation, intercept enabled,
  `max_iter=1000`, random_state `0`.
- Do not use class weights: the purpose is posterior comparability, not
  rebalancing.
- Select the class with maximum calibrated softmax probability.
- No threshold, temperature, hysteresis, architecture or seed grid.

## Evaluation

- Report 2025 as in-sample fusion calibration.
- Apply the frozen fusion unchanged to 2026.
- Use the same next-hour-open FLIP replay, costs, slippage and funding.
- Compare against the raw three-model argmax and fixed LONG/SHORT.

Fusion is useful only if 2026 PF and return improve over raw argmax without
raising turnover materially. Beating fixed SHORT remains the stronger
production criterion.
