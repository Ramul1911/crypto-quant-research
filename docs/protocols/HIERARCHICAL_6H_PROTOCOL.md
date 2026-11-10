# Hierarchical 6h MOVE + direction protocol

Frozen on 2026-08-23 before training.

## Fixed data and model setup

- Core-30 feature set.
- Sequence `60`, horizon `6h`, seed `0`.
- Entry reference `open[t+1]`, target reference `open[t+7]`.
- MOVE means absolute six-hour return above `2%`.
- Same Transformer shape as the core-30 pilot.
- Train through 2024, validation 2025, frozen diagnostic 2026.
- No Optuna, class weights, calibration, temperature or threshold grid.

## Models

1. MOVE gate:
   - train on every eligible sample;
   - target `1` when `abs(return_6h) > 2%`, otherwise `0`.
2. Direction:
   - train only on MOVE samples;
   - target `1=UP`, `0=DOWN`.

Both models use unweighted cross-entropy so their `0.5` decision thresholds
retain their natural class-prior meaning.

## Controller

- If flat: enter immediately when `p_move >= 0.5`, using direction threshold
  `p_up >= 0.5`.
- If positioned and the same side remains selected: HOLD.
- Close only after two consecutive FLAT decisions.
- FLIP only after two consecutive opposite-direction decisions.
- Reset pending confirmation whenever the current side is selected again.
- Execute every state change at next-hour DOGEUSDC open.

## Decision

The hierarchy supports further optimization only if PF is above `1.0`
separately in 2025 and 2026, net return is positive in both, and cost drag is
materially lower than the always-in-market no-FLAT controller.
