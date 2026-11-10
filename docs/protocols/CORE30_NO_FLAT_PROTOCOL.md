# Core-30 no-FLAT controller

Frozen on 2026-08-23 before replay.

- Use the existing frozen core-30 UP and DOWN probabilities.
- Ignore the FLAT model completely.
- `LONG` when `p_up >= p_down`; otherwise `SHORT`.
- Hold the same side and FLIP at the next-hour open when the winning
  direction changes.
- No retraining, calibration, thresholds, temperature, hysteresis or grid.
- Replay 2025 and 2026 with the same allocation, fees, slippage and funding.

This isolates whether the transferable UP/DOWN ranking edge is useful without
the dominant and poorly comparable FLAT score.
