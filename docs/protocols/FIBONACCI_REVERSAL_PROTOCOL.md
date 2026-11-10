# Causal Fibonacci reversal event study

Frozen on 2026-08-23 before running the study.

## Market and causality

- DOGEUSDC hourly futures candles.
- A pivot at hour `p` is confirmed only at `p+4`, after four right-hand bars.
- Levels become tradable starting at the hour after confirmation.
- Consecutive same-type pivots are consolidated causally; only confirmed
  opposite pivot pairs define an impulse.
- Minimum impulse size: `2 × ATR(14)` measured when the second pivot is
  confirmed.
- An impulse expires after 72 hours or when a new confirmed impulse replaces
  it.

## Levels

- Fibonacci: `0.382, 0.500, 0.618`.
- Linear controls: `0.333, 0.500, 0.667`.
- Random controls: three deterministic levels sampled uniformly from
  `[0.25, 0.75]` for every impulse with RNG seed `0`.
- Touch zone half-width: `0.15 × current ATR(14)`.
- Only the first reaction per level and impulse is counted.

## Reaction

- Bullish impulse: candle touches the zone, closes back at/above the level and
  closes above its open; event direction LONG.
- Bearish impulse: candle touches the zone, closes back at/below the level and
  closes below its open; event direction SHORT.
- Entry is the next-hour open.
- Outcomes are net returns after allocation, fees, 2 bps slippage and funding
  at fixed 6h, 12h and 24h exits.

## Evaluation

- Report 2024, 2025 and 2026 separately.
- Metrics: event count, mean/median net return, win rate, PF and bootstrap
  95% confidence interval for mean return.
- Fibonacci is supported only if its basket PF is above 1 and mean return
  exceeds both matched control baskets in 2025 and 2026 with at least 30
  events per period.
- No tuning after viewing results.
