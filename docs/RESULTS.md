# Live bots and how the numbers were computed

Author's own calculation from the bots' trade databases. The databases themselves are not published, so these numbers are not independently verified.

## Method

- Round trips are built from consecutive open and close fills at the real fill prices.
- Fee 4 bps per side. Slippage is already inside real fill prices. Funding is not included.
- Position size is treated as 100% of the notional, so the numbers show the strategy path, not the account balance. Account balances are not usable because of deposits and withdrawals.
- A position that is still open at the end of the period is not counted.

## Results

| Bot | Environment | Period | Return | Max drawdown | Round trips |
|---|---|---|---|---|---|
| FLIP DOGE 1h | Binance, real money | 2026-03-02 to 2026-09-22 | +60.7% | -22.4% | 32 |
| Target DOGE 4h | Binance, real money | 2026-02-01 to 2026-09-22 | +16.3% | -22.2% | 222 |
| H6 DOGE 4h | Bybit, demo | 2026-01-08 to 2026-03-19 | +12.7% | -23.2% | 51 |

- On the common dates 2026-03-09 to 2026-09-22 the 1h FLIP bot returned +38.0% and the 4h bot +2.6%.
- Until 2026-09-12 the FLIP bot ran on corrupted candles because of a cache bug (fixed). The bug and its diagnosis are described in the main log.

## Current out-of-sample period

Since 2026-09-30 the FLIP bot runs a version with three input features disabled at inference time (no retraining). The features were chosen using 2025-2026 data, so earlier results of this version are selection results, not a clean test.

From 2026-09-30 to 2026-11-10 on real fills: +4.4%, max drawdown -6.6%, 21 closed trades, win rate 67%. This is six weeks and within noise; it is reported as an ongoing observation, not as proof.

## Known limits

- Weights were trained on data up to 2024-12-31. Thresholds, features and execution parameters were chosen later using 2025 and early 2026 data.
- The model is one checkpoint chosen from many seeds. Other high-validation seeds lost almost everything in 2026 (see the main log, Russian).
- Honest retraining on a causal pipeline did not reproduce the edge (walk-forward -3.95%, monthly adaptation -68.5%).
