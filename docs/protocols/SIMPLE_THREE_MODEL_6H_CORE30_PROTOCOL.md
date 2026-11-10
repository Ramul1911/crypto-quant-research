# Three-model 6h — core-30 feature follow-up

Frozen on 2026-08-23 before training.

The `±2%` protocol remains unchanged except that the input is reduced from
129 features to the following frozen 30-feature set:

- DOGE: returns 1/3/6h, range, close position, upper/lower wick, volume z24,
  ATR14, RSI14, MA ratios 24/96, EMA ratios 50/200, MACD histogram,
  Bollinger width, ADX;
- BTC: returns 1/3/6h, ATR14, RSI14;
- ETH: returns 1/3/6h, ATR14;
- cross-market: DOGE/BTC correlation 24h and DOGE relative strength to BTC;
- funding: current rate and MA8.

Fixed experiment settings:

- FLAT band `±2%`;
- horizon `6h`;
- sequence length `60`;
- seed `0`;
- no Optuna, threshold grid, temperature, or seed search;
- train through 2024, validation 2025, frozen diagnostic 2026.

This isolates one question: does removing redundant and unavailable features
improve transferable raw edge?
