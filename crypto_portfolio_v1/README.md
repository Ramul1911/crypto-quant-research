# Hybrid Crypto Portfolio V2

Causal multi-expert Binance USD-M perpetual portfolio. V2 replaces the rejected generic
fixed-horizon prototype with independent trend, funding and shock experts, a GPU temporal
regime gate, 15m execution and a portfolio-level drawdown budget.

## Acceptance target

- Mean monthly return in active regimes: 10–30%.
- Median walk-forward month: at least 8%.
- Profitable walk-forward months: at least 65%.
- Maximum drawdown: at most 28%; a 25% fold breach is a terminal failure.
- No martingale, averaging down, final-period seed selection or retrospective satellites.

## Honest data boundary

- Default historical research universe (`HISTORICAL_CORE`) is fixed to:
  `BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT DOGEUSDT LINKUSDT ADAUSDT`.
- Post-hoc loser cuts after seeing OOS ledgers are diagnostic only and are not promoted.
- Satellites and late listings / memes (`SUI`, `HYPE`, `APT`, `PEPE`, `WIF`, …) are
  forward-only from archived daily Binance snapshots; never backfilled into historical claims.
- Historical mid-cap expansion is allowed only via pre-declared whitelists
  (`HISTORICAL_EXPANSION_CANDIDATES_WAVE1` / `_WAVE2`, CMC-filtered early Binance perps)
  plus a mechanical data eligibility audit (listing date, coverage, quality). Today's CMC
  late listings / memes / stables stay forward-only. Symbols may not be added or removed
  after OOS returns are observed.
- Historical 2025–2026 has already been inspected and is development walk-forward data,
  not a pristine holdout.
- The new untouched holdout begins with V2 paper collection after deployment.
- Free Binance OI and liquidation streams are collected forward only. They are never
  backfilled or silently substituted in historical experiments.

## Architecture

1. Completed 1h candles produce causal signals and a 168h temporal state.
2. The next 15m open executes target-weight changes.
3. Trend, funding-crowding and volatility-shock experts produce independent scores.
4. A BF16 TCN gate estimates trend-up, trend-down, chop and shock probabilities.
5. Regime probabilities combine expert scores; uncertainty and disagreement route to FLAT.
6. Volatility, correlation and drawdown budgets constrain portfolio weights.
7. A single stitched walk-forward engine evaluates all experts and the hybrid after costs.

## Commands

```powershell
python -m crypto_portfolio_v1.universe
python -m crypto_portfolio_v1.data --start 2021-01-01 --interval 1h
python -m crypto_portfolio_v1.data --start 2022-01-01 --interval 15m
python -m crypto_portfolio_v1.collectors.snapshot
python -m crypto_portfolio_v1.run_hybrid
python -m crypto_portfolio_v1.paper
```

The rejected V1 artifacts are listed under `_archive/prototype_20260809/` and must not be
imported by V2.
