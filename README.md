# crypto-quant-research

Research code and experiment logs of an independent researcher working on crypto perpetual futures (Binance USD-M, Bybit). The focus is on causal backtests with honest costs, pre-registered tests, and documented negative results.

**Author:** Ramil Yakhin, Kazan. Telegram: @ramul1911.

## What this is, and what it is not

- A self-taught, non-professional project. First experiments in 2022, systematic work since 2025. No prior work experience in a fund or trading firm.
- The code is written together with AI assistants (Copilot, Cursor, other models). I define tasks and validation protocols and check the results with tests, logs and real trade records. I do not claim to be a software developer.
- The point of this repository is the research process: protocols written before results, rejected hypotheses, and audits of my own mistakes. It does not claim a proven trading edge.

## Contents

| Path | What it is |
|---|---|
| `crypto_portfolio_v1/` | Portfolio research framework on Binance USD-M perpetuals: causal walk-forward engine, purged validation, experts (trend, funding carry, cross-sectional momentum, shock), universe eligibility checks, paper-trading ledger, 86 unit tests. |
| `crypto_portfolio_v1/RESEARCH_LOG.md` | Research log of that framework (English). Every experiment ends with a decision, mostly NO-GO. |
| `crypto_portfolio_v1/reports/` | Only small `summary.json` files of the key experiments. Full reports and market data are not included. |
| `research_log/RESEARCH_LOG.ru.md` | Main log of ~130 experiments on DOGE/BTC/ETH models (Russian). |
| `docs/protocols/` | Pre-registration documents written before running the experiments. |
| `docs/RESULTS.md` | Results of live bots and how they were computed. |

## Selected results

All numbers include 4 bps fee, 2 bps slippage per side and funding unless stated otherwise.

| Experiment | Result | Decision |
|---|---|---|
| Top-50 trend portfolio, first version | +3274% in 1 year | Invalid: one-hour look-ahead (hourly candles were indexed by open time) |
| Same, causal timestamps | -22.05%, max drawdown -25% | NO-GO. Independently rebuilt from 15m candles: identical result |
| Cross-sectional momentum, 8 coins | +45.9% (2021-2024), -24.8% (2025) | Kept as baseline for learned models |
| LightGBM quantile expert | -44.3% stitched OOS vs +5.9% for the baseline | Rejected |
| Funding carry (pre-registered) | -11.8%, 1 of 5 folds positive | Closed without tuning |
| Slow cross-sectional momentum (pre-registered) | +4.9%, profit factor 1.01 | Closed without tuning |
| Temporal-gate multi-expert | +15.1% but median month 0%, abstains most of the time | NO-GO |

More in `crypto_portfolio_v1/RESEARCH_LOG.md`.

## Reproducibility

- Python 3.10, dependencies in `requirements.txt`.
- Run the tests: `PYTHONPATH=. python -m pytest crypto_portfolio_v1/tests -q` (86 tests pass on the author's machine; one test needs CUDA and is skipped without it).
- Market data is not included. It can be downloaded from public Binance endpoints with `python -m crypto_portfolio_v1.data` (see the package README).

## Not included on purpose

- Model weights and the live trading bot code (private).
- API keys, server details and exchange account data.
- Raw market data and large experiment reports.

Parts of the Russian log refer to private files (`bot_binance_flip`, `causal_flip`) that are not in this repository, so those experiments cannot be rerun from here.

## Limits

- 2025 and 2026 were already used for development, so they are not a clean holdout. The log says so explicitly.
- Live results are short and come from a small account. See `docs/RESULTS.md`.
- English in this README was prepared with a translator; the Russian documents are the primary ones.
