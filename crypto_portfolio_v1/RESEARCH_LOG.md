# Hybrid Crypto Portfolio V2 — Research Log

## V2 reset — 2026-08-09

V1-BASE-001 and V1-MODEL-001 below are rejected audit evidence, not V2 components.
Their manifest is `_archive/prototype_20260809/MANIFEST.md`.

V2 historical tests use only the fixed eight-symbol core. Current satellites and all
2025–2026 outcomes have already been observed, so they cannot constitute an untouched
holdout. The new honest holdout starts at `2026-08-10 00:00 UTC` through the forward
paper collector. Historical OI/liquidation features are prohibited because free Binance
does not provide a reproducible multi-year archive.

## V2-UNIVERSE-001 — Locked universe policy and mid-cap whitelist

Frozen before any expanded-universe PnL inspection:

1. Default development baseline remains the eight-symbol `HISTORICAL_CORE`.
2. Post-hoc loser exclusions (V2.1 without DOGE/BNB; V1 without BTC/ETH) stay diagnostic
   only and are not promoted.
3. Late listings / memes remain satellite / forward-only and must never be backfilled into
   historical claims.
4. Historical expansion is allowed only through the pre-declared mid-cap whitelist
   `HISTORICAL_EXPANSION_CANDIDATES`:
   `LTCUSDT BCHUSDT ATOMUSDT DOTUSDT FILUSDT AVAXUSDT UNIUSDT AAVEUSDT`.
5. Candidates must pass a mechanical eligibility audit (still trading, local 1h+funding,
   listing start on/before 2021-01-01, coverage through development end, gap-rate gate).
   Performance metrics are not eligibility inputs. After the eligible set is locked, no
   further symbol surgery is allowed for that experiment.

## V2-UNIVERSE-002 — Expanded mid-cap walk-forward (single locked run)

Eligibility manifest: `reports/universe_expansion_eligibility.json`.

- All eight whitelist candidates passed mechanical gates; none rejected.
- Expanded universe (16): core + `LTC BCH ATOM DOT FIL AVAX UNI AAVE`.
- Report: `reports/trend_v21_expanded_midcaps/`.
- No further symbol adds/drops are permitted for this experiment after these results.

**Trend V2.1 expanded**

- Return +436.87%; MDD -13.80%; month win 73.1%; median month +6.08%; mean +7.14%.
- Trades 2,088; trade win 46.55%; PF 1.42.
- Folds: +130.1%, +55.1%, +38.5%, +9.6%, -0.9%.
- Decision: **NO-GO** (median/mean month still below 8%/10%).
- Strongest new contributors: UNI, AAVE, AVAX, ATOM; BCH slightly negative.

**Original Trend V1 expanded**

- Return +1897.67%; MDD -16.23%; month win 80.8%; median month +8.62%; mean +13.90%.
- Trades 6,767; trade win 45.90%; PF 1.17.
- Folds: +498.9%, +74.1%, +34.9%, +32.5%, +7.1%.
- Acceptance gates on this development stitch: **GO**.
- Dominant names: ATOM, XRP, UNI, BCH; ETH/AVAX/FIL/BTC remain net-negative.

**Interpretation**

Expanding the pre-declared liquid mid-cap set materially improves both strategies versus the
8-core baseline, and original trend clears the monthly acceptance band on development OOS.
This is still **not** a pristine holdout: 2025–2026 was already development data, and the
whitelist class was chosen with knowledge of alt-driven edge. Promotion requires freezing
this 16-symbol set and accumulating forward paper from `2026-08-10` without further
universe surgery. Late memes/satellites remain forward-only.

## V2-UNIVERSE-003 — Post-hoc PF&lt;1 cuts on expanded-16 (diagnostic)

Symbols with profit factor below 1 were removed after inspecting the expanded OOS
ledgers. This is development snooping, not a new holdout.

- V2.1 drop BCH: +436.9% → **+461.2%**, MDD −13.8% → −14.9%, median month +6.08% →
  **+5.46%**, still NO-GO. Report: `reports/trend_v21_expanded_no_pf_lt1/`.
- Original drop BTC/ETH/FIL/AVAX: +1897.7% → **+1827.4%**, MDD −16.2% → −16.4%,
  median +8.62% → +8.67%, mean +13.90% → +13.57%, still GO on development gates.
  LINK flips net-negative after reallocation. Report:
  `reports/trend_v1_expanded_no_pf_lt1/`.

Conclusion: hard-dropping PF&lt;1 names is not a free improvement. Keep the locked
16-symbol eligible set; do not promote these cuts.

## V2-UNIVERSE-004 — CMC wave-2 historical candidates (locked a priori)

CMC top was used only as a discovery list. Today's late listings / memes / stables
(`HYPE`, `SUI`, `APT`, `PEPE`, `TAO`, `ONDO`, USDT/USDC, …) remain forward-only.

Pre-declared wave-2 Binance USDT-M candidates (before this experiment's PnL):

`TRXUSDT XLMUSDT ETCUSDT ALGOUSDT VETUSDT DASHUSDT ZECUSDT THETAUSDT XTZUSDT
EOSUSDT IOTAUSDT NEOUSDT`

Combined candidate universe = core + wave-1 + wave-2 (28 names pre-audit). Eligibility
lock file for this run: `reports/universe_expansion_eligibility_wave2.json`. The prior
wave-1 16-set lock remains untouched.

### Wave-2 eligibility + single walk-forward result

- Eligible expansion: all wave-1 eight + 11/12 wave-2 names.
- Rejected: `EOSUSDT` (delisted / not active USDT-M perpetual; coverage ends 2025-05-21).
- Expanded universe size: **27** symbols.
- Report: `reports/trend_v21_expanded_cmc_wave2/`.

**Trend V2.1 (27)**

- Return +1319.3%; CAGR 251.8%; MDD −11.96%.
- Month win 80.8%; median month **+11.86%**; mean month **+11.22%**.
- Trades 2,795; trade win 46.7%; PF 1.53.
- Folds: +171.0%, +49.5%, +106.3%, +76.6%, −3.8%.
- Development acceptance: **GO**.
- Top contributors: ZEC, DASH, XLM, UNI, AAVE.

**Original Trend (27)**

- Return +58914%; CAGR ~1960%; MDD −11.44%.
- Month win 92.3%; median month +23.0%; mean month **+31.4%** (above the 30% upper band).
- Trades 7,835; trade win 46.8%; PF 1.45.
- Folds: +840.6%, +140.2%, +332.5%, +458.5%, +8.1%.
- Development acceptance: **NO-GO** (mean month exceeds `max_mean_monthly_return_pct=30`).
- Dominated by ZEC/XLM/DASH/ATOM; several majors still net-negative.

**Vs prior locked sets (same OOS window)**

| Universe | V2.1 return / median m / decision | Original return / median m / decision |
|---|---|---|
| 8-core | +178% / +2.6% / NO-GO | +361% / +2.6% / NO-GO |
| 16 wave-1 | +437% / +6.1% / NO-GO | +1898% / +8.6% / GO |
| 27 wave-1+2 | +1319% / +11.9% / GO | +58914% / +23.0% / NO-GO (mean too high) |

Interpretation: further honest breadth again lifts V2.1 into the monthly acceptance band on
development OOS, with lower MDD than the 16-set. Original becomes extremely concentrated in
a few winners (esp. ZEC) and fails the upper mean-month guardrail. Still **not** a pristine
holdout — freeze the 27-symbol eligible set and validate only via forward paper from
`2026-08-10`. No post-hoc symbol cuts after this ledger.

## V2-UNIVERSE-005 — Trend V2.1 gross leverage 1.5 vs 2.0 (27-set)

Same locked 27-symbol universe. Baseline wave-2 used `max_gross_exposure=1.5`.
Rerun with `max_gross_exposure=2.0` → `reports/trend_v21_expanded_cmc_wave2_lev2/`.

- 1.5x: +1319%, MDD −12.0%, median month +11.9%, mean +11.2%, month win 80.8%, GO.
  Avg gross exposure 28.3%. Profiles: BALANCED/BALANCED/SLOW/SLOW/SLOW.
- 2.0x: +428%, MDD −12.0%, median month +3.2%, mean +6.9%, month win 57.7%, NO-GO.
  Avg gross exposure **21.5%** (lower). 2024H2 validation flipped to FAST → fold return 0%.

**Correction:** this was not a clean leverage test. `max_gross_exposure` is also used to
seed per-side allocations before nonlinear symbol/cluster/volatility constraints. The
1.5 cap never bound (realized maximum gross 1.20); the 2.0 run also peaked near 1.21.
Changing the parameter altered portfolio composition and validation profile selection
rather than scaling final weights. Baseline V2.1 accounting remains valid, but a future
leverage study must freeze profiles/targets and apply a separate final-weight multiplier.

## V2-TOP50-001 — Point-in-time CMC top-50, Original Trend, one year

Universe source: CoinMarketCap historical snapshot on `2025-08-10`, fixed before PnL:
`https://coinmarketcap.com/historical/20250810/`.

- Test: `2025-08-10 00:00` → `2026-08-09 18:30` UTC (~0.999 years).
- CMC top 50 filtered to active Binance USDT-M perpetuals with full local 1h/15m/funding
  coverage at the test boundary: **37 eligible symbols**.
- Stablecoins and assets without a usable Binance perpetual were rejected mechanically.
- Strategy: frozen deterministic Original Trend; next-15m execution; 4 bps fee + 2 bps
  slippage per turnover plus funding; mandatory end liquidation.
- Report: `reports/original_top50_1y/`.

**Results**

- Return **+3274.55%**; CAGR ~3282%; MDD **−14.85%**.
- 12/13 profitable calendar observations (92.3%); median **+30.26%**;
  mean **+33.17%**. Excluding the two partial boundary months: 10/11 profitable,
  median **+35.57%**, mean **+37.55%**.
- Sharpe 7.88; Sortino 12.05; annualized volatility 46.0%.
- Average/max gross exposure 87.6% / 129.1%.
- 3,902 closed trades; win rate 47.26%; PF 1.65; payoff 1.84.
- Top contributors: XLM, HYPE, ATOM, 1000BONK, UNI, 1000SHIB, NEAR.
- Worst contributors: AVAX, DOGE, KAS, HBAR, LINK.

This is a point-in-time one-year development diagnostic, materially cleaner than
backfilling today's top 50, but it is not the forward holdout. No post-hoc symbol removal
is permitted for this result.

## V2-TOP50-002 — Causal timestamp correction

Audit found that Binance 1h rows are indexed by candle **open** time. V2-TOP50-001
used the close from the row stamped `t` and executed after `t+15m`, even though that
close is only known at `t+1h`. Its +3274.55% result therefore contains one-hour
look-ahead and is invalid for deployment.

Corrected run relabels every 1h row at candle-close availability (`open_time + 1h`)
before generating 4h decisions. Execution remains the first 15m open strictly after
the now-causal signal timestamp. Universe, costs, constraints, funding and mandatory
end liquidation are otherwise unchanged.

- Report: `reports/original_top50_1y_causal/`.
- Return **−22.05%**; MDD **−25.01%**; terminal drawdown governor triggered on
  2025-10-22 and the portfolio remained flat afterward.
- 0/13 profitable calendar observations; median **0.00%**, mean **−1.78%**.
- 822 closed trades; win rate **38.32%**; PF **0.70**; payoff **1.12**.
- Sharpe **−2.85**; Sortino **−3.77**.
- Total fees 6.11% of starting equity and slippage 3.05%; funding −0.39%.

Independent verification rebuilt every hourly signal candle from four completed 15m
rows, labeled at the right edge, without loading native 1h OHLC. Report:
`reports/original_top50_1y_causal_from_15m/`. Results match the shifted native-1h
run to floating-point precision: −22.0488% return, −25.0107% MDD, 822 trades and
PF 0.6959. This confirms that the correction uses real 15m-derived OHLC and that
the performance collapse is caused by removing future information, not by changing
market values.

**Decision: NO-GO.** The Original top-50 strategy must not be enabled with real
orders. The Demo bot may only remain in `DRY_RUN=1` while the research pipeline is
re-audited. The old dashboard and V2-TOP50-001 headline metrics are superseded.

## V2-CARRY-CSMOM-001 — Pre-registration

**Status at registration:** no experiment result inspected or generated.

**Primary evaluation:** `HISTORICAL_CORE`, existing five purged development folds.
**Secondary diagnostic:** frozen 37-symbol CMC manifest and the unchanged
2025-08-10 → 2026-08-09 window. No symbol may be removed after PnL.

All signal OHLC is rebuilt from complete 15m rows with right-edge availability
timestamps. Orders execute at the first 15m open strictly after a signal. Costs,
sparse 8h funding cashflows, default V2 portfolio limits, drawdown governor and
mandatory fold/window liquidation remain unchanged.

Frozen arms:

- **Funding carry:** after a known funding event, short the highest positive rates
  and long the lowest negative rates when magnitude clears the frozen
  `max(0.0001, 0.0008 / 3) = 0.0002667` hurdle inherited from the prior funding
  protocol. Re-evaluate every 8h. Select 2/2 on core and 5/5 on top-50.
- **Slow CSMOM:** equal-weight cross-sectional z-score of 7d, 14d and 30d log
  returns, active only when all three horizons exist. Re-evaluate daily at
  00:00 UTC. Select 2/2 on core and 5/5 on top-50.
- **Inverted carry:** exact sign inversion of funding carry, core-only sanity control.

No parameter grid, gate, profile selection, seed selection or post-result threshold
change is permitted. Promotion requires: no terminal kill, positive return after all
costs, PF ≥1.2 and at least 3/5 positive core folds. Existing strict protocol
acceptance metrics are reported separately. If both primary arms fail, both
hypotheses are closed without tuning.

### Results

Reports: `reports/v2_carry_csmom_001/`.

Primary core walk-forward:

- **Funding carry:** −11.76%, MDD −12.59%, PF 0.40, 1/5 positive folds.
  Only 6/26 months were active; active-month median −1.29%. **NO-GO.**
- **Slow CSMOM:** +4.90%, MDD −29.29%, PF 1.01, 3/5 positive folds,
  median month +0.86%. **NO-GO** (PF and drawdown fail).
- **Inverted-carry sanity control:** +8.87%, MDD −3.60%, PF 1.77,
  3/5 positive folds. This control passes the custom numeric promotion checks,
  but it was not a registered deployable arm and is not promoted. It shows that
  price movement associated with extreme funding dominated the direct carry
  payment in the opposite direction.

Frozen top-50 diagnostic with unchanged parameters:

- **Funding carry:** +9.45%, MDD −3.43%, PF 2.03; 7/13 active months,
  active-month median +0.61%. Despite the clean one-year diagnostic, it is
  rejected because the primary core walk-forward is negative and only 1/5 folds
  is positive.
- **Slow CSMOM:** +2.24%, MDD −24.53%, PF 1.01, median month −0.65%.
  **NO-GO.**

**Final decision:** both preregistered hypotheses are closed without tuning.
Funding carry is not robust across universes; slow CSMOM has no meaningful edge
after costs. Inverted carry may only be considered as a new, separately
preregistered hypothesis with an economic rationale and fresh validation.

## Protocol

- Venue: Binance USD-M perpetual futures.
- Data: futures candles and funding, never spot proxies.
- Signal timestamp: completed candle `t`; new position earns from `t` to `t+1`.
- Costs: 4 bps fee + 2 bps slippage per unit of turnover, plus funding.
- No final-period seed, threshold or model selection.
- Acceptance target: 10–30% monthly in active regimes, MDD below 30%.
- Hard portfolio governor: risk reduction at 10/15/20% DD, flat at 25%.

## Universe snapshot — 2026-08-09

Core:

`BTC ETH SOL XRP BNB DOGE LINK ADA`

Liquidity-selected satellites:

`HYPE SUI NEAR TRX`

TON is not included. Live Binance `exchangeInfo`, 24h quote volume and current open
interest were checked before download. The snapshot is suitable for forward research;
historical use still has survivorship bias and is not treated as final evidence.

## Data snapshot

- Interval: 1h
- Range: 2021-01-01 → 2026-08-09
- Full-history contracts: 10
- SUI starts: 2023-05-03
- HYPE starts: 2025-05-30
- Funding included for every available contract
- Total aligned panel rows: 49,119

## V1-BASE-001 — Causal cross-sectional momentum

**Signal**

- Cross-sectional z-score of 24h/72h/168h momentum.
- Relative score must agree with the asset's absolute 168h EMA trend.
- Top-2 long and bottom-2 short.
- BTC trend changes long/short gross allocation.
- Inverse-volatility sizing, 45% annualized volatility target.
- Rebalance every 12h, maximum weight 30% per symbol.

**Results**

| Period | Return | MDD | Kill-switch | Notes |
|---|---:|---:|---|---|
| 2021–2024 research | +45.9% | −25.0% | 2022-09-13 | Afterwards flat |
| 2025→latest | −24.8% | −25.1% | 2025-07-06 | Afterwards flat |
| 2025 | −24.8% | −25.1% | 2025-07-06 | Fails |
| 2026 YTD standalone | −0.7% | −23.1% | no | Median month +0.8% |

**Conclusion**

1. The hard risk governor works and prevents the strategy from drifting far beyond its
   design drawdown, but permanent flattening exposes that the underlying signal failed.
2. Simple momentum ranking is insufficient: it earns in strong trends and gives the gains
   back during regime transitions.
3. Turnover is already moderate; this is not primarily a fee problem.
4. This baseline is the mandatory benchmark for learned experts. A model must improve
   both walk-forward return and time-to-kill, not merely fit the strong 2021 trend.

**Next**

- Add causal regime probabilities rather than a binary BTC EMA state.
- Add funding/crowding and volatility-shock features.
- Train quantile return/downside models on rolling windows.
- Compare every learned expert against this unchanged baseline.

## V1-MODEL-001 — Shared LightGBM quantile expert

**Protocol**

- One shared multi-asset model with symbol as a categorical feature.
- Quantiles q10/q50/q90 of causal 24h forward return.
- 22 features: returns, volatility, EMA trend, volume, funding, BTC state,
  market breadth and time.
- Purged train/validation boundaries: a label's complete 24h horizon must remain
  inside its block.
- Five strictly forward half-year folds.
- Predicted median edge must exceed 12 bps estimated round-trip cost.
- Same portfolio layer and risk governor as the unchanged momentum baseline.

| OOS fold | Quantile model | MDD | Baseline | Baseline MDD |
|---|---:|---:|---:|---:|
| 2024H2 | −20.7% | −23.7% | +67.4% | −18.5% |
| 2025H1 | −19.5% | −24.6% | −22.9% | −23.3% |
| 2025H2 | −14.4% | −24.3% | −15.7% | −25.0% |
| 2026H1 | 0.0% | 0.0% | +1.0% | −23.1% |
| 2026H2 YTD | +1.8% | −6.2% | −3.6% | −10.4% |

**Stitched OOS**

- Quantile model: **−44.3%**, profitable 1/5 folds.
- Fixed momentum baseline: **+5.9%**, profitable 2/5 folds.

**Conclusion**

The generic 24h return-distribution expert is rejected. Validation stopped the median
model after very few trees in weak folds (one tree for 2026H1), which is direct evidence
that the current feature set contains little stable conditional 24h-return information.
The quantile width improves abstention and MDD, but abstention is not return.

Do not tune this model on the OOS folds. The next expert must represent a structurally
different event: regime transition or liquidation/volatility shock, rather than another
generic fixed-horizon return regressor.

## V2-WF-001 — Temporal-gated multi-expert portfolio

**Protocol**

- Fixed historical core: BTC/ETH/SOL/XRP/BNB/DOGE/LINK/ADA.
- Five expanding walk-forward folds, prior six-month validation and 72h purge.
- Three fixed seeds (42/59/76) averaged as consensus; no test-seed selection.
- Causal 168h shared TCN, BF16 on RTX 5080, multi-task regime/quantile/adverse/rank heads.
- Hourly completed-candle signals, decisions every 4h, next-15m-open execution.
- 4 bps fee + 2 bps slippage per turnover, sparse funding cash-flow events.
- Same vol target, weight caps and drawdown governor for every strategy.

**Stitched OOS results after costs**

- Frozen momentum: +81.1%, MDD -19.5%, mean month +2.56%, median +1.12%.
- Trend expert: +361.5%, MDD -22.4%, mean month +6.84%, median +2.58%.
- Funding expert: -11.3%, MDD -17.6%.
- Shock reversal expert: -68.1% stitched, with terminal failures in early folds.
- Hybrid: +15.1%, MDD -3.9%, mean month +0.58%, median 0.0%, profitable 5/26 months.

Hybrid folds: +15.0%, +0.1%, 0.0%, 0.0%, 0.0%. The gate became too uncertain after
2025H1 and correctly abstained, but abstention eliminated return. It did not beat frozen
momentum or the trend expert and therefore fails the V2 acceptance rule.

**Decision: NO-GO**

Passed MDD and no-kill checks. Failed profitable-month share, median monthly return,
minimum mean monthly return and mandatory baseline/expert outperformance. The historical
result must not be tuned further on these OOS folds.

**Forward state**

- Forward OI collection started 2026-08-09T16:47:01Z with 12 immutable symbol snapshots.
- Paper ledger initialized at 2026-08-09T16:51:27Z.
- The honest paper window is scheduled to open 2026-08-10T00:00:00Z; no pre-start
  observation was fabricated.

## V2-TREND-002 — Trend Expert V2.1

V2.1 adds causal efficiency/volatility-regime filtering, asymmetric short thresholds,
12h hysteresis, acceleration-based profit protection and a 168h rolling correlation
budget. FAST/BALANCED/SLOW horizon weights are selected strictly on the preceding
validation block with a predeclared lexicographic objective; test results are not inputs.

Selected profiles by OOS fold:

- 2024H2: BALANCED
- 2025H1: SLOW
- 2025H2: BALANCED
- 2026H1: SLOW
- 2026H2 YTD: FAST (validation selected the flat profile over negative-median candidates)

**Stitched development OOS**

- V2.1: +178.0%, MDD -13.2%, profitable months 19/26 (73.1%), median month +2.64%,
  mean month +4.28%, no fold kill.
- Frozen original trend: +361.5%, MDD -22.4%, profitable months 16/26 (61.5%),
  median month +2.58%, mean month +6.84%, no fold kill.
- V2.1 folds: +72.6%, +35.7%, +16.5%, +1.9%, 0.0%.
- Original folds: +195.4%, +36.8%, +11.2%, -4.3%, +7.3%.

**Decision: NO-GO for the original 10–30% monthly target**

V2.1 materially reduced drawdown and increased profitable-month frequency above the 65%
gate. It also converted 2026H1 from -4.3% to +1.9%. It did not materially improve the
median month and reduced mean/cumulative return, so it still fails the 8% median and 10%
mean monthly requirements. Because these development OOS results are now observed, no
further threshold adjustment may be called a new historical holdout. V2.1 is eligible
only as a separately frozen forward-paper candidate alongside the original trend.

## V2-TREND-003 — Exact trade ledger and complete diagnostics

All OOS folds were rerun with mandatory end-of-fold liquidation. Per-symbol market PnL,
funding, fees and slippage reconcile to portfolio equity. Fold-local monetary ledgers are
scaled by sequential stitched capital; combined net trade PnL differs from stitched equity
change by only `-5.86e-14`.

**Corrected V2.1 portfolio**

- Return +177.86%; CAGR 62.36%; MDD -13.16%.
- Annualized volatility 19.13%; Sharpe 2.63; Sortino 3.81; Calmar 4.74.
- Calendar-month win rate 73.08%; active-month win rate 79.17%.
- Median month +2.64%; mean month +4.28%.
- Maximum drawdown remained unrecovered at the end of development data.
- Average gross exposure 14.25%; total stitched fees/slippage 0.425/0.212 equity units.

**True closed-position statistics**

- 1,221 trades: 557 wins and 664 losses; trade win rate 45.62%.
- Profit factor 1.44; average win/loss payoff 1.72.
- Average/median duration 11.84h/8h.
- Average trade return on maximum notional +0.29%; median -0.19%.
- Average MFE +2.13%; average MAE -1.22%.
- Long/short win rates 44.44%/47.25%; both sides have positive net PnL.
- Maximum winning/losing streaks 9/11.
- Top ten trades contribute 51.88% of net PnL.

Edge decays monotonically by fold: profit factor 2.02, 1.71, 1.29 and 1.04 from
2024H2 through 2026H1. 2026H2 YTD had no trades because validation selected FAST/FLAT.
XRP is the dominant contributor (PF 3.00, net PnL 0.725); DOGE and BNB are approximately
break-even after costs. This is the correction baseline; further changes remain
development hypotheses until forward paper evidence accrues.

## V2-TREND-004 — Remove losing symbols (post-hoc diagnostic)

Excluded symbols were chosen from the already-observed OOS trade ledgers, so this is
**development snooping**, not a new holdout. True holdout remains forward paper from
`2026-08-10`.

Exclusions (version-specific):

- Trend V2.1: drop DOGEUSDT + BNBUSDT (weakest net contributors in the V2.1 ledger).
- Original Trend V1: drop BTCUSDT + ETHUSDT (only negative net contributors in the V1 ledger).

Reports:

- `reports/trend_v21_no_doge_bnb/`
- `reports/trend_v1_no_btc_eth/`

**V2.1 without DOGE/BNB — worse**

- Stitched return +177.86% → **+62.41%**; MDD -13.16% → -12.33%.
- Monthly win rate 73.1% → **50.0%**; median month +2.64% → +0.05%.
- Trades 1,221 → 759; trade win rate 45.62% → 44.93%; PF 1.44 → 1.37.
- Root cause: nested validation switched **2024H2 BALANCED → FAST**, wiping the
  +72.6% fold to **0.0%**. Remaining folds: +29.8%, +22.2%, +4.6%, -2.1%.
- Conclusion: near-zero DOGE/BNB were still useful for validation ranking /
  diversification; hard-dropping them is not a free improvement.

**Original V1 without BTC/ETH — modest gain, still NO-GO**

- Stitched return +360.77% → **+377.55%**; MDD -22.41% → -21.99%.
- Monthly win rate 61.5% → **69.2%**; median month +2.56% → +3.24%; mean +6.83% → +6.98%.
- Trades 5,044 → 4,362; trade win rate ~44.3% → 44.2%; PF 1.15 → 1.17.
- Fold returns: +195.3%, +22.3%, +32.1%, -3.7%, +3.9% (2025H1 weaker, 2025H2 stronger).
- After reallocation SOL/LINK flip net-negative; XRP/DOGE/ADA remain the engine.
- Still fails the 8% median / 10% mean monthly acceptance gates.

**Decision:** keep the full eight-symbol historical core as the frozen development
baseline. Do not promote either exclusion universe. Any future universe cut must be
locked before looking at OOS trade ledgers, or validated only on forward paper.

