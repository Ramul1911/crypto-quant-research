# DOGEUSDC whale-tape pilot

Frozen on 2026-08-23 before downloading or inspecting trade outcomes.

## Question

Do unusually large aggressive trades contain a causal, transferable signal
about continuation or absorption without order-book data?

## Data

- Binance USDT-M `DOGEUSDC aggTrades`, December 2024 through July 2026.
- December 2024 is warm-up only.
- Public fields: timestamp, price, quantity and `is_buyer_maker`.
- `is_buyer_maker=false` is buyer-initiated; `true` is seller-initiated.
- No claim is made that separate large trades belong to the same participant.

## Causal whale definition

- Calculate each calendar day's 99th percentile of individual trade notional.
- The threshold used on day D is the median daily 99th percentile from the
  preceding 30 complete days, shifted by one day.
- A trade above that threshold is called a whale trade.
- Aggregate all trades to one-minute bars with total and whale signed notional.

## Event definition

- Five-minute whale flow is the rolling sum of signed whale notional.
- A shock requires its absolute value to exceed the rolling 99th percentile of
  the preceding seven complete days of five-minute observations.
- Whale signed flow must also be at least 10% of total five-minute notional.
- Events are separated by a 60-minute refractory period.
- `continuation`: signed five-minute price impact is at least +0.10%; trade in
  the direction of whale flow.
- `absorption`: signed five-minute price impact is at most 0%; trade against
  whale flow.
- Observations between those definitions are reported but not traded.

## Execution and evaluation

- Signal exists only after the event minute closes.
- Entry is the next one-minute open.
- Fixed exits: 5, 15 and 60 minutes.
- Net outcome includes the existing 93.1% allocation, 4 bps taker fee and
  2 bps slippage per side. Funding is immaterial inside 60 minutes and omitted.
- Report 2025 and 2026 separately; 2025 is discovery and 2026 is the untouched
  transfer diagnostic.
- Metrics: events, mean/median net return, win rate, profit factor and bootstrap
  95% CI.
- Evidence requires PF above 1 and positive mean net return for the same event
  type and horizon in both years, at least 30 events per year, and a 2026
  confidence interval that does not cross zero.
- No parameter tuning after outcomes are viewed.
