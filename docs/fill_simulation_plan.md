# Fill Simulation Plan

## Context

Backtest and analyzer fills currently use the bar transition price with fixed
commission and fixed per-contract slippage. This is useful as a baseline, but it
does not test whether a strategy survives imperfect fills.

Phase 1 adds a cheap random adverse fill-cost stress mode. The notes below cover
the more realistic future phases.

## Phase 2: Intrabar Random Walk From OHLCV

Minute OHLCV can support a plausible intrabar simulator, but it cannot replay the
real market. OHLCV gives only the minute's open, high, low, close, and volume. It
does not preserve bid/ask spread, quote depth, trade ordering, queue position, or
whether the high happened before the low.

The best use of minute OHLCV is therefore stress testing:

- Build a synthetic path inside each bar, constrained to start at open, end at
  close, and stay within high/low.
- Randomize whether the path visits high before low or low before high.
- Add Brownian-bridge-like noise between anchor points, then clamp to the bar
  range and round to tick size.
- Apply configurable latency in substeps or bars before an order can fill.
- Fill market orders at the simulated path price plus adverse spread/slippage.
- Evaluate stop-loss and take-profit on the synthetic path instead of only on
  close-to-close prices.

Important implementation detail: execution price and mark price should be
separate. The environment should mark open positions against the bar's market
price, while realized trade costs use the simulated fill price/cost. Replacing
the bar close with a random fill price would distort mark-to-market PnL.

Phase 2 should expose settings such as:

- `substepsPerBar`
- `seed`
- `pathCount` or Monte Carlo simulations
- `tickSize`
- `tickValueUsd`
- `spreadTicks`
- `latencySubsteps` or `latencyBars`
- high/low ordering mode: random, optimistic, pessimistic
- conflict policy when stop and target are both touched in the same bar

For sweeps, use common random numbers: each combination should see the same
random scenarios for a given seed. This makes heatmaps compare strategy logic
instead of random luck.

## Phase 3: Granular Market Replay

The realistic mode should use market data that includes tradable prices, not only
bar summaries. The minimum useful upgrade is bid/ask quote data.

Useful data, from least to most realistic:

- Trade ticks: timestamp, price, size, exchange/session metadata.
- Top-of-book quotes: timestamp, bid, ask, bid size, ask size.
- Level 2/order book depth: price levels, sizes, update event type.
- Order flow events: add/cancel/modify/trade where available.
- Broker execution logs: submitted time, ack time, fill time, fill price,
  partial fills, rejected/cancelled orders.

Market-order simulation needs bid/ask, spread, available size, order size,
latency, tick size, tick value, fees, and session rules.

Limit-order simulation additionally needs queue position and depth depletion. A
simple first version can estimate fills when traded volume crosses the limit
price, but a stronger version should model queue priority and partial fills.

Stop and take-profit handling should move from bar-level checks to event-level
checks. If both stop and target are reachable inside one minute, granular replay
can determine which fired first; OHLCV cannot.

## Compute Impact

Random adverse cost is cheap because it samples only when position changes.

Intrabar random walks increase work roughly by:

`bars * substepsPerBar * simulations * combinations`

Tick/quote replay increases row count directly. Moving from one-minute bars to
one-second bars is about 60x more rows. Moving to trade ticks or order-book
events can be much larger on liquid futures.

Sweep defaults should therefore keep Phase 2 conservative. A practical starting
point is one path per combination for interactive sweeps, then an optional
robustness pass over the best candidates with more seeds.
