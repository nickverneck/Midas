# GCZ6 Minute-Bar Feed Lag

## Status

Root cause confirmed on 2026-07-30: the authenticated user has CME top-of-book
market data but no COMEX market-data subscription. GCZ6 is therefore delivered
on the broker's approximately 10-minute delayed stream. No production code has
been changed.

## Reported Behavior

- Contract: `GCZ6`.
- Source: live Tradovate/NinjaTrader Simulation market data.
- Chart: one-minute OHLC bars.
- The TUI chart and native EMA fast/slow values appeared roughly 11 minutes or 11 bars behind the NinjaTrader web chart.
- A crossover visible in the NinjaTrader web chart appeared materially later in the TUI.
- The same behavior has not been observed on ES. MES is out of scope until GCZ6 is understood.
- Volume-bar impact is unknown.

## Impact

The engine can evaluate and submit a strategy from delayed market state while the UI still reports a healthy subscription. The current UI does not distinguish a delayed-but-active stream from real-time data. Until COMEX real-time data is enabled and the application has a freshness guard, GCZ6 live strategy signals and timing should not be treated as current.

## Current Data Path

1. `market_data_worker_inner` requests `md/getChart` with the exact contract ID and the selected `BarType`.
2. Historical chart packets are inserted into `LiveSeries.closed_bars` by timestamp.
3. A realtime packet with the same timestamp replaces `forming_bar`.
4. A greater realtime timestamp closes the former bar and creates the next forming bar.
5. The strategy and TUI consume the same `MarketSnapshot`; there is no separate strategy-only delay in this path.

## Confirmed Findings

- The worker does not manufacture time bars. A newer minute exists only after Tradovate emits a realtime bar with a greater timestamp. On a contract with no trades, an old forming-bar timestamp can therefore be legitimate.
- The application does not expose or persist the newest provider bar timestamp, its age relative to wall time, historical/realtime chart IDs, EOH state, or websocket receive backlog.
- `market_update_age` measures when the application last received a snapshot; it does not measure the age of the market timestamp inside that snapshot. Revisions to an old bar can make the feed look fresh.
- There is no stale-chart watchdog and no `md/cancelChart` plus `md/getChart` resubscription when a time bar falls behind wall time.
- The EMA implementation consumes the same bar vector shown by the TUI. Matching old web-chart EMA values is therefore evidence that the input series itself is stale or structurally different, rather than a delayed EMA renderer.
- A local disarmed GCZ6 one-minute subscription measured the following provider timestamps:
  - At `2026-07-30T04:15:20Z`, the newest bar opened at `04:04:00Z` (11 minutes 20 seconds by bar-open time).
  - At `2026-07-30T04:18:05Z`, the newest bar opened at `04:07:00Z` (11 minutes 5 seconds by bar-open time).
- The bar timeline advanced while retaining the offset. This rules out an 11-element application buffer and does not match a frozen websocket.
- A read-only `marketDataSubscription/deps` query returned three non-archived subscriptions, all using plan ID `1`. A read-only plan lookup identified plan ID `1` as `CME_TOP`. No COMEX subscription was present.
- ES is listed on CME and GC is listed on COMEX. The account-level exchange entitlement therefore explains why ES is current while GCZ6 is delayed.
- The apparent 11-bar offset is consistent with a 10-minute delayed feed when comparing the open timestamps of completed one-minute bars: the current forming minute adds roughly one more minute to the visual count.

## Root Cause

The delay is upstream of the TUI and strategy engine. The account is entitled to
real-time CME data (`CME_TOP`) but not real-time COMEX data. Tradovate/NinjaTrader
delivers GCZ6 as delayed market data. The application faithfully consumes the
delayed timestamps but incorrectly presents the stream as simply "Subscribed,"
without exposing that it is approximately 10 minutes behind wall clock.

This also explains why the delayed candles and EMA values have exactly the same
shape as the web chart at an earlier time: the feed is real market data shifted
in time, not locally malformed candles.

Sparse volume may still explain occasional missing or thin GCZ6 bars, but it does
not explain the stable 10-minute feed offset and is not the cause of this report.

## Local Reproduction

The first 2026-07-30 attempt stopped at `account/list` with 401 because its token
had expired. After the token was refreshed, the replay-enabled build connected
to Simulation, selected exact contract GCZ6 with one-minute OHLC bars, and kept
the strategy disarmed. Read-only IPC snapshots produced the timestamp samples
above. No orders were submitted.

## Candidate Fix Directions

1. Enable real-time COMEX top-of-book market data for the authenticated account, or use a token/account that already has that entitlement.
2. Add market-timestamp freshness telemetry and a prominent delayed/stale-data state per engine/instrument.
3. Gate strategy arming when a time-based feed exceeds its permitted timestamp age. Keep the gate instrument-local so delayed GC cannot disarm an unrelated ES engine.
4. Allow an explicit user override, but show the measured delay continuously and record the override in logs.
5. Add chart lifecycle telemetry for request ID, historical ID, realtime ID, EOH, last packet time, and last timestamp advance.
6. Distinguish an active delayed stream from a frozen stream. A resubscription will not repair missing exchange entitlement.
7. For confirmed subscription stalls, cancel and resubscribe with bounded backoff while keeping only that instrument disarmed.
8. If empty minutes are the difference, define an explicit empty-time-bar policy and ensure live, replay, indicators, and tests use the same rule.
9. Never infer feed health from websocket heartbeat or snapshot receive age alone.
