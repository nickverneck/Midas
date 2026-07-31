# Task Group 04: Replay Engine and Fill Model

## Goal

Move replay from "bar stream plus immediate fills" toward deterministic virtual-time simulation.

Fast-forward must change wall-clock speed only. Accuracy should depend on market timestamps and the fill model, not on how fast the UI renders.

## Implementation Status (2026-07-31)

The first Phase 4 slice is implemented behind an explicit compatibility boundary:

- `ReplayEngineMode::Legacy` remains the default, preserving the existing replay path and fill behavior.
- `ReplayEngineMode::Deterministic` routes replay bar-close delivery through a virtual-time priority queue ordered by market timestamp, semantic phase, and stable insertion sequence.
- Same-timestamp phases are defined for raw market events, bar close, strategy evaluation, order submission, exchange arrival, ack, fill, and protection update.
- The virtual clock rejects events scheduled in its past or behind the phase already processed at the current timestamp.
- Replay speed remains a wall-clock pacer outside the queue. Tests verify that every supported speed produces the same virtual event trace.
- The replay controls, runtime status, and persisted session log identify the selected engine mode.
- Configuration supports `replay_engine_mode = "legacy"` or `"deterministic"`; `TRADER_REPLAY_ENGINE_MODE` provides the equivalent environment override.

The second Phase 4 slice adds a deterministic bar-only execution lifecycle:

- Service acknowledgements now serialize each raw replay bar, its market update, strategy evaluation, generated broker commands, and simulated broker events before replay advances.
- Simulated market entries, liquidations, broker-owned protection entries, and liquidation-then-entry transitions are deferred instead of filling immediately in deterministic mode.
- `signal_market_ts + fixed_latency` determines exchange eligibility. The first later raw source bar at or after that timestamp supplies the executable open and fill timestamp.
- Broker-owned TP/SL levels are rebased around the raw entry open, preserving their configured offsets rather than retaining a synthetic/reference-chart entry price.
- Simulated order acknowledgements record the configured fixed latency.
- Configuration supports `replay_fixed_latency_ms`; `TRADER_REPLAY_FIXED_LATENCY_MS` provides the environment override.
- Replay controls and persisted session logs identify the effective fill model and latency. Legacy mode explicitly reports that fixed latency is ignored.
- Legacy mode still fills immediately at its reference price and remains the default compatibility path.

The third Phase 4 slice adds a versioned, append-only execution ledger before any fee or account overlay:

- Replay fills now carry stable lifecycle sequence numbers plus signal, submission, exchange-arrival, acknowledgement, and fill market timestamps.
- Each fill records its execution source independently from the strategy chart source: legacy reference price, raw bar open, or raw bar OHLC.
- Broker-protection exits record the protection order id and normalized exit reason (`take_profit` or `stop_loss`).
- The service deduplicates fills by broker fill id and never mutates an accepted ledger entry.
- Ledger entries include account/contract identity, side, quantity, price, tick size, value per point, and fee-neutral gross realized PnL delta.
- Gross position accounting is isolated by account and contract, so fills from another replay instrument cannot affect the selected ledger path.
- Full versioned snapshots are available through replay-state events for headless consumers and TUI reattachment. Incremental UI updates send compact summaries instead of repeatedly cloning the full ledger.
- The dashboard and persisted session log report ledger schema, fill count, signal source, latency model, and gross realized PnL.

The fourth Phase 4 slice makes the deterministic broker lifecycle queue authoritative for execution ordering:

- Virtual-time ordering now includes a monotonic logical step between market timestamp and semantic phase. This separates bar-open execution from close-time strategy work without inventing fake nanoseconds.
- The coarse executable market event is explicitly named `RawBarOpen`; it is not mislabeled as a tick and does not claim tick-level fidelity.
- Duplicate provider timestamps remain valid. A strategy can submit after one cycle at timestamp `T`, and its order can execute on a later raw source bar that also reports `T`.
- Simulated order submission and exchange arrival are independent virtual events. A deferred command becomes executable only after its arrival event is popped from the queue.
- Acknowledgements and fill-bearing entity batches are queued independently and dispatched by their semantic phases instead of being emitted directly from the gateway loop.
- Broker-protection fills receive their own lifecycle sequence and pass through fill and protection-update phases before the replay-bar barrier is released.
- Bounded queue draining leaves future arrivals pending, so fixed latency is market-time behavior and is unaffected by replay wall-clock speed.

The fifth Phase 4 slice makes coarse-bar protection assumptions explicit and adds trailing parity:

- `replay_bar_protection_policy` selects `conservative`, `optimistic`, or `nearest_open` when raw OHLC reaches both TP and SL in one bar. Conservative stop-first is the deterministic default; Legacy keeps the previous nearest-open behavior.
- Protection fills record the selected policy and whether the source bar was ambiguous.
- Broker auto-trail trigger, stop offset, and frequency now pass into replay-owned protection state.
- A completed raw bar may activate or tighten a trailing stop; that new level becomes executable on the next raw bar. This deliberately avoids current-bar high/low lookahead.
- Trailing exits are labeled `trailing_stop`; fixed exits remain `stop_loss` or `take_profit`.
- Direct transitions, staged flatten/confirm/enter, and liquidation-then-entry reversal commands retain the same deterministic command lifecycle and barriers.

The sixth Phase 4 slice adds reproducible observed-latency models:

- `replay_latency_model` supports `fixed`, `observed_mean`, `observed_p95`, `observed_p99`, and `seeded_observed`.
- Observed modes consume the configured `replay_observed_latency_ms` sample population. Deterministic startup fails closed when a required population is empty.
- `seeded_observed` performs deterministic empirical sampling with `replay_latency_seed`; identical samples, seed, and order sequence produce the same latency trace.
- Each deferred command owns its sampled latency, and each fill records that actual sample rather than a session-wide placeholder.
- Ledger schema v3 records the latency model, fill model, optional seed, observed sample count, bar-protection policy, and execution precision. Older v2 snapshots deserialize with safe bar-approximate defaults for the new fields.

The seventh Phase 4 slice consolidates deterministic ordering under one broker coordinator queue:

- The replay worker now owns only dataset enumeration and wall-clock pacing; it no longer owns a second deterministic priority queue.
- The broker coordinator owns the sole `ReplayVirtualEventQueue` instance for raw-bar-open, bar-close, strategy-evaluation, submission, arrival, acknowledgement, fill, and protection phases.
- Worker barriers carry stable bar/evaluation ids. Those ids derive monotonic open/evaluation logical steps and preserve duplicate provider timestamps.
- Headless and TUI replay continue through the same worker/service/gateway path; rendering speed remains outside virtual-time ordering.

The eighth Phase 4 slice implements the Level 1 tick/trade/quote fill path:

- Raw replay frames carry their source ticks through the existing broker barrier without changing the strategy evaluation cadence.
- `ReplayFillModel::TickBidAsk` fills market buys at the ask and market sells at the bid when quotes are present; trade-only data uses the tick trade as an explicit fallback.
- Stops and limits are evaluated in source-tick order. A stop reached before a target wins by sequence, so the coarse-bar ambiguity policy is not applied when ticks are available.
- Tick-driven trailing activation and tightening become effective on the following tick, preventing same-tick lookahead after a strategy entry or trail update.
- Every fill records `replayFillSource` and `replayExecutionPrecision`: `quote_exact`, `tick_exact`, or `bar_approximate`.
- Cached raw-tick bid/ask data is available to the execution path. Tick mode never silently falls back to an empty bar open; it waits for the next usable tick.
- `replay_fill_model = "tick_bid_ask"` and `TRADER_REPLAY_FILL_MODEL` select Level 1. Bar-only and Legacy modes remain explicit compatibility choices.

The ninth Phase 4 slice adds optional Level 2 DOM-assisted execution:

- `ReplayMarketDom` represents timestamped full-book snapshots with visible bid and ask levels. DOM is an enrichment path, not a requirement: the default bar model and the Level 1 tick/quote model continue to work without a DOM file.
- `replay_fill_model = "dom"` (deterministic mode only) consumes visible ask levels for buys and visible bid levels for sells at the exchange-arrival timestamp. Multiple levels produce a deterministic volume-weighted average price.
- The queue assumption is explicit and conservative: `visible_levels_only`. If the requested quantity exceeds visible depth, the simulated order is rejected instead of inventing hidden liquidity or silently falling back to a bar/tick price.
- Full-book snapshots can be supplied through the optional `replay_dom_file_path` / `TRADER_REPLAY_DOM_FILE_PATH` JSONL sidecar. Each line contains `ts_ns`, `bids`, and `asks`; snapshots are normalized and grouped with replay bars before entering the broker coordinator.
- DOM updates are scheduled as `DomUpdate` virtual-time events. DOM top-of-book protection checks use executable bid/ask prices and are labeled `dom_top_of_book`; market fills are labeled `dom_visible_levels` with `dom_assisted` precision and depth-consumption metadata.
- This slice does not claim exact exchange queue position. A competition account may expose live DOM, but historical replay depth still depends on the provider's Market Replay entitlement and the snapshots returned for the requested contract/date.

The tenth Phase 4 slice adds explicit DOM capture paths:

- `capture-replay-dom` uses Tradovate's dedicated Market Replay WebSocket, checks replay entitlement, initializes the historical clock, subscribes with `md/subscribeDOM`, and writes normalized full-book snapshots to the documented JSONL sidecar.
- `capture-live-dom` is a separate opt-in command/process using the normal market-data WebSocket. The live trading engine does not start this subscription, task, or writer unless the user explicitly launches the capture command, so normal live execution has no added DOM overhead.
- Both capture paths require an exact contract symbol, preserve provider timestamps, map Tradovate `offers` to replay `asks`, reject unsafe output replacement unless `--overwrite` is supplied, and fail clearly when the provider returns no usable depth.
- Historical capture uses the disposable Market Replay session only for market data; it does not start account sync or send orders.

Capture examples:

```bash
cargo run --features replay -- capture-replay-dom \
  --contract GCZ6 \
  --start 2026-07-30T13:30:00Z \
  --end 2026-07-30T14:00:00Z \
  --output .run/replay-cache/GCZ6.dom.jsonl

# Separate, opt-in live recorder; the normal engine is not modified or slowed.
cargo run -- capture-live-dom \
  --contract GCZ6 \
  --duration-seconds 300 \
  --output .run/live-captures/GCZ6.dom.jsonl
```

RBT-030, RBT-031, and RBT-032 Levels 0/0b/1/2 are now implemented for deterministic replay. RBT-033 is implemented for the documented coarse-bar policy, tick sequence path, and optional DOM top-of-book path. Historical and standalone live DOM capture are now available through explicit CLI commands:

- In the coarse bar model, acknowledgement and fill occur together when the first eligible raw bar is processed.
- TP/SL checks still use raw bar OHLC. The configured ambiguity policy makes results reproducible but does not reconstruct the real intrabar path.
- Bar trailing updates intentionally become effective on the next source bar; tick trailing updates become effective on the next source tick.
- Tick/quote arrival within a bar is available when the selected dataset contains raw ticks. Cached server-bar datasets remain bar-approximate.
- DOM remains optional and is never inferred from Level 1 bid/ask metadata. Without a sidecar, selecting `dom` fails closed with a clear dataset error; selecting bar or tick/quote models remains unaffected. The sidecar can be produced by `capture-replay-dom` or `capture-live-dom`.
- Observed latency samples are supplied through configuration; automatically importing a population from saved live logs can be added as a convenience without changing the sampler.
- The ledger currently lives for the engine session. Durable `result.json`/CSV/Parquet output belongs to Phase 5.

Validation for this slice:

- Default suite: 279 tests passed.
- Replay-feature suite: 463 tests passed.
- All-feature suite: 466 tests passed.
- Level 1 regressions cover quote-side market pricing, trade-only fallback, empty-bar waiting, tick-order TP/SL selection, and next-tick trailing activation. Level 2 regressions cover multi-level VWAP consumption, insufficient-depth rejection, DOM top-of-book protection, JSONL normalization, and optional-config compatibility.
- The existing safe local TUI replay remains covered for cached MESU6, 1-minute OHLC, deterministic mode, strategy disarmed, manual orders unavailable, zero persisted errors, and clean engine shutdown.
- All-feature build, formatting check, and diff whitespace check passed.

## Current Code

- `ReplayLifecycleDispatchQueue` owns the sole deterministic `ReplayVirtualEventQueue` inside the broker gateway; submission/arrival control eligibility and ack/fill/protection phases control event dispatch.
- `ReplayBarSchedule` is now an ordered source cursor, not a second virtual-time queue.
- Logical steps preserve causality when open-time execution and close-time evaluation share a provider timestamp, including duplicate source bars.
- `wait_for_replay_bar` remains a separate wall-clock pacer in `src/tradovate/replay/worker.rs`.
- Deterministic replay waits for the service to finish strategy evaluation and waits for the broker gateway to register generated commands before advancing.
- Deterministic simulated entries become eligible from market time plus their sampled latency and fill from the next eligible raw source bar open.
- Legacy market entries still fill immediately at the current reference price.
- Deterministic acknowledgement RTT metadata records each command's sampled latency; Legacy acknowledgement RTT remains `0`.
- TP/SL fills use raw OHLC plus the explicit ambiguity policy. Trailing levels tighten after completed bars for next-bar effectiveness.
- `ReplayExecutionLedgerState` owns fee-neutral, append-only fill capture in `src/tradovate/replay/ledger.rs`.
- `ReplayExecutionLedgerSnapshot` is the backward-compatible serializable schema-v3 boundary consumed later by Phase 5 result persistence and analytics; v2 fills default to bar-approximate precision when the new field is absent.
- `ReplayMarketTick` carries last trade, size, bid/ask prices, and optional quote sizes from raw replay sources.
- `ReplayFillModel::TickBidAsk` is coordinated in `src/tradovate/gateway/broker.rs`; `ReplayBrokerState::simulate_replay_tick` owns tick-order protection and trailing transitions.
- `ReplayMarketDom` carries optional full-book snapshots. `ReplayBarFrame` and `BrokerCommand::ReplayBar` propagate those snapshots only when present; the default replay path carries an empty DOM slice.
- `ReplayFillModel::Dom` is coordinated in `src/tradovate/gateway/broker.rs`; `ReplayDomBook` consumes visible levels and records the queue/depth assumption on fills. `src/tradovate/replay/load.rs` owns the optional JSONL sidecar parser.

## Expected Cross-Strategy Behavior

Replay must separate signal calculation from simulated execution:

- Every strategy calculates indicators and signals from the selected chart source.
- If the selected chart source is Heikin Ashi, EMA/HMA/etc. use Heikin Ashi OHLC/close as their signal input.
- If the selected chart source is Renko, strategies use the selected Renko series as their signal input.
- If the selected chart source is a tick-count bar, strategies use the completed tick-count bars as their signal input.
- Fill prices never come from synthetic chart values such as Heikin Ashi open/close or Renko brick prices unless the user explicitly selects a diagnostic-only synthetic fill model.
- Executable fills come from tradable raw market data: raw OHLC bars, ticks, bid/ask quotes, or DOM/book state.
- For closed-bar timing with zero delay on time/tick/volume/range bars, the usual baseline is: signal becomes eligible after the completed prior bar, order is scheduled for the next tradable event plus configured latency, and fill uses raw tradable data at or after that order-arrival timestamp.
- Bar-based replay may use raw `t0` open as a coarse baseline fill. Tick/quote replay should use the first eligible tick/quote at or after `signal_ts + latency`.
- This rule applies to all strategies, not only EMA.

## Accounting Boundary

Replay fills should be stored as an immutable execution ledger before fees and account overlays.

- Gross fills, positions, and realized gross PnL are outputs of the deterministic replay engine.
- Fees, initial capital, margin requirements, and account-size calculations are analytics/accounting overlays unless the selected strategy explicitly depends on account state.
- A saved run can be repriced with another fee schedule without replaying market data when fees do not affect sizing, gating, liquidation, or fill decisions.
- If a strategy uses net equity, available margin, drawdown limits, compounding size, or liquidation simulation, then changing the fee model is path-dependent and should be treated as a rerun or a separate account-overlay simulation.

## Tasks

### RBT-030: Define Virtual-Time Event Queue

Status: implemented for bar-level, Level 1, and optional Level 2 replay. The broker coordinator owns one virtual-time queue for raw-bar, tick, DOM, evaluation, and execution lifecycle phases.

Events:

- tick
- quote update
- DOM update
- bar close
- strategy evaluation
- order submitted
- order arrives at exchange
- order ack
- fill
- stop/limit/trailing update

Dependencies:

- Raw tick/quote dataset support (available through the replay cache and local tick sources).

Acceptance criteria:

- Replay can run as fast as possible without changing event order.
- The same deterministic replay core is used by headless runs and TUI visual replay.
- TUI speed affects rendering cadence, not market-time timestamps.
- All generated fills have deterministic timestamps.

Risks:

- Reusing live async tasks too directly can make backtests nondeterministic.

### RBT-031: Add Latency Model

Status: implemented. Fixed/zero, observed mean, observed p95/p99, and seeded empirical sampling are available. Configuration and ledger metadata record the population size and seed required for reproducibility.

Latency options:

- fixed, such as `60ms` or `200ms`
- average REST/order RTT from live observations
- p95/p99 from saved logs
- seeded random sample from a latency distribution
- zero-latency baseline

Model:

```text
signal_market_ts + configured_latency = order_arrival_market_ts
```

Dependencies:

- RBT-030.

Acceptance criteria:

- Same config and seed produces identical fills.
- Fast-forward does not reduce or increase simulated latency.
- Run result records latency model and seed.

Risks:

- Average RTT is not enough for worst-case order behavior. Keep p95/p99 modes.

### RBT-032: Implement Fill Model Levels

Status: implemented through Level 2. Legacy reference-price, raw next-bar-open, tick trade/bid/ask, and optional DOM models are explicit, and ledger results separate signal chart source from fill source.

Level 0: Legacy bar close

- Fill at signal/reference close.
- Useful only for quick signal checks.
- Must be labeled diagnostic-only if the chart source is synthetic, because Heikin Ashi/Renko close is not an executable market price.

Level 0b: Raw next-bar open

- For closed-bar strategies on time bars, fill at the raw tradable next bar open after signal eligibility.
- This is the preferred coarse bar-only baseline when ticks/quotes are unavailable.
- Do not use Heikin Ashi/Renko open for fills.
- For non-time bars such as tick-count/volume/range, the coarse baseline should be the first raw tradable event after signal eligibility, not a synthetic chart price.

Level 1: Tick trade model

- Market buy fills at ask if available, else the arriving trade price.
- Market sell fills at bid if available, else the arriving trade price.
- Stops/limits are evaluated against the ordered tick sequence, not bar OHLC. Tick mode records quote-exact precision when either quote side is present and tick-exact precision for trade-only input.
- Tick-driven trailing updates are applied after protection evaluation and are executable beginning with the following tick.

Level 2: DOM snapshot model

- Market orders consume visible book levels at arrival timestamp.
- Market fills record weighted-average price, requested/consumed quantity, levels consumed, and the `visible_levels_only` queue assumption.
- Top-of-book protection uses executable bid/ask prices. Exact order queue position and hidden liquidity remain outside this slice; capture is available, but provider history/entitlement still determines whether snapshots exist.

Dependencies:

- RBT-030.
- Tradovate Market Replay entitlement for historical DOM capture, or an explicitly launched live capture process for current DOM data.

Acceptance criteria:

- Fill model is explicit in every result.
- Result metadata separately records signal chart source and fill price source.
- Fills are recorded before fees so analytics can apply alternate fee schedules without changing execution prices or timestamps.
- Heikin Ashi/Renko strategy signals can fill from raw OHLC/tick/quote data without using synthetic prices as execution prices.
- Closed-bar zero-delay time-bar replay has a documented raw next-bar-open baseline.
- Level 1 can reproduce deterministic TP/SL ordering within a bar.
- Level 2 documents queue assumptions and does not claim perfect exchange replay.

Risks:

- DOM snapshots do not give exact queue position, hidden liquidity, or order-by-order book changes.

### RBT-033: Simulate Broker-Native Protection

Status: implemented for the documented coarse-bar model, Level 1 tick sequence, and optional Level 2 top-of-book path. Fixed TP/SL, explicit both-reachable policy for OHLC, tick-order protection, DOM executable-side protection, next-event-effective auto-trailing, initial broker stop behavior, and all three reversal command paths share the deterministic coordinator. Exact queue/market-depth parity remains outside the visible-levels-only assumption.

The live app uses broker-native TP/SL/trailing when configured. Backtests need a simulation equivalent:

- fixed take profit
- fixed stop
- trailing trigger
- trailing offset
- initial broker stop behavior
- reversal modes

Dependencies:

- RBT-030.
- RBT-032.

Acceptance criteria:

- EMA and HMA Cross TP/SL/trailing configs produce deterministic simulated order events.
- Reversal modes match live intent: Direct, Flatten > Confirm > Enter, CloseAll > Enter.
- Results show entry, exit reason, protection order id, and fill source.

Risks:

- Live broker trailing behavior may differ from our model; validate against Market Replay or small live sim runs.

### RBT-034: Preserve Existing Replay as Compatibility Mode

Status: implemented. Legacy remains the default and has regression coverage beside deterministic mode.

Do not remove current replay immediately.

Dependencies:

- None.

Acceptance criteria:

- Existing local-file replay still runs.
- New engine can be enabled with a config/feature flag.
- Tests cover both compatibility and new deterministic modes during migration.

Risks:

- Maintaining two paths for too long can hide bugs. Set a removal criterion after parity.

## Open Questions

- Fixed zero remains the explicit compatibility/research baseline. Should a future saved-run preset default to observed p95 instead?
- Which fee schedule presets should ship first: observed live stats, broker standard, broker premium/lifetime, and custom per-side/round-turn?
