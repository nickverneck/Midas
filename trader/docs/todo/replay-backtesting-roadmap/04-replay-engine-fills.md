# Task Group 04: Replay Engine and Fill Model

## Goal

Move replay from "bar stream plus immediate fills" toward deterministic virtual-time simulation.

Fast-forward must change wall-clock speed only. Accuracy should depend on market timestamps and the fill model, not on how fast the UI renders.

## Implementation Status (2026-07-30)

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
- Ledger schema v2 records the latency model, optional seed, observed sample count, and bar-protection policy. Missing v2 fields deserialize to safe defaults for v1 compatibility.

The seventh Phase 4 slice consolidates deterministic ordering under one broker coordinator queue:

- The replay worker now owns only dataset enumeration and wall-clock pacing; it no longer owns a second deterministic priority queue.
- The broker coordinator owns the sole `ReplayVirtualEventQueue` instance for raw-bar-open, bar-close, strategy-evaluation, submission, arrival, acknowledgement, fill, and protection phases.
- Worker barriers carry stable bar/evaluation ids. Those ids derive monotonic open/evaluation logical steps and preserve duplicate provider timestamps.
- Headless and TUI replay continue through the same worker/service/gateway path; rendering speed remains outside virtual-time ordering.

RBT-030 and RBT-031 are now implemented for the deterministic bar engine, RBT-033 is implemented for the documented coarse-bar policy, and RBT-032 Level 0b remains the active fill level. It is not yet a tick-level replay engine:

- In the coarse bar model, acknowledgement and fill occur together when the first eligible raw bar is processed.
- TP/SL checks still use raw bar OHLC. The configured ambiguity policy makes results reproducible but does not reconstruct the real intrabar path.
- Bar trailing updates intentionally become effective on the next source bar; exact same-bar activation/update/fill requires ticks.
- Source bar timestamps are the available scheduling timestamps; tick/quote arrival within a bar is unavailable until the validated raw-tick path is connected.
- Tick/quote and DOM fills remain later Phase 4 work.
- Observed latency samples are supplied through configuration; automatically importing a population from saved live logs can be added as a convenience without changing the sampler.
- The ledger currently lives for the engine session. Durable `result.json`/CSV/Parquet output belongs to Phase 5.

Validation for this slice:

- Default suite: 273 tests passed.
- Replay-feature suite: 445 tests passed.
- All-feature suite: 448 tests passed.
- Safe local TUI replay: cached MESU6, 1-minute OHLC, 1,380 rows, deterministic mode, strategy disarmed, manual orders unavailable, zero persisted errors, and no engine left running after exit.
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
- `ReplayExecutionLedgerSnapshot` is the backward-compatible serializable schema-v2 boundary consumed later by Phase 5 result persistence and analytics.

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

Status: implemented for bar-level replay. The broker coordinator owns one virtual-time queue for every supported raw-bar, evaluation, and execution lifecycle phase. Validated raw tick/quote input remains the dependency for intrabar events.

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

- RBT-012 for tick-level mode.

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

Status: in progress. Legacy reference-price and raw next-bar-open models are explicit, and ledger results separate signal source from fill source. Tick and DOM levels remain.

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

- Market buy fills at ask if available, else next trade price plus slippage rule.
- Market sell fills at bid if available, else next trade price minus slippage rule.
- Stops/limits are evaluated against tick sequence, not bar OHLC.

Level 2: DOM snapshot model

- Market orders consume visible book levels at arrival timestamp.
- Limit orders use deterministic queue assumptions.

Dependencies:

- RBT-030.
- RBT-012 for Level 1.
- DOM capture/probe for Level 2.

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

Status: implemented for the documented coarse-bar model. Fixed TP/SL, explicit both-reachable policy, next-bar-effective auto-trailing, initial broker stop behavior, and all three reversal command paths share the deterministic coordinator. Exact intrabar parity remains dependent on Level 1 ticks.

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
