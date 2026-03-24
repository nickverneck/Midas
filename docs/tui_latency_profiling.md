# NinjaTrader TUI Latency Profiling Guide

Date: 2026-03-24

## Goal

Profile the current TUI before changing behavior, dependencies, or architecture, so we can identify where latency is actually coming from.

This guide is intentionally split into:

1. no-code-change profiling,
2. process isolation,
3. repeatable test scenarios,
4. tool-specific capture steps,
5. and hypotheses to verify in the current codebase.

## What We Need To Measure

For this app, "latency" is not one thing. There are at least five separate paths:

1. Input-to-paint latency in the foreground TUI process.
2. IPC latency between the TUI and the engine process.
3. Engine-side market-data ingestion latency.
4. Engine-side account/user-sync update latency.
5. Order-path latency from submit to broker-visible acknowledgements and fills.

The current app already exposes some network/order timing in the UI:

- `REST RTT`
- `Order Submit RTT`
- `Order Seen`
- `Exec Ack`
- `First Fill`
- `Market Update Age`

Those come from:

- `ninjatrader-tui/src/app/state.rs:195-205`
- `ninjatrader-tui/src/app/views.rs:89-120`
- `ninjatrader-tui/src/tradovate/latency.rs:1-49`
- `ninjatrader-tui/src/tradovate/gateway.rs:540-548`

Useful as they are, they do not tell us:

- how expensive `terminal.draw(...)` is,
- how much time is spent building chart datasets,
- how much time is spent serializing/deserializing IPC messages,
- how much time websocket JSON parsing consumes,
- how much time snapshot-building consumes,
- or whether one process is stalling the other.

## Current Architecture To Profile

The TUI is split into two processes:

- foreground TUI/client
- background engine/server

Relevant code:

- TUI loop: `ninjatrader-tui/src/main.rs:196-235`
- Engine spawn/connect path: `ninjatrader-tui/src/main.rs:243-287`
- IPC bridge: `ninjatrader-tui/src/ipc.rs:64-162`
- Engine service loop: `ninjatrader-tui/src/tradovate/service.rs:1-59`

That means profiling should be done in three passes, not one:

1. profile the TUI process alone,
2. profile the engine process alone,
3. profile both together to see cross-process effects.

## First Principles

Before using any profiler:

- always profile optimized code,
- use repeatable scenarios,
- isolate TUI and engine when possible,
- and capture both CPU cost and user-visible latency.

For this repo, start with release builds:

```sh
cargo build --release --manifest-path ninjatrader-tui/Cargo.toml
```

The crate already has a tuned release profile:

- `opt-level = 3`
- `lto = "thin"`
- `codegen-units = 1`

Source:

- `ninjatrader-tui/Cargo.toml:5-8`

## Recommended Profiling Order

### Phase 0: no-code-change baseline

Do this first.

Capture:

- per-process CPU %
- memory growth
- call stacks
- subjective UI responsiveness
- built-in latency fields in the header

Do not change code yet.

### Phase 1: isolate the TUI process

Run the engine separately, then attach the TUI without letting it spawn anything.

Why:

- makes the foreground process easier to sample,
- avoids mixing render cost with engine/network cost,
- and makes repeated runs more comparable.

Suggested setup:

Terminal 1, start the engine:

```sh
cargo run --release --manifest-path ninjatrader-tui/Cargo.toml -- \
  --config ninjatrader-tui/config.example.toml \
  --engine-socket /tmp/midas-tui-prof.sock \
  engine
```

Terminal 2, start the TUI client against that engine:

```sh
cargo run --release --manifest-path ninjatrader-tui/Cargo.toml -- \
  --config ninjatrader-tui/config.example.toml \
  --engine-socket /tmp/midas-tui-prof.sock \
  --no-spawn-engine
```

### Phase 2: isolate the engine process

Keep the same split-process setup, but profile the engine PID only while driving the app from the TUI.

Why:

- engine-side websocket parsing, snapshot work, and strategy logic all live outside the TUI process,
- so a "slow TUI" can actually be slow engine updates feeding the client late.

### Phase 3: full-system run

Profile both processes during the same test scenario.

Why:

- IPC serialization/deserialization cost only shows up clearly when both sides are active,
- and some regressions only appear when the engine is publishing frequent updates while the dashboard is rendering.

## Repeatable Test Scenarios

Use the same scenarios every time so captures are comparable.

### Scenario A: idle login screen

Purpose:

- baseline redraw cost with minimal state.

Expected components:

- `main.rs` tick loop,
- app draw,
- ratatui layout/render.

Relevant code:

- `ninjatrader-tui/src/main.rs:202-213`
- `ninjatrader-tui/src/app/render.rs:1-20`

### Scenario B: connected, no contract subscribed

Purpose:

- measure connect/auth/bootstrap overhead without market rendering.

Relevant code:

- `ninjatrader-tui/src/tradovate/service.rs:67-147`
- `ninjatrader-tui/src/tradovate/auth.rs`
- `ninjatrader-tui/src/tradovate/market.rs:52-100`

### Scenario C: contract search typing

Purpose:

- measure typing responsiveness plus contract search roundtrip.

Relevant code:

- `ServiceCommand::SearchContracts`
- `ninjatrader-tui/src/tradovate/service.rs:177-190`
- `ninjatrader-tui/src/tradovate/auth.rs:192-220`

### Scenario D: dashboard with live bars, no strategy armed

Purpose:

- measure the hot display path under normal streaming market data.

This is the most important baseline.

Relevant code:

- user websocket loop: `ninjatrader-tui/src/tradovate/market.rs:196-340`
- market websocket loop: `ninjatrader-tui/src/tradovate/market.rs:377-520`
- websocket JSON parse entry: `ninjatrader-tui/src/tradovate/protocol.rs:114-124`
- market snapshot emission: `ninjatrader-tui/src/tradovate/service.rs:382-395`
- chart rendering: `ninjatrader-tui/src/app/render.rs:904-1108`

### Scenario E: dashboard with live bars and native strategy armed

Purpose:

- measure strategy execution overhead on top of streaming updates.

Relevant code:

- strategy execution entry from market updates: `ninjatrader-tui/src/tradovate/service.rs:382-389`
- execution state emission sites: `ninjatrader-tui/src/tradovate/execution.rs`

### Scenario F: manual order submit

Purpose:

- measure local submit-to-UI and submit-to-broker timing.

Observe:

- built-in `Order Submit RTT`
- `Order Seen`
- `Exec Ack`
- `First Fill`

Relevant code:

- latency tracking: `ninjatrader-tui/src/tradovate/latency.rs:1-49`
- latency fields: `ninjatrader-tui/src/tradovate/types.rs:263-270`

## What To Watch During Each Run

Record these for each scenario:

- TUI PID
- engine PID
- average CPU %
- peak CPU %
- peak RSS
- p95 and worst-case input responsiveness by observation
- p95 and worst-case built-in latency fields from the header
- whether the dashboard stutters during bursts of bars or fills

If possible, save a short note with each profile:

- screen you were on,
- whether bars were flowing,
- whether strategy logic was armed,
- whether orders were submitted,
- and whether the run was `sim` or `live`.

## No-Code-Change Tooling

### macOS: built-in first

Start with `sample`, because it is already available and attaches to a running PID.

Capture a TUI sample:

```sh
sample <tui_pid> 10 -file /tmp/tui.sample.txt
```

Capture an engine sample:

```sh
sample <engine_pid> 10 -file /tmp/engine.sample.txt
```

Use Activity Monitor or `top` alongside it:

```sh
top -pid <tui_pid> -stats pid,command,cpu,mem,time
top -pid <engine_pid> -stats pid,command,cpu,mem,time
```

If you want timeline-style profiling, use Instruments:

- Time Profiler for CPU hotspots
- System Trace if the issue looks scheduler- or wakeup-related

### Linux: built-in and standard

If `perf` is available:

```sh
perf record -F 999 -g -p <tui_pid> -- sleep 15
perf report
```

and:

```sh
perf record -F 999 -g -p <engine_pid> -- sleep 15
perf report
```

If `cargo flamegraph` is available, it is useful for controlled single-process runs, but for this app the attach-to-running-PID model is often more useful because the TUI and engine are split.

### Cross-platform option

If `samply` is already installed, it is a good fit for Rust call stacks:

```sh
samply record --pid <tui_pid>
samply record --pid <engine_pid>
```

Do not make `samply` a prerequisite for the first round. Use it only if it is already installed or easy to add later.

## How To Isolate The Suspected Hot Paths

### 1. Foreground draw loop

Relevant code:

- `ninjatrader-tui/src/main.rs:209-213`
- `ninjatrader-tui/src/app/render.rs:1-20`

What to verify:

- how much CPU the unconditional redraw loop consumes,
- whether idle redraw is already expensive,
- and whether active rendering grows sharply when the dashboard is visible.

Important detail:

- the TUI redraws every loop iteration and also keeps a 125 ms interval tick.
- Even if no user input happens, the client still repaints on that cadence.

Source:

- `ninjatrader-tui/src/main.rs:202-213`

### 2. Chart rendering and per-frame allocation

Relevant code:

- `ninjatrader-tui/src/app/render.rs:904-1108`

What to verify:

- allocations from cloning recent bars,
- cost of rebuilding point vectors and segment datasets every frame,
- cost of scanning trade markers each frame,
- and whether the dashboard screen is materially slower than other screens.

Why this is a likely hotspot:

- it clones and rebuilds multiple `Vec`s on every chart render,
- and it does so in the most visually important screen.

### 3. Market websocket decode path

Relevant code:

- `ninjatrader-tui/src/tradovate/market.rs:401-520`
- `ninjatrader-tui/src/tradovate/protocol.rs:114-158`

What to verify:

- CPU in JSON parse,
- CPU in bar extraction,
- CPU in repeated `Value` traversal,
- and whether bursts of incoming data align with UI stutter.

### 4. User sync websocket and envelope processing

Relevant code:

- `ninjatrader-tui/src/tradovate/market.rs:196-340`
- `ninjatrader-tui/src/tradovate/service.rs:340-365`
- `ninjatrader-tui/src/tradovate/store.rs:1-86`

What to verify:

- cost of applying entity envelopes,
- cost of cloning `Value`s into account/order/fill stores,
- and whether fill bursts trigger visible frame stalls.

### 5. Snapshot building

Relevant code:

- snapshot refresh trigger: `ninjatrader-tui/src/tradovate/gateway.rs:552-572`
- account snapshot builder: `ninjatrader-tui/src/tradovate/store.rs:88-108`
- account snapshot emission: `ninjatrader-tui/src/tradovate/service.rs:367-373`

What to verify:

- cost of cloning `accounts`, `market`, `managed_protection`, and `user_store`,
- cost of rebuilding account snapshots,
- and whether snapshot work correlates with spikes in either process.

### 6. Full market snapshot emission over IPC

Relevant code:

- internal incremental update build: `ninjatrader-tui/src/tradovate/gateway.rs:583-642`
- apply + emit full snapshot: `ninjatrader-tui/src/tradovate/service.rs:382-395`
- IPC serialization/deserialization: `ninjatrader-tui/src/ipc.rs:73-118` and `ninjatrader-tui/src/ipc.rs:132-158`

Important detail:

- engine-side market handling is incremental,
- but after each market update, the service sends a full `ServiceEvent::MarketSnapshot(snapshot)` to the client.

What to verify:

- time spent cloning `session.market`,
- time spent serializing that event to JSON,
- time spent deserializing it in the client,
- and whether this dominates over render cost once bars are flowing.

### 7. Order path

Relevant code:

- tracker update: `ninjatrader-tui/src/tradovate/latency.rs:1-49`
- tracker state: `ninjatrader-tui/src/tradovate/types.rs:283-301`

What to verify:

- whether local processing is negligible compared with broker/network time,
- or whether submit handling and follow-up state updates add noticeable local delay.

## Practical Capture Plan

### Pass 1: TUI-only CPU hotspot capture

1. Start engine separately.
2. Start TUI with `--no-spawn-engine`.
3. Connect and subscribe to a contract.
4. Switch between `Selection` and `Dashboard`.
5. While bars are flowing, capture a 10 to 15 second sample of the TUI PID.

Success criteria for this pass:

- know whether `terminal.draw`, `render_chart`, or IPC decode dominates the client process.

### Pass 2: engine-only CPU hotspot capture

1. Keep the same run active.
2. Capture a 10 to 15 second sample of the engine PID while:
   - bars are flowing,
   - no strategy is armed,
   - then with a strategy armed.

Success criteria for this pass:

- know whether market websocket parsing, snapshot refresh, or strategy execution dominates the engine process.

### Pass 3: order path observation

1. Submit one manual order in `sim`.
2. Watch the header latency fields.
3. Capture both PIDs during the submit and acknowledgement window.

Success criteria for this pass:

- separate broker/network delay from local CPU delay.

## Hypotheses To Verify In This Codebase

These are not conclusions. They are the first things worth checking.

### Hypothesis 1: dashboard rendering is expensive

Reason:

- `render_chart` clones bars, builds multiple vectors, scans markers, and builds many datasets every frame.

Relevant code:

- `ninjatrader-tui/src/app/render.rs:919-1108`

### Hypothesis 2: full market snapshot IPC is a major contributor

Reason:

- engine work is incremental,
- but a full `MarketSnapshot` is still cloned and sent to the client on each market update.

Relevant code:

- `ninjatrader-tui/src/tradovate/gateway.rs:583-642`
- `ninjatrader-tui/src/tradovate/service.rs:382-395`
- `ninjatrader-tui/src/ipc.rs:73-118`

### Hypothesis 3: websocket JSON parsing is a measurable engine hotspot

Reason:

- both user-data and market-data workers parse JSON in tight async loops using `serde_json::Value`.

Relevant code:

- `ninjatrader-tui/src/tradovate/market.rs:252-334`
- `ninjatrader-tui/src/tradovate/market.rs:406-508`
- `ninjatrader-tui/src/tradovate/protocol.rs:114-158`

### Hypothesis 4: account snapshot rebuilding is noisy under fill/order churn

Reason:

- the engine clones multiple state objects and rebuilds snapshots in a spawned task after store updates.

Relevant code:

- `ninjatrader-tui/src/tradovate/gateway.rs:552-572`
- `ninjatrader-tui/src/tradovate/store.rs:88-108`

### Hypothesis 5: idle redraw cadence is wasting CPU

Reason:

- the TUI loop keeps a 125 ms interval and redraws continuously even when nothing visible changed.

Relevant code:

- `ninjatrader-tui/src/main.rs:202-213`

## What We Should Not Do Yet

Until the baseline captures are done, do not:

- change JSON libraries,
- change the IPC payload format,
- add new caching layers,
- rewrite the chart renderer,
- or change async task structure.

Those may end up being the right fixes, but the current split-process design means we can easily optimize the wrong layer if we do not profile first.

## Deliverables From The First Profiling Round

At the end of the first round we should have:

1. one TUI CPU profile while idle,
2. one TUI CPU profile on the live dashboard,
3. one engine CPU profile on the live dashboard,
4. one engine CPU profile with a strategy armed,
5. one short order-path capture,
6. and a short write-up with the top 3 latency contributors.

If the first round clearly identifies a dominant hotspot, the next document should propose fixes against that specific path only.

## Likely Next Step After This Document

If the captures are readable with current symbols, proceed directly to analysis.

If the captures are too coarse, the next change should be minimal and temporary:

- improve release-symbol visibility for profiling,
- then re-run the exact same scenarios.

That should happen only after the no-code-change pass above.
