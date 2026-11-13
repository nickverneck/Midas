# Trading workspace plan

## Goal and boundaries

Create a Svelte trading research workspace independent of the training UI and trader TUI. It should chart stored and live market data, run and compare strategies, and save a reproducible candidate for later use in the TUI. It must not place orders. The TUI remains responsible for account selection and execution.

## Existing foundations

- `frontend/` is a SvelteKit 2 / Svelte 5 training application with reusable UI patterns, but has a separate purpose and navigation.
- `trader/src/replay_cache/` contains dataset manifests, provenance, Parquet raw ticks and server bars, time-range resolution, and a dataset library. `trader/src/tradovate/replay/bars.rs` derives minute, second, tick, volume and range bars from cached ticks. These APIs are currently internal to the trader binary.
- `trader/src/tradovate/hosted_replay.rs` provides NinjaTrader hosted **replay**, including a sampled quote capture path. It is not evidence of a complete live NinjaTrader trade tape. Source and fidelity must be visible in the workspace; a live connector needs separate verification.
- `trader/src/strategies/` and `trader/src/strategy.rs` define native strategies and execution configuration. `trader/src/tradovate/replay/` implements the replay simulator, sweep planning, execution, result artifacts and ranking. Its fill, cost, timing and protection semantics should remain authoritative.
- `src/script/` implements a limited Lua sandbox for a separate lightweight backtester. It is not yet integrated with the trader replay engine or its sweep runner. The existence of a Lua strategy enum in the trader does not establish replay support.
- The TUI holds Lua editor state but `ExecutionStrategyConfig` does not carry the program. The trader execution loop and sweep runner currently gate on native strategies, and the prepared CPU sweep path has specialized native implementations. Lua support therefore needs a serialized, versioned program and bounded VM, an explicit replay evaluator path and script identity in results/resume fingerprints; optimized paths must reject or fall back until they support it. Position protections and live execution require separate integration and tests.
- The trader engine exposes local IPC and replay commands; there is no existing public web chart API. Existing replay types and bar builders are not presently a stable external library.

## Execution choice

Use **server-side execution first**, on the machine hosting the trader data and replay engine. This retains one implementation of strategy evaluation, fills, fees, risk and sweep scheduling; avoids transferring large datasets to browsers; and permits durable, cancellable jobs. It requires a running local service, careful authentication and resource limits, and network latency for remote clients. Bind to loopback by default; never expose credentials or order operations through the research API.

Browser WebAssembly could offer low-latency, private offline experiments and distribute compute to the client, but it would require a portable core, explicit data transfer and worker orchestration, and matching native numerical and execution semantics. Browser memory, lifecycle and device capacity also limit large sweeps. Consider it only after extracting a pure replay core and proving parity with the server on fixed fixtures. A browser worker communicates through messages and cannot directly manipulate the DOM ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers)).

## Delivery sequence

1. **Contracts and data access.** Define a versioned read-only API for dataset discovery, source provenance and chart bars, plus explicit errors for missing tick/volume data. Expose existing cache readers and bar builders through a narrow supported boundary, preserving nanosecond ordering and session semantics. Define feed status and reconnection separately from replay. Verify a genuine live NinjaTrader source and its permissions before advertising it as live.
2. **Independent workspace.** Create a separate Svelte application with responsive chart navigation, time/volume/tick/range controls, normal and Heikin Ashi display, source/status feedback and empty/error states. Heikin Ashi is a display transform; execution must use the original bars unless explicitly specified. Choose a chart library that handles large streams, zoom, pan and touch; preserve its license attribution. Non-time bars may share timestamps, so the chart adapter must preserve bar identity rather than silently discard data.
3. **Native backtests.** Expose existing replay configuration validation and job lifecycle through a local service. Support progress, cancellation, resource limits, and result provenance. Show equity, drawdown, trades, fees and strategy parameters from actual artifacts, with clear handling of missing metrics. Cross-check results against a matching trader CLI run.
4. **Lua integration.** Define a versioned Lua callback contract, resource limits and deterministic execution in the trader replay engine. Reuse sandboxing ideas, but do not present the lightweight backtester as equivalent. Test signals, fills, cancellation and errors against fixtures and document what, if anything, can run in the TUI.
5. **Sweeps and candidate handoff.** Reuse the existing sweep planner, runner and ranking. Show progress, failed/partial runs and a heatmap for chosen parameter axes, with filters and exact metric definitions. Save a versioned candidate containing strategy, parameters, data, execution assumptions and provenance; validate TUI compatibility and add an explicit import path before calling it deployable. Keep selection and live/sim activation in the TUI.
6. **Quality and release.** Exercise real data end to end; compare each aggregation with known fixtures and replay results; verify browser desktop and mobile interactions, keyboard access, errors and loading states. A fresh critic should inspect the running UI and attempt the actual chart, backtest, sweep and save workflows. Fix actionable findings and repeat with a fresh review. Document remaining limitations honestly; a review cannot guarantee that no defects exist.

## Agent rounds

- Initial independent audits cover data/feed, strategy/replay and frontend infrastructure.
- Fresh implementation agents work on non-overlapping chart/data service, strategy workflow, and Svelte UI tasks with an agreed API contract. Integration ownership remains centralized.
- Fresh reviewers inspect each implementation for correctness, preservation of existing work and meaningful tests. Resolve findings before integration.
- A fresh browser critic exercises desktop and mobile, compares the implementation with the complete feature list, and reports concrete defects. Repeat implementation and review rounds as needed.

## Acceptance criteria

- A user can select an identified stored dataset and source, inspect truthful provenance, and view all supported bar aggregations with normal or Heikin Ashi candles. Live mode clearly distinguishes connected, delayed, disconnected and unsupported feeds.
- A user can configure and run a supported native or Lua strategy on selected data, inspect its actual statistics and trades, and see execution assumptions. Unsupported combinations explain why instead of returning invented results.
- A user can run a bounded sweep, inspect and rank actual results, and save a candidate with reproducible settings. A documented, tested TUI import validates the candidate before account use.
- The interface works with mouse, keyboard and touch at desktop and mobile sizes, and exposes errors and long-running progress without blocking the application.
