# Backtest script viability (frontend)

## Goal
Add a frontend section where a user can load a strategy script, run it against local `.parquet` datasets, and compare its metrics to the trained model (GA/RL) on the same data and env settings.

## Feasibility in this repo
- The frontend already runs Rust CLIs from SvelteKit server routes (see `frontend/src/routes/api/*`).
- Adding a `/api/backtest` endpoint that spawns a new Rust binary (e.g., `backtest_script`) is consistent with the current architecture.
- The script execution should happen server-side (Node -> Rust). Running scripts in the browser would require streaming full parquet data to the client, which is heavy and defeats the current local-data model.

## Option A: rust-script (Rust scripts via CLI)
What it is:
- `rust-script` is a command-line tool (not a library) that runs Rust scripts by unpacking them into a Cargo package and compiling them. It supports inline dependency declarations in the script itself. Sources: docs.rs and lib.rs. [1][2]

Pros:
- Max performance at runtime once compiled.
- Full access to existing Rust crates (Polars, env, TA, etc.) and types.

Cons / risks:
- Heavy compile latency per script change; requires a full Rust toolchain on the host.
- Not sandboxed: scripts can access filesystem/network via Rust stdlib.
- Hard to safely run untrusted scripts (even on local machine, mistakes can be destructive).

Use it if:
- Scripts are strictly local and trusted (power-user mode).
- You are okay with compile latency and toolchain dependency.

## Option B: Lua via `mlua` (embedded runtime)
What it is:
- `mlua` is a Rust binding that supports Lua 5.4/5.3/5.2/5.1, LuaJIT, and Luau via feature flags. It can vendor Lua/LuaJIT at build time for easier distribution. [3][4]
- `mlua` has a safe `Lua::new()` that disables loading external C modules (use `Lua::unsafe_new()` to re-enable). [5]
- It also exposes `Lua::set_memory_limit()` for memory caps. [5]
- Lua itself provides instruction-count hooks (`debug.sethook` / `lua_sethook`) that can be used to stop runaway scripts. [6]

Pros:
- Fast startup, no compile step per script.
- Good sandboxing story: limit memory, instruction counts, and what globals/libs are exposed.
- API surface can be intentionally small: only expose TA and trading commands.

Cons:
- Need to design a binding layer for data and indicators.
- Slightly less raw performance than compiled Rust, but still likely fast enough for backtest loops.

Use it if:
- You want user-loadable scripts with reasonable safety limits.
- You want rapid iteration and a simple UX.

## Recommendation
- **Primary**: Lua via `mlua` is the most viable for a UI-driven “load script and run” flow.
- **Optional**: Keep `rust-script` as a power-user local mode (clearly labeled “unsafe / trusted scripts only”).

## Proposed script API (Lua)
Expose a small, stable surface so scripts are short and deterministic:

- `on_init(ctx)`
  - Called once with metadata (symbol, timeframe, start/end).
- `on_bar(ctx, bar)`
  - Called each bar; return an action: `"buy" | "sell" | "revert" | "hold"`.

Suggested globals/modules:
- `bar`: `{ ts, open, high, low, close, volume, ...features }`
- `ctx`: `{ position, cash, equity, unrealized_pnl, realized_pnl, step }`
- `ta`: indicator helpers that operate on Rust-managed arrays or rolling windows:
  - `ta.ema(period, series)` or `ta.ema(period)` for streaming state
  - `ta.sma(period)`, `ta.hma(period)`, `ta.atr(period)` etc.
- `trade`: command helpers if you prefer explicit calls instead of return values:
  - `trade.buy()`, `trade.sell()`, `trade.revert()`, `trade.hold()`

Implementation detail: precompute indicators in Rust (using existing `features.rs` or `velora_ta`) and pass slices/values into Lua per bar. This avoids Lua re-implementing math and keeps results consistent with the trained model.

## Backend flow (high level)
1. `POST /api/backtest` with: dataset path, env params, script text, and optional model baseline.
2. Rust binary:
   - Load parquet (Polars).
   - Build indicators/features (same as training).
   - Initialize script runtime (Lua).
   - Loop bars -> call `on_bar` -> map to `Action` -> run `TradingEnv` step.
   - Compute metrics (Sharpe, PnL, drawdown) using existing backtest helpers.
3. Return metrics + equity curve + optional action trace.
4. Frontend compares to baseline (e.g., last GA/RL run metrics on same dataset).

## Safety / guardrails (Lua)
- Expose only selected libraries (no `os`, `io`, `package`).
- Set memory limit via `Lua::set_memory_limit()`.
- Set instruction hooks (count-based) to abort runaway scripts.
- Bound runtime per request and return a clear error when limits are hit.

## Notes on baseline comparison
“Better than the model” should mean same dataset slice + env config + metrics. Reuse metrics already computed in Rust for GA/RL runs so the comparison is apples-to-apples.

## Sources
[1] rust-script crate docs (usage, CLI tool): https://docs.rs/crate/rust-script/0.36.0
[2] rust-script overview (inline dependencies, CLI): https://lib.rs/crates/rust-script
[3] mlua overview (Lua versions, LuaJIT): https://github.com/mlua-rs/mlua
[4] mlua feature flags (Lua/LuaJIT/luau, vendored): https://docs.rs/crate/mlua/%3E%3D0.7.0
[5] mlua changelog (safe `Lua::new()`, memory limits): https://docs.rs/crate/mlua/latest/source/CHANGELOG.md
[6] Lua 5.4 manual (instruction-count hook): https://www.lua.org/manual/5.4/manual.html
