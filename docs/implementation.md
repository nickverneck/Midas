# Lua scripting + backtest frontend implementation

## Objective
Provide a UI-driven backtest section where users can paste or load Lua scripts, run them against local `.parquet` data, and compare results to existing model baselines (GA/RL) using the same environment settings.

## Architecture
Frontend (SvelteKit) -> API route -> Rust binary -> results JSON -> frontend charts.

- **Frontend**: `frontend/src/routes/backtest/+page.svelte` (new)
- **API**: `frontend/src/routes/api/backtest/+server.ts` (new)
- **Rust**: `src/bin/backtest_script.rs` (new) + `src/script/` (new module)

## Rust implementation

### New dependencies (Cargo)
- `mlua` with a fixed Lua version for determinism (e.g., `lua54` + `vendored` feature).
- Optional: `serde` for result JSON serialization (already likely present).

### Script runtime module
Create `src/script/mod.rs` (or `src/lua/`) that exposes a minimal interface:

- `ScriptRunner::new(script_text, limits) -> Result<ScriptRunner>`
- `ScriptRunner::on_init(ctx) -> Result<()>`
- `ScriptRunner::on_bar(ctx, bar) -> Result<Action>`

Safety limits:
- Use `Lua::new()` (safe mode) so external C modules cannot load.
- Strip standard libraries: do not open `os`, `io`, `package`.
- Apply `Lua::set_memory_limit()`.
- Add instruction-count hook and abort if exceeded.

### Script API (Lua)
Global functions:
- `on_init(ctx)` (optional)
- `on_bar(ctx, bar)` -> action string: `"buy" | "sell" | "revert" | "hold"`

Data tables passed into Lua per bar:
- `ctx = { step, position, cash, equity, unrealized_pnl, realized_pnl }`
- `bar = { ts, open, high, low, close, volume, ...features }`

Indicators:
- Precompute in Rust using existing pipelines (`features.rs` and/or velora_ta) and pass the values in `bar`.
- Keep the Lua API small and deterministic. Avoid exposing data loaders or file system access.

### Backtest loop
`src/bin/backtest_script.rs`:
- Parse args: parquet path, script path or inline, env config, slice ranges.
- Load parquet (existing loader or Polars path).
- Build features (same as training).
- Initialize `TradingEnv` with `EnvConfig`.
- Loop bars:
  - Build `ctx` and `bar` tables.
  - Call `on_bar` -> map to `Action`.
  - `env.step(action, next_price, StepContext)`.
- Compute metrics (reuse `backtesting.rs` helpers; may need to make `compute_metrics` public).
- Return JSON with metrics, equity curve, and optional action trace.

### Result JSON
- `metrics`: total_pnl, sharpe, drawdown, profit_factor, win_rate, max_consecutive_losses.
- `equity_curve`: array
- `actions`: optional array of `{ idx, action }`
- `errors`: script error message if any

## API route (SvelteKit)
`POST /api/backtest`:
- Body: `{ script, dataset, path, env, limit, offset }`
- Validate and resolve path (copy the safe path logic from `/api/parquet`).
- Spawn `cargo run --bin backtest_script -- ...` or a prebuilt binary if present.
- Return JSON result.

Notes:
- This should be a normal JSON response, not SSE. Backtests are typically quick; add SSE later if needed.
- Fail fast with clear error messaging if script compile or runtime errors occur.

## Frontend UI
Create a new Backtest page:

Sections:
- **Dataset selector** (train/val/custom path)
- **Script editor** (textarea, optional load/save)
- **Env settings** (commission, slippage, max position, etc.)
- **Run button**
- **Results panel** (metrics table + equity curve chart)
- **Compare baseline** (optional selection of GA/RL run and overlay metrics)

Implementation detail:
- Reuse existing UI components (`Button`, `Table`, charts used in GA/RL pages).
- Provide a sample script snippet inline for faster onboarding.

## Baseline comparison
- Use existing GA/RL logs or metrics output for comparison on the same dataset.
- If baseline metrics are not stored in a standard JSON, add a small parsing step.

## Open choices
- Choose a single Lua engine (Lua 5.4 + vendored preferred).
- Decide whether to store scripts in `runs_backtest/` for reproducibility.
- Decide if we expose action trace by default (larger payloads).
