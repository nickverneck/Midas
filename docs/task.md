# Lua scripting + backtest frontend tasks

## Phase 1: Rust scripting core
- [ ] Add `mlua` dependency + features to `Cargo.toml`.
- [ ] Create `src/script/mod.rs` with a safe Lua runtime wrapper.
- [ ] Define Lua API surface (`on_init`, `on_bar`) and action mapping.
- [ ] Expose metrics helper (make `compute_metrics` reusable).
- [ ] Implement `src/bin/backtest_script.rs` CLI.
- [ ] Add unit tests for action mapping + script error handling.

## Phase 2: API route
- [ ] Add `frontend/src/routes/api/backtest/+server.ts`.
- [ ] Validate parquet paths (reuse `/api/parquet` logic).
- [ ] Spawn `backtest_script` and return JSON.
- [ ] Handle errors and timeouts gracefully.

## Phase 3: Frontend UI
- [ ] Add `frontend/src/routes/backtest/+page.svelte`.
- [ ] Script editor + sample script.
- [ ] Dataset selector + env parameter form.
- [ ] Results view (metrics + equity curve).
- [ ] Baseline comparison toggle.

## Phase 4: Polish
- [ ] Save script snapshots with results (optional).
- [ ] Add SSE streaming if backtests become long-running.
- [ ] Add docs with example scripts and safety limits.
