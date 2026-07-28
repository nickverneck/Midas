# TODO to reach backtesting + RL/GA goal (Dec 16, 2025)

## Core env & data
- [x] Observation API: expose reset and step_batch in PyO3; obs includes t-1 price/vol, time sin/cos, MA features, position.
- [x] Data sampler: chronological window sampler in Rust (no lookahead) accessible from Python.
- [x] Add time encoding (sin/cos hour) into feature builder and PyO3 obs.
- [x] Add session/margin masks to obs for action masking in agents.

## Features/indicators
- [x] Compute ATR (multiple periods) as features (no thresholds).
- [x] Include volume-derived features (vol_t1 done; add rolling VWAP/volatility later if needed).
- [ ] Ensure feature warmup handling (NaNs) is masked or trimmed before training.

## Backtester enhancements
- [x] Equity/reward export (CSV) from Rust runner for quick inspection.
- [ ] Batch stepping performance profiling; reduce allocations in env step.
- [ ] Slippage/commission configs per instrument; load from config file.

## RL/GA logic
- [ ] Implement PPO-style rollout/advantage calc with configurable reward weights (Sharpe/PnL/drawdown).
- [ ] Implement GA fitness over multiple regimes: mean return/Sharpe, penalties for drawdown/std returns.
- [x] Action space: keep discrete {buy/sell/hold/revert}; optional hybrid size head later.
- [ ] Hard rule enforcement: finalize margin/session rules and penalties.

## Data splits / evaluation
- [ ] Walk-forward evaluation helper (train window -> validate -> test) using provided parquet streams.
- [ ] Keep external validation dataset untouched until final evaluation.

## CI / tests
- [ ] Add Rust benches for env step and indicator computation.
- [ ] Basic property tests for PnL accounting and drawdown metrics.

## Documentation
- [x] Update README with build instructions (`uv` + `maturin develop --features python`).
- [x] Update README to reflect Python deprecation and Rust as the main engine.
- [ ] Quickstart for running EMA rule vs. feature-only RL training.

## Frontend QoL (SvelteKit)
### High priority
- [x] [001] Fix active top nav state — make nav highlight correct section based on current route
- [ ] [003] Prevent accidental training submit — remove implicit form submits, only Step 3 button triggers run
- [ ] [004] Show supported training backends — disable unavailable backends in train forms
- [ ] [009] Link validation errors to fields — make errors actionable per field, not just a generic list

### Medium priority
- [ ] [002] Align train analytics back link — back link should depend on trainMode (GA vs RL)
- [ ] [005] Improve collapsed sidebars — visible expand buttons, labels, and tooltips
- [ ] [006] Add GA run folder picker — mirror the RL folder picker experience
- [ ] [007] Make backtest demo state unmistakable — clearly separate demo from real run data
- [ ] [008] Wire script load or remove disabled button — implement .lua loading or remove the button
- [ ] [010] Simplify analyzer setup flow — guided flow with advanced settings secondary
- [ ] [011] Highlight selected analyzer cell — persistent selected state on heatmap cells
- [ ] [012] Add RL chart inspection controls — zoom/window/reset, increase height, show epoch range

### Low priority
- [ ] [013] Clarify RL fitness weight controls — readable labels, reset action, formula summary
- [ ] [014] Stabilize training log timestamps — store timestamp at append time, not render time
