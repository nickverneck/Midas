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

## PyO3 / Python tooling
- [ ] Replace deprecated PyO3/numpy APIs with bound versions to remove warnings.
- [ ] Provide wheels/build script for Python 3.12+ (avoid ABI3 workaround if possible).
- [ ] Add `python/examples`:
  - [x] `train_ppo.py` using PyTorch (MLP policy) against batched env.
  - [ ] `train_ga.py` illustrating GA fitness over multiple windows.

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
- [ ] Python tests for feature parity and PyO3 bindings.
- [ ] Basic property tests for PnL accounting and drawdown metrics.

## Documentation
- [x] Update README with build instructions (`uv` + `maturin develop --features python`).
- [ ] Quickstart for running EMA rule vs. feature-only RL training.
