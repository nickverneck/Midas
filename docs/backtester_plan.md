# Backtester + RL/GA Plan
Updated: 2025-12-16

## Objectives
- Rust-first, low-latency backtester with PyO3 bridge for model training in Python.
- Support short intraday futures trading (e.g., MES) with dense short-period MAs and ATR-based risk signals.
- Provide RL environment + GA fitness eval with reward functions centered on Sharpe, PnL, and drawdown.

## Architecture Overview
- **Core backtester (Rust)**: data loading (Parquet via Polars/Arrow), indicator calc (velora_ta), order simulation, risk checks, metrics, episode rollouts.
- **RL/GA training (Python)**: policy models in PyTorch/Lightning; GA loop in Python for convenience; PyO3 bindings expose Rust env step/reset + batched rollouts.
- **Interop**: PyO3 module `midas_env` wrapping Rust env; zero-copy Arrow buffers where possible; fall back to ndarray copies for GPU handoff.
- **Deployment targets**: dev on macOS (CPU), training on Linux + NVIDIA 1080 Ti (CUDA 12.x). Keep CUDA-conditional code on Python side only.

## Data + Features
- **Source**: Parquet OHLCV + microstructure fields. Lazy scan with Polars to slice date ranges and columns.
- **Time encoding**: add `sin(2π*hour/24)`, `cos(2π*hour/24)`; keep explicit timestamp for possible LSTM/temporal models.
- **Assets**: start with MES; design schema to extend to multi-asset later.

## Indicators (velora_ta)
- Period set: 3, 5, 7, 11, 13, 19, 23, 29, 31, 37, 41, 43, 47, 53.
- For each period: EMA, SMA, HMA (15 each = 45 total).
- Add volatility/risk signals: ATR (multiple windows, e.g., 7, 14, 21), true range, rolling stdev; later optional: RSI, MACD, Bollinger Bands for pattern filters.

## Trading Constraints (hard rules)
- Margin requirements, position size limits, buying power, short-selling rules, session hours (align with NinjaTrader-style futures constraints).
- Reject invalid actions in Rust env; return penalty flag for RL step.

## Rewards
- **Step reward**: `pnl_change - commission*|Δpos| - slippage_estimate - risk_penalty*|pos/max_pos| - 1000*margin_call_violation - 1000*position_limit_violation - idle_penalty`.
- **Episode reward**: `10*sharpe + total_pnl -5*max_drawdown -2*max_consecutive_losses +0.5*profit_factor +0.3*win_rate +0.2*strategy_diversity`.
- **GA fitness (across episodes)**: mean returns/sharpes, penalize std of returns, percentile(25) uplift, penalize mean/max drawdown, adaptability term (corr with market regimes), survival bonus (no margin calls).

## Backtester Mechanics
- Order types: market, limit with simple book/slippage model; configurable commission per contract.
- Position accounting: FIFO lots, intraday flat option; track MFE/MAE, drawdown, consecutive losses.
- Episode handling: multi-episode batches; supports random start offsets and regime sampling.

## RL/GA Interface
- Rust env exposes `reset(seed) -> State`, `step(action) -> (State, reward, done, info)`; batched `step_batch` for vectorized rollouts.
- Action space: discrete (flat/long/short) initially; extend to continuous size later.
- GA loop in Python calling into Rust for fast evaluation; fitness aggregation in Python/NumPy.

## Performance Targets
- Indicator calc: precompute windows on load; incremental updates for live stepping.
- Memory: keep OHLCV + features in Arrow/Polars columns; avoid per-step heap allocs; reuse buffers.
- Throughput goal: ≥1e6 steps/min on laptop CPU; higher on training server.

## Tooling & Build
- Rust: edition 2021/2024; feature flags `python` (PyO3), `sim-only` (pure Rust CLI).
- Python: minimal deps (pandas/polars, torch, numpy); optional CUDA wheels for training host.
- Testing: Rust unit benches for indicators and reward math; property tests for PnL accounting; Python pytest for bindings.

## Milestones
1) Skeleton crates: `midas-sim` (core), `midas-env-py` (PyO3), `midas-cli` (runner).
2) Parquet loader + feature builder (Polars) with indicator set above.
3) Order simulation + constraint enforcement + per-step reward plumbing.
4) Episode metrics (Sharpe, PnL, drawdown, profit factor, win rate).
5) Python bindings + basic PPO/DQN training loop stub; GA fitness harness.
6) Benchmark pass on macOS; then on 1080 Ti box with CUDA.
7) Add more indicators/filters (ATR variants, RSI/MACD/BBands) and slippage/latency refinements.

## Open Questions / Decisions
- Final choice of lazy engine: stick with Polars or switch to DataFusion for SQL-style pipelines?
- Exact commission/slippage model for MES; need empirical values.
- Whether to include LSTM/temporal encoder vs. only cyclic time features for first pass.
- Action space granularity (discrete vs. continuous sizing) for RL v1.
