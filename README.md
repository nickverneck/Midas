# Midas

Rust-first backtesting and RL/GA playground for intraday futures (MES/ES).

## Setup
- Rust nightly/stable, Python 3.12+ (PyO3 built with ABI3 for 3.13 support).
- Install maturin via uv: `UV_CACHE_DIR=.uv-cache UV_TOOL_DIR=.uv-tools uv tool install maturin`
- Build/install Python bindings:  
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 UV_CACHE_DIR=.uv-cache UV_TOOL_DIR=.uv-tools uv run maturin develop --features python`

## CLI examples
- Load parquet + compute features (default feature-only mode):  
  `cargo run -- --file data/ES.parquet`
- EMA rule backtest:  
  `cargo run -- --file data/ES.parquet --mode ema_rule --ema-fast 5 --ema-slow 21 --commission 1.6 --slippage 0.25`

## Python examples
- Dump features to npz:  
  `python python/examples/feature_dump.py --parquet data/ES.parquet --out /tmp/es_features.npz`
- PPO stub (uses Globex hours, infers margin $50 MES / $500 ES):  
  `uv run --python 3.13 python python/examples/train_ppo.py --parquet data/ES.parquet --epochs 3`

## Features/obs
- SMA/EMA/HMA periods: 3,5,7,11,13,19,23,29,31,37,41,43,47,53
- ATR periods: 7,14,21
- vol_t1 (previous bar volume), time sin/cos (hour-of-day), session/margin masks, position

## Env/Actions
- Discrete actions: buy, sell, hold, revert (flip). Commission/slippage configurable; margin enforcement per contract ($50 MES default, $500 ES if symbol is ES). Session hard block when closed.

## Sampling
- Chronological window sampler exposed to Python (`list_windows`) for walk-forward GA/RL evaluation.

## Todo (short list)
- Sharpe/drawdown logging in PPO loop; config-driven session/margin per symbol.
- Window-based evaluation and checkpointing improvements.
