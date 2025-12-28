# Midas

Rust-first backtesting and RL/GA playground for intraday trading (Stocks/Futures).

## Setup
- Rust nightly/stable, Python 3.12+ (Built and tested on 3.13).
- Install `uv` for Python management if not already installed.
- Build and install Python bindings:  
  `uv run maturin develop --features python`

## CLI examples (Rust)
- Load parquet + compute features (default feature-only mode):  
  `cargo run -- --file data/train/SPY0.parquet`
- EMA rule backtest:  
  `cargo run -- --file data/train/SPY0.parquet --mode ema_rule --ema-fast 5 --ema-slow 21 --commission 1.6 --slippage 0.25`

## Python examples

### Genetic Algorithm + PPO Training
Run the hybrid GA-PPO trainer with parallel evaluation:
```bash
uv run python/examples/train_hybrid.py \
  --train-parquet data/train/SPY0.parquet \
  --val-parquet data/val/SPY.parquet \
  --test-parquet data/val/SPY.parquet \
  --outdir runs_ga \
  --workers 4 \
  --pop-size 12
```

### GA-Only Neuroevolution
Run GA-only training that evolves policy weights directly:
```bash
uv run python/examples/train_ga.py \
  --train-parquet data/train/SPY0.parquet \
  --val-parquet data/val/SPY.parquet \
  --test-parquet data/val/SPY.parquet \
  --outdir runs_ga \
  --workers 4 \
  --pop-size 12
```
Notes:
- GA selection fitness uses `train-parquet`; validation metrics use `val-parquet`.
- Add `--skip-val-eval` to skip validation during GA for faster iterations.

### PPO Only Training
`uv run python/examples/train_ppo.py --parquet data/train/SPY0.parquet --epochs 3`

## Environment & Observations
- **Initial Balance**: Configurable starting cash (default $10,000).
- **Observations**:
  - `open[t]` (Current bar entry price)
  - `close[t-1]`, `volume[t-1]`
  - Real-time `equity` (Cash + Unrealized PnL)
  - Indicators (SMA/EMA/HMA periods: 3,5,7,11,13,19,23,29,31,37,41,43,47,53)
  - ATR (periods: 7,14,21)
  - Time encoding (sin/cos of hour), Session/Margin masks, Current Position.

## Env/Actions
- **Discrete actions**: buy, sell, hold, revert (flip).
- **Core Logic**: Commission/slippage configurable; margin enforcement per contract.
- **Parallel Workers**: Training can be parallelized across CPU/GPU cores using the `--workers` flag.

## Development
To recompile the Rust environment for Python:
`uv run maturin develop --features python`

## Artifacts
- **Logs**: `runs_ga/ga_log.csv` contains performance metrics for every individual.
- **Traces**: `runs_ga/trace_gen{G}_rank{R}.csv` contains step-by-step action history for top performers.
- **Weights**: `.pt` files for the neural networks are saved for top performers each generation.
