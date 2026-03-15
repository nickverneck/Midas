# Midas

Rust-first backtesting and RL/GA playground for intraday trading (Stocks/Futures).

## Setup
- Rust nightly/stable, Python 3.12+ (Built and tested on 3.13).
- Install `uv` for Python management if not already installed.
- Build and install Python bindings:  
  `uv run maturin develop --features python`
- Build Rust with torch from the local uv venv:  
  `python scripts/cargo-build.py`
  - Set `MIDAS_PLATFORM=windows` or `MIDAS_PLATFORM=unix` in `.env` to pick the helper script.

## Windows CUDA notes (tch-rs)
- Ensure the venv CUDA build is used: set `LIBTORCH_USE_PYTORCH=1` and avoid pointing `LIBTORCH` at a CPU-only libtorch.
- Match the CUDA wheel version at build time, e.g. `TORCH_CUDA_VERSION=cu126`.
- `LIBTORCH_BYPASS_VERSION_CHECK=1` is set by the helper scripts and frontend runner.
- If CUDA is still unavailable, confirm the NVIDIA driver is installed (`nvcuda.dll` in `C:\Windows\System32`) and the venv `torch\lib` directory is on `PATH`.

## ML backend selection
- Both Rust trainers now accept `--backend libtorch|burn|candle|mlx` and `--device auto|cpu|cuda|mps`.
- `libtorch` is implemented for both Rust trainers today.
- `candle` now runs both the GA trainer and the RL PPO/GRPO trainer in this branch, saving `.safetensors` checkpoints.
- `burn` now runs the GA trainer in this branch. The current runtime split is Burn CPU via `burn-cpu`, native Burn CUDA via the optional `backend-burn-cuda` Cargo feature, and Apple GPU via `burn-mlx` with the optional `backend-burn-mlx` Cargo feature.
- `mlx` is still a separate planned backend slot rather than the Burn Apple GPU path.
- Successful runs write `training_stack.json` beside the log files so benchmark tooling can group results by backend/runtime/algorithm/host.
- The training page diagnostics now run both the existing libtorch probe and `python/examples/mlx_probe.py`, so Apple MLX viability can be checked before a full MLX trainer exists.
- Candle frontend runs now compile with `backend-candle` automatically, add `backend-candle-accelerate` on macOS unless `MIDAS_CANDLE_ACCELERATE=0`, and can opt into CUDA on Linux with `MIDAS_CANDLE_CUDA=1`.
- Burn frontend runs always compile with `backend-burn`, add `backend-burn-cuda` when `MIDAS_BURN_CUDA=1`, add `backend-burn-mlx` only when you explicitly target `mps` or opt into it with `MIDAS_BURN_MLX=1`, and add `backend-burn-ndarray` only when `MIDAS_BURN_NDARRAY=1` or `MIDAS_BURN_CPU_BACKEND=ndarray`.
- `burn-mlx` currently needs both `cmake` and an active Xcode Metal Toolchain. On this machine the MLX source build progressed after installing `cmake`, but `xcrun metal` still reports the Metal toolchain as unavailable.
- Rollout details live in [`docs/ml_backend_rollout.md`](docs/ml_backend_rollout.md).

## CLI examples (Rust)
- Load parquet + compute features (default feature-only mode):  
  `cargo run -- --file data/train/SPY0.parquet`
- EMA rule backtest:  
  `cargo run -- --file data/train/SPY0.parquet --mode ema_rule --ema-fast 5 --ema-slow 21 --commission 1.6 --slippage 0.25`
- Rust GA-only trainer (requires libtorch, CUDA/MPS optional):  
  `LIBTORCH=/path/to/libtorch cargo run --features torch --bin train_ga -- --backend libtorch --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga --device cuda --workers 8 --drawdown-penalty 0.05 --drawdown-penalty-growth 0.02`
- Rust GA-only trainer on Candle CPU:  
  `cargo run --features backend-candle --bin train_ga -- --backend candle --device cpu --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga_candle`
- Rust RL trainer on Candle CPU (PPO):  
  `cargo run --features backend-candle --bin train_rl -- --backend candle --device cpu --algorithm ppo --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --test-parquet data/val/SPY.parquet --outdir runs_rl_candle`
- Rust GA-only trainer on Burn CPU:  
  `cargo run --features backend-burn --bin train_ga -- --backend burn --device cpu --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga_burn_cpu`
- Rust GA-only trainer on Burn legacy ndarray CPU:  
  `MIDAS_BURN_CPU_BACKEND=ndarray cargo run --features backend-burn,backend-burn-ndarray --bin train_ga -- --backend burn --device cpu --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga_burn_ndarray`
- Rust GA-only trainer on Burn CUDA (Linux box):  
  `cargo run --features backend-burn,backend-burn-cuda --bin train_ga -- --backend burn --device cuda --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga_burn_cuda`
- Rust GA-only trainer on Burn MLX (macOS Apple GPU, toolchain required):  
  `cargo run --features backend-burn,backend-burn-mlx --bin train_ga -- --backend burn --device mps --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga_burn_mlx`

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
  - Indicators (SMA/EMA/HMA periods: 3,5,7,11,13,19,23,29,31,37,41,43,47,53,100,150,200,250,300)
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
- **Run folders**: default outdirs create timestamped subfolders like `runs_ga/20260124_153045` (and `runs_rl/...`) to avoid overwriting.
- **Logs**: `runs_ga/<timestamp>/ga_log.csv` contains performance metrics for every individual.
- **Traces**: `runs_ga/<timestamp>/trace_gen{G}_rank{R}.csv` contains step-by-step action history for top performers.
- **Weights**: `libtorch` saves `.pt`; Candle GA/RL saves `.safetensors`.
- **Portable artifacts**: Candle and Burn GA also write backend-neutral JSON policy exports for cross-runtime comparison work.
