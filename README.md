# Midas

Rust-first backtesting and RL/GA playground for intraday trading (Stocks/Futures).

## Setup

**Python is deprecated.** All training, backtesting, and tooling has been migrated to Rust. The Python dependency exists only for legacy bindings and build scaffolding; new work should be done in Rust.

- Rust nightly/stable.
- Python 3.12+ (Built and tested on 3.13) is only needed to build the PyO3 bindings:  
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
- `burn` now runs GA, RL (manual PPO/GRPO), and supervised event classification in this branch. Burn GA, the CPU RL inference path, and supervised CPU use `burn-ndarray` for reliable small-matrix execution. Native Burn CUDA is enabled with the optional `backend-burn-cuda` Cargo feature, and Apple GPU via `burn-mlx` with the optional `backend-burn-mlx` Cargo feature.
- `mlx` is still a separate planned backend slot rather than the Burn Apple GPU path.
- Successful runs write `training_stack.json` beside the log files so benchmark tooling can group results by backend/runtime/algorithm/host.
- `python/examples/mlx_probe.py` remains as a Python-side MLX runtime probe. It is the only remaining Python example in active use.
- Candle frontend runs now compile with `backend-candle` automatically, add `backend-candle-accelerate` on macOS unless `MIDAS_CANDLE_ACCELERATE=0`, and add `backend-candle-cuda` when CUDA is selected or `MIDAS_CANDLE_CUDA=1` is set.
- Burn frontend runs always compile with `backend-burn`, add `backend-burn-cuda` when CUDA is selected or `MIDAS_BURN_CUDA=1` is set, and add `backend-burn-mlx` when you explicitly target `mps` or opt into it with `MIDAS_BURN_MLX=1`.
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
- Rust GA-only trainer on deterministic Burn CPU (`burn-ndarray`):
  `cargo run --features backend-burn --bin train_ga -- --backend burn --device cpu --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga_burn_cpu`
- Rust GA-only trainer on Burn CUDA (Linux box):  
  `CUDARC_CUDA_VERSION=13000 cargo run --features backend-burn,backend-burn-cuda --bin train_ga -- --backend burn --device cuda --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga_burn_cuda`
- Rust GA-only trainer on Burn MLX (macOS Apple GPU, toolchain required):  
  `cargo run --features backend-burn,backend-burn-mlx --bin train_ga -- --backend burn --device mps --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --outdir runs_ga_burn_mlx`
- Rust RL trainer on Burn CPU (manual PPO):
  `cargo run --features backend-burn --bin train_rl -- --backend burn --device cpu --algorithm ppo --train-parquet data/train/SPY0.parquet --val-parquet data/val/SPY.parquet --test-parquet data/val/SPY.parquet --outdir runs_rl_burn`
- Supervised trainer on Burn CPU:
  `cargo run --features backend-burn --bin supervised -- train --backend burn --device cpu --input .run/supervised/datasets/example.parquet --outdir .run/supervised/runs/burn-cpu`
- Supervised long-run checkpoints:
  add `--checkpoint-every 1000`; resumable policies and train/validation metrics are recorded at each checkpoint while holdout is reserved for the final policy. Burn/Candle resume artifacts preserve deterministic initialization and AdamW state.

## Python examples (deprecated)

The following Python examples have been removed. Their functionality is fully replaced by Rust binaries:

- `python/examples/train_ga.py` — replaced by `train_ga` (Rust).
- `python/examples/train_ppo.py` — replaced by `train_rl` (Rust).
- `python/examples/train_hybrid.py` — replaced by `train_rl` / `train_ga` (Rust).
- `python/examples/check_mps.py` — replaced by `tch_mps_check` (Rust).

The remaining Python files serve limited purposes:
- `python/examples/mlx_probe.py` — MLX runtime probe (kept as-is).
- `python/examples/benchmark_policy_inference.py` — cross-backend inference benchmark (kept as-is).
- `python/examples/feature_dump.py` — feature computation (kept; no Rust CLI replacement yet).
- `python/examples/train_stub.py` — minimal teaching stub (kept as documentation).
- `scripts/cargo-build.py` — build utility (kept).

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
To recompile the Rust environment for Python bindings (only needed when adding new PyO3 interfaces):
`uv run maturin develop --features python`

## Artifacts
- **Run folders**: default outdirs create timestamped subfolders like `runs_ga/20260124_153045` (and `runs_rl/...`) to avoid overwriting.
- **Logs**: `runs_ga/<timestamp>/ga_log.csv` contains performance metrics for every individual.
- **Traces**: `runs_ga/<timestamp>/trace_gen{G}_rank{R}.csv` contains step-by-step action history for top performers.
- **Weights**: `libtorch` saves `.pt`; Candle GA/RL saves `.safetensors`.
- **Portable artifacts**: Candle and Burn GA also write backend-neutral JSON policy exports for cross-runtime comparison work.
