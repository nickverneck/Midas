# ML Backend Rollout

This branch establishes a single backend-selection surface for GA, RL, and supervised training so we can benchmark different tensor/runtime stacks without rewriting the surrounding training workflow each time.

## Current state

- `libtorch` is implemented for GA and RL.
- `candle` is implemented for GA training and RL PPO/GRPO training in this branch. It targets CPU by default and native CUDA when built with the optional `backend-candle-cuda` Cargo feature. Candle Metal is not wired into the training binaries.
- `burn` is implemented for GA, RL PPO/GRPO, and supervised event classification. GA, the CPU RL inference path, and supervised CPU use deterministic `burn-ndarray` for reliable small-matrix execution; there is no separate CPU backend selector. Linux CUDA is available with the optional `backend-burn-cuda` Cargo feature, and Apple GPU through `burn-mlx` with the optional `backend-burn-mlx` Cargo feature.
- On Linux hosts without `nvcc`, Cargo reads the project default from
  `.cargo/config.toml`. This checkout uses `12080` (CUDA 12.8 API), which is
  compatible with the local 580.142 driver and avoids cudarc requesting the
  CUDA 13.1-only `cuDevSmResourceSplit` symbol. Override it for another GPU
  or toolkit with `CUDARC_CUDA_VERSION=13000 cargo build ...`; the project
  setting uses `force = false`, so an explicit environment value wins.
- The supervised CLI exposes the deterministic `cpu-linear` reference plus parity-preserving Candle and Burn linear classifiers. Candle can run on CPU or native CUDA when the optional `backend-candle-cuda` feature is compiled; Burn uses autodiff over burn-ndarray on CPU or native Burn CUDA.
- `mlx` remains a separate first-class CLI/UI option, but it still intentionally fails fast until a dedicated runner exists.
- GA orchestration now calls through a backend runner boundary in `src/bin/train_ga/backends/` instead of reaching directly into the `tch` policy code. RL now has matching Candle and Burn runners in `src/bin/train_rl/candle.rs` and `src/bin/train_rl/burn.rs` for PPO and GRPO.
- Every successful training run now writes `training_stack.json` into the run directory so benchmark scripts can compare backend/runtime/algorithm combinations later.
- `python/examples/mlx_probe.py` is wired into the frontend diagnostics flow so Mac MLX viability can be checked before a full MLX trainer exists. (Only remaining Python example in active use.)

## Shared contract

Both `train_ga` and `train_rl` now accept:

```bash
--backend libtorch|burn|candle|mlx
--device auto|cpu|cuda|mps
```

Current runtime policy:

- `auto` prefers `cuda`, then `mps`, then `cpu`.
- `libtorch` resolves to the effective runtime at startup and records it in `training_stack.json`.
- `candle` resolves to `cpu` by default, rejects `mps` explicitly, and can target `cuda` when compiled with `backend-candle-cuda`. An explicit CUDA request fails before data loading when that feature is absent.
- `burn` probes a compiled CUDA backend for `auto`, falls back to deterministic `burn-ndarray` CPU when no usable device is present, and can target `mps` through `burn-mlx` on macOS. An explicit CUDA/Metal request fails before data loading when the corresponding feature or usable runtime is absent.
- `mlx` still reserves the same interface for a later dedicated runner.

## Machine strategy

- Dev machine (macOS): use `burn --device cpu` for the new Burn CPU path, `burn --device mps` with `backend-burn-mlx` for Apple GPU viability, `libtorch --device auto` for the existing MPS path, and `candle --device cpu` for a second Rust-native CPU comparison.
- Training machine (Linux + NVIDIA): use `burn --device cuda` with `backend-burn,backend-burn-cuda` for native Burn CUDA, then compare that against `libtorch --device cuda` and the Candle CUDA path.
- Inference machine (lightweight CPU): prioritize `--device cpu`; Candle writes `.safetensors` for both GA and RL, while Burn GA writes portable JSON policy artifacts today.

## Toolchain notes

- `burn-mlx` is sourced from the Burn 0.20 compatibility branch of `eidolons-ai/burn-mlx`.
- On this machine, `burn-mlx` required `cmake` before it could start building the MLX source tree.
- After installing `cmake`, the local build progressed into Apple's Metal compilation step, where `xcrun metal` still reported a missing Metal Toolchain. Burn CPU and Burn CUDA builds do not require either of those Apple-specific prerequisites.

## Recommended implementation order

1. Add an `mlx` runner path after the probe work proves the Mac path is worth it. Treat it as a separate runtime adapter, not a drop-in replacement for `tch`.
2. Add a benchmark harness that iterates over:
   - trainer: `ga`, `rl`
   - backend: `libtorch`, `burn`, `candle`, `mlx`
   - device: `cpu`, `mps`, `cuda`
   - dataset split / window config

## Benchmark output

Use `training_stack.json` with the existing logs (`ga_log.csv`, `rl_log.csv`) to group benchmark results by:

- trainer
- backend
- requested runtime
- effective runtime
- algorithm
- host OS / arch

That keeps the benchmark aggregation separate from whichever backend implementation details we add next.
