# ML Backend Rollout

This branch establishes a single backend-selection surface for both GA and RL training so we can benchmark different tensor/runtime stacks without rewriting the surrounding training workflow each time.

## Current state

- `libtorch` is implemented for GA and RL.
- `candle` is implemented for GA training and RL PPO/GRPO training in this branch. It currently targets CPU by default, exports `.safetensors`, and keeps CUDA behind the optional `backend-candle-cuda` Cargo feature.
- `burn` is implemented for GA training in this branch. It currently targets CPU via `burn-cpu`, can target Linux CUDA with the optional `backend-burn-cuda` Cargo feature, can target Apple GPU through `burn-mlx` with the optional `backend-burn-mlx` Cargo feature, and can still compile the legacy `burn-ndarray` CPU path behind `backend-burn-ndarray`.
- `mlx` remains a separate first-class CLI/UI option, but it still intentionally fails fast until a dedicated runner exists.
- GA orchestration now calls through a backend runner boundary in `src/bin/train_ga/backends/` instead of reaching directly into the `tch` policy code. RL now has a matching Candle runner in `src/bin/train_rl/candle.rs` for PPO and GRPO.
- Every successful training run now writes `training_stack.json` into the run directory so benchmark scripts can compare backend/runtime/algorithm combinations later.
- `python/examples/mlx_probe.py` is wired into the frontend diagnostics flow so Mac MLX viability can be checked before a full MLX trainer exists.

## Shared contract

Both `train_ga` and `train_rl` now accept:

```bash
--backend libtorch|burn|candle|mlx
--device auto|cpu|cuda|mps
```

Current runtime policy:

- `auto` prefers `cuda`, then `mps`, then `cpu`.
- `libtorch` resolves to the effective runtime at startup and records it in `training_stack.json`.
- `candle` resolves to `cpu` by default, rejects `mps` explicitly, and can target `cuda` when compiled with `backend-candle-cuda`.
- `burn` resolves to `cpu` by default, can target `cuda` when compiled with `backend-burn-cuda`, and can target `mps` through `burn-mlx` when compiled with `backend-burn-mlx`.
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

1. Mirror the Burn GA seam into RL.
2. Add an `mlx` runner path after the probe work proves the Mac path is worth it. Treat it as a separate runtime adapter, not a drop-in replacement for `tch`.
3. Add a benchmark harness that iterates over:
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
