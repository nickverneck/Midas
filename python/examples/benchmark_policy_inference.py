#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark portable GA policy inference across torch, MLX, and NumPy."
    )
    parser.add_argument("--artifact", required=True, help="Path to *.portable.json policy export")
    parser.add_argument(
        "--backend",
        default="all",
        choices=["all", "torch", "mlx", "numpy"],
        help="Backend to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--torch-device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device when benchmarking torch",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    return parser.parse_args()


def load_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("architecture") != "mlp-tanh":
        raise ValueError(f"unsupported architecture: {payload.get('architecture')!r}")
    if not payload.get("layers"):
        raise ValueError("portable policy has no layers")
    return payload


def build_input(batch_size: int, input_dim: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [rng.uniform(-1.0, 1.0) for _ in range(input_dim)]
        for _ in range(batch_size)
    ]


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return sorted_values[low]
    frac = idx - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def summarise(name: str, times: list[float], batch_size: int, checksum: float, extra: dict[str, Any]) -> dict[str, Any]:
    times_ms = [value * 1000.0 for value in times]
    ordered = sorted(times_ms)
    mean_ms = statistics.fmean(times_ms)
    throughput = batch_size / (mean_ms / 1000.0) if mean_ms > 0 else 0.0
    result = {
        "backend": name,
        "iterations": len(times),
        "batch_size": batch_size,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "p95_ms": percentile(ordered, 0.95),
        "throughput_obs_per_sec": throughput,
        "checksum": checksum,
    }
    result.update(extra)
    return result


def print_results(results: list[dict[str, Any]], json_output: bool) -> None:
    payload = {
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "results": results,
    }
    if json_output:
        print(json.dumps(payload, indent=2))
        return

    print(json.dumps(payload["host"], indent=2))
    for result in results:
        print(
            f"{result['backend']:>5} | mean {result['mean_ms']:.3f} ms | "
            f"p95 {result['p95_ms']:.3f} ms | "
            f"{result['throughput_obs_per_sec']:.1f} obs/s | "
            f"checksum {result['checksum']:.6f}"
        )
        if device := result.get("device"):
            print(f"      device={device}")
        if error := result.get("error"):
            print(f"      error={error}")


def benchmark_numpy(policy: dict[str, Any], inputs: list[list[float]], warmup: int, iters: int) -> dict[str, Any]:
    import numpy as np

    weights = [
        np.asarray(layer["weight"], dtype=np.float32).reshape((layer["out_dim"], layer["in_dim"]))
        for layer in policy["layers"]
    ]
    biases = [
        np.asarray(layer["bias"], dtype=np.float32)
        for layer in policy["layers"]
    ]
    x = np.asarray(inputs, dtype=np.float32)

    def forward() -> Any:
        out = x
        last = len(weights) - 1
        for idx, (weight, bias) in enumerate(zip(weights, biases)):
            out = out @ weight.T + bias
            if idx != last:
                out = np.tanh(out)
        return out

    for _ in range(warmup):
        _ = forward()

    times: list[float] = []
    output = None
    for _ in range(iters):
        start = time.perf_counter()
        output = forward()
        end = time.perf_counter()
        times.append(end - start)

    checksum = float(output.mean()) if output is not None else 0.0
    return summarise("numpy", times, len(inputs), checksum, {"device": "cpu"})


def resolve_torch_device(torch: Any, requested: str) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torch CUDA is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("torch MPS is not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_synchronise(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def benchmark_torch(
    policy: dict[str, Any],
    inputs: list[list[float]],
    warmup: int,
    iters: int,
    requested_device: str,
) -> dict[str, Any]:
    import torch

    device = resolve_torch_device(torch, requested_device)
    weights = [
        torch.tensor(layer["weight"], dtype=torch.float32, device=device).reshape(
            layer["out_dim"], layer["in_dim"]
        )
        for layer in policy["layers"]
    ]
    biases = [
        torch.tensor(layer["bias"], dtype=torch.float32, device=device)
        for layer in policy["layers"]
    ]
    x = torch.tensor(inputs, dtype=torch.float32, device=device)

    @torch.no_grad()
    def forward() -> Any:
        out = x
        last = len(weights) - 1
        for idx, (weight, bias) in enumerate(zip(weights, biases)):
            out = out @ weight.T + bias
            if idx != last:
                out = torch.tanh(out)
        return out

    for _ in range(warmup):
        _ = forward()
    torch_synchronise(torch, device)

    times: list[float] = []
    output = None
    for _ in range(iters):
        start = time.perf_counter()
        output = forward()
        torch_synchronise(torch, device)
        end = time.perf_counter()
        times.append(end - start)

    checksum = float(output.mean().item()) if output is not None else 0.0
    return summarise("torch", times, len(inputs), checksum, {"device": str(device)})


def benchmark_mlx(policy: dict[str, Any], inputs: list[list[float]], warmup: int, iters: int) -> dict[str, Any]:
    import mlx.core as mx

    weights = [
        mx.array(layer["weight"], dtype=mx.float32).reshape((layer["out_dim"], layer["in_dim"]))
        for layer in policy["layers"]
    ]
    biases = [
        mx.array(layer["bias"], dtype=mx.float32)
        for layer in policy["layers"]
    ]
    x = mx.array(inputs, dtype=mx.float32)

    def forward() -> Any:
        out = x
        last = len(weights) - 1
        for idx, (weight, bias) in enumerate(zip(weights, biases)):
            out = out @ weight.T + bias
            if idx != last:
                out = mx.tanh(out)
        return out

    for _ in range(warmup):
        y = forward()
        mx.eval(y)

    times: list[float] = []
    output = None
    for _ in range(iters):
        start = time.perf_counter()
        output = forward()
        mx.eval(output)
        end = time.perf_counter()
        times.append(end - start)

    checksum = float(output.mean().item()) if output is not None else 0.0
    return summarise("mlx", times, len(inputs), checksum, {"device": "mlx-default"})


def main() -> int:
    args = parse_args()
    artifact = Path(args.artifact)
    policy = load_policy(artifact)
    inputs = build_input(args.batch_size, int(policy["input_dim"]), args.seed)

    requested_backends = ["torch", "mlx", "numpy"] if args.backend == "all" else [args.backend]
    results: list[dict[str, Any]] = []

    for backend in requested_backends:
        try:
            if backend == "torch":
                result = benchmark_torch(policy, inputs, args.warmup, args.iters, args.torch_device)
            elif backend == "mlx":
                result = benchmark_mlx(policy, inputs, args.warmup, args.iters)
            else:
                result = benchmark_numpy(policy, inputs, args.warmup, args.iters)
        except Exception as exc:  # pragma: no cover - benchmark fallback reporting
            result = {
                "backend": backend,
                "artifact": str(artifact),
                "error": str(exc),
            }
        results.append(result)

    print_results(results, args.json)
    return 0 if any("mean_ms" in result for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
