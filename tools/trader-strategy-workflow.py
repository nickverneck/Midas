#!/usr/bin/env python3
"""Fixed-command adapter for the Trader native offline replay-sweep CLI.

This is a local process adapter, not a broker client or HTTP server. It accepts
only validate/run/rank operations and never forwards caller-supplied arguments
to a shell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


RANK_METRICS = {
    "robustness", "net-pnl", "gross-pnl", "fees", "max-drawdown",
    "required-account-size", "return-required", "profit-factor", "win-rate",
    "trade-count", "average-trade", "average-giveback", "largest-giveback",
    "mfe-capture",
}
EXECUTION_MODES = {"isolated_services", "batch_cpu", "prepared_cpu"}


def existing_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a file: {resolved}")
    return resolved


def trader_binary(repo: Path) -> Path:
    binary = (repo / "trader" / "target" / "release" / "trader").resolve()
    if not binary.is_file():
        raise ValueError(f"Trader release binary not found: {binary}; build trader with --features replay")
    return binary


def base_command(binary: Path, config: Path | None) -> list[str]:
    command = [str(binary)]
    if config is not None:
        command += ["--config", str(existing_file(config, "config"))]
    return command


def invoke(command: list[str], timeout: int) -> dict[str, object]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    return {"ok": result.returncode == 0, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr, "argv": command}


def run(args: argparse.Namespace) -> int:
    repo = args.repo.resolve(strict=True)
    binary = trader_binary(repo)
    prefix = base_command(binary, args.config)
    if args.operation == "rank":
        summary = existing_file(args.summary, "summary")
        if args.metric not in RANK_METRICS:
            raise ValueError(f"unsupported ranking metric: {args.metric}")
        command = prefix + ["rank-replay-sweep", "--summary", str(summary), "--metric", args.metric,
                            "--limit", str(args.limit)]
        if args.plan:
            command += ["--plan", str(existing_file(args.plan, "plan"))]
        if args.minimum_trades:
            command += ["--min-closed-trades", str(args.minimum_trades)]
        if args.max_drawdown_pct is not None:
            command += ["--max-drawdown-pct", str(args.max_drawdown_pct)]
        if args.output:
            command += ["--output", str(args.output.resolve())]
        result = invoke(command, args.timeout)
        print(json.dumps({"operation": "rank", **result}, indent=2))
        return int(result["exit_code"])
    spec = existing_file(args.spec, "spec")
    validate_result = invoke(prefix + ["validate-replay-sweep", "--spec", str(spec)], args.timeout)
    if not validate_result["ok"]:
        print(json.dumps({"operation": "validate", **validate_result}, indent=2))
        return int(validate_result["exit_code"])
    if args.operation == "validate":
        print(json.dumps({"operation": "validate", **validate_result}, indent=2))
        return 0
    if args.operation == "run":
        if args.execution_mode is not None and args.execution_mode not in EXECUTION_MODES:
            raise ValueError(f"unsupported execution mode: {args.execution_mode}")
        command = prefix + ["run-replay-sweep", "--spec", str(spec)]
        if args.execution_mode:
            command += ["--execution-mode", args.execution_mode]
        result = invoke(command, args.timeout)
        print(json.dumps({"operation": "run", "validation": validate_result, **result}, indent=2))
        return int(result["exit_code"])
    raise AssertionError("unreachable operation")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("validate", "run", "rank"))
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--config", type=Path, help="Trader config used for replay-cache location")
    p.add_argument("--spec", type=Path, help="durable native replay-sweep JSON spec")
    p.add_argument("--timeout", type=int, default=1800, help="wall-clock bound in seconds (1..3600)")
    p.add_argument("--execution-mode", choices=sorted(EXECUTION_MODES))
    p.add_argument("--summary", type=Path, help="required for rank")
    p.add_argument("--plan", type=Path)
    p.add_argument("--metric", default="robustness")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--minimum-trades", type=int, default=0)
    p.add_argument("--max-drawdown-pct", type=float)
    p.add_argument("--output", type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    if not 1 <= args.timeout <= 3600:
        parser().error("--timeout must be between 1 and 3600 seconds")
    if not 1 <= args.limit <= 1000:
        parser().error("--limit must be between 1 and 1000")
    if not 0 <= args.minimum_trades <= 1_000_000:
        parser().error("--minimum-trades must be between 0 and 1000000")
    if args.operation == "rank" and args.summary is None:
        parser().error("rank requires --summary")
    if args.operation != "rank" and args.spec is None:
        parser().error("validate and run require --spec")
    if args.operation != "rank" and any((args.summary, args.plan, args.output)):
        parser().error("--summary, --plan, and --output are rank-only options")
    if args.operation != "run" and args.execution_mode is not None:
        parser().error("--execution-mode is run-only")
    if args.operation == "rank" and args.execution_mode is not None:
        parser().error("--execution-mode is run-only")
    try:
        return run(args)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
