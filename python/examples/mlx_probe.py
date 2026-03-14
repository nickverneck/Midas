#!/usr/bin/env python3

import json
import platform
import sys
import time


def main() -> int:
    report = {
        "platform": platform.platform(),
        "python": sys.executable,
        "installed": False,
        "matmul_ok": False,
    }

    try:
        import mlx
        import mlx.core as mx

        report["installed"] = True
        report["version"] = getattr(mlx, "__version__", "unknown")

        lhs = mx.random.uniform(shape=(256, 256))
        rhs = mx.random.uniform(shape=(256, 256))

        started = time.perf_counter()
        out = lhs @ rhs
        mx.eval(out)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        report["matmul_ok"] = True
        report["elapsed_ms"] = round(elapsed_ms, 3)
        report["shape"] = list(out.shape)
        report["dtype"] = str(out.dtype)
    except ImportError as err:
        report["error"] = f"MLX import failed: {err}"
    except Exception as err:
        report["installed"] = report.get("installed", False)
        report["error"] = f"MLX runtime probe failed: {err}"

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
