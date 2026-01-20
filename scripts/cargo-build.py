#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            env[key] = val
    return env


def resolve_platform(env: dict[str, str]) -> str:
    raw = env.get("MIDAS_PLATFORM", "").strip().lower()
    if not raw:
        return "windows" if os.name == "nt" else "unix"
    if raw in {"windows", "win", "win32"}:
        return "windows"
    if raw in {"unix", "macos", "darwin", "linux"}:
        return "unix"
    raise SystemExit(f"Unsupported MIDAS_PLATFORM={env['MIDAS_PLATFORM']!r}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    dotenv = load_dotenv(root / ".env")

    env = os.environ.copy()
    for key, val in dotenv.items():
        env.setdefault(key, val)

    platform = resolve_platform(env)
    args = sys.argv[1:]

    if platform == "windows":
        script = root / "scripts" / "with-venv-libtorch.cmd"
        cmd = ["cmd", "/c", str(script), "cargo", "build", *args]
    else:
        script = root / "scripts" / "cargo-build.sh"
        cmd = [str(script), *args]

    if not script.exists():
        raise SystemExit(f"Missing helper script: {script}")

    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
