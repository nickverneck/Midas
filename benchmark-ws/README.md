# Strategy Execution Engine (NinjaScript-style)

This crate now runs a bar-by-bar execution engine for a converted
`UptrickZeroHMAAngleFiltertrailing`-style strategy using existing TA utilities
from the main `midas_env` crate (`ATR`, `WMA`).

Modes:
- `backtest`: parquet-only execution (local).
- `sim`: Tradovate demo session execution (bearer token + websocket market data + REST orders).
- `live`: same strategy and credentials flow, but live Tradovate REST/session URLs.

## What it supports
- Zero-lag HMA (WMA-based) signal line.
- ATR-normalized angle filter.
- `CrossAbove`/`CrossBelow` entry logic.
- Optional longs-only mode.
- NY session time filter.
- Fixed dollar take-profit / stop-loss.
- Tick-based and bar-based trailing stops.
- TOML config + environment variable overrides.
- Explicit ticker setting in config/env.

## Run
From repo root:

```sh
cargo run --release --manifest-path benchmark-ws/Cargo.toml -- --config benchmark-ws/strategy.example.toml
```

Optional overrides:

```sh
cargo run --release --manifest-path benchmark-ws/Cargo.toml -- --config benchmark-ws/strategy.example.toml --ticker MES --data-file data/train/MES.parquet
```

## Realtime modes (Tradovate sim/live)
1. Set `mode = "sim"` or `mode = "live"` in `benchmark-ws/strategy.example.toml` (or `MIDAS_EXEC_MODE`).
2. For `mode = "live"`, you must explicitly confirm with one of:
   - `confirm_live = true` in TOML
   - `--confirm-live` on CLI
   - `MIDAS_EXEC_CONFIRM_LIVE=true`
3. Ensure bearer token exists at `.auth/bearer-token.json` (or override `sim_token_path`).
4. Set `sim_contract` to the Tradovate contract symbol (example: `MESH6`).
5. Start with `sim_dry_run = true` to validate signal flow without sending orders.
6. Keep credentials/token path the same, switch URLs by mode:
   - `sim_rest_url` (demo)
   - `live_rest_url` (live)
   - market-data ws defaults to `wss://md.tradovateapi.com/v1/websocket`

Run:

```sh
cargo run --release --manifest-path benchmark-ws/Cargo.toml -- --config benchmark-ws/strategy.example.toml
```

## Config
- TOML example: `benchmark-ws/strategy.example.toml`
- ENV example: `benchmark-ws/.env.example`

Load env values by copying `.env.example` to `.env` (or exporting variables)
and using prefix `MIDAS_EXEC_`.
