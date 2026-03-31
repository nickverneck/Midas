# Trader

Terminal client for Tradovate and Ironbeam, built with `ratatui`.

Current broker coverage:
- Tradovate: token-file auth, credential auth, account/instrument selection, live market data, and local replay mode.
- Ironbeam: token-file auth, credential auth, account/symbol selection, and live 1-minute websocket bars.
- When the build includes both brokers, the app opens on a broker selection screen before login.

Build options:

```sh
# Default build: both brokers enabled.
cargo run --manifest-path trader/Cargo.toml -- --config trader/config.example.toml

# Tradovate only, with replay support.
cargo run --manifest-path trader/Cargo.toml --no-default-features --features tradovate,replay -- --config trader/config.example.toml

# Ironbeam only.
cargo run --manifest-path trader/Cargo.toml --no-default-features --features ironbeam -- --config trader/config.example.toml
```

Engine management on Linux and macOS:

```sh
trader list
trader attach 1455
trader kill 1455
trader kill 1455 -c
trader killall
trader killall -c
trader --help
trader kill --help
trader killall --help
```

Notes:
- `list`, `attach`, `kill`, and `killall` work on Linux and macOS.
- `attach` reuses the existing engine socket and does not spawn a new engine.
- `kill -c` and `killall -c` disarm the selected strategy, send a close on the selected market, wait for that selected market to go flat, and only then kill the engine process.
- `TRADER_*` environment variables are preferred; legacy `MIDAS_TUI_*` variables still fall back where supported.

Suggested config flow:
1. Pick `broker = "tradovate"` or `broker = "ironbeam"` for single-broker builds. On dual-broker builds you can still switch from the broker selection screen.
2. Start with token-file mode if you already have a valid bearer token, or switch to credentials mode to request and cache a new token.
3. Select the account and instrument you want to watch or trade.
4. Continue to strategy setup, then open the dashboard.
