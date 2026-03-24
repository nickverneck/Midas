# NinjaTrader TUI

Terminal client for the Tradovate/NinjaTrader API, built with `ratatui`.

Current phase:
- Login using either a bearer-token file or API credentials.
- Override the token directly inside the login screen when you want to ignore the token file.
- Use a dedicated selection screen for account and instrument choice immediately after login.
- Use a dedicated strategy screen after instrument selection.
- Choose `Native Rust`, `Lua`, or `Machine Learning`.
- For Lua, either load a script from a file or type it directly in a vim-style editor.
- Toggle `sim` vs `live`.
- Discover and select accounts.
- Search/select contracts by text using `contract/suggest`.
- Load 1-minute history plus live updates with `md/getChart`.
- Show account-focused stats from `account`, `accountRiskStatus`, `cashBalance`, and `position` sync data.
- Keep manual-trading hotkeys and automation backends scaffolded, but not enabled yet.

Run from the repo root:

```sh
cargo run --manifest-path ninjatrader-tui/Cargo.toml -- --config ninjatrader-tui/config.example.toml
```

Engine management on Linux and macOS:

```sh
# List running engine processes and their sockets.
ninjatrader-tui list

# Attach a full TUI session to a running engine by ID from `list`.
ninjatrader-tui attach 1455

# Kill one engine immediately.
ninjatrader-tui kill 1455

# Disarm, close the selected market, then kill one engine.
ninjatrader-tui kill 1455 -c

# Kill every running engine immediately.
ninjatrader-tui killall

# Disarm, close the selected market on each engine, then kill them.
ninjatrader-tui killall -c

# Show the generated clap help, including all engine commands.
ninjatrader-tui --help
ninjatrader-tui kill --help
ninjatrader-tui killall --help
```

Notes:
- `list`, `attach`, `kill`, and `killall` work on Linux and macOS. Other platforms are not supported yet.
- `attach` reuses the existing engine socket and does not spawn a new engine.
- The attached TUI remains fully interactive, including strategy changes and other controls.
- `kill -c` and `killall -c` disarm the selected strategy, send a close on the selected market, wait for that selected market to go flat, and only then kill the engine process.

Key controls:
- `F1`: login screen
- `F2`: selection screen
- `F3`: strategy screen
- `F4`: dashboard
- `Up` / `Down`: move between login fields or through account/contract lists
- `Left` / `Right`: change login toggles like environment/auth mode and switch the selection screen bar type between `1 Min` and `1 Range`
- `Tab` / `Shift+Tab`: cycle strategy/selection focus
- `Enter`: trigger focused action
- `q`: quit

Lua editor:
- Normal mode: `h` `j` `k` `l` move, `i` insert, `a` append, `o` open line, `x` delete
- Insert mode: type text, `Enter` newline, `Backspace` delete, `Esc` back to normal

Suggested config flow:
1. Start with token-file mode and point `token_path` at `.auth/bearer-token.json` if you already have a valid session token.
2. Switch to credentials mode if you want the TUI to request and cache access tokens directly.
3. Select the account and instrument you want to watch or trade.
4. Pick the strategy backend and, for Lua, either load a file or type it directly in the editor.

Live-order note:
- Strategy orders do not send `customTag50` by default.
- If your broker setup requires a registered Tag 50 value, set `custom_tag50` in config or `MIDAS_TUI_CUSTOM_TAG50` in the environment.
