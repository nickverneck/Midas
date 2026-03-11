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

Key controls:
- `F1`: login screen
- `F2`: selection screen
- `F3`: strategy screen
- `F4`: dashboard
- `Up` / `Down`: move between login fields or through account/contract lists
- `Left` / `Right`: change login toggles like environment and auth mode
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
