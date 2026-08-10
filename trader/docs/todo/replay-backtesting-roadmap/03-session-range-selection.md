# Task Group 03: Session and Range Selection

## Slice Status

Implemented first slice (RBT-020):

- Dataset views are versioned JSON documents stored under `<cache-root>/.views/`; source Parquet/JSONL files and their manifests are not copied or rewritten.
- A view persists a cache-root-relative source manifest ID plus provider, environment, instrument, and contract identity. Re-running fails closed if the path escapes the cache root, the manifest identity changes, or the requested range no longer has source coverage.
- Evaluation timestamps use half-open UTC semantics (`[start, end)`), while the IANA timezone used for user input and the selected session preset are retained in the document.
- Warmup is currently expressed as a duration before `evaluation_start`. The complete warmup-plus-evaluation range must exist in the source cache.
- The default and currently supported warmup execution policy is `flat_until_evaluation`. Warmup rows seed the replay series without emitting a strategy-triggering market update; the first emitted market event belongs to the evaluation window. Carrying a warmup position is explicitly rejected until that opt-in behavior is implemented.
- Replay startup accepts an optional saved view path. Server-bar reads use exact range filtering; raw-tick views carry the same range into bounded Parquet streaming.

Implemented second slice (RBT-021):

- Strongly typed selection inputs now resolve full-source, one-trading-date Globex, New York RTH, Chicago RTH, custom-local, and custom-UTC presets into exact half-open UTC ranges.
- Globex uses the same New York clock as the live execution gate: 18:00 on the preceding calendar date through 17:00 on the trading date. A Monday session therefore starts Sunday evening, and Friday ends before the weekend.
- New York RTH resolves 09:30-16:00 local; Chicago RTH resolves 08:30-15:00 local. Both retain their IANA input timezone in the saved view.
- IANA timezone conversion follows DST. Ambiguous fall-back inputs and nonexistent spring-forward inputs fail closed and direct the caller to choose another local time or use UTC.
- Display labels include the local timezone abbreviation on both boundaries, such as EDT/EST or CDT/CST.
- Futures presets accept Monday-Friday trading dates only. Static presets deliberately do not claim exchange-holiday or early-close awareness.

Implemented third slice (RBT-022):

- View-backed replay state carries a serializable window snapshot containing the preset, input timezone, warmup/evaluation UTC boundaries, warmup rows, total evaluation rows, and processed evaluation rows.
- Rows before `evaluation_start` are inserted into the indicator series but are never sent to the simulated broker and do not emit an execution-triggering market update. The simulated broker, strategy execution, fills, trade markers, and PnL therefore begin with the evaluation slice.
- The first evaluation update contains the warmed indicator series and progress starts at evaluation row 1. A view with warmup data but zero evaluation rows fails closed.
- Replay status and the dashboard distinguish `warmup N` from `evaluation processed/total` and display the evaluation range with timezone abbreviations.
- Persisted `.run/trader-logs` include the preset, timezone, all three UTC boundaries, the local labeled range, and final warmup/evaluation row counters.
- Carrying a position from warmup remains rejected. The currently supported policy is flat until evaluation.

Implemented fourth slice (RBT-023):

- The Replay dataset picker exposes a dedicated saved-view manager with `V`. Views are listed only for the selected source manifest; malformed, stale, or identity-mismatched view files are excluded with warnings.
- The TUI can create and edit all Phase 3 presets: full source, Globex, New York RTH, Chicago RTH, custom local time with an IANA timezone, and custom UTC.
- View IDs, dates, timestamps, timezones, warmup duration, source coverage, and the flat-until-evaluation policy are validated by the shared view model before an atomic save. A failed validation leaves the editor open and does not alter the selected view.
- Saving selects the view, `Enter` selects an existing view, and `C` explicitly returns to full-source coverage. Replay startup now passes both the selected manifest and view path so the loader fails closed if their source identities differ.
- The Replay screen labels whether startup will use full source coverage or a saved view. Automatic cache resolution recognizes both current Parquet server-bar downloads and legacy JSONL caches.
- End-to-end validation used 1,380 downloaded MESU6 one-minute bars for 2026-07-27. A TUI-created New York RTH view loaded exactly 390 evaluation rows from 09:30 through 16:00 EDT without starting live user or order streams.
- Replay views also support an optional recurring weekday local-time filter over cached server bars. The filter keeps the immutable source manifest/Parquet intact and applies a half-open local interval on each weekday, including DST-aware timezone conversion. This supports custom windows such as 08:30-16:15 America/New_York across a multi-day or multi-week source range. The current TUI editor does not create this advanced field; CLI/spec users can persist it directly.

Still pending:

- Phase 05 still owns a dedicated persisted backtest-result model and higher-level trade/signal analytics; RBT-022 now supplies the authoritative evaluation boundaries and row counts that model will consume.

## Goal

Let the user replay/backtest a subset of a cached dataset without mutating the cached source file.

The downloader owns broad source coverage, such as a whole contract range or a multi-month window. This task group owns replay-time views over that cached data, so the user can test one session, one RTH window, or one custom period without redownloading or rewriting the source file.

Examples:

- Full Globex session.
- New York RTH only.
- Chicago-local window.
- Custom start/end.
- Specific volatile window after news.

## Current Code

- Session profile exists in `InstrumentSessionProfile` in `src/broker/types/sessions.rs`.
- Futures session logic currently uses New York time for Globex open/close/blockout.
- Replay text parser currently assumes America/New_York for local timestamps in `src/tradovate/replay/ticks.rs`.

## Tasks

### RBT-020: Define Dataset View Model

A dataset view should point at source data plus a filter:

- source manifest id
- start timestamp
- end timestamp
- timezone used for user input
- session preset
- warmup policy

Dependencies:

- RBT-010.

Acceptance criteria:

- Views can be saved and re-run.
- Views preserve warmup data before the visible/test window when indicators need it.
- Views do not duplicate source data unless explicitly exported.

Risks:

- Backtest windows without warmup bars make EMA/HMA first signals unreliable.

### RBT-021: Add Session Presets

Presets:

- Full source file.
- Futures Globex.
- Futures RTH, New York.
- Futures RTH, Chicago.
- Custom local timezone.
- Custom UTC.

Dependencies:

- RBT-020.

Acceptance criteria:

- DST transitions are tested.
- Sunday open and Friday close behavior is explicit.
- Displayed timestamps show timezone labels.

Risks:

- CME product sessions vary around holidays and early closes. Static presets are not enough for perfect exchange calendars.

### RBT-022: Split Warmup and Evaluation Windows

Backtests should support:

- warmup_start <= evaluation_start
- evaluation_end
- analytics only counts trades/signals in evaluation window
- indicators can use warmup bars/ticks before evaluation start

Dependencies:

- RBT-020.

Acceptance criteria:

- Strategy signals before evaluation start can seed runtime state but do not count as trades unless configured.
- Analytics clearly shows warmup rows and evaluation rows.

Risks:

- If runtime state is not warmed consistently, replay will not match live behavior at the start of a selected window.

### RBT-023: Add TUI Dataset View Controls

Controls:

- `V` opens saved views for the selected cached dataset.
- `N` creates a view; `E` edits the highlighted view.
- `Enter` saves from the editor or selects from the list.
- `C` clears the selected view and restores full-source replay.
- `Esc` cancels an edit without saving, then returns to the Replay screen.

Acceptance criteria:

- Create, edit, save, list, select, and clear flows work without mutating source data.
- Invalid inputs fail closed in the editor with an actionable error.
- Startup uses the selected view's evaluation and warmup ranges.
- Replay-disabled builds remain unaffected.

## Open Questions

- Decision: runs force flat until evaluation start by default. Carrying an open warmup position is a later explicit opt-in and is rejected by the current view schema/runtime.
- Should session holidays be maintained manually, pulled from broker metadata, or added later?
