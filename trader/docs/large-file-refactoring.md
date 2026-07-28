# Large-file refactoring plan

This document records the source files that exceeded 1,000 lines on July 28,
2026 and the intended decomposition for each one. The refactor is structural:
public behavior, feature gates, serialization formats, CLI behavior, broker
behavior, and test coverage must remain unchanged.

## Completion status

Completed on July 28, 2026. All twelve Rust files identified below were
decomposed, their existing public or crate-visible APIs were preserved through
module re-exports, and all original tests were retained. A final source scan
found no Rust file at or above 1,000 lines; the largest is now
`src/replay_cache/raw_ticks/parquet.rs` at 918 lines.

| Original file | Largest resulting file | Lines |
| --- | --- | ---: |
| `src/replay_cache.rs` | `src/replay_cache/raw_ticks/parquet.rs` | 918 |
| `src/app/render/tests.rs` | `src/app/render/tests/strategy.rs` | 839 |
| `src/tradovate/download.rs` | `src/tradovate/download/metadata.rs` | 543 |
| `src/tradovate/replay.rs` | `src/tradovate/replay/tests/aggregation.rs` | 486 |
| `src/app/state.rs` | `src/app/state/logging.rs` | 442 |
| `src/tradovate/execution/tests/strategy_loop.rs` | `strategy_loop/guarded.rs` | 451 |
| `src/tradovate/execution/strategy.rs` | `strategy/run_loop.rs` | 517 |
| `src/main.rs` | `src/replay_cli.rs` | 412 |
| `src/tradovate/service/tests.rs` | `service/tests/replay_jobs.rs` | 450 |
| `src/tradovate/service/commands.rs` | `service/commands/replay_jobs.rs` | 418 |
| `src/broker/types.rs` | `src/broker/types/market.rs` | 436 |
| `src/app/render/replay.rs` | `src/app/render/replay/downloader.rs` | 492 |

Final validation passed both the default 265-test suite and the all-feature
382-test suite. `openapi.json` remains intentionally intact as a generated
specification.

## Working rules

- Move code before redesigning it. Each extraction should compile and preserve
  the existing tests before any further cleanup.
- Keep public re-exports stable so callers do not need broad simultaneous
  changes.
- Prefer a directory module when a file owns three or more distinct concerns.
- Keep feature gates on the smallest relevant modules.
- Keep unit tests near the component they exercise, or use a sibling `tests/`
  directory when fixtures are shared across multiple components.
- Do not split `openapi.json`; it is a generated API specification artifact.

## Production files

### `src/replay_cache.rs` — 6,405 lines

Convert it to `src/replay_cache/mod.rs` with these components:

- `model.rs`: manifests, cache metadata, coverage, market shapes, data-file
  records, checkpoint records, and normalized row types.
- `manifest.rs`: manifest parsing, derived fields, metadata merging, hashing,
  locking, atomic manifest updates, and validation.
- `paths.rs`: dataset paths, safe path segments, symlink/path-escape checks,
  cache-relative paths, and superseded-file cleanup.
- `server_bars.rs`: server-bar normalization, JSONL/Parquet writing, reading,
  range loading, and metadata validation.
- `raw_ticks.rs`: raw-tick normalization, checkpoint/chunk lifecycle, Parquet
  writing, streaming, pruning, and sequence validation.
- `dataset.rs`: `ReplayCacheDataset` resolution and serving APIs.
- `library.rs`: cache-library scanning, matching, and unique-dataset loading.
- `tests/`: model/manifest, paths/security, server bars, raw ticks, checkpoint,
  and library test modules with shared fixtures in `tests/support.rs`.

`mod.rs` should primarily declare modules and re-export the existing public API.

### `src/tradovate/download.rs` — 2,158 lines

Convert it to `src/tradovate/download/mod.rs`:

- `types.rs`: public download request/result/session/progress types.
- `session.rs`: authenticated preparation and high-level download entry points.
- `chunked.rs`: chunk planning, cancellation, progress, checkpoint commits, and
  finalization.
- `metadata.rs`: account/contract/product/fee metadata requests, parsing, and
  suggested coverage.
- `protocol.rs`: chart request bodies, historical message parsing, telemetry,
  and sanitized failures.
- `websocket.rs`: server-bar and raw-tick WebSocket fetch loops plus subscription
  cleanup.
- `tests/`: request, parser, metadata, chunking, and fetch-loop tests.

Preserve the current `tradovate::download::*` exports through `mod.rs`.

### `src/tradovate/replay.rs` — 1,819 lines

Convert it to `src/tradovate/replay/mod.rs`:

- `state.rs`: `ReplayState`, account/contract exposure, and source abstraction.
- `load.rs`: local/cache loading, replay path resolution, and source selection.
- `worker.rs`: replay task lifecycle, timing, speed scaling, and broker command
  delivery.
- `ticks.rs`: raw tick parsing and cache-row conversion.
- `bars.rs`: time, tick-count, volume, and range bar builders.
- `instrument.rs`: inferred symbol, contract ID, tick size, and point value.
- `tests/`: loading, timing, aggregation, and instrument inference tests.

Keep the `replay` feature gates localized and preserve the non-replay stubs.

### `src/app/state.rs` — 1,590 lines

Keep `App` itself in its current owning module, but split its method
implementations into focused sibling modules:

- `state/capabilities.rs`: broker capability and visibility decisions.
- `state/focus.rs`: focus order, next/previous focus, and text-focus predicates.
- `state/logging.rs`: log sanitization, persistence, review summaries, and saved
  session sections.
- `state/engine.rs`: active-engine identity, labels, and engine counts.
- `state/strategy.rs`: readiness, configuration normalization, arming/disarming,
  and strategy summaries.
- `state/market.rs`: selected snapshots, live selected-contract P&L, session
  windows, position quantity, and projected protection levels.

Use `src/app/state/mod.rs` for shared private structures and module declarations.

### `src/tradovate/execution/strategy.rs` — 1,446 lines

Convert it to `src/tradovate/execution/strategy/mod.rs`:

- `broker_path.rs`: live-path detection, grace periods, pending-target gates,
  and stale-target cleanup.
- `evaluation.rs`: active strategy evaluation and signal-to-target decisions.
- `signals.rs`: dispatched/consumed signal tracking and crossover seeding.
- `debug.rs`: strategy evaluation context and debug formatting.
- `reversal.rs`: staged reversal continuation and transition handling.
- `account_sync.rs`: broker position synchronization and drift handling.
- `run_loop.rs`: the top-level execution loop and its orchestration helpers.

Keep the current `pub(crate)` surface re-exported by `mod.rs` so order and
service modules are unaffected.

### `src/main.rs` — 1,354 lines

Leave only module declarations, process initialization, and the top-level
dispatch in `main.rs`:

- `cli.rs`: `Cli`, command/mode definitions, parsing, and validation.
- `replay_cli.rs`: replay-download planning, parsing, and reporting.
- `engine_cli.rs`: list, kill, kill-all, and attach configuration commands.
- `tui_runtime.rs`: terminal initialization/restoration and the main TUI loop.
- `engine_session.rs`: engine connection/spawn, observation, event relays, and
  lifecycle actions.
- Move tests beside the component being tested.

### `src/tradovate/service/commands.rs` — 1,309 lines

Convert it to `src/tradovate/service/commands/mod.rs`:

- `dispatch.rs`: `ServiceCommand` matching and routing only.
- `connection.rs`: live connection, replay entry, and session reset.
- `replay_jobs.rs`: lookup/download ownership, cancellation, commit state, and
  replay-state publication.
- `market.rs`: account selection, contract search, subscriptions, and replay
  speed.
- `orders.rs`: manual orders, target position, profiling, and protection sync.
- `strategy.rs`: strategy configuration normalization, arm/disarm, and probes.

Keep command handlers `pub(super)` only where sibling service modules require
them.

### `src/broker/types.rs` — 1,182 lines

Convert it to `src/broker/types/mod.rs`:

- `service.rs`: `ServiceCommand`, `ServiceEvent`, manual order actions, and
  replay operation/progress types.
- `capabilities.rs`: broker kinds and capability descriptions.
- `market.rs`: bars, bar types, candle transformations, market snapshots, and
  trade markers.
- `contracts.rs`: contract suggestions, maturity/trade status, and maturity
  parsing.
- `account.rs`: account snapshots, engine history, latency, and execution probes.
- `sessions.rs`: session profiles, session windows, and schedule calculations.
- `replay.rs`: replay speed and cache target types.
- Move the inline tests into component-specific test modules.

Maintain `crate::broker::*` compatibility through re-exports.

### `src/app/render/replay.rs` — 1,004 lines

Convert it to `src/app/render/replay/mod.rs`:

- `input.rs`: replay picker keyboard handling and replay startup.
- `downloader.rs`: downloader focus transitions, validation, lookup, inspection,
  submission, cancellation, and selected-manifest helpers.
- `view.rs`: replay picker rendering.
- `downloader_view.rs`: downloader rendering and progress/guidance text.

## Test files

### `src/app/render/tests.rs` — 4,442 lines

Convert it to `src/app/render/tests/mod.rs`, with shared builders/assertions in
`support.rs` and test modules for:

- `engine.rs`
- `broker_login.rs`
- `selection.rs`
- `replay.rs`
- `strategy.rs`
- `dashboard.rs`
- `session_stats.rs`
- `logging.rs`

### `src/tradovate/execution/tests/strategy_loop.rs` — 1,553 lines

Replace it with `src/tradovate/execution/tests/strategy_loop/mod.rs` and split
tests into:

- `reversal.rs`
- `timing.rs`
- `guarded.rs`
- `blockout.rs`
- `broker_sync.rs`
- `protection.rs`

Keep position/order seeding helpers in `support.rs`.

### `src/tradovate/service/tests.rs` — 1,343 lines

Convert it to `src/tradovate/service/tests/mod.rs`, with common state/session
fixtures in `support.rs` and modules for:

- `replay_jobs.rs`
- `configuration.rs`
- `rejections.rs`
- `engine_history.rs`
- `pending_targets.rs`

## Generated specification

`openapi.json` is 20,162 lines. It should not be manually split because tooling
expects a single OpenAPI document. If repository size becomes a concern, handle
it through the generation workflow rather than source-module decomposition.

## Suggested implementation order

1. `replay_cache.rs`, `tradovate/execution/strategy.rs`, and `app/state.rs`.
2. `tradovate/download.rs`, `main.rs`, and `broker/types.rs`.
3. `tradovate/replay.rs`, `tradovate/service/commands.rs`, and
   `app/render/replay.rs`.
4. Split the three large test files after their production boundaries settle.
5. Run formatting, default and feature builds, the full test suite, and verify
   that no non-generated Rust source file remains above 1,000 lines.
