# `serde_json` to `simd_json` Migration Report

Date: 2026-03-24

## Executive Summary

This repo should not do a repo-wide replacement of `serde_json` with `simd_json`.

The best fit for `simd_json` is the repeated websocket decode path in:

- `benchmark-ws/src/main.rs:1492-1502` and `benchmark-ws/src/main.rs:1843-1928`
- `ninjatrader-tui/src/tradovate/protocol.rs:114-124`
- `ninjatrader-tui/src/tradovate/market.rs:252-334`
- `ninjatrader-tui/src/tradovate/market.rs:401-520`

Everywhere else, the current JSON work is either:

- one-off config loading,
- small REST responses,
- IPC messages that are probably small,
- or pretty-printed report output where parsing speed is irrelevant.

The practical recommendation is:

1. Keep `serde_json` as the default JSON library across the repo.
2. Add `simd_json` only to `benchmark-ws` and `ninjatrader-tui`.
3. Use `simd_json::serde::from_slice` only at high-frequency ingress points.
4. Re-measure before expanding the migration further.

## How This Assessment Was Done

I searched the full repo for `serde_json` and `simd_json` usage, then classified each call site by:

- parse frequency,
- expected payload size,
- whether the code runs inside a live loop,
- whether the code is parse-heavy or serialize-heavy,
- and how much refactoring would be required because of current `serde_json::Value` usage.

I also checked the current checked-in JSON artifact sizes under `runs/**/*.json`. The largest checked-in JSON file is about 42 KB, and the total checked-in JSON corpus is about 621 KB. That makes the report/config paths poor candidates for a parser-focused migration.

External API note: the current `simd_json` README shows that it can deserialize into `serde_json::Value` through `simd_json::serde::from_slice(&mut bytes)`, but it expects a mutable byte buffer and parses in place. That makes selective adoption feasible, but not a drop-in swap for the current `from_str(&str)` pattern.

Source:

- <https://github.com/simd-lite/simd-json/blob/main/README.md>

## Current Usage Inventory

### High-value candidates

#### 1. `benchmark-ws` realtime websocket parsing

Relevant code:

- `benchmark-ws/src/main.rs:1492-1502`
- `benchmark-ws/src/main.rs:1843-1928`

Why it is a good candidate:

- `parse_frame` runs inside the live websocket loop.
- The code parses every incoming market-data frame into `serde_json::Value`.
- Parsed payloads are then walked repeatedly to extract status, chart IDs, and bars.
- This is one of the few places where JSON parsing is on the latency-sensitive path.

Expected benefit:

- High, relative to the rest of the repo.
- Best chance of reducing CPU spent on repeated DOM parsing during live and simulated streaming.

Migration cost:

- Medium.
- `simd_json` wants a mutable byte buffer, so `parse_frame(&str)` likely becomes something closer to `parse_frame(String)` or `parse_frame(Vec<u8>)`.
- The current code can still keep using `serde_json::Value` downstream if it parses via `simd_json::serde::from_slice`.

Recommendation:

- Migrate this path first if you want to test `simd_json` in this repo.

#### 2. `ninjatrader-tui` user-data and market-data websocket parsing

Relevant code:

- `ninjatrader-tui/src/tradovate/protocol.rs:114-124`
- `ninjatrader-tui/src/tradovate/market.rs:252-334`
- `ninjatrader-tui/src/tradovate/market.rs:401-520`

Why it is a good candidate:

- Same shape as `benchmark-ws`: repeated websocket JSON parsing inside long-running loops.
- `ninjatrader-tui` explicitly optimizes for low-latency sockets in `ninjatrader-tui/src/tradovate/market.rs:102-159`, so parser cost is more likely to matter here than in the batch/reporting tools.
- Incoming payloads contain arrays of messages and bars, which is exactly the sort of workload where a faster parser can help.

Expected benefit:

- High for market-data flow.
- Medium-to-high for user-data sync flow.

Migration cost:

- Medium to high.
- This crate is heavily coupled to `serde_json::Value` and `json!`.
- `ninjatrader-tui/src/tradovate/protocol.rs` alone has 49 `Value`/`json!` occurrences by grep count, and `orders.rs` has 28. A full type-level swap would spread quickly.

Recommendation:

- Use `simd_json` only at the frame ingress boundary.
- Keep returning `serde_json::Value` from the parser for now.
- Do not attempt a full `serde_json::Value` to `simd_json::OwnedValue` rewrite unless profiling proves it is worth the blast radius.

### Secondary candidates

#### 3. `ninjatrader-tui` IPC decode loops

Relevant code:

- `ninjatrader-tui/src/ipc.rs:73-118`
- `ninjatrader-tui/src/ipc.rs:132-158`

Why it might help:

- These loops parse JSON continuously if the TUI and engine exchange many events.

Why it is not a top candidate:

- IPC messages are likely much smaller than websocket market-data payloads.
- Serialization is still present on both sides via `serde_json::to_string`.
- If IPC is not a measurable bottleneck, the migration adds complexity without visible payoff.

Expected benefit:

- Low to medium.

Recommendation:

- Leave this on `serde_json` unless profiling shows IPC decode cost is meaningful.

#### 4. `ninjatrader-tui` REST bootstrap and lookup paths

Relevant code:

- `ninjatrader-tui/src/tradovate/auth.rs:51-83`
- `ninjatrader-tui/src/tradovate/auth.rs:107-159`
- `ninjatrader-tui/src/tradovate/auth.rs:192-220`
- `ninjatrader-tui/src/tradovate/market.rs:8-70`

Why it might help:

- Some endpoints return arrays or object graphs.
- `fetch_entity_list` can materialize lists of accounts, positions, fills, and related objects.

Why it is not a top candidate:

- These are startup/admin flows, not continuous hot loops.
- Several call sites use `reqwest` ergonomics today. Switching from `response.json().await?` or `from_str(&body)` to manual byte-buffer parsing is more ceremony for limited gain.

Expected benefit:

- Low to medium.

Recommendation:

- Consider only after websocket parsing is measured and already migrated successfully.

### Low-value or poor-fit candidates

#### 5. Root crate config and report JSON

Relevant code:

- `src/ml.rs:242-260`
- `src/bin/backtest_script.rs:207-228`
- `src/bin/strategy_analyzer.rs:196-202`
- `src/bin/strategy_analyzer.rs:311-331`
- `src/bin/inspect_rl_policy.rs:375-376`
- `src/bin/inspect_rl_policy.rs:895-897`
- `src/bin/inspect_ga_policy.rs:354-355`
- `src/bin/inspect_ga_policy.rs:372-374`

Why it is a poor fit:

- These are one-shot config loads and pretty-printed outputs.
- The time spent here is dominated by the actual analysis, training, parquet work, or replay logic, not JSON parsing.
- `simd_json` does not solve the main cost in `to_string_pretty` call sites.

Expected benefit:

- Low.

Recommendation:

- Keep these on `serde_json`.

#### 6. `benchmark-ws` small REST helpers

Relevant code:

- `benchmark-ws/src/main.rs:1472-1480`
- `benchmark-ws/src/main.rs:1561-1583`
- `benchmark-ws/src/main.rs:1586-1645`
- `benchmark-ws/src/main.rs:1684-1739`

Why it is a poor fit:

- Token load, contract lookup, account list, and order-response parsing happen occasionally.
- These are not the dominant JSON costs compared with the streaming websocket loop.

Expected benefit:

- Low.

Recommendation:

- Leave them on `serde_json` unless you are already standardizing helper functions in that crate.

#### 7. Portable GA policy JSON

Relevant code:

- `src/bin/train_ga/portable.rs:35-50`

Why this is a special case:

- This is the only non-network JSON path that could become materially large, because it stores model weights as arrays of floats.

Why `simd_json` is still not the first fix:

- If this file becomes large enough to hurt, the deeper problem is the format choice, not the parser.
- For large model payloads, a binary format or an existing tensor-safe format is a better optimization than swapping JSON parsers.

Expected benefit:

- Medium only if these JSON policy files are large and loaded often.

Recommendation:

- Do not migrate this to `simd_json` first.
- If it becomes hot, evaluate replacing JSON for this artifact entirely.

## Pros of Introducing `simd_json`

- Faster deserialization on repeated or larger JSON payloads.
- Best fit for the websocket-heavy code paths that already operate like streaming systems.
- Can be introduced selectively because it can deserialize into `serde_json::Value`.
- A selective migration keeps most of the existing business logic unchanged.
- In `ninjatrader-tui`, lower decode overhead aligns with the crate's explicit low-latency networking posture.

## Cons of Introducing `simd_json`

- It is not a clean drop-in for the current code style because it expects mutable bytes and parses in place.
- The repo relies heavily on `serde_json::Value`, `serde_json::json!`, and pretty-print helpers, especially in `ninjatrader-tui`.
- A partial migration means carrying both libraries, which increases cognitive and maintenance overhead.
- Several current call sites use `reqwest` JSON helpers; replacing those with `simd_json` requires more manual response handling.
- Real gains depend on payload shape and CPU characteristics, so the speedup should be measured rather than assumed.
- Most of the root crate would get complexity without meaningful benefit.

## Recommended Migration Shape

### Phase 1: targeted adoption only

Add `simd_json` to:

- `benchmark-ws/Cargo.toml`
- `ninjatrader-tui/Cargo.toml`

Do not add it to the root crate yet.

### Phase 2: replace only websocket ingress parsing

Start with:

- `benchmark-ws/src/main.rs:1492-1502`
- `ninjatrader-tui/src/tradovate/protocol.rs:114-124`

Preferred approach:

- keep downstream types as `serde_json::Value`,
- change ingress helpers to parse from mutable bytes,
- avoid a full rewrite to `simd_json::OwnedValue` or `BorrowedValue` on the first pass.

### Phase 3: benchmark before expanding

Measure:

- websocket messages per second,
- CPU in decode-heavy loops,
- and end-to-end latency before and after.

Only then consider:

- `ninjatrader-tui/src/ipc.rs`
- `ninjatrader-tui` REST bootstrap helpers
- `benchmark-ws` REST helpers

## Final Recommendation

Use `simd_json` as a targeted optimization, not as a wholesale replacement for `serde_json`.

The only clearly justified migration targets in this repo today are the repeated websocket frame parsers in `benchmark-ws` and `ninjatrader-tui`. Everything else should stay on `serde_json` unless a benchmark shows otherwise.
