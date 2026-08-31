# Trader replay proxy

`trader-replay-proxy` is a standalone, loopback-only Tradovate protocol
facade. It is intentionally outside the trader GUI and outside the normal
replay engine. It serves one explicit bar fixture through the REST, user-data
WebSocket, and market-data WebSocket surfaces used by the native Tradovate
engine. Orders are accepted into a bounded synthetic account, so the live
transport path can be tested without contacting Tradovate or changing the
engine's production execution code.

For execution-accurate tests, pass a raw-tick fixture with `--ticks`. The chart
bars still drive the native strategy, but the raw tick stream becomes the
single broker clock: market-order entries fill on the next source tick, and
broker-owned TP/SL/trailing children are evaluated on every executable quote.
This keeps chart-bar delivery and execution matching separate, which is
important for range bars and other bars that can backtrack or be timestamped
at their close.

The boundary is explicit:

```text
trader (unchanged live-style path)
        │  simulation_proxy.enabled = true
        ▼
trader-replay-proxy ── fixture bars ── synthetic account / fills
        │
        ├── optional raw ticks ── quote-accurate fills and broker brackets
        │
        └── JSONL protocol trace (optional)
```

The proxy binds only to loopback addresses. The trader routes to it only when
the environment is `sim` and the proxy block is explicitly enabled. A live
configuration or a non-loopback proxy URL is rejected, and there is no
fallback from a failed proxy to a real broker endpoint.

## Build

From the Midas repository root:

```bash
cargo build --release --manifest-path tools/trader-replay-proxy/Cargo.toml
cargo build --release --manifest-path trader/Cargo.toml \
  --no-default-features --features tradovate
# Add manual-orders when using the headless swipe-profile order probe.
cargo build --release --manifest-path trader/Cargo.toml \
  --no-default-features --features 'tradovate manual-orders'
```

## Run an offline transport test

Use a **server-bars** parquet file, not a feature/training parquet. The
fixture must contain `timestamp`, `ts_ns`, `open`, `high`, `low`, `close`, and
optionally `volume`. Rows are consumed in file order: they are not sorted or
deduplicated, so same-timestamp corrections remain observable. Raw-tick mode
rejects a bar file with a timestamp regression because a forward-only tick
cursor cannot replay a range-bar correction honestly.

```bash
export TRADER_PROXY_SOURCE_COMMIT="$(git rev-parse HEAD)"
export TRADER_PROXY_CLIENT_COMMIT="$(git rev-parse HEAD)"
tools/trader-replay-proxy/target/release/trader-replay-proxy \
  --bars /path/to/server-bars-from-the-same-window.parquet \
  --ticks /path/to/raw-ticks-from-the-same-window.parquet \
  --fixture-manifest /path/to/manifest.json \
  --require-quote-ticks \
  --history-bars 500 \
  --raw-tick-bar-timestamps close \
  --protection-precedence stop \
  --speed 20 \
  --trace /tmp/gcz6-proxy-trace.jsonl
```

The bars and ticks in this command must cover the same selected realtime
window. A manifest whose bar file spans a week while its tick sidecar covers
only a short sample is intentionally rejected.

`--ticks` accepts the repository raw-tick JSONL/CSV/Parquet schema. Stable
`tick_id` values are retained when present; older nullable-ID Parquet files
use source order plus the proxy sequence. Use bars generated from the same
tick tape and the same contract and selected realtime window. The proxy
verifies the manifest file hash, full row count, source kind, chart shape, and
contract metadata; it also requires the raw tape to reach both sides of the
selected realtime window. If `--max-bars` or `--max-ticks` truncates that
window, startup fails rather than leaving orders Working on an incomplete
tape. A trade-only tape has no executable bid and ask; the proxy then uses
the trade price as an explicitly marked fallback in the fill entity
(`replayFillSource=tick_trade_fallback`,
`replayExecutionPrecision=trade_time_only`). A tape containing both bid and
ask fields is required for quote-side accuracy. Use `--require-quote-ticks`
to fail closed instead of accepting trade-only or partial-quote rows. The
repository Databento-trade importer intentionally produces trade-only rows,
so those fixtures are tick-time accurate but not quote-side accurate.

When a replay-cache bar filename or manifest identifies its shape (for example
`10range`), the proxy rejects a native `md/getChart` request for a different
shape, contract, or history size. Unknown chart shapes are rejected instead
of bypassing validation. The headless `swipe-profile` probe defaults to 1
Range and accepts `--bar-value` so its request can be matched to the selected
fixture. `swipe-profile` requires the loopback proxy by default; pass
`--allow-real-sim` only for an explicitly authorized real Tradovate simulation
probe. Its exit status is also non-zero when a scenario is unsettled or has a
validation finding; pass `--allow-findings` only when collecting a diagnostic
report intentionally.

For replay-cache data the proxy discovers the nearest `manifest.json` and
checks the contract symbol, contract ID, tick size, point value, selected file
paths, source kinds, and bar/tick time overlap against the fixture metadata.
Use `--fixture-manifest PATH` to select it explicitly. If `--contract-id` is
omitted, a verified manifest supplies the contract ID; an explicit mismatched
ID is rejected. Hand-authored fixtures without a manifest must opt in with
`--allow-unverified-fixture`.

In another terminal, start the normal trader binary with the checked-in
example config:

```bash
trader/target/release/trader \
  --config tools/trader-replay-proxy/examples/proxy-trader.toml
```

Create a broker engine in the TUI, select the synthetic account and contract,
and subscribe to bars. The proxy's stdout and trace show every endpoint,
request id, response, order acceptance, fill, and emitted entity update. The
normal `trader engine` subcommand receives configuration from the TUI parent
over IPC; it is not a standalone TOML loader. For a headless order-path
check, use the existing `swipe-profile` command with the same config:

```bash
trader/target/release/trader \
  --config tools/trader-replay-proxy/examples/proxy-trader.toml \
  swipe-profile --account-filter PROXY --contract-query GCZ6 \
  --contract-exact GCZ6 --bar-value 10 --require-simulation-proxy \
  --delays-ms 0 --iterations 1 \
  --settle-timeout-ms 5000 --output-dir /tmp/proxy-swipe
```

Useful fault/latency controls:

```text
--speed 0                 send bars as fast as possible (finite replays only)
--speed 1                 follow fixture timestamp gaps (capped by --max-sleep-ms)
--loop-boundary-delay-ms 1000  pace same-timestamp loop boundaries
--rest-delay-ms 25        delay every REST response
--ack-delay-ms 25         delay order WebSocket acknowledgements
--fill-delay-ms 100       delay fill/entity publication after acknowledgement
--max-bars 10000           bound a stress run
--start-file /tmp/ready    hold realtime until this local marker exists
--loop-replay              repeat bar-only realtime data at EOF (raw ticks disallowed)
--max-clients 1            expose accidental concurrent clients immediately
--max-state-entities 100000 bound retained synthetic broker entities
--trace-max-bytes 67108864 cap a trace file
--max-ticks 1000000        bound raw-tick memory for a diagnostic run
--fixture-manifest PATH    explicitly select the replay-cache identity manifest
--allow-unverified-fixture permit raw ticks without a discoverable manifest
--require-quote-ticks       reject any raw tick missing a valid bid and ask
--raw-tick-bar-timestamps start  process ticks before each chart timestamp
--raw-tick-bar-timestamps close  process ticks through each chart timestamp
--protection-precedence stop     conservative same-quote TP/SL resolution
--protection-precedence target   favorable same-quote TP/SL resolution
```

The initial `--history-bars` rows are returned from `md/getChart`; remaining
rows are emitted one at a time as realtime chart updates. With `--ticks`, raw
ticks from the realtime boundary to each chart timestamp are processed before
that chart update (`start` is exclusive; `close` is inclusive). A parent
market order is eligible only on a later raw tick, preventing an
acceptance/quote race. The selected raw-tick window ends at the final selected
bar timestamp; there is no unbounded EOF drain that can accidentally execute
orders against a later session. A broker-owned bracket is installed only
after the parent fill. Its synthetic child orders are marked
`replayBrokerOwned=true`, use bid for long exits and ask for short exits, and
cancel their OCO sibling atomically after a TP, stop, or trailing fill. The
trace records source sequence, tick identity, fill source, execution precision,
exit reason, and fixture hashes.

Without `--ticks`, the proxy retains the legacy bar-quote mode. A value larger
than the fixture returns all rows as historical and leaves no realtime stream.
A looped replay requires at least one realtime row and paces loop boundaries,
including one-bar fixtures, so it cannot hot-loop. The market stream also
keeps reading while it waits between bars, so a disconnected client is
noticed without waiting for the full sparse-bar sleep to expire.

Trace output is capped and flushed per record. Tracing is useful for forensics but adds
serialization overhead; run latency comparisons once with `--trace` and once
without it. The trace metadata records both `TRADER_PROXY_SOURCE_COMMIT` (the
proxy checkout) and `TRADER_PROXY_CLIENT_COMMIT` (the trader checkout, when
supplied).

The candidate runner also records a run id, proxy/client toolchain, and the
candidate binary SHA-256 in the trace metadata.

## Comparing commits

Keep the fixture, proxy arguments, synthetic IDs, and trace settings constant.

The supervised candidate runner automates the safe part of that process:

```bash
tools/trader-replay-proxy/scripts/bisect-candidate.sh \
  --bars trader/.run/replay-cache/.../server-bars/session.parquet \
  --fail-regex 'avg_fill=Some\(([5-9][0-9]{2}|[1-9][0-9]{3,})' \
  --output-dir .run/proxy-bisect/current
```

It defaults to `--commit HEAD`, so it can also be passed directly to
`git bisect run`; an explicit `--commit REV` is useful for a one-off candidate.
It creates a detached worktree, builds the candidate into a temporary target
directory, starts a fresh proxy instance, runs the existing headless
`swipe-profile` probe, and leaves the build log, candidate output, proxy log,
and trace in a fresh output directory. Reusing an output root creates a new
timestamped child, so stale artifacts cannot be mistaken for the current run.
The synthetic account and contract passed to the helper are also passed to the
proxy, keeping selection and fixture identity aligned. `--pass-regex` can be used instead when a
candidate must contain a required report marker. The script returns `125` for
a missing/unbuildable/pre-seam candidate so `git bisect run` skips it, and
returns `1` for a failed assertion or candidate crash. It never modifies a
candidate checkout. Each candidate is bounded by `--candidate-timeout-ms`
(120 seconds by default); a timeout is a failed candidate rather than an
unbounded bisect run.

The proxy seam was introduced after the earlier HMA/range changes, so binaries
that predate `simulation_proxy` are deliberately skipped. Do not apply an
unreviewed patch to an old commit and do not silently fall back to a real
broker. To compare a pre-seam commit, first port the small, reviewed routing
seam into a dedicated candidate branch and record that fact in the trace.

The trace is JSONL and starts with fixture SHA-256, row count, timestamps,
proxy settings, delays, checkout commit metadata, the proxy binary SHA-256,
and a workspace-dirty marker. That makes a run reproducible and prevents an
apparent code regression from actually being a fixture mismatch.

Each proxy process represents one test run. Restart it between candidates so
account position, order IDs, fills, and PnL cannot leak between binaries. The
default `--max-clients 1` admits one client per protocol listener; a trader uses
one REST, one user-WebSocket, and one market-WebSocket connection. Separate
trader processes must use separate proxy instances and ports.

## Current scope and deliberate limitations

The proxy models the protocol needed by the native connection, market,
user-sync, direct order, liquidation, cancellation, and
`startorderstrategy` paths. It includes deterministic IDs, client-order-ID and
strategy-UUID deduplication, bounded client/state/trace admission,
fail-closed WebSocket broadcast lag handling, and (when `--ticks` is present)
deterministic quote-side entry and broker-owned TP/SL/trailing lifecycle.
Accepted orders emit native order, execution-report, fill, and zero-fee events
with the identity fields the trader latency tracker uses.

It is not a full Tradovate matching engine. It does not model DOM depth,
exchange rejects, partial fills, queue priority, slippage beyond the supplied
quote, or every broker error code. Orders are intentionally full-quantity
synthetic fills; quote/trade size is retained in the fixture but is not a
depth/partial-fill model. A trade-only fixture cannot reconstruct a bid/ask
spread. The proxy is a transport and lifecycle regression harness, not a PnL
oracle. DOM capture and replay-download commands fail closed while proxy mode
is enabled because those provider-specific surfaces are not implemented here.
Use `--max-clients 1` for commit comparisons; each proxy process is
intended for one trader session, not an isolated multi-session simulation.

Never enable this block in a live config. Never put real credentials in the
proxy command line or fixture. Use a synthetic account and contract unless a
future test explicitly adds a separate, reviewed fixture mapping.
