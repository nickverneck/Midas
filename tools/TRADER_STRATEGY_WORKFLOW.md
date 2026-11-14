# Trader replay workflow adapter

`trader-strategy-workflow.py` is a small local process adapter for Trader's
existing offline replay-sweep CLI. The durable candidate/config contract is the
native JSON sweep specification; native validation expands it into a rerunnable
plan, and the native runner writes result and summary artifacts beneath the
spec's configured output directory. Ranking reads those native summaries.

The adapter exposes only `validate`, `run`, and `rank`. It builds a fixed
argument vector for the corresponding Trader commands, never invokes a shell,
and returns one JSON envelope containing the exit status, output, and exact
argument vector. It does not expose engine launch, trader orders, broker
credentials, or arbitrary CLI pass-through. Runs have a wall-clock timeout of
at most one hour; the native spec's guardrails remain authoritative. The
adapter intentionally does not enable `--allow-large` or
`--override-guardrails`.

Build the native CLI with its replay feature, then call the adapter from a local
web UI's server-side process layer:

```bash
cargo build --release --manifest-path trader/Cargo.toml --features replay

python3 tools/trader-strategy-workflow.py validate \
  --spec path/to/replay-sweep.json

python3 tools/trader-strategy-workflow.py run \
  --spec path/to/replay-sweep.json --execution-mode batch_cpu

python3 tools/trader-strategy-workflow.py rank \
  --summary path/to/sweep-summary.json --metric robustness \
  --output path/to/ranking.json
```

The spec and output artifacts should live in an application-controlled durable
directory. A UI may present the native fields and artifacts, but this adapter
is not an HTTP service and does not define a TUI/UI compatibility protocol.
`run-replay-sweep` uses Trader's replay simulator against the configured local
cache; it does not connect to a broker or place orders. `rank` supports the
native ranking metrics listed in the adapter's `RANK_METRICS` allowlist.

The adapter assumes the release executable is at
`trader/target/release/trader` under `--repo`. Config is optional and only
supplies normal Trader config such as the replay-cache directory. Native CLI
validation is the source of truth for strategy parameters, cache views,
resource estimates, and sweep guardrails.

## Lua cached replay callback

Lua strategies run only in cached replay through `isolated_services`. Each run
owns one sandboxed Lua 5.4 VM; globals persist between bars, while cloning or
starting a sweep child creates an independent VM. The runtime rebuilds state
from the replay prefix when needed. During prefix warmup, `ctx.is_warmup` is
true and reading `ctx.position_qty` fails the run because historical position
values are not available to reconstruct safely. On the evaluated bar,
`ctx.position_qty` is the current signed position. `on_bar(ctx, bar)` must return
`buy`, `sell`, or `hold`.

The sandbox caps memory at the configured limit (64 KiB to 32 MiB; 8 MiB by
default) and instructions at the configured limit (1 to 1,000,000; 100,000 by
default) for source initialization and each callback. I/O, OS, module loading,
randomness, and quota bypass libraries are unavailable. Lua is not enabled for
live or hosted replay execution. Replay sweeps require `isolated_services`;
batch and prepared CPU evaluators reject Lua strategies.

Lua replay requires closed-bar timing. Exact duplicate service updates are
idempotent when the service supplies the same update sequence; revised bars,
ambiguous duplicate updates without a stable sequence, and retained windows
that no longer overlap the last processed bar fail the run. When several
unseen bars arrive in one update, earlier bars run as warmup with unavailable
position context, so scripts that need historical positions fail closed.

Run its unit tests with:

```bash
python3 -m unittest tools/test_trader_strategy_workflow.py
```
