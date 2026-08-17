# Supervised crossover-event training

The supervised path is deliberately event-based. It emits one row when the
configured trigger average crosses, rather than one row per bar.

## Causal boundary

The crossover is observed after the source bar closes. The first executable
price is the next bar open in the same New York Globex session. Every feature
column is computed from bars at or before the crossover row. The following are
targets/audit columns and are never model inputs:

- `action_value_normal`, `action_value_skip`, `action_value_invert`;
- `label_action`, `label_name`, `oracle_*`;
- `decision_price`, `interval_end_price`, and terminal/session-close values.

Sessions use `America/New_York`, from 18:00 on the prior local date through
17:00 on the trading date. The 17:00–18:00 maintenance interval is excluded.
The CLI excludes a trailing session whose final bar is more than five minutes
before 17:00, rather than treating the last available bar as a fictitious
session close.

## Recursive labels

For each session, the labeler runs backward over the crossover schedule. Its
state is the current target position (`-1`, `0`, `+1`). At each event it
evaluates:

- normal: target the raw crossover direction;
- skip: hold the current target position;
- invert: target the opposite direction.

The selected action maximizes current interval PnL plus the best value of the
next event. Entry, reversal, and forced session-close fees are charged as
position transitions. Ties prefer skip, then normal, then invert. This is a
hindsight label generator, not a live strategy result.

The current v1 artifact stores the optimal-path label and its three action
values for the position on that path. `oracle_position_*` is audit-only and is
explicitly rejected if someone tries to include it in the feature schema. A
future state-expanded artifact should emit reachable `(event, position,
protection)` rows when protection or other runtime state becomes a model
input.

`evaluate` accepts a different prepared holdout artifact when its feature
schema and causal configuration match the policy. Resume training is stricter:
it requires the original source hash and dataset fingerprint.

## CLI

Prepare directly from a replay-cache server-bar parquet:

```bash
cargo run --bin supervised -- prepare \
  --config config/supervised.example.yaml \
  --input trader/.run/replay-cache/.../server-bars/one-minute.parquet \
  --output trader/.run/supervised/datasets/gc-events.parquet \
  --instrument GC --contract GCZ6
```

CLI flags override YAML/JSON/TOML config values. The feature builder accepts
`--features` as a JSON or YAML `FeatureSpec` array. Supported feature families
include SMA, EMA, HMA, KAMA, ALMA, ATR, ADX, RVOL, Kaufman ER, and close.
Each spec can include its value, causal lookbacks, deltas, and ATR
normalization. Every artifact also includes normalized per-bar slopes for the
trigger/context fast and slow averages, plus the trigger/context separation
and distance from the trigger slow average.

The loader accepts canonical OHLCV parquet/CSV. It also derives minute/second
bars from a Databento raw-trade parquet containing `ts_event` (or `ts_ns`),
`price`, and optional `size`. Native Databento fixed-point prices are
auto-scaled by 1e9 when their magnitude is unambiguously large; pass
`--databento-price-scale` when the source needs an explicit scale. Volume/range
raw aggregation remains in Trader, where the existing exact replay aggregator
handles oversized trades and equal-timestamp logical bars. Existing
pre-aggregated volume/range files are accepted.

NinjaTrader `.Last.txt` files are bucketed into sparse one-minute OHLC bars.
Those files do not provide trusted trade size in the supported format, so any
RVOL feature causes preparation to fail instead of inventing volume.

Train and continue a CPU reference policy:

```bash
cargo run --bin supervised -- train \
  --input trader/.run/supervised/datasets/gc-events.parquet \
  --outdir trader/.run/supervised/runs/gc-baseline \
  --epochs 50 --learning-rate 0.01 --seed 42

cargo run --bin supervised -- train \
  --input trader/.run/supervised/datasets/gc-events.parquet \
  --outdir trader/.run/supervised/runs/gc-continued \
  --resume-policy trader/.run/supervised/runs/gc-baseline/policy.json \
  --epochs 10

# Evaluate the uninterrupted trajectory without touching holdout metrics.
cargo run --features backend-burn --bin supervised -- train \
  --input trader/.run/supervised/datasets/gc-events.parquet \
  --outdir trader/.run/supervised/runs/gc-long \
  --backend burn --device cpu --epochs 100000 \
  --checkpoint-every 1000 --seed 42
```

The same event classifier can run through Candle. Use `--device cuda` on a
machine where the CUDA toolchain is installed; the build must include
`backend-candle-cuda` (the web flow adds it automatically when CUDA is selected):

```bash
cargo run --features backend-candle --bin supervised -- train \
  --input trader/.run/supervised/datasets/gc-events.parquet \
  --outdir trader/.run/supervised/runs/gc-candle \
  --backend candle --device cpu --epochs 50

cargo run --features backend-candle,backend-candle-cuda --bin supervised -- train \
  --input trader/.run/supervised/datasets/gc-events.parquet \
  --outdir trader/.run/supervised/runs/gc-candle-cuda \
  --backend candle --device cuda --epochs 50
```

The reference trainer is a deterministic three-class logistic policy. It
splits whole ET sessions, requires at least three complete sessions, fits
scaling on the training sessions only, and writes `policy.json` and
`metrics.json`. It refuses one- or two-session fixtures rather than reporting
an overlapping holdout. The dataset also stores a provenance JSON record and a
fingerprint covering the source, feature config, timestamp transformation, and
aggregation mode.

`--checkpoint-every N` evaluates the same uninterrupted model trajectory every
N epochs, records train/validation metrics in `metrics.json`, and writes a
resumable policy at `checkpoints/epoch-XXXXXXXX/policy.json`; holdout is
evaluated only for the final policy. Burn and Candle policies now carry
explicit seed-derived initial weights plus a backend-neutral AdamW state
(moments and step count), so `--resume-policy` continues the optimizer rather
than silently resetting it. Existing policies without optimizer state remain
loadable as weight-only resumes, but they are not exact optimizer continuations.
On resume, the original policy seed is retained as provenance; `--seed` only
controls a fresh run.

On the local replay sample (19,191 one-minute bars, 18 complete sessions, 579
events, 56 features), an optimized CPU build prepared the artifact in about
7.4 seconds and trained 50 epochs in about 0.3 seconds. The supervised CLI
now provides the deterministic `cpu-linear` reference plus parity-preserving
Candle and Burn linear classifiers. Candle uses CPU by default and can use
CUDA when built with `backend-candle-cuda`; Burn uses `burn-ndarray` on CPU
and native Burn CUDA when built with `backend-burn-cuda`. The GTX 1080 Ti has
a working Burn CUDA FP32 smoke path, but Candle CUDA is not a viable local
backend for this Pascal `sm_61` device in the tested build.

CUDA probe status on the local GTX 1080 Ti is recorded in the implementation
work log: Candle CUDA first requires a GCC version supported by CUDA 12.8; with
GCC 14 it then fails to compile its scalar-half `atomicAdd` kernel for `sm_61`.
Burn CUDA successfully ran an FP32 matmul. The local UI therefore advertises
Burn CUDA but keeps Candle CUDA disabled on this Pascal card.
