# Cross-conditioned meta-gate training design

Status: design baseline, 2026-08-12

## Decision

The proposed model is viable, but it should be implemented as an event-level
meta-gate over an already-defined signal schedule. It should not initially be
implemented by adding another unconstrained action to the current bar-by-bar
policy trainer.

The existing Midas trainer is useful infrastructure: it already has GA and RL
runners, Candle model code, checkpoint formats, Parquet loading, and a four-way
bar action environment. The missing abstraction is a deterministic event
dataset and event simulator for decisions made only when a base crossover
occurs.

The first production-shaped experiment should be:

1. Generate causal rows for EMA 10/30 and the higher-timeframe context family
   (initially EMA 210/240, with HMA/KAMA/ADX families selectable later).
2. Replay the rows sequentially with fixed baselines: always normal, always
   skip, always inverted, and the current hand-written gate.
3. Train a small normal/skip classifier or GA policy on earlier windows.
4. Validate on a later, untouched window by event and by complete day.
5. Add invert only after skip and normal are stable; inversion has a larger
   action and leakage risk than abstention.

## Event row contract

Every event row must be self-contained and reproducible. The exporter should
accept the server-bar schema (`timestamp`, `ts_ns`, `open`, `high`, `low`,
`close`, `volume`, `row_idx`) and add instrument/contract metadata rather than
requiring those columns to be present in the raw replay cache.

Required identity and decision fields:

- event id, timestamp, nanosecond timestamp, instrument, contract, session and
  source file;
- cross family and parameters (`ema`, `hma`, `adx`, fast period, slow period);
- raw direction (`+1` bullish, `-1` bearish), cross age, and the raw fast/slow
  values;
- current position and whether a protection/session exit is active;
- the exact feature schema version and source-bar row index.

The initial feature groups should be independently selectable:

- price/volatility: close returns, true range, ATR-normalized distance;
- trigger structure: fast-minus-slow spread, each EMA slope normalized by ATR,
  bars since cross, and the trigger family;
- regime: EMA 210/240 spread and slopes, ADX and ADX slope, DI imbalance,
  Kaufman efficiency ratio, and choppiness/realized-volatility regime;
- activity/context: relative volume, VWAP distance, session, hour, and
  instrument/contract identity.

All rolling values must be computed using bars at or before the event. Warmup
bars are context, not training examples. Scaling parameters are fit on the
training partition only and stored beside the model.

## Action and lifecycle semantics

The model action is interpreted relative to the raw crossover direction:

- `normal`: target the raw direction;
- `invert`: target the opposite direction;
- `skip`: make no position change and keep the current position;
- protection and session-close rules always remain active.

This produces deterministic behavior at consecutive conflicting events:

- bullish + normal while flat -> long;
- bearish + skip while long -> remain long;
- bearish + normal while long -> close/reverse according to the configured
  execution policy;
- bearish + invert while long -> target long, so it is a no-op rather than a
  duplicate buy;
- the first accepted signal after a stop may enter again.

The gate must not silently close a position on `skip`. If a separate policy is
desired later, expose it explicitly as `skip_hold` versus `skip_flatten` and
evaluate both. The default is `skip_hold`, because the question being tested is
whether to accept or reject a new cross, not whether to override risk exits.

## Labels, reward, and leakage controls

There are two useful targets and they must not be conflated:

1. A fixed-horizon diagnostic label, such as net return after N bars with
   configured TP/SL/costs. This is easy to inspect and useful for feature
   screening.
2. A sequential simulator reward, where the action is applied to the current
   position and the next event/protection exit determines the realized result.
   This is the acceptance criterion for a trading policy.

The first implementation should write both counterfactual outcomes for each
   event (`normal_pnl`, `invert_pnl`, `skip_pnl`) and then run the sequential
   simulator independently. It must not select the best counterfactual action
   in the same window used to report performance.

Use chronological train/validation/holdout windows, grouped by day and
contract where possible. Apply a purge/embargo around split boundaries when
the label horizon overlaps. Never randomly shuffle rows across time. Report
per-day and per-window results, not only an aggregate across all weeks.

Useful diagnostics are action occupancy, accepted/skipped/inverted event
counts, turnover, costs, max drawdown, worst day, median day, and the result
relative to always-normal. Accuracy alone is not a trading metric.

## Training order

GA should come first for the event gate. It can optimize sequential reward and
action-occupancy penalties with a small MLP without pretending that sparse
cross events are a dense bar-level control problem. A supervised baseline is
also valuable before GA: it provides a fast leakage check and a reference for
whether the features contain signal at all.

RL should follow after the event environment and transition semantics are
frozen. The RL observation should occur at event/protection boundaries, include
position state, and transition to the next event or forced exit. Reusing the
current bar-step PPO environment without this adapter would train the wrong
problem. There is no reason to assume five continuous days of training is
needed; checkpointed short runs with repeatable validation are preferable until
learning curves and seed stability justify larger runs.

The first model should be small enough for CPU inference. Since inference is
needed only at a crossover (not every tick), the one-minute execution budget is
dominated by data handling and order routing, not a small MLP forward pass.

## Data sufficiency

The current five-day/one-week replay windows are appropriate for holdout
experiments, not for claiming a reliable learned model. We need many
chronological windows and multiple contracts/instruments. A practical minimum
for the first serious run is at least hundreds of events per split and
preferably more than 1,000 non-overlapping event outcomes across GC, ES, and
NQ. The model must receive instrument identity, but splits must prevent the same
contract/session pattern from appearing on both sides of a validation boundary.

## Reuse boundaries

Reuse from Midas:

- Candle/GA/RL model and checkpoint backends;
- device selection and backend reporting;
- Parquet utilities, logging, and configuration patterns;
- feature kernels where their period and causal behavior match the event row.

Reuse from Trader:

- replay-cache server bars;
- existing signal/protection/replay diagnostics for validating the exported
  schedule;
- the eventual read-only runtime adapter that loads a CPU model artifact.

Keep the event exporter and evaluator in a shared Midas-facing library or
standalone CLI. Do not make the parent training crate depend on the Trader
application crate, which would create a dependency cycle. Pass explicit paths
and metadata across that boundary.

## UI and CLI acceptance criteria

The same configuration must be runnable from both interfaces. It needs:

- source bar paths or replay-cache selection;
- instrument/contract and event families;
- trigger/context periods;
- selectable feature groups and a visible feature list;
- action set (`normal`, `skip`, optional `invert`) and lifecycle semantics;
- TP/SL/cost/session settings;
- chronological train/validation/holdout windows and purge bars;
- GA/RL mode, backend, device, seed, checkpoint interval, and run directory;
- baseline comparison and an event/trade/chart report.

The UI should show dataset provenance, warmup, row/event counts, split dates,
action occupancy, and whether a displayed result is train, validation, or
holdout. A large empty chart with hidden configuration is not sufficient for
this workflow.

## Staged implementation

1. Add a versioned event schema, causal exporter, fixed-action evaluator, and
   golden tests for the lifecycle rules.
2. Add a small supervised/GA event model using the shared schema and compare it
   with frozen baselines across chronological windows.
3. Add an event-level RL adapter only after the evaluator is trusted.
4. Add persistent jobs and the event-gate controls to the web UI; keep the CLI
   as the reference path.
5. Add a read-only Trader runtime adapter and TUI diagnostics showing the model
   action, raw direction, feature/model version, and reason for skip/invert.
6. Run fresh backend, frontend, and UX reviewers after each implementation
   slice. Use `bdg` for rendered checks and the Trader TUI-control workflow for
   safe replay validation.

## Known local GPU issue

The local machine has a GTX 1080 Ti and a CUDA 12.8 toolkit. Candle CUDA
compilation currently reaches a `candle-kernels` half-precision `atomicAdd`
failure for compute capability 6.1. This is a build compatibility issue, not
evidence that the model is too large. Do not “solve” it by compiling only for a
newer architecture that the 1080 Ti cannot execute. Test Burn/libtorch CUDA or
patch/use a kernel path that is valid for sm_61, and retain CPU as the reliable
fallback until that is verified.

## Local and hosted training plan

The GTX 1080 Ti is a development and smoke-test target, not a reason to run
multi-day jobs before the event environment is trustworthy. The current
event-level prototype is deliberately CPU-deterministic and small; its model
forward pass is cheap enough for one-minute inference because inference occurs
at crossover events, not on every bar. GA can later use the existing Burn CUDA
path if the sm_61 runtime smoke test succeeds. RL still needs a dedicated
event-transition adapter before moving to a GPU; putting the current dense
bar-step PPO runner on CUDA would optimize the wrong problem.

The staged compute decision is:

1. Run exporter, fixed baselines, CPU GA, and CPU RL smoke tests locally with a
   fixed seed. Repeat across chronological windows and contracts before paying
   for compute.
2. Benchmark the event adapter and validate one Burn CUDA GA run on the 1080 Ti.
   Record wall time, VRAM, seed, backend, driver, and artifact hashes. A CUDA
   build that compiles but has no runtime smoke result is not considered GPU
   support.
3. Only after the sequential Trader schedule replay is green, use a rented GPU
   for parallel seeds or larger feature sweeps. Checkpoint frequently and keep
   the local CPU path as the reproducibility reference.

For short experiments, a marketplace RTX 4090/5090 or a Colab L4 is more
appropriate than an H100. An H100 is reserved for a measured workload that is
actually bottlenecked by training throughput. Prices change, so the launcher
should display the provider and date used for any estimate rather than baking a
price into the application. Current reference pages are RunPod pricing,
Google Colab Enterprise pricing, and Lambda GPU instances.

## Review findings folded into the build

The parallel ML, replay, infrastructure, and product reviews found the same
failure modes in the existing console and trainers:

- the current GA/RL environments decide on every bar, so an event gate must
  have its own episode/transition adapter;
- existing trainer loaders can silently reuse a path when a requested split is
  missing, which is unacceptable for holdout research;
- checkpoints do not carry a complete feature/action/dataset manifest and must
  be treated as warm starts until that metadata is added;
- the web train request is tied to the browser connection and has no durable
  job ID or reconnectable run record;
- the frontend does not yet expose feature/label provenance, baseline
  comparison, or a decision-level replay inspector;
- the replay cache has the right immutable server-bar source, but Trader does
  not yet consume an external decision schedule.

Therefore the first implementation slice is intentionally narrower: a strict
event exporter, fixed diagnostic evaluator, and web preparation/evaluation
workflow. The next slice must add chronological split manifests and a
sequential Trader schedule replay before any learned action is called
deployable. A learned GA/RL prototype may use the fixed-horizon rows for
feature-signal screening, but its report must retain the diagnostic label
warning and be validated through Trader replay afterward.

For the current documented GC sample, the exporter produced 864 EMA10/30
events over the replay-cache span after warmup and a 30-bar horizon. With a GC
contract multiplier of 100 and a $3.10 event cost, the fixed-horizon evaluator
reported always-normal -$8,448.40, always-invert +$3,091.60, and a hindsight
oracle +$605,290.20. These numbers are not a strategy result: the oracle is
hindsight, the event outcomes overlap, and no position/protection lifecycle is
replayed. They are a useful smoke test that confirms the event direction and
units are being written, and they show why a model cannot be judged from the
oracle column.

As a separate schema smoke test, the same exporter ran on ESU6 for Aug 3–7
and produced 245 events. With a 50-point multiplier and $3.10 event cost,
always-normal was -$2,997.00 and always-invert was +$1,478.00. The small CPU
GA/RL prototypes completed and wrote artifacts, but their validation splits
were negative; that is consistent with the current conclusion that one-week
fixed-horizon rows are useful for plumbing and feature screening, not for
claiming a deployable learned gate.

## Current validation status

The current web/CLI training slice is a research harness, not a live strategy
runner. Its fixed-horizon labels intentionally do not model the current
position, TP/SL, trailing protection, forced session exits, pending reversals,
or intrabar fill ordering. `skip_pnl = 0` therefore means “no new diagnostic
trade,” not “the existing position earned zero.” A policy artifact must not be
loaded by Trader as an executable schedule until the sequential adapter below
has reproduced the normal replay ledger.

The next acceptance gate is a sequential event replay that consumes the same
event timestamps and policy actions, carries current target quantity across
events, lets protection/session rules run in their normal order, and writes a
decision ledger. It must test at least: flat → long, opposite cross + skip,
opposite cross + normal reversal, and an opposite-side `invert` while already
holding. The output must report fills, costs, exit reasons, action occupancy,
and per-day PnL against a frozen always-normal schedule.

The contextual GA/RL runner now belongs in the “feature screening” stage. RL
in that stage means a one-step contextual policy-gradient baseline; it is not
episodic RL. The eventual RL environment must transition from one crossover or
forced-exit boundary to the next and include position/protection state.

## Final vertical-slice validation (2026-08-12)

The current implementation has been exercised through the parent CLI, the web
route, and the Trader read-only schedule inspector. The GC July 5–August 8
source produced 864 events. With a 30-bar purge, the retained split was
518/167/170 for train/validation/holdout. On the holdout, fixed normal was
`+$2,713.00`, fixed invert was `-$3,767.00`, the CPU GA prototype was
`-$2,612.90`, and the CPU contextual-RL prototype was `-$16,513.50`. A second
ESU6 August 3–7 source produced 245 events and retained 147/46/46; its GA and
RL holdouts were `+$1,347.30` and `+$478.70`, respectively, while both had
negative validation PnL. These are still fixed-horizon diagnostics, not live
strategy results.

The browser flow was tested with `bdg`: prepare → fixed baselines → GA
prototype completed, the diagnostic warning and train/validation/holdout cards
were visible, and an intentionally missing source after a successful run
cleared the old dataset/results and disabled both downstream buttons. The
frontend passed `npm run check` with zero diagnostics and `npm run build`.
The Trader command
`print-meta-gate-schedule --events ... --policy ... --initial-side flat`
reconstructed a bounded stateful normal/skip/invert target-side schedule
without loading account, broker, engine, or order-routing paths. It is an
inspection boundary only; sequential fills, protections, and session exits
remain the next required acceptance gate.

The final integrity pass also makes event artifacts self-describing and
rejects mixed datasets: instrument, contract, feature schema, serialized gate
configuration, source path/size/row count, and timestamp source must be
non-empty and consistent across every row. `ts_ns` is preferred; Datetime
units and Date-at-midnight are converted explicitly, while ambiguous numeric
`timestamp`/`date` columns are rejected. The evaluator now requires the same
event schema and provenance rather than accepting arbitrary PnL columns.
Prepare, evaluate, and train web subprocesses are bounded to ten minutes,
terminate with SIGTERM on timeout, and inherit request-abort cancellation.
