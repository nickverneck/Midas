# ES Candle CPU Training Pipeline

This is the current fastest path on the Mac dev machine.

- Backend: `candle`
- Device: `cpu`
- Build profile: `release`
- Instrument preset: `ES Mini`
- Data mode: `windowed`
- Window: `700`
- Step: `128`

## Why this path

- Candle CPU has been the fastest backend on this Mac in both runtime and binary size.
- Running through the frontend API keeps the launch path aligned with the UI and avoids backend-specific env drift.
- The new helper scripts write JSON summaries so you do not have to inspect `ga_log.csv` or `rl_log.csv` by hand.

## Data split used for the test run

The ES data is currently stored by week under [data/train](/Users/nick/Documents/dev/Midas/data/train), not under clean `train/val/test` ES folders.

Weekly ES files:

- [data/train/12-29-2025/ES_F.parquet](/Users/nick/Documents/dev/Midas/data/train/12-29-2025/ES_F.parquet)
- [data/train/01-04-2026/ES_F.parquet](/Users/nick/Documents/dev/Midas/data/train/01-04-2026/ES_F.parquet)
- [data/train/01-12-2026/ES_F.parquet](/Users/nick/Documents/dev/Midas/data/train/01-12-2026/ES_F.parquet)
- [data/train/01-20-2026/ES_F.parquet](/Users/nick/Documents/dev/Midas/data/train/01-20-2026/ES_F.parquet)

Walk-forward split used here:

1. GA fold 1
   Train: `12-29-2025`
   Eval: `01-04-2026`
   Test: `01-12-2026`
2. RL PPO fold 2
   Train: `01-04-2026`
   Eval: `01-12-2026`
   Test: `01-20-2026`

Important:

- I did **not** continue weights from fold 1 into fold 2.
- The fold shift is chronological, but each fold retrains fresh.
- That keeps the walk-forward cleaner and avoids training on what was previously used as a reported holdout result.

## ES preset used

The run used the ES preset explicitly instead of relying on symbol inference:

- `--margin-mode per-contract`
- `--margin-per-contract 500`
- `--contract-multiplier 50`
- `--auto-close-minutes-before-close 5`

This matters because [config/symbols.yaml](/Users/nick/Documents/dev/Midas/config/symbols.yaml) currently has `ES`, not `ES=F` or `ES_F`.

## Windowing choice

Why `700 / 128`:

- `700` bars is about `11.7` hours of 1-minute data, which is long enough to include more than one intraday regime.
- `128` bars is about `2.1` hours, which still gives many windows per weekly file without the extreme overlap of a `64` step.
- For scalping-style experiments this is a better first pass than the frontend default of a full-week window.

## Helper scripts

New scripts added for this workflow:

- [scripts/run_train_via_api.mjs](/Users/nick/Documents/dev/Midas/scripts/run_train_via_api.mjs)
- [scripts/run_es_candle_walkforward.mjs](/Users/nick/Documents/dev/Midas/scripts/run_es_candle_walkforward.mjs)

What they do:

- Launch training through `POST /api/train`
- Stream the live SSE output to the terminal
- Discover the real run directory from trainer stdout
- Pull structured rows from `GET /api/logs`
- Write `run_summary.json` into each run directory
- Write fold summaries plus `pipeline_summary.json` under `runs/pipeline_reports/...`

## How to run it

Start the frontend server:

```bash
cd frontend
bun run dev --host 127.0.0.1 --port 4173
```

From the repo root, run the ES Candle walk-forward:

```bash
node scripts/run_es_candle_walkforward.mjs --host http://127.0.0.1:4173 --profile release --mode both
```

That will run:

- GA on the first weekly split
- PPO RL on the shifted weekly split

If you want a single run instead of the two-stage pipeline:

```bash
node scripts/run_train_via_api.mjs \
  --engine ga \
  --host http://127.0.0.1:4173 \
  --profile release \
  --params-file /path/to/params.json
```

## How to inspect runs without opening CSVs

The existing API surface is usable, but incomplete.

Useful endpoints:

- `GET /api/files?dir=runs_ga`
- `GET /api/files?dir=runs_rl`
- `GET /api/logs?dir=<runDir>&log=ga|rl`
- `GET /api/logs?dir=<runDir>&log=ga|rl&mode=summary&key=gen|epoch`
- `GET /api/behavior?mode=list&dir=<runDir>` for GA behavior CSVs

Examples:

```bash
curl -s 'http://127.0.0.1:4173/api/files?dir=runs_ga'
curl -s 'http://127.0.0.1:4173/api/logs?dir=runs_ga/es_candle_fold1_2026-03-15T08-33-09-973Z&log=ga&mode=summary&key=gen'
curl -s 'http://127.0.0.1:4173/api/logs?dir=runs_rl/es_candle_fold2_2026-03-15T08-33-09-973Z&log=rl&limit=200'
curl -s 'http://127.0.0.1:4173/api/behavior?mode=list&dir=runs_ga/es_candle_fold1_2026-03-15T08-33-09-973Z'
```

Current gaps:

- `/api/train` is only an SSE stream, not a proper run-status API.
- `/api/logs?mode=summary` only summarizes `fitness`.
- RL summary rows still come back with a `gen` key even when the grouping key is `epoch`.
- There is no endpoint that directly exposes final checkpoint paths, `training_stack.json`, or final test metrics.

The helper scripts work around those gaps by writing:

- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T08-33-09-973Z/run_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-15T08-33-09-973Z/run_summary.json)
- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T08-33-09-973Z/pipeline_summary.json)

## Test run findings

Latest pipeline report:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T08-33-09-973Z/pipeline_summary.json)

GA fold 1 artifacts:

- [runs_ga/es_candle_fold1_2026-03-15T08-33-09-973Z](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T08-33-09-973Z)
- [ga_log.csv](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T08-33-09-973Z/ga_log.csv)
- [best_overall_policy.safetensors](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T08-33-09-973Z/best_overall_policy.safetensors)

GA fold 1 findings:

- Best selection row: `gen 1 / idx 6`
- Selection fitness: `-4110.5587`
- Eval total PnL on that row: `+101.175`
- Eval drawdown on that row: `10725.0`
- Best eval-PnL row: `gen 0 / idx 2`
- Eval total PnL there: `+4966.9875`
- But that row had very poor selection fitness: `-530788.1956`
- Held-out test result was still bad: `fitness -263223.93`, `pnl -259623.75`, `mdd 7200.30`

Interpretation:

- The GA selection metric is doing what it should: it refused to pick the highest raw eval-PnL candidate because drawdown/risk was much worse.
- Even so, the fold-1 policy did not generalize to the next week.

RL PPO fold 2 artifacts:

- [runs_rl/es_candle_fold2_2026-03-15T08-33-09-973Z](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-15T08-33-09-973Z)
- [rl_log.csv](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-15T08-33-09-973Z/rl_log.csv)
- [ppo_final.safetensors](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-15T08-33-09-973Z/ppo_final.safetensors)

RL PPO fold 2 findings:

- Best fitness epoch: `1`
- Fitness there: `-775.8990`
- Eval PnL there: `+834.375`
- Eval drawdown there: `5643.575`
- Best eval-PnL epoch: `0`
- Eval PnL there: `+5309.375`
- But the fitness there was still negative: `-1847.0132`
- Held-out test result was also bad: `pnl -12068.75`, `sortino -0.0266`, `mdd 15166.675`

Interpretation:

- PPO produced some positive eval weeks, but the risk-adjusted objective still penalized them heavily.
- The held-out week remained clearly negative, so this is not a deployable ES model.

## Practical conclusions

- Candle CPU is the correct default backend on this Mac for fast iteration.
- Use `release` builds only for any serious comparison or training pass.
- Use GA first to sanity-check the environment and reward shaping.
- Then run PPO on the next chronological fold.
- Do not treat these specific ES policies as valid. The current environment/features/reward still need work.

## What to improve next

1. Add a real run manifest/status endpoint instead of parsing `stdout` for the run directory.
2. Add a JSON endpoint for final metrics plus checkpoint paths.
3. Add an RL behavior/rollout endpoint similar to GA behavior traces.
4. Add instrument-specific presets for `NQ` and `MNQ` before copying this pipeline to them.
5. Keep the walk-forward split pattern, but retrain fresh on each shifted fold.

## If you bypass the frontend API

For this Candle CPU path, direct `cargo run --release` is fine technically.

For libtorch, use the existing wrapper so the PyTorch/libtorch env is set correctly:

```bash
scripts/with-venv-libtorch.sh cargo run --release --features torch --bin train_ga -- ...
```

For consistency, the recommended path is still the frontend API plus the helper scripts above.
