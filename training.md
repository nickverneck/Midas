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
- For serious validation, use a very high `eval-windows` value. The current trainers cap both validation and final test to `min(eval_windows, available_windows)`, so a small value only scores the first few windows of the week.

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

Longer all-window evaluation run used for the latest pass:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode both \
  --ga-generations 12 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-eval-windows 999 \
  --rl-epochs 18 \
  --rl-train-windows 24 \
  --rl-ppo-epochs 4 \
  --rl-eval-windows 999
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

## Longer all-window run findings

Latest longer-run report:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T08-43-08-762Z/pipeline_summary.json)

GA longer fold 1 artifacts:

- [runs_ga/es_candle_fold1_2026-03-15T08-43-08-762Z](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T08-43-08-762Z)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T08-43-08-762Z/run_summary.json)

GA longer fold 1 findings:

- Run size: `12` generations, `12` candidates, `eval-windows=999`
- Runtime: about `31s`
- Best selection row: `gen 10 / idx 0`
- Eval total PnL on that row: `+1276.2091`
- Eval drawdown on that row: `23101.05`
- PnL-to-drawdown ratio on that row: about `0.055`
- Held-out test result: `pnl -106723.69`, `mdd 36830.25`
- Held-out test PnL-to-drawdown ratio: about `-2.90`

Interpretation:

- Longer GA training did not make the ES policy effective.
- Even when validation found some positive-PnL rows, the drawdown was far too large and the next-week test stayed strongly negative.

RL PPO longer fold 2 artifacts:

- [runs_rl/es_candle_fold2_2026-03-15T08-43-08-762Z](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-15T08-43-08-762Z)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-15T08-43-08-762Z/run_summary.json)
- [ppo_final.safetensors](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-15T08-43-08-762Z/ppo_final.safetensors)

RL PPO longer fold 2 findings:

- Run size: `18` epochs, `24` train windows per epoch, `eval-windows=999`
- Runtime: about `15.5s`
- Best fitness epoch: `11`
- Eval PnL there: `+173.7069`
- Eval drawdown there: `10752.4655`
- PnL-to-drawdown ratio there: about `0.016`
- Best eval-PnL epoch: `16`
- Eval PnL there: `+3327.8017`
- Eval drawdown there: `10208.8922`
- PnL-to-drawdown ratio there: about `0.326`
- Held-out test result: `pnl -658.1019`, `mdd 9937.1685`
- Held-out test PnL-to-drawdown ratio: about `-0.066`

Interpretation:

- Longer PPO training was materially better than the short PPO pass on the held-out week.
- But it still does **not** meet the bar for effectiveness because held-out PnL remained negative.
- The drawdown is still too large relative to the test result to call this a reasonable ES policy.

## 50-generation GA follow-up

Latest 50-generation GA report:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T08-47-05-273Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T08-47-05-273Z/run_summary.json)

Command used:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-eval-windows 999 \
  --window 700 \
  --step 128
```

Results:

- Runtime: about `120.6s`
- Best selection fitness reached: `-5539.7592` at `gen 25`
- Final generation best-of-gen fitness: `-7114.1627`
- Held-out test result: `pnl -12712.86`, `mdd 8311.20`, `sortino 0.01`
- Held-out test PnL-to-drawdown ratio: about `-1.53`

Interpretation:

- Going from `12` generations to `50` materially improved the held-out test from roughly `-106k` PnL to about `-12.7k` PnL.
- That is real progress, but it is still **not effective** by the standard we care about because held-out PnL remained negative.
- The selection curve also flattened after the mid-20s. The run mostly oscillated in the same negative-fitness basin instead of finding a new regime.
- Based on this run, I would not jump straight to `100` generations yet unless we first change the objective, features, or penalties. More generations alone are not clearly solving the problem.

## 50-generation GA with eval-based selection tuning

This rerun fixed an important mistake in the earlier experiments: previous GA runs were not using eval-aware selection at all because `selection_use_eval` was off. The tuned rerun explicitly enabled it.

Latest tuned GA report:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T09-00-18-795Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T09-00-18-795Z/run_summary.json)

Command used:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-eval-windows 999 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.1 \
  --selection-eval-weight 0.9 \
  --selection-gap-penalty 0.1 \
  --w-pnl 1 \
  --w-sortino 1 \
  --w-mdd 0.25
```

Results:

- Runtime: about `140s`
- Best selection row: `gen 28 / idx 9`
- Eval fitness there: `-7390.7238`
- Eval total PnL there: `-812.7966`
- Eval drawdown there: `8459.85`
- Held-out test result: `pnl -64518.09`, `mdd 30432.75`, `sortino -0.01`
- Held-out test PnL-to-drawdown ratio: about `-2.12`

Interpretation:

- Eval-aware selection did improve the *validation-fitness band* during the run. It spent a lot of time around roughly `-10k` to `-15k` eval fitness instead of the much worse earlier ranges.
- But the actual held-out week got worse than the train-only 50-generation GA run.
- So this particular tuning did **not** improve real out-of-sample effectiveness.
- The main conclusion is that objective tuning without fixing evaluation noise is not reliable enough yet.

## Stochasticity note

Current GA evaluation is stochastic, and that is a real problem.

Why:

- Candle GA samples actions with `rand::thread_rng()` during candidate evaluation in [src/bin/train_ga/backends/candle.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/backends/candle.rs).
- That means the same genome can score differently across different evaluations.
- In a GA, noisy fitness is especially harmful because parent selection, elite retention, and mutation pressure all depend on rank ordering.

Practical impact:

- A genome can win because of rollout luck, not because it is actually better.
- Selection-weight tuning can look promising on one run and collapse on the next.
- Comparing two GA runs is less trustworthy than it should be.

Recommendation:

- The next high-value fix is to make GA evaluation deterministic or at least reproducibly seeded per genome/window.
- Until that is fixed, treat GA tuning results as directional only, not authoritative.

## Deterministic GA follow-up

I implemented deterministic GA action selection across all three GA backends:

- `candle`
- `burn`
- `libtorch`

The change is simple: candidate evaluation now uses `argmax(logits)` with a stable first-index tie break instead of stochastic action sampling.

Relevant code paths:

- [src/bin/train_ga/backends/candle.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/backends/candle.rs)
- [src/bin/train_ga/backends/burn.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/backends/burn.rs)
- [src/bin/train_ga/ga.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/ga.rs)

I then reran the same ES Candle CPU walk-forward validation used for the earlier 50-generation comparison:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-eval-windows 999 \
  --window 700 \
  --step 128
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T09-23-58-174Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T09-23-58-174Z/run_summary.json)
- [ga_log.csv](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T09-23-58-174Z/ga_log.csv)

Results:

- Runtime: about `138.6s`
- Best selection row: `gen 20 / idx 10`
- Best selection fitness: `-12160.7024`
- That row's eval total PnL: `+2009.8727`
- That row's eval drawdown: `58879.25`
- Best raw eval-PnL row: `gen 28 / idx 7`
- That row's eval total PnL: `+13157.8375`
- That row's eval drawdown: `59073.0`
- Held-out test result: `pnl -142424.94`, `mdd 49030.90`, `sortino -0.00`
- Held-out test PnL-to-drawdown ratio: about `-2.90`

Comparison versus the earlier stochastic 50-generation run:

- Earlier stochastic held-out test: `pnl -12712.86`, `mdd 8311.20`
- Deterministic held-out test: `pnl -142424.94`, `mdd 49030.90`

Interpretation:

- Deterministic fitness did exactly what it was supposed to do mechanically: the ranking became stable and reproducible.
- But on this ES split it did **not** improve walk-forward generalization. It made it much worse.
- The search also stagnated early. The best selection score improved by generation `20` and then stayed flat through generation `49`.
- That suggests the previous stochastic policy sampling was adding exploratory behavior during evaluation, while pure greedy action selection collapses too quickly into a brittle policy class for this current objective/setup.

Current conclusion:

- Deterministic GA is still the correct baseline if the goal is trustworthy fitness ranking.
- But deterministic greedy evaluation alone is not enough to make this ES pipeline effective.
- The next improvement should not be “go back to noisy scoring.” The next improvement should be a better deterministic search target:
  - evaluate across more folds
  - improve reward/penalty shaping
  - consider deterministic action smoothing or logit-margin penalties
  - possibly re-score finalists under multiple controlled perturbations after the deterministic search phase

## GA search-loop improvement

After the deterministic GA change, the next problem was clear: the search loop was converging too hard.

What changed in code:

- Added a configurable GA parent pool fraction in [src/bin/train_ga/args.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/args.rs).
- Changed reproduction in [src/bin/train_ga.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga.rs) so crossover can draw from the top `parent_pool_frac` of the scored population instead of only the elite set.
- Kept support for random immigrants and surfaced the new parent-pool control in [scripts/run_es_candle_walkforward.mjs](/Users/nick/Documents/dev/Midas/scripts/run_es_candle_walkforward.mjs).

Why:

- The old deterministic loop still preserved elites and then bred only from those elites.
- With a small population and a deterministic evaluator, that caused early convergence and long plateaus.
- Broadening the mating pool while injecting a small number of new genomes each generation is a standard way to keep GA exploration alive without throwing away the best candidates.

## Seeded comparison

To make the comparison fair, I ran the same ES fold with a fixed seed: `42`.

### Seeded baseline

Settings:

- `elite-frac 0.33`
- `parent-pool-frac 0.33`
- `immigrant-frac 0`
- train-only selection
- `window 700`
- `step 128`

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T09-52-02-384Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T09-52-02-384Z/run_summary.json)

Result:

- Held-out test: `pnl -292624.04`, `mdd 79739.45`

This was worse than the earlier unseeded deterministic run and confirms how fragile the old search loop was.

### Improved search

Settings:

- `elite-frac 0.17`
- `parent-pool-frac 0.67`
- `immigrant-frac 0.17`
- `selection-use-eval`
- `selection-train-weight 0.2`
- `selection-eval-weight 0.8`
- `selection-gap-penalty 0.1`
- `window 700`
- `step 128`
- same seed: `42`

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T09-54-15-387Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T09-54-15-387Z/run_summary.json)

Result:

- Held-out test: `pnl -15776.78`, `mdd 2900.00`, `sortino 0.04`

Interpretation:

- This is a large improvement over the seeded baseline.
- The run is still not profitable, so it still fails the “effective” bar.
- But this is the first change in the current GA pipeline that materially improved the out-of-sample result without relying on stochastic scoring.

Practical takeaway:

- For GA on ES, keep the improved search loop settings:
  - `elite-frac 0.17`
  - `parent-pool-frac 0.67`
  - `immigrant-frac 0.17`
  - eval-aware selection enabled

## Overlap test

I also tested whether reducing overlap would help generalization by increasing `step` from `128` to `256`, while keeping the improved search settings and the same seed.

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T17-20-33-926Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T17-20-33-926Z/run_summary.json)

Result:

- Held-out test: `pnl -77480.60`, `mdd 2777.10`, `sortino 0.02`

Interpretation:

- Reducing overlap did lower runtime a lot.
- It did **not** improve held-out PnL on this fold.
- So for now, the better ES GA setting remains `700 / 128`, not `700 / 256`.

## Updated recommendation

If I were running the GA leg again on ES right now, I would use:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.67 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.05 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.2 \
  --selection-eval-weight 0.8 \
  --selection-gap-penalty 0.1
```

That is not profitable yet, but it is the best GA configuration found so far in this ES Candle CPU walk-forward process.

## 100-generation check

I then kept the same improved deterministic GA settings and only increased generations from `50` to `100`.

Run command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 100 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.67 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.05 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.2 \
  --selection-eval-weight 0.8 \
  --selection-gap-penalty 0.1
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T21-11-03-127Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T21-11-03-127Z/run_summary.json)
- [best_overall_policy.safetensors](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T21-11-03-127Z/best_overall_policy.safetensors)

Results:

- Runtime: about `240.6s`
- Best selection row: `gen 94 / idx 7`
- Best selection score: `-12927.7560`
- That row's eval total PnL: `+1503.6136`
- That row's eval drawdown: `4812.5`
- Held-out test result: `pnl -302577.46`, `mdd 37835.60`, `sortino 0.01`

Interpretation:

- The longer run did keep improving validation after the earlier `50`-generation plateau.
- But that extra search time did **not** improve the only metric that matters here: the next-week holdout result.
- It was much worse than the best `50`-generation run on the same fold, which ended at `pnl -15776.78`, `mdd 2900.00`.
- So for this ES Candle CPU GA setup, `100` generations is not a better default than `50`.

## 500-generation check

I then pushed the same configuration much further to `500` generations, while disabling per-generation save/checkpoint/behavior outputs so the run stayed focused on the search itself instead of artifact churn.

Run command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 500 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.67 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.05 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --ga-save-top-n 0 \
  --ga-save-every 0 \
  --ga-checkpoint-every 0 \
  --ga-behavior-every 0 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.2 \
  --selection-eval-weight 0.8 \
  --selection-gap-penalty 0.1
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T21-22-39-980Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T21-22-39-980Z/run_summary.json)
- [best_overall_policy.safetensors](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T21-22-39-980Z/best_overall_policy.safetensors)

Results:

- Runtime: about `16` minutes
- Best selection row: still `gen 94 / idx 7`
- Best selection score: still `-12927.7560`
- That row's eval total PnL: still `+1503.6136`
- That row's eval drawdown: still `4812.5`
- Best raw eval-PnL row moved later to `gen 472 / idx 3`
- That row's eval total PnL: `+12698.2398`
- But that row's eval drawdown was `46360.8`
- Held-out test result: `pnl -302577.46`, `mdd 37835.60`, `sortino 0.01`

Interpretation:

- For this split, pushing from `100` to `500` generations did not improve the selected policy at all.
- The best selected row was unchanged from the `100`-generation run, which means the extra `400` generations mostly explored without finding a better risk-adjusted candidate.
- The high raw eval-PnL late in the run came with very large drawdown, so selection correctly rejected it.
- Most importantly, the held-out next week stayed exactly as bad as the `100`-generation run.
- This is strong evidence that “just train GA much longer” is not the lever that fixes ES here.

## Search-diversity follow-up

I added two practical GA controls after the deterministic run:

- `--immigrant-frac` to inject fresh random genomes each generation
- `--behavior-every` so walk-forward experiments can skip the huge per-generation behavior CSV dump

I also exposed the existing GA evolution knobs in the helper script so the ES pipeline can be rerun without hand-editing JSON:

- `elite-frac`
- `parent-pool-frac`
- `immigrant-frac`
- `mutation-sigma`
- `init-sigma`
- `seed`
- `save-top-n`
- `save-every`
- `checkpoint-every`
- `behavior-every`

Relevant code:

- [src/bin/train_ga/args.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/args.rs)
- [src/bin/train_ga.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga.rs)
- [scripts/run_es_candle_walkforward.mjs](/Users/nick/Documents/dev/Midas/scripts/run_es_candle_walkforward.mjs)

This also fixed a practical pipeline problem: the earlier 50-generation experiments were producing about `1G` of artifacts per run because the top candidate behavior was being written for both train and val every generation.

## Tuned deterministic run: `700 / 128`

Run command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.33 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.1 \
  --ga-init-sigma 0.5 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --ga-save-top-n 0 \
  --ga-save-every 0 \
  --ga-checkpoint-every 0 \
  --ga-behavior-every 0 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.15 \
  --selection-eval-weight 0.85 \
  --selection-gap-penalty 0.25
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T17-12-11-565Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T17-12-11-565Z/run_summary.json)

Results:

- Runtime: about `125.9s`
- Best selection row: `gen 28 / idx 10`
- Best selection score: `-15496.3502`
- That row's eval total PnL: `-112.3727`
- That row's eval drawdown: `2926.05`
- Held-out test result: `pnl -267603.22`, `mdd 39607.55`

Interpretation:

- This run was a real improvement in *validation behavior* relative to the deterministic train-only baseline.
- It found much lower-drawdown validation rows and avoided the very early score freeze.
- But the next-week test was dramatically worse than the earlier deterministic baseline.
- So better search diversity plus eval-aware ranking still did **not** produce a usable ES policy.

## Tuned deterministic run: `700 / 256`

Because `700 / 128` heavily overlaps windows, I also tried a sparser walk-forward with the same tuned GA and `step=256`.

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T17-20-33-925Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T17-20-33-925Z/run_summary.json)

Results:

- Runtime: about `56.1s`
- Best selection row: `gen 37 / idx 11`
- Best selection score: `-16826.7762`
- That row's eval total PnL: `+70.3`
- That row's eval drawdown: `913.4`
- Held-out test result: `pnl -285933.31`, `mdd 67728.45`

Interpretation:

- The sparser step reduced runtime a lot and still found very low-drawdown validation rows.
- But held-out test got even worse.
- That makes it unlikely that search diversity or overlap geometry is the primary blocker anymore.

## Reward / environment / model pass: `700 / 128`

I then changed the GA setup structurally instead of just tuning search hyperparameters.

Code changes:

- [src/bin/train_ga/data.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/data.rs)
  - normalized the large raw price-level feature block relative to `close` / `atr_14`
  - replaced raw volume and raw OBV with stationary transforms
  - switched time features to ET instead of UTC
  - added `minutes_to_close`, normalized position, `bars_in_position`, and `flat_steps`
- [src/bin/train_ga/metrics.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/metrics.rs)
  - added a shared forced-flat liquidation cost helper
  - made final candidate fitness explicitly use net PnL, Sortino, and drawdown together
- [src/bin/train_ga/backends/candle.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/backends/candle.rs)
- [src/bin/train_ga/backends/burn.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/backends/burn.rs)
- [src/bin/train_ga/ga.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/ga.rs)
  - all three GA backends now score candidates off the same forced-flat net outcome instead of realized-only PnL
- [src/env.rs](/Users/nick/Documents/dev/Midas/src/env.rs)
  - enabled the existing `session_close_penalty` knob instead of leaving it hardcoded off
- [scripts/run_es_candle_walkforward.mjs](/Users/nick/Documents/dev/Midas/scripts/run_es_candle_walkforward.mjs)
- [frontend/src/routes/train/+page.svelte](/Users/nick/Documents/dev/Midas/frontend/src/routes/train/+page.svelte)
  - changed ES-friendly defaults to `hidden=64` and `max-position=1`

Run command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.67 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.05 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --ga-save-top-n 0 \
  --ga-save-every 0 \
  --ga-checkpoint-every 0 \
  --ga-behavior-every 0 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.2 \
  --selection-eval-weight 0.8 \
  --selection-gap-penalty 0.1
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T22-01-02-572Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T22-01-02-572Z/run_summary.json)

Results:

- Runtime: about `89.7s`
- Best selection row: `gen 48 / idx 6`
- That row's eval total PnL: `-140.4045`
- That row's eval drawdown: `3606.7`
- Held-out test result: `pnl -19497.30`, `mdd 4104.60`, `sortino -0.02`

Interpretation:

- This was a large improvement versus the earlier catastrophic tuned deterministic run at the same `700 / 128` geometry, which ended at `pnl -267603.22`, `mdd 39607.55`.
- It also kept the next-week drawdown much tighter than the `100` and `500` generation runs.
- But it still did **not** beat the best earlier 50-generation deterministic-search baseline, which was about `pnl -15776.78`, `mdd 2900.00`.
- So the reward / environment / model pass made the GA objective more honest and the policy behavior less explosive, but it still did not produce a profitable held-out ES policy.

## GA reward-shape follow-up

After that reward / environment / model pass, I exposed the missing GA reward knobs in both the ES helper and the frontend:

- [scripts/run_es_candle_walkforward.mjs](/Users/nick/Documents/dev/Midas/scripts/run_es_candle_walkforward.mjs)
- [frontend/src/routes/train/+page.svelte](/Users/nick/Documents/dev/Midas/frontend/src/routes/train/+page.svelte)

The immediate reason was practical: the frontend previously did not expose `drawdown-penalty`, `session-close-penalty`, or any of the flat-hold controls, which made fast GA reward experiments awkward.

### Experiment A: remove flat-hold pressure, add drawdown/session pressure

Run command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.67 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.05 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --ga-save-top-n 0 \
  --ga-save-every 0 \
  --ga-checkpoint-every 0 \
  --ga-behavior-every 0 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.2 \
  --selection-eval-weight 0.8 \
  --selection-gap-penalty 0.1 \
  --drawdown-penalty 0.5 \
  --drawdown-penalty-growth 0.05 \
  --session-close-penalty 0.5 \
  --flat-hold-penalty 0 \
  --flat-hold-penalty-growth 0 \
  --max-flat-hold-bars 0
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T22-14-19-404Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T22-14-19-404Z/run_summary.json)

Results:

- Held-out test result: `pnl -61803.44`, `mdd 3868.35`, `sortino -0.02`

Interpretation:

- This was worse than the current best GA baseline.
- The added drawdown/session penalties made the search more conservative on drawdown, but they also pushed held-out PnL much lower.
- So that combined reward change was not an improvement.

### Experiment B: remove flat-hold pressure only

Run command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.67 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.05 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --ga-save-top-n 0 \
  --ga-save-every 0 \
  --ga-checkpoint-every 0 \
  --ga-behavior-every 0 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.2 \
  --selection-eval-weight 0.8 \
  --selection-gap-penalty 0.1 \
  --flat-hold-penalty 0 \
  --flat-hold-penalty-growth 0 \
  --max-flat-hold-bars 0
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T22-16-15-303Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T22-16-15-303Z/run_summary.json)

Results:

- Best selection row: `gen 44 / idx 4`
- That row's eval total PnL: `-600.2295`
- That row's eval drawdown: `2889.6`
- Held-out test result: `pnl -28327.54`, `mdd 3753.90`, `sortino -0.01`

Interpretation:

- This was better than Experiment A, but still worse than the earlier reward / environment / model pass (`pnl -19497.30`, `mdd 4104.60`).
- So simply removing flat-hold pressure did not unlock a profitable or even clearly better ES GA policy.
- The earlier suspicion that flat-hold penalties were the main remaining bottleneck is not supported by the capped, normalized GA setup.

### Updated takeaway from the reward sweeps

- The new helper/frontend controls are useful and should stay.
- In the **old** GA regime, flat-hold penalties were clearly pathological.
- In the **current** capped + normalized GA regime, reward retuning alone did not beat the best prior 50-generation result.
- The next GA optimization should not be another blind weight sweep. It should be a structural change, most likely one of:
  1. simplify the action semantics for `max-position = 1`
  2. inspect per-step behavior of the selected policy on the test fold
  3. reduce model/action ambiguity rather than adding more penalties

## Structural GA change: `short / flat / long`

I then changed the GA action surface itself instead of tuning penalties further.

Code changes:

- [src/bin/train_ga/actions.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/actions.rs)
  - added a shared GA intent mapper for `short / flat / long`
  - policy outputs now represent target position intent instead of `buy / sell / hold / revert`
- [src/bin/train_ga/model.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/model.rs)
- [src/bin/train_ga/backends/candle.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/backends/candle.rs)
- [src/bin/train_ga/backends/burn.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/backends/burn.rs)
- [src/bin/train_ga/ga.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/ga.rs)
- [src/bin/train_ga/portable.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/portable.rs)
- [src/bin/train_ga/backends/mlx.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/backends/mlx.rs)
  - switched all GA backends and exports from 4 logits to 3 logits
- [src/bin/train_ga.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga.rs)
- [src/bin/train_ga/types.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/types.rs)
  - behavior CSV now records the policy intent in `action` and the actual env operation in `effective_action`

The environment still executes with the existing low-level actions, but the policy now chooses only:

- `short`
- `flat`
- `long`

Run command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 50 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.67 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.05 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --ga-save-top-n 0 \
  --ga-save-every 0 \
  --ga-checkpoint-every 0 \
  --ga-behavior-every 0 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.2 \
  --selection-eval-weight 0.8 \
  --selection-gap-penalty 0.1
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T22-27-17-997Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T22-27-17-997Z/run_summary.json)

Results:

- Runtime: about `93.7s`
- Best selection row: `gen 37 / idx 8`
- That row's eval total PnL: `+639.1136`
- That row's eval drawdown: `878.2`
- Held-out test result: `pnl -550.33`, `mdd 2877.30`, `sortino -0.01`

Interpretation:

- This is the strongest GA result so far on the ES fold.
- It is still not positive on held-out test, so it is still not deployable.
- But it is a very large improvement versus the earlier best GA result (`pnl -19497.30`, `mdd 4104.60`) and an even larger improvement versus the older catastrophic runs.
- More importantly, the search behavior looks sane now:
  - the fitness scale is no longer dominated by giant action-ambiguity penalties
  - validation found multiple near-flat or positive rows
  - the selected candidate no longer collapses immediately on the next week

### Updated takeaway after the action-space change

- For GA on this project, `short / flat / long` is materially better than `buy / sell / hold / revert`.
- The next GA work should build on this simpler action surface, not revert to the old one.
- The remaining gap is now much smaller and looks more like model/search quality than a fundamentally broken action interface.

## Longer GA run: 120 generations on `short / flat / long`

After the `short / flat / long` action-surface change, I ran a longer ES Candle CPU walk-forward on the same fold to see whether the cleaner action semantics would actually benefit from a deeper GA search.

Run command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode ga \
  --ga-generations 120 \
  --ga-pop-size 12 \
  --ga-workers 2 \
  --ga-elite-frac 0.17 \
  --ga-parent-pool-frac 0.67 \
  --ga-immigrant-frac 0.17 \
  --ga-mutation-sigma 0.05 \
  --ga-eval-windows 999 \
  --ga-seed 42 \
  --ga-save-top-n 0 \
  --ga-save-every 0 \
  --ga-checkpoint-every 0 \
  --ga-behavior-every 0 \
  --window 700 \
  --step 128 \
  --selection-use-eval \
  --selection-train-weight 0.2 \
  --selection-eval-weight 0.8 \
  --selection-gap-penalty 0.1
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-15T22-31-31-761Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_ga/es_candle_fold1_2026-03-15T22-31-31-761Z/run_summary.json)

Results:

- Runtime: about `244s`
- Best selection row: `gen 88 / idx 10`
- That row's train total PnL: `+308.0788`
- That row's eval total PnL: `+978.9273`
- That row's eval drawdown: `884.25`
- Held-out test result: `pnl -119.10`, `mdd 2544.65`, `sortino 0.01`

Interpretation:

- This is better than the 50-generation `short / flat / long` baseline.
- The held-out test improved from `pnl -550.33` to `pnl -119.10`.
- Held-out drawdown also improved from `2877.30` to `2544.65`.
- So, unlike the old `buy / sell / hold / revert` regime, the cleaner action surface *does* appear to benefit from a longer GA search.
- It still does **not** meet the bar for an effective trading policy, because the held-out PnL remains slightly negative.
- But this is now close enough to flat that further GA work should focus on improving next-week generalization, not on basic action semantics or obviously broken search dynamics.

### Updated takeaway after the 120-generation run

- `100+` generations are worth trying on the new `short / flat / long` policy surface.
- The search did not simply overfit harder; it found a candidate with materially better validation and a better held-out result.
- The remaining gap is now small enough that the next likely gains are in:
  1. broader walk-forward validation across more folds
  2. observation/reward refinements that improve regime transfer
  3. inspecting the selected policy's trade behavior on the held-out week

## Broader walk-forward validation and full-week policy replay

I then expanded the ES check from one chronological fold to all available three-week ES folds in `data/train/*/ES_F.parquet`.

I also added two utilities to make this repeatable:

- [scripts/run_candle_ga_multifold_walkforward.mjs](/Users/nick/Documents/dev/Midas/scripts/run_candle_ga_multifold_walkforward.mjs)
  - runs GA Candle CPU across every chronological 3-week fold for the chosen instrument file
  - keeps the same ES preset and 120-generation settings
  - writes one consolidated report under `runs/pipeline_reports/...`
- [src/bin/inspect_ga_policy.rs](/Users/nick/Documents/dev/Midas/src/bin/inspect_ga_policy.rs)
  - replays a saved GA `best_overall_policy.portable.json` on a parquet file
  - dumps a single behavior CSV plus a compact JSON summary of trades, turnover, fees, and drawdown
- [src/bin/train_ga/portable.rs](/Users/nick/Documents/dev/Midas/src/bin/train_ga/portable.rs)
  - now loads portable GA policies, not just saving them

I also pruned the bulky generated GA artifacts before this pass:

- `runs_ga` dropped from about `2.0G` to about `5.1M`
- I removed old per-generation behavior dumps, checkpoints, and intermediate ranked policy files, while keeping the final summaries and best-policy artifacts

Run command:

```bash
node scripts/run_candle_ga_multifold_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T22-51-00-761Z/pipeline_summary.json)
- [fold_1.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T22-51-00-761Z/fold_1.summary.json)
- [fold_2.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T22-51-00-761Z/fold_2.summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T22-51-00-761Z/fold_1_test_replay/replay_summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T22-51-00-761Z/fold_2_test_replay/replay_summary.json)

### Fold results

Fold 1:

- train `12-29-2025`
- val `01-04-2026`
- test `01-12-2026`
- windowed held-out test from training run: `pnl -119.10`, `mdd 2544.65`
- full-week replay on the held-out ES week: `pnl -1895.80`, `mdd 5294.00`

Fold 2:

- train `01-04-2026`
- val `01-12-2026`
- test `01-20-2026`
- windowed held-out test from training run: `pnl +144.28`, `mdd 2454.35`
- full-week replay on the held-out ES week: `pnl +280.50`, `mdd 4377.60`

Aggregate:

- average full-week replay PnL across both ES folds: about `-807.65`
- average full-week replay max drawdown across both ES folds: about `4835.80`

### What the replay behavior showed

Fold 1 replay:

- `979` entries and `978` exits in one week
- average hold length about `5.01` bars
- realized PnL change: `+987.5`
- commission paid: about `2196.0`
- slippage paid: about `686.25`
- net full-week PnL after costs: about `-1895.8`

Fold 2 replay:

- `887` entries and `886` exits in one week
- average hold length about `5.17` bars
- realized PnL change: `+3525.0`
- commission paid: about `2471.2`
- slippage paid: about `772.25`
- net full-week PnL after costs: about `+280.5`

Shared behavior:

- both folds spent about `49%` of bars flat
- both folds had roughly even win/loss exit counts
- both folds flipped direction frequently
- both folds were effectively high-turnover scalpers, not patient directional policies

### Interpretation

- The ES preset is correct: `margin-per-contract=500`, `contract-multiplier=50`, `max-position=1`, `auto-close-minutes-before-close=5`.
- The main issue is **not** that the GA never finds directional edge.
- The main issue is that the policy trades too often for the amount of edge it has.
- On fold 1, the strategy was approximately break-even before friction and then lost to fees and slippage.
- On fold 2, it stayed barely positive after costs, but the PnL-to-drawdown ratio was still poor and not close to deployment quality.

### Updated takeaway after the broader validation

- The current GA ES pipeline still does **not** generalize well enough to call effective.
- The full-week replays are more informative than the window-averaged held-out line by itself.
- The next GA work should target **turnover control** and **selection on full-week net behavior**, not just more generations.
- The most likely high-value changes now are:
  1. add an explicit turnover / trade-count penalty to selection or fitness
  2. penalize frequent direction flips more strongly
  3. add a minimum hold / action-stability bias for ES
  4. compare selection on full-week replay metrics instead of only averaged windows

## Updated conclusion

- I was able to improve the *search mechanics*:
  - deterministic ranking
  - more diverse reproduction
  - much cheaper experiment artifacts
- I was also able to improve the *environment/model realism*:
  - forced-flat net scoring instead of realized-only scoring
  - normalized, more stationary observations
  - smaller model and capped ES position size
- I was **not** able to improve actual held-out ES GA performance.
- The bottleneck now looks more like environment/reward/model mismatch than GA exploration itself.
- Based on these runs, the next high-value work should not be more GA search tuning. It should be one of:
  1. change the reward/penalty structure
  2. change the observation/model setup
  3. move effort to PPO, which earlier came much closer on held-out data than GA

## Practical conclusions

- Candle CPU is the correct default backend on this Mac for fast iteration.
- Use `release` builds only for any serious comparison or training pass.
- Use GA first to sanity-check the environment and reward shaping.
- Then run PPO on the next chronological fold.
- Use a high `eval-windows` value for any serious walk-forward check, otherwise the reported val/test only cover the first few windows.
- Do not treat these specific ES policies as valid. The current environment/features/reward still need work.
- Based on the longer run, current training is still **not effective** by the standard you set: held-out PnL is still negative.

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

## Hold-biased multifold GA rerun on ES

I added explicit GA hold-bias controls and reran the ES Candle CPU multifold walk-forward in `release` mode with:

- `max-hold-bars-positive=100`
- `max-hold-bars-drawdown=50`
- `hold-duration-penalty=0.1`
- `hold-duration-penalty-growth=0`
- `min-hold-bars=25`
- `early-exit-penalty=1`
- `early-flip-penalty=2`
- `flat-hold-penalty=0.25`
- `flat-hold-penalty-growth=0`
- `max-flat-hold-bars=200`

Command:

```bash
node scripts/run_candle_ga_multifold_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T23-54-45-892Z/pipeline_summary.json)
- [fold_1.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T23-54-45-892Z/fold_1.summary.json)
- [fold_2.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T23-54-45-892Z/fold_2.summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T23-54-45-892Z/fold_1_test_replay/replay_summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T23-54-45-892Z/fold_2_test_replay/replay_summary.json)

### Fold results

Fold 1:

- train `12-29-2025`
- val `01-04-2026`
- test `01-12-2026`
- windowed held-out test from training run: `pnl -144.53`, `mdd 2244.60`
- full-week replay on the held-out ES week: `pnl -2060.60`, `mdd 3402.55`
- replay behavior: `730` entries, `406` flips, average hold `7.43` bars, commissions `1817.6`, slippage `568.0`

Fold 2:

- train `01-04-2026`
- val `01-12-2026`
- test `01-20-2026`
- windowed held-out test from training run: `pnl +295.81`, `mdd 3051.70`
- full-week replay on the held-out ES week: `pnl +2316.60`, `mdd 5156.80`
- replay behavior: `528` entries, `851` flips, average hold `8.47` bars, commissions `2206.4`, slippage `689.5`

Aggregate:

- average full-week replay PnL across both ES folds: about `+128.00`
- average full-week replay max drawdown across both ES folds: about `4279.67`
- average entries per week: about `629`
- average hold length: about `7.95` bars

### Comparison against the prior multifold baseline

Baseline report:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ga_multifold_2026-03-15T22-51-00-761Z/pipeline_summary.json)

Before vs after:

- replay average PnL improved from about `-807.65` to `+128.00`
- replay average MDD improved from about `4835.80` to `4279.67`
- average entries fell from about `933` to `629`
- average hold length rose from about `5.09` bars to `7.95` bars
- average commission fell from about `2333.60` to `2012.00`
- average slippage fell from about `729.25` to `628.75`
- average realized PnL before friction improved from about `2256.25` to `2768.75`

Important nuance:

- this did reduce churn
- this did improve full-week aggregate replay PnL
- this still did **not** get anywhere near the `~100` bar hold style you described
- the policy is still too reversal-heavy, especially on fold 2 where `851` flips is still excessive
- the PnL-to-drawdown ratio is still not good enough to call the GA pipeline effective yet

### What this means

- The new `min_hold` and early exit/flip penalties are worth keeping.
- They moved the ES GA policy in the right direction mechanically.
- They are not sufficient by themselves.
- The next GA step should target **flip suppression and selection quality**, not more raw generations.

Recommended next changes:

1. Penalize direction changes directly in fitness or selection, not just early exits.
2. Select finalists using full-week replay metrics, not only averaged validation windows.
3. Raise the action-stability bias further if the target style is genuinely `~100` bars per trade.
4. Consider a cooldown after flattening or after a flip so the policy cannot churn immediately back.

Current verdict:

- This is the first GA ES rerun that produced a slightly positive average full-week replay across folds.
- It is real progress.
- It is still not strong enough for the standard we want, because drawdown remains too large relative to net PnL and trade duration is still far shorter than the intended strategy style.

## PPO retry on ES

After the GA hold-bias pass, I ran a fresh ES Candle CPU PPO fold on the shifted weekly split through the frontend/API pipeline in `release` mode.

Command:

```bash
node scripts/run_es_candle_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --mode rl \
  --window 700 \
  --step 128 \
  --rl-epochs 24 \
  --rl-train-windows 24 \
  --rl-eval-windows 999 \
  --rl-ppo-epochs 4
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_candle_2026-03-16T00-48-31-014Z/pipeline_summary.json)
- [run_summary.json](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-16T00-48-31-014Z/run_summary.json)
- [ppo_final.safetensors](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-16T00-48-31-014Z/ppo_final.safetensors)

Results:

- run size: `24` epochs, `24` train windows per epoch, `eval-windows=999`
- runtime: about `12.8s`
- best fitness epoch: `8`
- best fitness there: `31.5389`
- eval PnL there: `-58.1897`
- eval drawdown there: `595.0672`
- best eval-PnL epoch: `3`
- eval PnL there: `+213.7931`
- eval drawdown there: `671.7164`
- held-out test result: `pnl +100.9259`, `mdd 1383.6065`, `sortino 0.0081`

Comparison against the earlier longer PPO baseline:

- earlier run: [run_summary.json](/Users/nick/Documents/dev/Midas/runs_rl/es_candle_fold2_2026-03-15T08-43-08-762Z/run_summary.json)
- earlier held-out test: `pnl -658.1019`, `mdd 9937.1685`
- new held-out test: `pnl +100.9259`, `mdd 1383.6065`

Interpretation:

- This PPO retry is clearly better than the earlier RL baseline.
- It is the first documented PPO ES test in this repo that finished positive on the held-out fold.
- The drawdown is also far lower than the earlier `18 x 24` PPO run.
- It is still not strong enough to call production-worthy, because the absolute PnL is small relative to the time/risk budget and the sortino remains weak.

Current RL takeaway:

- PPO is currently more promising than GA on this ES setup.
- If I were choosing where to keep pushing, I would move effort to PPO before spending more time on GA micro-tuning.

## PPO multifold replay on ES

I added a proper RL replay and walk-forward path so PPO can be assessed the same way as GA:

- [inspect_rl_policy.rs](/Users/nick/Documents/dev/Midas/src/bin/inspect_rl_policy.rs)
- [run_candle_rl_multifold_walkforward.mjs](/Users/nick/Documents/dev/Midas/scripts/run_candle_rl_multifold_walkforward.mjs)

Important replay caveat:

- deterministic `argmax` replay is currently too pessimistic for these PPO checkpoints and often collapses to near-flat behavior
- the practical replay mode for current PPO runs is sampled replay with a fixed seed, for example `--sample --seed 42`
- that keeps the replay reproducible while still reflecting how the stochastic PPO policy actually behaves

### Baseline multifold PPO

Command:

```bash
node scripts/run_candle_rl_multifold_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --sample \
  --seed 42
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-51-07-837Z/pipeline_summary.json)
- [fold_1.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-51-07-837Z/fold_1.summary.json)
- [fold_2.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-51-07-837Z/fold_2.summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-51-07-837Z/fold_1_test_replay/replay_summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-51-07-837Z/fold_2_test_replay/replay_summary.json)

Fold 1:

- train `12-29-2025`
- val `01-04-2026`
- test `01-12-2026`
- training test line: `pnl -153.8793`, `mdd 1145.9578`
- sampled full-week replay: `pnl -337.50`, `mdd 3593.70`
- replay behavior: `60` entries, `317` flips, average hold `59.53` bars, commissions `603.20`, slippage `188.50`

Fold 2:

- train `01-04-2026`
- val `01-12-2026`
- test `01-20-2026`
- training test line: `pnl +118.9815`, `mdd 1143.1065`
- sampled full-week replay: `pnl -150.00`, `mdd 7631.50`
- replay behavior: `755` entries, `353` flips, average hold `5.95` bars, commissions `1772.00`, slippage `553.75`

Aggregate:

- average sampled full-week replay PnL: about `-243.75`
- average sampled full-week replay MDD: about `5612.60`

Interpretation:

- PPO is capable of finishing positive on the training test line, but the full-week replay is still weak
- fold 1 already shows the kind of longer-hold behavior we want, but it still loses money
- fold 2 is still too churn-heavy and gives back too much to friction and unstable positioning

### Hold-biased multifold PPO

I then ported the hold-bias controls into RL and reran PPO with the same ES-style settings that improved GA behavior:

- `max-hold-bars-positive=100`
- `max-hold-bars-drawdown=50`
- `hold-duration-penalty=0.1`
- `hold-duration-penalty-growth=0`
- `min-hold-bars=25`
- `early-exit-penalty=1`
- `early-flip-penalty=2`
- `flat-hold-penalty=0.25`
- `flat-hold-penalty-growth=0`
- `max-flat-hold-bars=200`

Command:

```bash
node scripts/run_candle_rl_multifold_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --sample \
  --seed 42 \
  --max-hold-bars-positive 100 \
  --max-hold-bars-drawdown 50 \
  --hold-duration-penalty 0.1 \
  --hold-duration-penalty-growth 0 \
  --min-hold-bars 25 \
  --early-exit-penalty 1 \
  --early-flip-penalty 2 \
  --flat-hold-penalty 0.25 \
  --flat-hold-penalty-growth 0 \
  --max-flat-hold-bars 200
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-53-05-078Z/pipeline_summary.json)
- [fold_1.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-53-05-078Z/fold_1.summary.json)
- [fold_2.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-53-05-078Z/fold_2.summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-53-05-078Z/fold_1_test_replay/replay_summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_ppo_multifold_2026-03-16T03-53-05-078Z/fold_2_test_replay/replay_summary.json)

Fold 1:

- training test line: `pnl +198.4914`, `mdd 472.7560`
- sampled full-week replay: `pnl -112.50`, `mdd 3737.60`
- replay behavior: `6` entries, `4` flips, average hold `306.33` bars, commissions `16.00`, slippage `5.00`

Fold 2:

- training test line: `pnl +391.8981`, `mdd 817.1454`
- sampled full-week replay: `pnl -650.00`, `mdd 6238.60`
- replay behavior: `668` entries, `34` flips, average hold `6.41` bars, commissions `1122.40`, slippage `350.75`

Aggregate:

- average sampled full-week replay PnL: about `-381.25`
- average sampled full-week replay MDD: about `4988.10`

### PPO comparison and takeaway

Baseline vs hold-biased PPO:

- replay average PnL worsened from about `-243.75` to `-381.25`
- replay average MDD improved from about `5612.60` to `4988.10`
- fold 1 improved mechanically a lot: entries dropped from `60` to `6`, average hold rose from `59.53` to `306.33` bars, and replay PnL improved from `-337.50` to `-112.50`
- fold 2 did not respond the same way: replay PnL worsened from `-150.00` to `-650.00` even though commissions, slippage, and flips all came down

Important behavior-count note:

- `flips` in the replay summary are direct sign changes while already in a position
- they are tracked separately from flat-to-position `entries` and position-to-flat `exits`
- so a summary like `entries 60` with `flips 317` is possible and does not mean the counter is broken

Current PPO verdict:

- PPO is still the more promising path than GA on ES
- the new RL replay tooling is worth keeping
- the hold-bias controls are not a better default for PPO yet
- on this two-fold ES test, they make fold 1 much more sensible but hurt aggregate replay PnL because fold 2 still fails to generalize

Recommended next PPO steps:

1. Keep the replay tooling and fixed-seed sampled evaluation path.
2. Select PPO checkpoints using replay metrics, not just training-window fitness.
3. Try a lower `ent-coef` next, because fold 2 still looks too noisy even after the hold penalties.
4. Add more chronological folds before making any claim about PPO stability on ES.

## GRPO multifold replay on ES

I extended the RL multifold helper so it can run either PPO or GRPO through the same frontend/API path:

- [run_candle_rl_multifold_walkforward.mjs](/Users/nick/Documents/dev/Midas/scripts/run_candle_rl_multifold_walkforward.mjs)

Then I ran baseline Candle CPU GRPO on the same ES folds with the same replay mode used for PPO:

- `algorithm=grpo`
- `epochs=24`
- `train-windows=24`
- `grpo-epochs=4`
- `group-size=8`
- sampled replay with fixed seed `42`

Command:

```bash
node scripts/run_candle_rl_multifold_walkforward.mjs \
  --host http://127.0.0.1:4173 \
  --profile release \
  --algorithm grpo \
  --sample \
  --seed 42
```

Artifacts:

- [pipeline_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_grpo_multifold_2026-03-16T04-27-00-864Z/pipeline_summary.json)
- [fold_1.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_grpo_multifold_2026-03-16T04-27-00-864Z/fold_1.summary.json)
- [fold_2.summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_grpo_multifold_2026-03-16T04-27-00-864Z/fold_2.summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_grpo_multifold_2026-03-16T04-27-00-864Z/fold_1_test_replay/replay_summary.json)
- [replay_summary.json](/Users/nick/Documents/dev/Midas/runs/pipeline_reports/es_f_grpo_multifold_2026-03-16T04-27-00-864Z/fold_2_test_replay/replay_summary.json)

Fold 1:

- train `12-29-2025`
- val `01-04-2026`
- test `01-12-2026`
- training test line: `pnl -110.1293`, `mdd 1375.7112`
- sampled full-week replay: `pnl +8725.00`, `mdd 2614.50`, `sortino 0.0420`
- replay behavior: `353` entries, `504` flips, average hold `17.62` bars, commissions `1370.40`, slippage `428.25`

Fold 2:

- train `01-04-2026`
- val `01-12-2026`
- test `01-20-2026`
- training test line: `pnl -79.6296`, `mdd 1524.2824`
- sampled full-week replay: `pnl +6512.50`, `mdd 3846.40`, `sortino 0.0243`
- replay behavior: `189` entries, `818` flips, average hold `25.02` bars, commissions `1611.20`, slippage `503.50`

Aggregate:

- average sampled full-week replay PnL: about `+7618.75`
- average sampled full-week replay MDD: about `3230.45`

### GRPO vs PPO

Across the current two-fold ES test:

- PPO baseline replay average: about `-243.75 pnl / 5612.60 mdd`
- PPO hold-biased replay average: about `-381.25 pnl / 4988.10 mdd`
- GRPO baseline replay average: about `+7618.75 pnl / 3230.45 mdd`

So with the current data, baseline GRPO is materially better than both PPO variants on full-week replay.

### Important GRPO finding

The training-window metrics do **not** reflect the replay result well here:

- both GRPO folds had negative training test lines
- both GRPO folds had strongly positive full-week replay PnL

That means the current RL checkpoint-selection setup is still mismatched to what we actually care about. In this run, selecting the checkpoint by training `fitness` still happened to produce the best replay we have seen so far, but the reported training metrics would have made GRPO look worse than PPO if replay had not been checked.

### Current RL verdict after GRPO

- With the current weekly ES data, GRPO is the best RL result so far.
- The replay tooling was necessary to see that.
- The next RL step should not be more blind hyperparameter tuning first.
- The next RL step should be selection based on replay behavior and then retesting GRPO across more chronological data.

Current best interpretation:

- We are not mainly blocked by optimizer choice anymore.
- We are blocked by limited regime coverage and by the mismatch between windowed training metrics and full-period replay outcomes.
