# Simplify Analyzer Setup Flow

## Problem
The Strategy Analyzer presents signal setup, exits, dataset, bar settings, environment, run summary, and results as several equally weighted cards. Important first-run decisions compete visually with advanced settings.

## Why it matters
The analyzer is powerful but cognitively heavy. Users need a clear path from `choose dataset` to `define signal` to `run sweep`, with advanced tuning available without dominating the first-screen workflow.

## Proposed change
Rework the Analyzer setup into a more guided flow:

- Put the core path in a clear order: dataset, signal pair, exits, run sweep.
- Move environment and bar settings into an `Advanced` section or visually secondary panel.
- Keep the combination count visible near the range controls and run button.
- Add a compact run summary that explains what will be swept before the user runs.
- Preserve the current configurability; do not remove advanced inputs.

## Likely files/components to touch
- `frontend/src/routes/backtest/_components/BacktestAnalyzerView.svelte`
- `frontend/src/routes/backtest/_components/AnalyzerHeader.svelte`
- `frontend/src/routes/backtest/_components/SignalBuilderCard.svelte`
- `frontend/src/routes/backtest/_components/ExitRangesCard.svelte`
- `frontend/src/routes/backtest/_components/RunAnalyzerCard.svelte`
- `frontend/src/routes/backtest/_components/AnalyzerBarSettingsCard.svelte`
- `frontend/src/routes/backtest/_components/AnalyzerEnvironmentCard.svelte`

## Acceptance criteria
- A first-time user can identify the minimum required Analyzer setup without opening unrelated sections.
- Advanced settings remain accessible but are visually secondary.
- The combination count stays visible while editing sweep ranges.
- The run card summarizes the selected dataset, indicators, exit ranges, and combination count.
- Existing Analyzer payload behavior is unchanged.

## Priority
Medium
