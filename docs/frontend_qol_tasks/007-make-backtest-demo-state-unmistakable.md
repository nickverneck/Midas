# Make Backtest Demo State Unmistakable

## Problem
Before any script run, the Backtest page displays demo equity and demo performance metrics via `demoEquitySeries` and `demoMetrics`. Some components label this as demo, but the page still presents plausible numbers in the normal results layout.

## Why it matters
Backtest output drives decision-making. Demo data should never be mistaken for a real run, especially when the surrounding controls and cards look production-like.

## Proposed change
Replace the pre-run result area with an unmistakable empty/demo state.

Suggested approach:

- Show a clear empty state in the chart area until a real run exists, or
- Keep sample data only if it is visually watermarked and separated from real result styling.
- Make the `Performance Snapshot` values unavailable or visibly sample-only until `runResult` exists.
- Keep helpful guidance about what will populate after the first run.

## Likely files/components to touch
- `frontend/src/routes/backtest/+page.svelte`
- `frontend/src/routes/backtest/_components/EquityCurveCard.svelte`
- `frontend/src/routes/backtest/_components/PerformanceSnapshotCard.svelte`
- `frontend/src/routes/backtest/backtest.ts`

## Acceptance criteria
- Before the first run, no plausible metric value is shown without an obvious sample/demo treatment.
- After a run, the real metrics and equity curve replace the empty/demo state.
- The chart and snapshot cards clearly distinguish `no run yet` from `latest run loaded`.
- Existing sample script behavior is unchanged.

## Priority
Medium
