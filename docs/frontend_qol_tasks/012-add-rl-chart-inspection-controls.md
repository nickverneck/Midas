# Add RL Chart Inspection Controls

## Problem
GA Analytics has chart zoom/window controls and large chart areas, while RL Analytics shows all chart tabs in a fixed `320px` height with no visible reset/window controls. The underlying `GaChart` supports zoom interactions, but RL does not expose comparable controls or guidance beyond the charts themselves.

## Why it matters
RL runs can span many epochs and noisy metrics. Users need the same inspection affordances they have in GA to compare training phases, losses, gradients, policy behavior, and probe metrics.

## Proposed change
Bring RL chart exploration closer to GA parity:

- Add visible zoom/window/reset controls or a time-window range control.
- Increase chart height or make it responsive to available viewport space.
- Show the active epoch range and total epoch count.
- Keep chart tabs usable on narrower screens; eight tabs in one row should not overflow awkwardly.

## Likely files/components to touch
- `frontend/src/routes/rl/+page.svelte`
- `frontend/src/routes/rl/_components/RlTrainingCurvesCard.svelte`
- `frontend/src/routes/rl/charts.ts`
- `frontend/src/lib/components/GaChart.svelte` only if shared reset APIs are needed

## Acceptance criteria
- RL users can inspect a subset of epochs without relying only on modifier-key chart gestures.
- RL charts provide a clear reset path after zooming or changing windows.
- The current displayed epoch range is visible.
- Chart tabs remain readable and usable on desktop and mobile widths.
- Existing chart data and labels are preserved.

## Priority
Medium
