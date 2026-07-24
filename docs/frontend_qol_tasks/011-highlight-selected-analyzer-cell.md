# Highlight Selected Analyzer Cell

## Problem
`HeatmapCard.svelte` lets users click a 2D heatmap cell or 3D point to populate `SelectionDetailsCard`, but the 2D heatmap cells do not show a persistent selected state. Slice controls also mix labels like `All` with value-specific controls in a way that can be hard to reason about after a run.

## Why it matters
Analyzer users compare many similar cells. Without a persistent selected visual, it is easy to lose track of which combination the details panel describes.

## Proposed change
Add a durable selected state to result exploration:

- Selected 2D cell gets a visible ring/border and `aria-pressed` or equivalent state.
- Selection details repeats the selected axis values, take-profit slice, stop-loss slice, and metric.
- Slice selectors clearly show whether the user is viewing one slice or all slices.
- If an `All` slice is supported, make it an explicit selectable option. If not, remove unreachable or misleading `All` display branches.

## Likely files/components to touch
- `frontend/src/routes/backtest/_components/HeatmapCard.svelte`
- `frontend/src/routes/backtest/_components/SelectionDetailsCard.svelte`
- `frontend/src/routes/backtest/+page.svelte`

## Acceptance criteria
- Clicking a 2D heatmap cell visibly marks it as selected.
- Changing slice controls clears or updates the selection predictably.
- The details card always identifies the selected combination without relying on the user remembering the clicked cell.
- Keyboard focus and selected state are visually distinct.
- 3D point selection continues to populate the same details panel.

## Priority
Medium
