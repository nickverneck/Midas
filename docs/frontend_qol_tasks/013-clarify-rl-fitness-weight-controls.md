# Clarify RL Fitness Weight Controls

## Problem
`RlPageHeader.svelte` exposes raw `w_pnl`, `w_sortino`, and `w_mdd` inputs directly in the page header. The labels are implementation-oriented and compete with run-folder controls.

## Why it matters
Fitness weights affect chart interpretation, but in the current header they look like low-level settings with little explanation. Users can change the meaning of the displayed fitness curve without an obvious summary or reset path.

## Proposed change
Move fitness-weight editing into a clearer control group:

- Use readable labels such as `PnL`, `Sortino`, and `Max drawdown`.
- Add a reset-to-default action.
- Show the active weight formula or compact summary near the Fitness chart.
- Consider moving the controls into an expandable `Fitness weights` panel instead of the main header row.

## Likely files/components to touch
- `frontend/src/routes/rl/_components/RlPageHeader.svelte`
- `frontend/src/routes/rl/_components/RlTrainingCurvesCard.svelte`
- `frontend/src/routes/rl/+page.svelte`
- `frontend/src/routes/rl/snapshot.ts`

## Acceptance criteria
- Users can understand what each fitness weight means without reading raw field names.
- There is a one-click reset to default weights.
- The active chart interpretation is visible near the Fitness chart or Latest Snapshot.
- Run-folder controls remain the dominant action in the header.
- Existing computed chart values remain unchanged for the same weights.

## Priority
Low
