# Align Train Analytics Back Link

## Problem
`TrainPageHeader.svelte` always shows `Back to GA Analytics`, even when the selected training mode is RL and the page title says `Train RL Model`.

## Why it matters
When users switch from GA to RL training, the header sends them back to the wrong analytics area. That breaks the mental model that each training mode has a matching analysis page.

## Proposed change
Make the header back link depend on `trainMode`:

- GA mode links to `/ga` with label `Back to GA Analytics`
- RL mode links to `/rl` with label `Back to RL Analytics`

If product wants a broader navigation affordance, add a compact secondary link to the other analytics page, but the primary back link should match the active training mode.

## Likely files/components to touch
- `frontend/src/routes/train/_components/TrainPageHeader.svelte`
- `frontend/src/routes/train/+page.svelte` if additional props are needed

## Acceptance criteria
- In GA mode, the header link points to `/ga` and says `Back to GA Analytics`.
- In RL mode, the header link points to `/rl` and says `Back to RL Analytics`.
- Switching the GA/RL tab updates the header link without a page refresh.
- The page title still reads `Train GA Model` or `Train RL Model` correctly.

## Priority
Medium
