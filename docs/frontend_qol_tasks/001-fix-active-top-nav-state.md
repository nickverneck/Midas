# Fix Active Top Navigation State

## Problem
The global top nav hard-codes `Train` as `font-semibold text-foreground` in `frontend/src/routes/+layout.svelte`, while `Backtest`, `GA Analytics`, and `RL Analytics` stay muted regardless of the current route.

This validates the reported example: when the user is on `/backtest`, `/ga`, or `/rl`, `Train` still appears visually selected.

## Why it matters
Users lose trust in navigation cues when the strongest nav item does not match the current page. This is especially confusing because the app also has internal sidebars and tabs with their own selected states.

## Proposed change
Make top-level nav styling route-aware using the current SvelteKit page URL. Apply the selected visual treatment only to the matching route group:

- `/` and `/train` select `Train`
- `/backtest` selects `Backtest`
- `/ga` selects `GA Analytics`
- `/rl` selects `RL Analytics`

Add `aria-current="page"` to the active top-level link and make inactive links visually consistent.

## Likely files/components to touch
- `frontend/src/routes/+layout.svelte`

## Acceptance criteria
- `Train` is not bold/foreground on `/backtest`, `/ga`, or `/rl`.
- The current top-level section has the active visual treatment.
- Only the active top-level link has `aria-current="page"`.
- Hover styling still works for inactive links.
- Root `/` still correctly highlights `Train`.

## Priority
High
