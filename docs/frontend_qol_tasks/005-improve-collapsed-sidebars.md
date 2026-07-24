# Improve Collapsed Sidebar Affordances

## Problem
The Train and Backtest sidebars collapse into narrow rails, but the collapsed states do not carry enough context.

In Train, `TrainSidebar.svelte` collapses to a single `Setup` or `Stop` button and an invisible full-sidebar click target. In Backtest, `SideMenu.svelte` keeps icons but hides labels and does not expose a persistent section title beyond the `BT` mark.

## Why it matters
Collapsed sidebars are useful only if users can still tell what area they are in and how to get back to controls. Hidden click targets and unlabeled rails make the layout feel unpredictable.

## Proposed change
Create a clearer collapsed sidebar pattern for both Train and Backtest:

- Show a visible expand button in a consistent location.
- Keep a compact section label or icon with tooltip.
- Preserve active mode/view context in the collapsed rail.
- Avoid relying on invisible full-area buttons as the primary affordance.

## Likely files/components to touch
- `frontend/src/routes/train/_components/TrainSidebar.svelte`
- `frontend/src/routes/backtest/_components/SideMenu.svelte`
- Shared UI primitives only if a reusable collapsed rail pattern is introduced

## Acceptance criteria
- A collapsed Train sidebar visibly communicates `Training Controls` and the current action state.
- A collapsed Backtest sidebar visibly communicates the current Backtest sub-view.
- Expand/collapse controls have clear accessible labels and visible focus states.
- Tooltips or titles identify icon-only controls.
- No primary interaction depends on clicking an invisible overlay.

## Priority
Medium
