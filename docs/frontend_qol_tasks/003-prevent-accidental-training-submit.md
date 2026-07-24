# Prevent Accidental Training Submit

## Problem
`TrainConfigCard.svelte` wraps the GA and RL parameter forms in `<form onsubmit=...>` handlers that call `onSubmitGa` or `onSubmitRl`. Because there is no visible submit button inside those forms, pressing Enter in a numeric/text input can unexpectedly start training.

## Why it matters
Starting a training run is expensive and stateful. It should only happen from the explicit Step 3 run button, not as a side effect of editing a field.

## Proposed change
Make the Step 3 `TrainRunCard` button the only way to start or resume training from the UI. Treat Enter in configuration fields as field editing only.

Possible implementation directions:

- Remove the form submit handlers from `TrainConfigCard.svelte`, or
- Keep forms for semantics but prevent submit from triggering a run unless the explicit run button is activated.

## Likely files/components to touch
- `frontend/src/routes/train/_components/TrainConfigCard.svelte`
- `frontend/src/routes/train/+page.svelte`
- `frontend/src/routes/train/_components/TrainRunCard.svelte`

## Acceptance criteria
- Pressing Enter inside GA or RL config inputs does not start training.
- The Step 3 start/resume button still starts the selected mode.
- Existing validation messages still appear when the user explicitly tries to start.
- Keyboard users can still reach and activate the Step 3 start/resume button.

## Priority
High
