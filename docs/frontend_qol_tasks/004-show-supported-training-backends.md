# Show Supported Training Backends

## Problem
The Train forms expose backend options that the UI text says are not runnable for the current mode. For example, `RlTrainingForm.svelte` lets users select `burn` and `mlx`, while the inline status says Burn RL is not implemented and MLX is reserved.

## Why it matters
Users can choose a backend that appears valid in the control but is expected to fail or not run. That creates avoidable failed jobs and makes the backend rollout status feel contradictory.

## Proposed change
Make backend availability explicit at the point of selection.

Suggested behavior:

- Keep runnable backends selectable.
- Mark unavailable backends as disabled or label them as `Coming soon` / `Not available for RL`.
- If disabled options are not desirable, show a blocking inline warning and disable the Start button for unsupported mode/backend combinations.
- Keep the rollout note, but make the select control itself impossible to misread.

## Likely files/components to touch
- `frontend/src/routes/train/_components/GaTrainingForm.svelte`
- `frontend/src/routes/train/_components/RlTrainingForm.svelte`
- `frontend/src/routes/train/+page.svelte`
- `frontend/src/routes/train/types.ts`

## Acceptance criteria
- RL users cannot accidentally start a run with a backend identified by the UI as unavailable.
- GA users see which backend options are runnable versus reserved.
- The selected backend state, warning copy, and Start button enabled state agree.
- Existing runnable defaults still work without extra clicks.

## Priority
High
