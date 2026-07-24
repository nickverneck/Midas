# Stabilize Training Log Timestamps

## Problem
`TrainConsoleCard.svelte` renders each console line timestamp with `new Date().toLocaleTimeString()` during rendering. That means old lines can appear to have the current render time rather than the time the message actually arrived.

## Why it matters
Training logs are used for monitoring long-running work. Incorrect or shifting timestamps make it harder to understand progress, pauses, failures, and backend startup time.

## Proposed change
Store a timestamp when each console line is appended, then render that stored timestamp.

Suggested behavior:

- Add a timestamp field to the console line model.
- Assign it at the point stdout/stderr/system/error messages are appended.
- Display a stable formatted time in `TrainConsoleCard`.
- Keep existing colors and message grouping.

## Likely files/components to touch
- `frontend/src/routes/train/types.ts`
- `frontend/src/routes/train/+page.svelte`
- `frontend/src/routes/train/_components/TrainConsoleCard.svelte`

## Acceptance criteria
- Existing console lines keep their original arrival time after re-renders.
- System, stdout, stderr, and error lines all use the same timestamp behavior.
- New lines appear with the correct current time when appended.
- Empty console state remains unchanged.

## Priority
Low
