# Add GA Run Folder Picker

## Problem
GA Analytics requires users to type a log folder into `GaPageHeader.svelte`, while RL Analytics has a browse dialog (`RlFolderPicker.svelte`) that lists run folders sorted newest-first.

## Why it matters
Manual folder entry is slower and error-prone, especially when run folders include timestamps. GA and RL analytics also feel inconsistent despite solving the same run-selection problem.

## Proposed change
Add a GA run folder picker that mirrors the RL folder picker experience:

- Browse `runs_ga`.
- Show run folders sorted by modification time, newest first.
- Selecting a folder updates the log folder and loads logs.
- Keep manual path entry for advanced/custom paths.

If feasible, extract a reusable run-folder picker that can support both `runs_ga` and `runs_rl`.

## Likely files/components to touch
- `frontend/src/routes/ga/_components/GaPageHeader.svelte`
- `frontend/src/routes/ga/+page.svelte`
- `frontend/src/routes/rl/_components/RlFolderPicker.svelte` if reused/generalized
- `frontend/src/routes/api/files/+server.ts`

## Acceptance criteria
- GA Analytics has a `Browse` affordance next to the log folder input.
- The picker lists only folders from `runs_ga`, sorted newest-first.
- Selecting a folder updates the input and triggers the same load behavior as `Load Logs`.
- Typing a custom folder manually still works.
- RL folder picking behavior is not regressed.

## Priority
Medium
