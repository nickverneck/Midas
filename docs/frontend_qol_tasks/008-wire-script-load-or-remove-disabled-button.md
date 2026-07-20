# Wire Script Load Or Remove Disabled Button

## Problem
`ScriptEditorCard.svelte` shows a `Load .lua` button that is permanently disabled.

## Why it matters
A disabled command with no explanation looks broken. It also suggests an important workflow, loading an existing strategy script, that users cannot complete from the UI.

## Proposed change
Choose one product direction and make the UI honest:

- Implement `Load .lua` with the existing file browser pattern and populate the script editor, or
- Remove the button until the workflow is available, or
- Keep the button disabled only with a clear tooltip/reason and a tracked follow-up.

The preferred QoL outcome is to implement loading because script iteration is a core Backtest workflow.

## Likely files/components to touch
- `frontend/src/routes/backtest/_components/ScriptEditorCard.svelte`
- `frontend/src/routes/backtest/+page.svelte`
- `frontend/src/routes/backtest/_components/BacktestFileBrowserModal.svelte`
- `frontend/src/routes/api/files/+server.ts`
- A new endpoint may be needed if script file contents are not currently exposed

## Acceptance criteria
- The UI no longer shows an unexplained permanently disabled `Load .lua` button.
- If implemented, users can browse for `.lua` files and load one into the editor.
- Loading a file does not overwrite unsaved editor text without an explicit user action.
- Reset Sample still restores the built-in sample script.

## Priority
Medium
