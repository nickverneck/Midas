# Link Validation Errors To Fields

## Problem
Backtest and Strategy Analyzer run cards list validation errors, but the messages are not connected to the fields that need attention. Users must scan large forms to find the invalid input.

## Why it matters
The Analyzer form contains many numeric fields across signal, exit, bar, and environment sections. A generic `Fix before running` list slows iteration and increases frustration.

## Proposed change
Make validation errors actionable:

- Each validation error should identify the related field or section.
- Clicking or activating an error should focus the field or scroll the section into view.
- Invalid fields should have visible inline messages, not only `aria-invalid`.
- If a section contains an error, its card/header should show a subtle error indicator.

## Likely files/components to touch
- `frontend/src/routes/backtest/+page.svelte`
- `frontend/src/routes/backtest/_components/RunBacktestCard.svelte`
- `frontend/src/routes/backtest/_components/RunAnalyzerCard.svelte`
- `frontend/src/routes/backtest/_components/ScriptEditorCard.svelte`
- `frontend/src/routes/backtest/_components/DatasetCard.svelte`
- `frontend/src/routes/backtest/_components/SignalBuilderCard.svelte`
- `frontend/src/routes/backtest/_components/ExitRangesCard.svelte`
- `frontend/src/routes/backtest/_components/AnalyzerBarSettingsCard.svelte`
- `frontend/src/routes/backtest/_components/BacktestEnvironmentCard.svelte`
- `frontend/src/routes/backtest/_components/AnalyzerEnvironmentCard.svelte`

## Acceptance criteria
- Validation errors are actionable from both `Run Backtest` and `Run Analyzer` cards.
- Keyboard users can navigate from an error to the related field.
- Invalid fields show local helper/error text.
- Large cards with invalid descendants have a visible section-level marker.
- Existing validation rules are preserved.

## Priority
High
