<script lang="ts">
	import { Clock3, ShieldCheck } from 'lucide-svelte';
	import FeatureBuilder from './FeatureBuilder.svelte';
	import SourceFilePicker from './SourceFilePicker.svelte';
	import type { BarKind, FeatureSpec, TrainingDefinition, TrainOptions, StartMode } from './types';

	type Props = {
		definition: TrainingDefinition;
		options: TrainOptions;
		startMode: StartMode;
		disabled?: boolean;
		onDefinitionChange: (patch: Partial<TrainingDefinition>) => void;
		onOptionsChange: (patch: Partial<TrainOptions>) => void;
	};

	let { definition, options, startMode, disabled = false, onDefinitionChange, onOptionsChange }: Props = $props();

	const inputClass = 'h-8 min-w-0 w-full rounded-md border bg-background px-2 text-xs disabled:cursor-not-allowed disabled:opacity-60 sm:h-9 sm:text-sm';
	const valueOf = (event: Event) => (event.currentTarget as HTMLInputElement).value;
	const numberOf = (event: Event) => Number(valueOf(event));
	const barKindOf = (event: Event) => valueOf(event) as BarKind;
	const timestampUnitOf = (event: Event) => valueOf(event) as TrainingDefinition['timestampUnit'];
</script>

<section class="rounded-xl border bg-card p-3 shadow-xs sm:p-4" aria-labelledby="definition-title">
	<div class="mb-2 flex flex-wrap items-center justify-between gap-2">
		<div>
			<h2 id="definition-title" class="text-sm font-semibold">Event definition</h2>
			<p class="text-[11px] text-muted-foreground">EMA crossover events · normal, skip, or invert</p>
		</div>
		<div class="flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-[10px] text-muted-foreground"><Clock3 size={12} /> 18:00–17:00 ET</div>
	</div>

	<div class="grid gap-2 sm:grid-cols-2">
		<div class="sm:col-span-2">
			<label class="mb-1 block text-[11px] font-medium" for="source-path">Source data</label>
			<SourceFilePicker
				id="source-path"
				value={definition.sourcePath}
				onChange={(value) => onDefinitionChange({ sourcePath: value })}
				disabled={disabled}
				label="Choose source bars"
				placeholder="data/bars.parquet, raw Databento parquet, or Ninja .Last.txt"
			/>
			<p class="mt-1 hidden text-[10px] leading-4 text-muted-foreground sm:block">
				Canonical OHLCV bars work directly. Raw Databento trades can be converted to minute/second bars; NinjaTrader <code>.Last.txt</code> can be converted to sparse minute bars. Use canonical Trader output for tick, volume, or range bars.
			</p>
		</div>

		<div class="grid grid-cols-2 gap-2">
			<label class="text-[11px] font-medium">Instrument<input value={definition.instrument} oninput={(event) => onDefinitionChange({ instrument: valueOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">Contract<input value={definition.contract} oninput={(event) => onDefinitionChange({ contract: valueOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
		</div>
		<div class="grid grid-cols-[minmax(0,1fr)_5rem] gap-2">
			<label class="text-[11px] font-medium">Bar kind
				<select value={definition.barKind} onchange={(event) => onDefinitionChange({ barKind: barKindOf(event) })} disabled={disabled} class={`${inputClass} mt-1`}>
					<option value="minute">Minute</option><option value="second">Second</option><option value="tick">Tick</option><option value="volume">Volume</option><option value="range">Range</option>
				</select>
			</label>
			<label class="text-[11px] font-medium">Value<input type="number" min="0.000001" step="any" value={definition.barValue} oninput={(event) => onDefinitionChange({ barValue: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
		</div>

		<div class="grid grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)] items-end gap-1.5 rounded-lg border p-2">
			<span class="pb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Trigger</span>
			<label class="text-[10px] text-muted-foreground">Fast EMA<input type="number" min="1" value={definition.triggerFast} oninput={(event) => onDefinitionChange({ triggerFast: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[10px] text-muted-foreground">Slow EMA<input type="number" min="2" value={definition.triggerSlow} oninput={(event) => onDefinitionChange({ triggerSlow: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
		</div>
		<div class="grid grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)] items-end gap-1.5 rounded-lg border p-2">
			<span class="pb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Context</span>
			<label class="text-[10px] text-muted-foreground">Fast EMA<input type="number" min="1" value={definition.contextFast} oninput={(event) => onDefinitionChange({ contextFast: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[10px] text-muted-foreground">Slow EMA<input type="number" min="2" value={definition.contextSlow} oninput={(event) => onDefinitionChange({ contextSlow: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
		</div>

		{#if startMode === 'continue'}
			<div class="sm:col-span-2 rounded-lg border border-dashed bg-muted/20 p-2.5">
				<div class="mb-2 grid gap-1 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-start sm:gap-2">
					<div class="min-w-0"><h3 class="text-xs font-semibold">Continue from saved artifacts</h3><p class="text-[10px] leading-4 text-muted-foreground">Select the prepared dataset and policy to resume.</p></div>
					<span class="whitespace-nowrap text-[10px] text-muted-foreground sm:justify-self-end">.run/supervised</span>
				</div>
				<div class="grid gap-2 sm:grid-cols-2">
					<div class="min-w-0"><span class="mb-1 block text-[10px] font-medium">Prepared dataset</span><SourceFilePicker value={options.datasetPath} onChange={(value) => onOptionsChange({ datasetPath: value })} disabled={disabled} initialDir=".run/supervised/datasets" extensions={['parquet']} label="Choose prepared dataset" buttonLabel="Browse dataset" placeholder=".run/supervised/datasets/*.parquet" emptyText="No prepared parquet datasets in this folder." /></div>
					<div class="min-w-0"><span class="mb-1 block text-[10px] font-medium">Resume policy</span><SourceFilePicker value={options.resumePolicy} onChange={(value) => onOptionsChange({ resumePolicy: value })} disabled={disabled} initialDir=".run/supervised/runs" extensions={['json']} label="Choose resume policy" buttonLabel="Browse policy" placeholder=".run/supervised/runs/*/policy.json" emptyText="No JSON policy artifacts in this folder." /></div>
				</div>
			</div>
		{/if}

		<div class="sm:col-span-2">
			<FeatureBuilder features={definition.features} disabled={disabled} onFeaturesChange={(features: FeatureSpec[]) => onDefinitionChange({ features })} />
		</div>
	</div>

	<details class="mt-2 rounded-lg border bg-muted/10" aria-disabled={disabled}>
		<summary class="cursor-pointer px-2.5 py-1.5 text-[11px] font-medium">Advanced fields</summary>
		<div class="grid gap-2 border-t p-2.5 sm:grid-cols-3">
			<label class="text-[11px] font-medium">ATR period<input type="number" min="1" value={definition.atrPeriod} oninput={(event) => onDefinitionChange({ atrPeriod: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">Slope lookback<input type="number" min="1" value={definition.slopeLookback} oninput={(event) => onDefinitionChange({ slopeLookback: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">Timestamp unit
				<select value={definition.timestampUnit} onchange={(event) => onDefinitionChange({ timestampUnit: timestampUnitOf(event) })} disabled={disabled} class={`${inputClass} mt-1`}><option value="">Auto / typed</option><option value="ns">ns</option><option value="us">µs</option><option value="ms">ms</option><option value="s">seconds</option></select>
			</label>
			<label class="text-[11px] font-medium">Contract multiplier<input type="number" min="0.000001" step="any" value={definition.contractMultiplier} oninput={(event) => onDefinitionChange({ contractMultiplier: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">Round-trip cost<input type="number" min="0" step="any" value={definition.roundTripCost} oninput={(event) => onDefinitionChange({ roundTripCost: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="flex items-center gap-2 pt-4 text-[11px]"><input type="checkbox" checked={definition.allowIndexTimestamps} onchange={(event) => onDefinitionChange({ allowIndexTimestamps: (event.currentTarget as HTMLInputElement).checked })} disabled={disabled} /> Allow legacy index timestamps</label>

			<label class="text-[11px] font-medium">Epochs<input type="number" min="1" value={options.epochs} oninput={(event) => onOptionsChange({ epochs: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">Checkpoint every<input type="number" min="0" value={options.checkpointEvery} oninput={(event) => onOptionsChange({ checkpointEvery: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /><span class="mt-1 block text-[9px] font-normal text-muted-foreground">0 disables saved curves</span></label>
			<label class="text-[11px] font-medium">Learning rate<input type="number" min="0.0000001" max="1" step="any" value={options.learningRate} oninput={(event) => onOptionsChange({ learningRate: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">L2<input type="number" min="0" max="1" step="any" value={options.l2} oninput={(event) => onOptionsChange({ l2: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">Seed<input type="number" min="0" value={options.seed} oninput={(event) => onOptionsChange({ seed: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">Train fraction<input type="number" min="0.05" max="0.9" step="0.05" value={options.trainFraction} oninput={(event) => onOptionsChange({ trainFraction: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
			<label class="text-[11px] font-medium">Validation fraction<input type="number" min="0.05" max="0.9" step="0.05" value={options.validationFraction} oninput={(event) => onOptionsChange({ validationFraction: numberOf(event) })} disabled={disabled} class={`${inputClass} mt-1`} /></label>
		</div>
	</details>

	<div class="mt-2 flex items-start gap-2 rounded-lg bg-emerald-50 px-2.5 py-1.5 text-[10px] leading-4 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-100">
		<ShieldCheck size={14} class="mt-0.5 shrink-0" />
		<span>Leakage guard: inputs use only values at or before the event. Labels and future-action audit columns stay outside the feature registry.</span>
	</div>
</section>
