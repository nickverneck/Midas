<script lang="ts">
	import { Plus, Trash2 } from 'lucide-svelte';
	import type { FeatureSpec, IndicatorKind } from './types';

	type Props = {
		features: FeatureSpec[];
		disabled?: boolean;
		onFeaturesChange: (features: FeatureSpec[]) => void;
	};

	let { features, disabled = false, onFeaturesChange }: Props = $props();

	const indicators: IndicatorKind[] = ['ema', 'sma', 'hma', 'kama', 'alma', 'atr', 'adx', 'rvol', 'er'];
	let indicator = $state<IndicatorKind>('ema');
	let period = $state(14);

	const inputValue = (event: Event) => (event.currentTarget as HTMLInputElement).value;
	const inputNumber = (event: Event) => Number(inputValue(event));
	const selectValue = (event: Event) => (event.currentTarget as HTMLSelectElement).value as IndicatorKind;

	const replaceFeature = (index: number, update: (feature: FeatureSpec) => FeatureSpec) => {
		if (!features[index]) return;
		onFeaturesChange(features.map((feature, itemIndex) => (itemIndex === index ? update(feature) : feature)));
	};

	const addFeature = () => {
		if (disabled) return;
		onFeaturesChange([
			...features,
			{
				indicator,
				period: Math.max(indicator === 'hma' ? 2 : 1, Math.round(Number(period) || 14)),
				lookbacks: [1, 3, 5],
				include_value: true,
				include_delta: true,
				normalize_by_atr: true
			}
		]);
	};

	const removeFeature = (index: number) => {
		if (disabled || features.length === 1) return;
		onFeaturesChange(features.filter((_, itemIndex) => itemIndex !== index));
	};

	const updateLookbacks = (index: number, value: string) => {
		const parsed = value
			.split(',')
			.map((item) => Number(item.trim()))
			.filter((item) => Number.isSafeInteger(item) && item > 0);
		if (parsed.length > 0) {
			replaceFeature(index, (feature) => ({ ...feature, lookbacks: [...new Set(parsed)] }));
		}
	};

	const updateIncludeValue = (index: number, checked: boolean) => {
		replaceFeature(index, (feature) => {
			const next = { ...feature, include_value: checked };
			if (!checked && !next.include_delta) next.include_delta = true;
			return next;
		});
	};

	const updateIncludeDelta = (index: number, checked: boolean) => {
		replaceFeature(index, (feature) => {
			const next = { ...feature, include_delta: checked };
			if (!checked && !next.include_value) next.include_value = true;
			return next;
		});
	};
</script>

<details class="group rounded-lg border bg-muted/20" aria-disabled={disabled}>
	<summary class="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2 text-sm font-medium marker:hidden">
		<span>Feature builder <span class="ml-1 rounded-full bg-foreground px-2 py-0.5 text-[10px] text-background">{features.length} selected</span></span>
		<span class="text-xs font-normal text-muted-foreground group-open:hidden">Edit specs</span>
	</summary>
	<div class="border-t p-2 sm:p-3">
		<p class="mb-2 hidden text-xs leading-5 text-muted-foreground sm:block">
			Values and deltas are causal at the closed event bar. Labels, future prices, oracle values, and future action values stay outside the feature schema.
		</p>
		<div class="mb-2 grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_4.5rem_auto]">
			<label class="sr-only" for="new-feature-indicator">Indicator</label>
			<select id="new-feature-indicator" bind:value={indicator} disabled={disabled} class="h-8 min-w-0 rounded-md border bg-background px-2 text-xs">
				{#each indicators as name}<option value={name}>{name.toUpperCase()}</option>{/each}
			</select>
			<label class="sr-only" for="new-feature-period">Period</label>
			<input id="new-feature-period" type="number" min={indicator === 'hma' ? 2 : 1} bind:value={period} disabled={disabled} class="h-8 min-w-0 rounded-md border bg-background px-2 text-xs" />
			<button type="button" onclick={addFeature} disabled={disabled} class="inline-flex h-8 items-center justify-center gap-1 rounded-md bg-foreground px-2 text-xs font-medium text-background disabled:cursor-not-allowed disabled:opacity-50">
				<Plus size={13} /> Add
			</button>
		</div>

		<div class="space-y-1.5">
			{#each features as feature, index (index)}
				<div class="min-w-0 rounded-md border bg-background p-2">
					<div class="grid min-w-0 gap-2 sm:grid-cols-[5rem_4.5rem_minmax(7rem,1fr)_auto_auto_auto_2rem] sm:items-center">
						<select value={feature.indicator} onchange={(event) => replaceFeature(index, (item) => ({ ...item, indicator: selectValue(event) }))} disabled={disabled} aria-label={`Indicator ${index + 1}`} class="h-8 min-w-0 rounded border bg-background px-1 text-xs uppercase disabled:opacity-60">
							{#each indicators as name}<option value={name}>{name}</option>{/each}
						</select>
						<input type="number" min={feature.indicator === 'hma' ? 2 : 1} value={feature.period} oninput={(event) => replaceFeature(index, (item) => ({ ...item, period: inputNumber(event) }))} disabled={disabled} aria-label={`Period for ${feature.indicator}`} class="h-8 min-w-0 rounded border px-2 text-xs disabled:opacity-60" />
						<input value={feature.lookbacks.join(', ')} onchange={(event) => updateLookbacks(index, inputValue(event))} disabled={disabled} aria-label={`Causal lookbacks for ${feature.indicator}`} class="h-8 min-w-0 rounded border px-2 text-xs disabled:opacity-60" />
						<label class="flex items-center gap-1 whitespace-nowrap text-[10px] sm:col-auto"><input type="checkbox" checked={feature.include_value} onchange={(event) => updateIncludeValue(index, (event.currentTarget as HTMLInputElement).checked)} disabled={disabled} /> Value</label>
						<label class="flex items-center gap-1 whitespace-nowrap text-[10px] sm:col-auto"><input type="checkbox" checked={feature.include_delta} onchange={(event) => updateIncludeDelta(index, (event.currentTarget as HTMLInputElement).checked)} disabled={disabled} /> Delta</label>
						<label class="flex items-center gap-1 whitespace-nowrap text-[10px] sm:col-auto"><input type="checkbox" checked={feature.normalize_by_atr} onchange={(event) => replaceFeature(index, (item) => ({ ...item, normalize_by_atr: (event.currentTarget as HTMLInputElement).checked }))} disabled={disabled || !feature.include_delta} /> ATR norm</label>
						<button type="button" onclick={() => removeFeature(index)} disabled={disabled || features.length === 1} aria-label={`Remove ${feature.indicator} ${feature.period}`} class="grid size-11 place-items-center rounded text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:cursor-not-allowed disabled:opacity-30 sm:size-8 sm:col-auto">
							<Trash2 size={14} />
						</button>
					</div>
				</div>
			{/each}
		</div>
		<details class="mt-2">
			<summary class="cursor-pointer text-[11px] text-muted-foreground">Exact <code>--features</code> JSON</summary>
			<pre class="mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-all rounded bg-foreground p-2 text-[10px] text-background">{JSON.stringify(features, null, 2)}</pre>
		</details>
	</div>
</details>
