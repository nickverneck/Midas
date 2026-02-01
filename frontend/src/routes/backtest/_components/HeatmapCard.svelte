<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Select from "$lib/components/ui/select";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import type { AnalyzerCell, AnalyzerMetrics, AnalyzerResult } from "../types";

	type HeatmapMetricOption = { value: keyof AnalyzerMetrics; label: string };

	type Props = {
		analyzerResult: AnalyzerResult | null;
		heatmapMetric: keyof AnalyzerMetrics;
		heatmapMetricOptions: HeatmapMetricOption[];
		heatmapRangeMin: number | string;
		heatmapRangeMax: number | string;
		heatmapRangePlaceholderMin: string;
		heatmapRangePlaceholderMax: string;
		selectedTakeProfit: string | null;
		selectedStopLoss: string | null;
		analyzerAxisA: number[];
		analyzerAxisB: number[];
		findCell: (aPeriod: number, bPeriod: number) => AnalyzerCell | undefined;
		heatmapColor: (metrics: AnalyzerMetrics) => string;
		isHeatmapValueInRange: (metrics: AnalyzerMetrics) => boolean;
		metricValue: (metrics: AnalyzerMetrics, metric: keyof AnalyzerMetrics) => number;
		formatMetricValue: (value: number, metric: keyof AnalyzerMetrics) => string;
		selectedCell: AnalyzerCell | null;
	};

	let {
		analyzerResult,
		heatmapMetric = $bindable(),
		heatmapMetricOptions,
		heatmapRangeMin = $bindable(),
		heatmapRangeMax = $bindable(),
		heatmapRangePlaceholderMin,
		heatmapRangePlaceholderMax,
		selectedTakeProfit = $bindable(),
		selectedStopLoss = $bindable(),
		analyzerAxisA,
		analyzerAxisB,
		findCell,
		heatmapColor,
		isHeatmapValueInRange,
		metricValue,
		formatMetricValue,
		selectedCell = $bindable()
	}: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title>Combination Heatmap</Card.Title>
		<Card.Description>
			{#if analyzerResult}
				Select a metric, set a range, and filter by exits to inspect the grid.
			{:else}
				Run the analyzer to populate the heatmap.
			{/if}
		</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-4">
		<div class="flex flex-wrap gap-3">
			<div class="min-w-[180px] space-y-2">
				<Label>Metric</Label>
				<Select.Root bind:value={heatmapMetric}>
					<Select.Trigger class="w-full">
						{heatmapMetricOptions.find((m) => m.value === heatmapMetric)?.label ?? heatmapMetric}
					</Select.Trigger>
					<Select.Content>
						{#each heatmapMetricOptions as option}
							<Select.Item value={option.value}>{option.label}</Select.Item>
						{/each}
					</Select.Content>
				</Select.Root>
			</div>
			<div class="min-w-[220px] space-y-2">
				<Label>Range{heatmapMetric === "winRate" ? " (%)" : ""}</Label>
				<div class="flex items-center gap-2">
					<Input
						type="number"
						step="any"
						class="w-full"
						placeholder={heatmapRangePlaceholderMin}
						bind:value={heatmapRangeMin}
						disabled={!analyzerResult}
					/>
					<span class="text-xs text-muted-foreground">to</span>
					<Input
						type="number"
						step="any"
						class="w-full"
						placeholder={heatmapRangePlaceholderMax}
						bind:value={heatmapRangeMax}
						disabled={!analyzerResult}
					/>
				</div>
			</div>
			{#if analyzerResult && analyzerResult.axes.takeProfitValues.length > 0}
				<div class="min-w-[160px] space-y-2">
					<Label>Take Profit</Label>
					<Select.Root bind:value={selectedTakeProfit}>
						<Select.Trigger class="w-full">
							{selectedTakeProfit === null ? "All" : `${selectedTakeProfit}%`}
						</Select.Trigger>
						<Select.Content>
							{#each analyzerResult.axes.takeProfitValues as tp}
								<Select.Item value={String(tp)}>{tp}%</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				</div>
			{/if}
			{#if analyzerResult && analyzerResult.axes.stopLossValues.length > 0}
				<div class="min-w-[160px] space-y-2">
					<Label>Stop Loss</Label>
					<Select.Root bind:value={selectedStopLoss}>
						<Select.Trigger class="w-full">
							{selectedStopLoss === null ? "All" : `${selectedStopLoss}%`}
						</Select.Trigger>
						<Select.Content>
							{#each analyzerResult.axes.stopLossValues as sl}
								<Select.Item value={String(sl)}>{sl}%</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				</div>
			{/if}
		</div>

		{#if analyzerResult}
			<div class="overflow-auto rounded-lg border">
				<div
					class="grid min-w-[720px] text-xs"
					style={`grid-template-columns: 140px repeat(${analyzerAxisA.length}, minmax(48px, 1fr));`}
				>
					<div class="border-b border-r bg-muted/30 px-3 py-2 font-semibold text-muted-foreground">
						{analyzerResult.axes.indicatorA.kind.toUpperCase()} ->
					</div>
					{#each analyzerAxisA as period}
						<div class="border-b bg-muted/30 px-3 py-2 text-center font-semibold text-muted-foreground">
							{period}
						</div>
					{/each}
					{#each analyzerAxisB as rowPeriod}
						<div class="border-r bg-muted/20 px-3 py-2 font-semibold text-muted-foreground">
							{analyzerResult.axes.indicatorB.kind.toUpperCase()} {rowPeriod}
						</div>
							{#each analyzerAxisA as colPeriod}
								{@const cell = findCell(colPeriod, rowPeriod)}
								{#if cell}
									{@const inRange = isHeatmapValueInRange(cell.metrics)}
									<button
										type="button"
										class={`border px-2 py-2 text-center transition hover:opacity-90 ${
											inRange ? "" : "text-muted-foreground"
										}`}
										style={`background:${inRange ? heatmapColor(cell.metrics) : "hsl(var(--background))"}`}
										onclick={() => (selectedCell = cell)}
									>
										{formatMetricValue(metricValue(cell.metrics, heatmapMetric), heatmapMetric)}
									</button>
								{:else}
									<div class="border bg-muted/10 px-2 py-2 text-center text-muted-foreground">-</div>
								{/if}
							{/each}
					{/each}
				</div>
			</div>
		{:else}
			<div class="rounded-lg border border-dashed px-4 py-10 text-center text-sm text-muted-foreground">
				Run the analyzer to render the heatmap.
			</div>
		{/if}
	</Card.Content>
</Card.Root>
