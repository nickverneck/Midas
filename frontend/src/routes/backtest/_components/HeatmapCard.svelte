<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Select from "$lib/components/ui/select";
	import * as Tabs from "$lib/components/ui/tabs";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import Sweep3DPlot, { type SweepDimension } from "./Sweep3DPlot.svelte";
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
		selectedTakeProfit: string | undefined;
		selectedStopLoss: string | undefined;
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

	type VizMode = "2d" | "3d";
	let vizMode = $state<VizMode>("2d");
	let xDimension = $state<SweepDimension>("aPeriod");
	let yDimension = $state<SweepDimension>("bPeriod");
	let zMetric3d = $state<keyof AnalyzerMetrics>("fitness");
	let threeDTakeProfitSlice = $state("all");
	let threeDStopLossSlice = $state("all");

	type DimensionOption = { value: SweepDimension; label: string };

	const sweepParamLabel = (value: string) => {
		switch (value) {
			case "period":
				return "Period";
			case "fast":
				return "Fast";
			case "slow":
				return "Slow";
			case "offset":
				return "Offset";
			case "sigma":
				return "Sigma";
			default:
				return value;
		}
	};

	const formatAxisValue = (value: number) => {
		if (!Number.isFinite(value)) return "n/a";
		if (Number.isInteger(value)) return String(value);
		return value.toFixed(4);
	};

	const dimensionLabel = (result: AnalyzerResult | null, key: SweepDimension) => {
		if (!result) {
			if (key === "aPeriod") return "Indicator A";
			if (key === "bPeriod") return "Indicator B";
			if (key === "takeProfit") return "Take Profit";
			return "Stop Loss";
		}
		if (key === "aPeriod") {
			return `A ${result.axes.indicatorA.kind.toUpperCase()} ${sweepParamLabel(result.axes.indicatorA.sweepParam)}`;
		}
		if (key === "bPeriod") {
			return `B ${result.axes.indicatorB.kind.toUpperCase()} ${sweepParamLabel(result.axes.indicatorB.sweepParam)}`;
		}
		if (key === "takeProfit") return "Take Profit";
		return "Stop Loss";
	};

	let dimensionOptions = $derived.by(() => {
		if (!analyzerResult) return [] as DimensionOption[];
		const options: DimensionOption[] = [
			{ value: "aPeriod", label: dimensionLabel(analyzerResult, "aPeriod") },
			{ value: "bPeriod", label: dimensionLabel(analyzerResult, "bPeriod") }
		];
		if (analyzerResult.axes.takeProfitValues.length > 0) {
			options.push({ value: "takeProfit", label: dimensionLabel(analyzerResult, "takeProfit") });
		}
		if (analyzerResult.axes.stopLossValues.length > 0) {
			options.push({ value: "stopLoss", label: dimensionLabel(analyzerResult, "stopLoss") });
		}
		return options;
	});

	$effect(() => {
		dimensionOptions;
		if (dimensionOptions.length === 0) return;
		const values = dimensionOptions.map((opt) => opt.value);
		if (!values.includes(xDimension)) xDimension = values[0];
		if (!values.includes(yDimension) || yDimension === xDimension) {
			yDimension = values.find((value) => value !== xDimension) ?? values[0];
		}
	});

	$effect(() => {
		if (!analyzerResult) {
			threeDTakeProfitSlice = "all";
			threeDStopLossSlice = "all";
			return;
		}
		if (
			threeDTakeProfitSlice !== "all" &&
			!analyzerResult.axes.takeProfitValues.some((value) => String(value) === threeDTakeProfitSlice)
		) {
			threeDTakeProfitSlice = "all";
		}
		if (
			threeDStopLossSlice !== "all" &&
			!analyzerResult.axes.stopLossValues.some((value) => String(value) === threeDStopLossSlice)
		) {
			threeDStopLossSlice = "all";
		}
	});

	$effect(() => {
		heatmapMetric;
		if (zMetric3d !== heatmapMetric) {
			zMetric3d = heatmapMetric;
		}
	});

	let threeDCells = $derived.by(() => {
		if (!analyzerResult) return [] as AnalyzerCell[];
		const takeProfitFilter = threeDTakeProfitSlice === "all" ? null : Number(threeDTakeProfitSlice);
		const stopLossFilter = threeDStopLossSlice === "all" ? null : Number(threeDStopLossSlice);
		return analyzerResult.results.filter((cell) => {
			if (xDimension !== "takeProfit" && yDimension !== "takeProfit" && takeProfitFilter !== null) {
				if (cell.takeProfit !== takeProfitFilter) return false;
			}
			if (xDimension !== "stopLoss" && yDimension !== "stopLoss" && stopLossFilter !== null) {
				if (cell.stopLoss !== stopLossFilter) return false;
			}
			return true;
		});
	});
</script>

<Card.Root>
	<Card.Header>
		<Card.Title>Combination Heatmap</Card.Title>
		<Card.Description>
			{#if analyzerResult}
				Select a metric, set a range, and inspect in either 2D or 3D.
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
					<Label>Take Profit (2D Slice)</Label>
					<Select.Root bind:value={selectedTakeProfit}>
						<Select.Trigger class="w-full">
							{selectedTakeProfit === undefined ? "All" : `${selectedTakeProfit}%`}
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
					<Label>Stop Loss (2D Slice)</Label>
					<Select.Root bind:value={selectedStopLoss}>
						<Select.Trigger class="w-full">
							{selectedStopLoss === undefined ? "All" : `${selectedStopLoss}%`}
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
			<Tabs.Root bind:value={vizMode} class="space-y-4">
				<Tabs.List class="grid w-[260px] grid-cols-2">
					<Tabs.Trigger value="2d">2D Heatmap</Tabs.Trigger>
					<Tabs.Trigger value="3d">3D Sweep</Tabs.Trigger>
				</Tabs.List>

				<Tabs.Content value="2d" class="space-y-3">
					<div class="overflow-auto rounded-lg border">
						<div
							class="grid min-w-[720px] text-xs"
							style={`grid-template-columns: 140px repeat(${analyzerAxisA.length}, minmax(48px, 1fr));`}
						>
							<div class="border-b border-r bg-muted/30 px-3 py-2 font-semibold text-muted-foreground">
								{analyzerResult.axes.indicatorA.kind.toUpperCase()}
								{sweepParamLabel(analyzerResult.axes.indicatorA.sweepParam)} ->
							</div>
							{#each analyzerAxisA as period}
								<div class="border-b bg-muted/30 px-3 py-2 text-center font-semibold text-muted-foreground">
									{formatAxisValue(period)}
								</div>
							{/each}
							{#each analyzerAxisB as rowPeriod}
								<div class="border-r bg-muted/20 px-3 py-2 font-semibold text-muted-foreground">
									{analyzerResult.axes.indicatorB.kind.toUpperCase()}
									{sweepParamLabel(analyzerResult.axes.indicatorB.sweepParam)} {formatAxisValue(rowPeriod)}
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
				</Tabs.Content>

				<Tabs.Content value="3d" class="space-y-3">
					<div class="grid gap-3 md:grid-cols-4">
						<div class="space-y-1">
							<Label class="text-xs">X Axis</Label>
							<select class="h-9 w-full rounded-md border bg-background px-2 text-sm" bind:value={xDimension}>
								{#each dimensionOptions as option}
									{#if option.value !== yDimension}
										<option value={option.value}>{option.label}</option>
									{/if}
								{/each}
							</select>
						</div>
						<div class="space-y-1">
							<Label class="text-xs">Y Axis</Label>
							<select class="h-9 w-full rounded-md border bg-background px-2 text-sm" bind:value={yDimension}>
								{#each dimensionOptions as option}
									{#if option.value !== xDimension}
										<option value={option.value}>{option.label}</option>
									{/if}
								{/each}
							</select>
						</div>
						<div class="space-y-1">
							<Label class="text-xs">Z Axis (Metric)</Label>
							<select class="h-9 w-full rounded-md border bg-background px-2 text-sm" bind:value={zMetric3d}>
								{#each heatmapMetricOptions as option}
									<option value={option.value}>{option.label}</option>
								{/each}
							</select>
						</div>
						<div class="space-y-1">
							<Label class="text-xs">Points</Label>
							<div class="h-9 rounded-md border bg-muted/20 px-3 text-sm leading-9">
								{threeDCells.length.toLocaleString()}
							</div>
						</div>
					</div>

					<div class="grid gap-3 md:grid-cols-2">
						{#if analyzerResult.axes.takeProfitValues.length > 0 && xDimension !== "takeProfit" && yDimension !== "takeProfit"}
							<div class="space-y-1">
								<Label class="text-xs">Take Profit Slice (3D)</Label>
								<select class="h-9 w-full rounded-md border bg-background px-2 text-sm" bind:value={threeDTakeProfitSlice}>
									<option value="all">All</option>
									{#each analyzerResult.axes.takeProfitValues as tp}
										<option value={String(tp)}>{tp}%</option>
									{/each}
								</select>
							</div>
						{/if}
						{#if analyzerResult.axes.stopLossValues.length > 0 && xDimension !== "stopLoss" && yDimension !== "stopLoss"}
							<div class="space-y-1">
								<Label class="text-xs">Stop Loss Slice (3D)</Label>
								<select class="h-9 w-full rounded-md border bg-background px-2 text-sm" bind:value={threeDStopLossSlice}>
									<option value="all">All</option>
									{#each analyzerResult.axes.stopLossValues as sl}
										<option value={String(sl)}>{sl}%</option>
									{/each}
								</select>
							</div>
						{/if}
					</div>

					<Sweep3DPlot
						cells={threeDCells}
						{xDimension}
						{yDimension}
						zMetric={zMetric3d}
						bind:selectedCell
					/>
					<p class="text-xs text-muted-foreground">
						Drag to orbit, scroll to zoom, click a point to inspect details.
					</p>
				</Tabs.Content>
			</Tabs.Root>
		{:else}
			<div class="rounded-lg border border-dashed px-4 py-10 text-center text-sm text-muted-foreground">
				Run the analyzer to render the heatmap.
			</div>
		{/if}
	</Card.Content>
</Card.Root>
