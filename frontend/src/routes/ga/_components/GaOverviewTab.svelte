<script lang="ts">
	import type { ChartConfiguration } from "chart.js";
	import { TrendingUp } from "lucide-svelte";
	import GaChart from "$lib/components/GaChart.svelte";
	import * as Card from "$lib/components/ui/card";
	import * as Tabs from "$lib/components/ui/tabs";
	import { formatNum } from "../analytics";
	import type { ActiveChartMeta, ChartTab, GenSummary, PopulationFilter } from "../types";

	type Props = {
		activeChartMeta: ActiveChartMeta;
		sourceLabel: string;
		metricLabel: string;
		chartTab: ChartTab;
		populationFilter: PopulationFilter;
		generationCount: number;
		zoomRangeLabel: string;
		zoomWindowSize: number;
		maxZoomStart: number;
		zoomStart: number;
		canZoomIn: boolean;
		canZoomOut: boolean;
		fitnessChartData: ChartConfiguration["data"];
		pnlChartData: ChartConfiguration["data"];
		realizedChartData: ChartConfiguration["data"];
		drawdownChartData: ChartConfiguration["data"];
		frontierChartData: ChartConfiguration["data"];
		frontierOptions: ChartConfiguration["options"];
		trainRealizedAvg: number | null;
		trainRealizedBest: number;
		evalRealizedAvg: number | null;
		evalRealizedBest: number;
		recentGenerations: GenSummary[];
		onZoomIn: () => void;
		onZoomOut: () => void;
		onZoomReset: () => void;
	};

	let {
		activeChartMeta,
		sourceLabel,
		metricLabel,
		chartTab = $bindable(),
		populationFilter = $bindable(),
		generationCount,
		zoomRangeLabel,
		zoomWindowSize,
		maxZoomStart,
		zoomStart = $bindable(),
		canZoomIn,
		canZoomOut,
		fitnessChartData,
		pnlChartData,
		realizedChartData,
		drawdownChartData,
		frontierChartData,
		frontierOptions,
		trainRealizedAvg,
		trainRealizedBest,
		evalRealizedAvg,
		evalRealizedBest,
		recentGenerations,
		onZoomIn,
		onZoomOut,
		onZoomReset
	}: Props = $props();

	let zoomResetDisabled = $derived(
		generationCount === 0 || (zoomWindowSize >= generationCount && zoomStart === 0)
	);
</script>

<Card.Root>
	<Card.Header class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
		<div>
			<Card.Title class="flex items-center gap-2">
				<TrendingUp size={20} class="text-blue-500" />
				{activeChartMeta.title}
			</Card.Title>
			<Card.Description>{activeChartMeta.description}</Card.Description>
		</div>
		<div class="flex w-full flex-col gap-2 text-xs lg:w-auto lg:items-end">
			<div class="flex flex-wrap items-center gap-2">
				<span class="text-muted-foreground">Population</span>
				<select
					bind:value={populationFilter}
					class="rounded-md border bg-background px-2 py-1 text-xs"
				>
					<option value="all">All individuals</option>
					<option value="top5">Top 5 by fitness</option>
					<option value="top10">Top 10 by fitness</option>
					<option value="top20p">Top 20% by fitness</option>
				</select>
			</div>
			<div class="flex flex-wrap items-center gap-2">
				<span class="text-muted-foreground">{zoomRangeLabel}</span>
				<button
					class="rounded-md border px-2.5 py-1 font-medium disabled:opacity-40"
					onclick={onZoomIn}
					disabled={!canZoomIn}
				>
					Zoom In
				</button>
				<button
					class="rounded-md border px-2.5 py-1 font-medium disabled:opacity-40"
					onclick={onZoomOut}
					disabled={!canZoomOut}
				>
					Zoom Out
				</button>
				<button
					class="rounded-md border px-2.5 py-1 font-medium disabled:opacity-40"
					onclick={onZoomReset}
					disabled={zoomResetDisabled}
				>
					Reset Window
				</button>
			</div>
			<div class="flex w-full items-center gap-2 lg:w-auto">
				<span class="text-muted-foreground">Window</span>
				<input
					type="range"
					min="0"
					max={maxZoomStart}
					step="1"
					bind:value={zoomStart}
					disabled={zoomWindowSize >= generationCount || maxZoomStart === 0}
					class="w-full accent-primary disabled:opacity-40 lg:w-48"
					aria-label="Pan window across generations"
				/>
			</div>
			<div class="text-[10px] text-muted-foreground">
				Ctrl + scroll or pinch to zoom. Shift + drag to pan.
			</div>
		</div>
	</Card.Header>
	<Card.Content class="pt-0">
		<Tabs.Root bind:value={chartTab} class="w-full">
			<Tabs.List class="mb-4 grid w-full grid-cols-2 lg:w-[900px] lg:grid-cols-5">
				<Tabs.Trigger value="fitness">Fitness</Tabs.Trigger>
				<Tabs.Trigger value="performance">Performance</Tabs.Trigger>
				<Tabs.Trigger value="realized">Realized</Tabs.Trigger>
				<Tabs.Trigger value="drawdown">Drawdown</Tabs.Trigger>
				<Tabs.Trigger value="frontier">Frontier</Tabs.Trigger>
			</Tabs.List>

			<Tabs.Content value="fitness">
				{#if chartTab === "fitness"}
					<div class="h-[65vh] min-h-[420px]">
						<GaChart data={fitnessChartData} />
					</div>
				{/if}
			</Tabs.Content>

			<Tabs.Content value="performance">
				{#if chartTab === "performance"}
					<div class="h-[65vh] min-h-[420px]">
						<GaChart data={pnlChartData} />
					</div>
				{/if}
			</Tabs.Content>

			<Tabs.Content value="realized">
				{#if chartTab === "realized"}
					{#if realizedChartData.datasets.length > 0}
						<div class="h-[65vh] min-h-[420px]">
							<GaChart data={realizedChartData} />
						</div>
					{:else}
						<div class="flex h-[50vh] min-h-[320px] items-center justify-center text-muted-foreground">
							No realized PNL columns found in this run.
						</div>
					{/if}
				{/if}
			</Tabs.Content>

			<Tabs.Content value="drawdown">
				{#if chartTab === "drawdown"}
					<div class="h-[65vh] min-h-[420px]">
						<GaChart data={drawdownChartData} />
					</div>
				{/if}
			</Tabs.Content>

			<Tabs.Content value="frontier">
				{#if chartTab === "frontier"}
					{#if frontierChartData.datasets.length > 0}
						<div class="h-[65vh] min-h-[420px]">
							<GaChart
								data={frontierChartData}
								options={frontierOptions}
								type="scatter"
							/>
						</div>
					{:else}
						<div class="flex h-[50vh] min-h-[320px] items-center justify-center text-muted-foreground">
							Not enough data to build the frontier yet.
						</div>
					{/if}
				{/if}
			</Tabs.Content>
		</Tabs.Root>
	</Card.Content>
</Card.Root>

<div class="grid grid-cols-1 gap-6 md:grid-cols-2">
	<Card.Root class="border-l-4 border-l-emerald-500">
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium text-muted-foreground">
				Train Realized PNL (Avg)
			</Card.Title>
		</Card.Header>
		<Card.Content>
			<p class="text-3xl font-bold text-emerald-500">{formatNum(trainRealizedAvg, 2)}</p>
			<p class="mt-1 text-xs text-muted-foreground">Best: {formatNum(trainRealizedBest, 2)}</p>
		</Card.Content>
	</Card.Root>

	<Card.Root class="border-l-4 border-l-rose-500">
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium text-muted-foreground">
				Eval Realized PNL (Avg)
			</Card.Title>
		</Card.Header>
		<Card.Content>
			<p class="text-3xl font-bold text-rose-500">{formatNum(evalRealizedAvg, 2)}</p>
			<p class="mt-1 text-xs text-muted-foreground">Best: {formatNum(evalRealizedBest, 2)}</p>
		</Card.Content>
	</Card.Root>
</div>

<Card.Root>
	<Card.Header>
		<Card.Title>Recent Highlights</Card.Title>
	</Card.Header>
	<Card.Content>
		<div class="grid grid-cols-1 gap-4 md:grid-cols-3">
			{#each recentGenerations as generation}
				<div class="rounded-lg border bg-muted/50 p-4">
					<h4 class="text-lg font-bold">Generation {generation.gen}</h4>
					<div class="mt-2 space-y-1 text-sm">
						<div class="flex justify-between">
							<span>Max Fitness:</span>
							<span class="font-mono text-blue-500">{generation.bestFitness.toFixed(2)}</span>
						</div>
						<div class="flex justify-between">
							<span>Max {sourceLabel} Fitness PNL:</span>
							<span class="font-mono text-green-500">{generation.bestPnl.toFixed(2)}</span>
						</div>
						<div class="flex justify-between">
							<span>Max Realized PNL:</span>
							<span class="font-mono text-emerald-500">
								{generation.bestRealizedPnl.toFixed(2)}
							</span>
						</div>
						<div class="flex justify-between">
							<span>Max Total PNL:</span>
							<span class="font-mono text-teal-500">
								{generation.bestTotalPnl.toFixed(2)}
							</span>
						</div>
						<div class="flex justify-between">
							<span>Max {metricLabel}:</span>
							<span class="font-mono text-purple-500">
								{generation.bestMetric.toFixed(2)}
							</span>
						</div>
						<div class="flex justify-between">
							<span>Worst Drawdown:</span>
							<span class="font-mono text-red-500">
								{formatNum(generation.maxDrawdown, 2)}%
							</span>
						</div>
					</div>
				</div>
			{/each}
		</div>
	</Card.Content>
</Card.Root>
