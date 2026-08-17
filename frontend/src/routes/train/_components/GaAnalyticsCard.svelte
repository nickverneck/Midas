<script lang="ts">
	import { ShieldCheck, TriangleAlert } from "lucide-svelte";
	import type { ChartConfiguration } from "chart.js";
	import GaChart from "$lib/components/GaChart.svelte";
	import * as Card from "$lib/components/ui/card";
	import { Badge } from "$lib/components/ui/badge";
	import type { GaGenerationPoint } from "../types";

	type Props = {
		training: boolean;
		series: GaGenerationPoint[];
	};

	let { training, series }: Props = $props();

	const finite = (value: number | null): value is number =>
		value !== null && Number.isFinite(value);

	const format = (value: number | null, digits = 2) =>
		finite(value)
			? value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
			: "—";

	const latest = $derived(series.at(-1) ?? null);
	const usesSelectionFitness = $derived(
		series.some((point) => point.anchorSource === "selection_fitness")
	);
	const evalAvailable = $derived(
		series.some(
			(point) =>
				finite(point.evalFitness) ||
				finite(point.evalPnl) ||
				finite(point.evalRealizedPnl) ||
				finite(point.evalTotalPnl)
		)
	);
	const gapTone = $derived(
		latest?.fitnessGap !== null && latest?.fitnessGap !== undefined && latest.fitnessGap > 0
			? "text-amber-500"
			: "text-emerald-500"
	);

	const fitnessChartData = $derived.by<ChartConfiguration["data"]>(() => ({
		labels: series.map((point) => `Gen ${point.gen}`),
		datasets: [
				{
					label: "Matched train fitness",
				data: series.map((point) => point.trainFitness),
				borderColor: "rgb(59, 130, 246)",
				backgroundColor: "rgba(59, 130, 246, 0.16)",
				tension: 0.2,
				spanGaps: true
			},
			...(evalAvailable
				? [
						{
							label: "Matched eval fitness",
							data: series.map((point) => point.evalFitness),
							borderColor: "rgb(236, 72, 153)",
							backgroundColor: "rgba(236, 72, 153, 0.12)",
							borderDash: [5, 4],
							tension: 0.2,
							spanGaps: true
						}
					]
				: [])
		]
	}));

	const pnlChartData = $derived.by<ChartConfiguration["data"]>(() => ({
		labels: series.map((point) => `Gen ${point.gen}`),
		datasets: [
			{
				label: "Matched train net PNL",
				data: series.map((point) => point.trainPnl),
				borderColor: "rgb(16, 185, 129)",
				backgroundColor: "rgba(16, 185, 129, 0.14)",
				tension: 0.2,
				spanGaps: true
			},
			...(evalAvailable
				? [
						{
							label: "Matched eval net PNL",
							data: series.map((point) => point.evalPnl),
							borderColor: "rgb(249, 115, 22)",
							backgroundColor: "rgba(249, 115, 22, 0.12)",
							borderDash: [5, 4],
							tension: 0.2,
							spanGaps: true
						}
					]
				: [])
		]
	}));

	const chartOptions = (axisTitle: string): ChartConfiguration["options"] => ({
		plugins: {
			legend: {
				position: "top",
				labels: { boxWidth: 12, usePointStyle: true, padding: 12 }
			}
		},
		scales: {
			y: {
				title: { display: true, text: axisTitle },
				grid: { color: "rgba(148, 163, 184, 0.12)" }
			},
			x: {
				grid: { display: false },
				ticks: { maxTicksLimit: 8 }
			}
		}
	});

	const fitnessOptions = $derived(chartOptions("Fitness"));
	const pnlOptions = $derived(chartOptions("Net PNL"));
</script>

<Card.Root>
	<Card.Header class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
		<div>
			<Card.Title>GA Generalization</Card.Title>
			<Card.Description>
				Matched metrics come from the candidate selected by the GA: highest
				{usesSelectionFitness ? " selection_fitness" : " train fitness"} per generation.
				{#if usesSelectionFitness}
					If selection_fitness is absent, the chart falls back to the best train-fitness candidate.
				{/if}
			</Card.Description>
		</div>
		<div class="flex shrink-0 items-center gap-2">
			{#if training}
				<Badge variant="outline" class="animate-pulse">Active</Badge>
			{/if}
			{#if evalAvailable}
				<Badge variant="secondary" class="gap-1"><ShieldCheck size={13} /> Validation logged</Badge>
			{:else}
				<Badge variant="outline" class="gap-1 text-muted-foreground"><TriangleAlert size={13} /> Train only</Badge>
			{/if}
		</div>
	</Card.Header>
	<Card.Content class="space-y-4">
		{#if series.length === 0}
			<div class="flex min-h-[220px] items-center justify-center text-sm text-muted-foreground">
				Waiting for the first GA generation...
			</div>
		{:else}
			<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
				<div class="rounded-md border bg-muted/30 p-3">
					<div class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Latest gen</div>
					<div class="mt-1 font-mono text-lg font-semibold">{latest?.gen ?? "—"}</div>
				</div>
				<div class="rounded-md border bg-muted/30 p-3">
					<div class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Fitness gap</div>
					<div class={`mt-1 font-mono text-lg font-semibold ${gapTone}`}>
						{evalAvailable ? format(latest?.fitnessGap ?? null) : "—"}
					</div>
					<div class="text-[10px] text-muted-foreground">train − eval</div>
				</div>
				<div class="rounded-md border bg-muted/30 p-3">
					<div class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Train net PNL</div>
					<div class="mt-1 font-mono text-lg font-semibold text-emerald-500">{format(latest?.trainPnl ?? null)}</div>
				</div>
				<div class="rounded-md border bg-muted/30 p-3">
					<div class="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Eval net PNL</div>
					<div class="mt-1 font-mono text-lg font-semibold text-orange-500">{evalAvailable ? format(latest?.evalPnl ?? null) : "—"}</div>
				</div>
			</div>
			{#if evalAvailable}
				<p class="text-xs text-muted-foreground">
					A positive gap means the matched train result is ahead of validation; watch for a widening gap while train fitness rises.
				</p>
			{:else}
				<p class="text-xs text-muted-foreground">No eval_* values were logged. This run has no validation comparison to diagnose.</p>
			{/if}
			<div class="grid gap-4 lg:grid-cols-2">
				<div class="min-w-0 rounded-md border bg-background/40 p-2 sm:p-3">
					<div class="mb-1 text-xs font-medium text-muted-foreground">Fitness by generation</div>
					<div class="h-[205px] min-h-[180px] sm:h-[230px]"><GaChart data={fitnessChartData} options={fitnessOptions} /></div>
				</div>
				<div class="min-w-0 rounded-md border bg-background/40 p-2 sm:p-3">
					<div class="mb-1 text-xs font-medium text-muted-foreground">Net PNL by generation</div>
					<div class="h-[205px] min-h-[180px] sm:h-[230px]"><GaChart data={pnlChartData} options={pnlOptions} /></div>
				</div>
			</div>
		{/if}
	</Card.Content>
</Card.Root>
