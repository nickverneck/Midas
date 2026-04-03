<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { formatNum } from "../analytics";
	import type { PlateauResult } from "../types";

	type Props = {
		generationCount: number;
		bestFitness: number;
		metricLabel: string;
		avgMetric: number | null;
		logsCount: number;
		plateauResult: PlateauResult;
		plateauMetricLabel: string;
		plateauWindow: number;
		plateauMinDelta: number;
	};

	let {
		generationCount,
		bestFitness,
		metricLabel,
		avgMetric,
		logsCount,
		plateauResult,
		plateauMetricLabel,
		plateauWindow = $bindable(),
		plateauMinDelta = $bindable()
	}: Props = $props();
</script>

<div class="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-5">
	<Card.Root class="border-l-4 border-l-blue-500">
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium text-muted-foreground">Generations</Card.Title>
		</Card.Header>
		<Card.Content>
			<p class="text-3xl font-bold">{generationCount}</p>
		</Card.Content>
	</Card.Root>

	<Card.Root class="border-l-4 border-l-green-500">
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium text-muted-foreground">Global Best Fitness</Card.Title>
		</Card.Header>
		<Card.Content>
			<p class="text-3xl font-bold text-green-500">{formatNum(bestFitness, 2)}</p>
		</Card.Content>
	</Card.Root>

	<Card.Root class="border-l-4 border-l-purple-500">
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium text-muted-foreground">Avg {metricLabel}</Card.Title>
		</Card.Header>
		<Card.Content>
			<p class="text-3xl font-bold">{formatNum(avgMetric, 2)}</p>
		</Card.Content>
	</Card.Root>

	<Card.Root class="border-l-4 border-l-orange-500">
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium text-muted-foreground">Total Processed</Card.Title>
		</Card.Header>
		<Card.Content>
			<p class="text-3xl font-bold">{logsCount.toLocaleString()}</p>
		</Card.Content>
	</Card.Root>

	<Card.Root class="border-l-4 border-l-slate-500">
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium text-muted-foreground">Plateau Gen</Card.Title>
		</Card.Header>
		<Card.Content>
			<p class="text-3xl font-bold">
				{plateauResult.plateauGen ? `Gen ${plateauResult.plateauGen}` : "Not reached"}
			</p>
			<p class="mt-1 text-xs text-muted-foreground">Metric: {plateauMetricLabel}</p>
			<p class="text-xs text-muted-foreground">
				Last improvement:
				{plateauResult.lastImprovementGen ? `Gen ${plateauResult.lastImprovementGen}` : "—"}
			</p>
			<div class="mt-2 flex items-center gap-2 text-[10px] text-muted-foreground">
				<span>Window</span>
				<input
					type="number"
					min="50"
					step="50"
					bind:value={plateauWindow}
					class="w-16 rounded-md border bg-background px-1 py-0.5 text-xs"
				/>
				<span>Delta</span>
				<input
					type="number"
					min="0"
					step="0.1"
					bind:value={plateauMinDelta}
					class="w-16 rounded-md border bg-background px-1 py-0.5 text-xs"
				/>
			</div>
		</Card.Content>
	</Card.Root>
</div>
