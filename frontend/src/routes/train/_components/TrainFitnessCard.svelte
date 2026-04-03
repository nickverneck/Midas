<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Badge } from "$lib/components/ui/badge";
	import GaChart from "$lib/components/GaChart.svelte";
	import type { ChartConfiguration } from "chart.js";
	import type { FitnessPoint } from "../types";

	type Props = {
		training: boolean;
		logLabel: string;
		fitnessSeries: FitnessPoint[];
		chartData: ChartConfiguration["data"];
		chartOptions: ChartConfiguration["options"];
	};

	let { training, logLabel, fitnessSeries, chartData, chartOptions }: Props = $props();
</script>

<Card.Root>
	<Card.Header class="flex flex-row items-center justify-between">
		<Card.Title>Best Fitness by {logLabel}</Card.Title>
		{#if training}
			<Badge variant="outline" class="animate-pulse">Active</Badge>
		{/if}
	</Card.Header>
	<Card.Content>
		{#if fitnessSeries.length > 0}
			<div class="h-[260px] min-h-[220px]">
				<GaChart data={chartData} options={chartOptions} />
			</div>
		{:else}
			<div class="flex h-[220px] items-center justify-center text-muted-foreground">
				Waiting for the first {logLabel.toLowerCase()} results...
			</div>
		{/if}
	</Card.Content>
</Card.Root>
