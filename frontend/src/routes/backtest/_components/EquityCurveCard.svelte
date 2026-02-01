<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import GaChart from "$lib/components/GaChart.svelte";
	import type { ChartConfiguration } from "chart.js";
	import type { BacktestResult } from "../types";

	type Props = {
		runResult: BacktestResult | null;
		equityChartData: ChartConfiguration["data"];
		chartOptions: ChartConfiguration["options"];
	};

	let { runResult, equityChartData, chartOptions }: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title>Equity Curve</Card.Title>
		<Card.Description>
			{#if runResult}
				Latest script run.
			{:else}
				Demo preview (run a backtest to update).
			{/if}
		</Card.Description>
	</Card.Header>
	<Card.Content>
		<GaChart data={equityChartData} options={chartOptions} />
	</Card.Content>
</Card.Root>
