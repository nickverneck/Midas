<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Table from "$lib/components/ui/table";
	import { formatNum } from "../analytics";
	import type { GenSummary } from "../types";

	type Props = {
		genData: GenSummary[];
		sourceLabel: string;
		metricLabel: string;
	};

	let { genData, sourceLabel, metricLabel }: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title>Detailed Generation Analysis</Card.Title>
	</Card.Header>
	<Card.Content>
		<Table.Root>
			<Table.Header>
				<Table.Row>
					<Table.Head>Gen</Table.Head>
					<Table.Head>Selected</Table.Head>
					<Table.Head>Best Fitness</Table.Head>
					<Table.Head>Avg Fitness</Table.Head>
					<Table.Head>Best {sourceLabel} Fitness PNL</Table.Head>
					<Table.Head>Best Realized</Table.Head>
					<Table.Head>Best Total</Table.Head>
					<Table.Head>Worst DD</Table.Head>
					<Table.Head>Avg DD</Table.Head>
					<Table.Head>Best {metricLabel}</Table.Head>
				</Table.Row>
			</Table.Header>
			<Table.Body>
				{#each genData.slice().reverse() as generation}
					<Table.Row>
						<Table.Cell class="font-bold">{generation.gen}</Table.Cell>
						<Table.Cell>
							{generation.count === generation.population
								? generation.population
								: `${generation.count}/${generation.population}`}
						</Table.Cell>
						<Table.Cell class="font-medium text-blue-500">
							{generation.bestFitness.toFixed(4)}
						</Table.Cell>
						<Table.Cell class="text-muted-foreground">
							{generation.avgFitness.toFixed(4)}
						</Table.Cell>
						<Table.Cell class={generation.bestPnl >= 0 ? "text-green-500" : "text-red-500"}>
							{generation.bestPnl.toFixed(4)}
						</Table.Cell>
						<Table.Cell
							class={generation.bestRealizedPnl >= 0 ? "text-emerald-500" : "text-red-500"}
						>
							{formatNum(generation.bestRealizedPnl)}
						</Table.Cell>
						<Table.Cell class={generation.bestTotalPnl >= 0 ? "text-teal-500" : "text-red-500"}>
							{formatNum(generation.bestTotalPnl)}
						</Table.Cell>
						<Table.Cell class="text-red-500">{formatNum(generation.maxDrawdown)}%</Table.Cell>
						<Table.Cell class="text-muted-foreground">
							{formatNum(generation.avgDrawdown)}%
						</Table.Cell>
						<Table.Cell class="text-purple-500">{generation.bestMetric.toFixed(4)}</Table.Cell>
					</Table.Row>
				{/each}
			</Table.Body>
		</Table.Root>
	</Card.Content>
</Card.Root>
