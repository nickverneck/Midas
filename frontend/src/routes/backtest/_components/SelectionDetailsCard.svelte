<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Table from "$lib/components/ui/table";
	import type { AnalyzerCell, AnalyzerMetrics, AnalyzerResult } from "../types";

	type Props = {
		selectedCell: AnalyzerCell | null;
		analyzerResult: AnalyzerResult | null;
		formatMetricValue: (value: number, metric: keyof AnalyzerMetrics) => string;
	};

	let { selectedCell, analyzerResult, formatMetricValue }: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title>Selection Details</Card.Title>
		<Card.Description>Inspect metrics for a chosen cell.</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-4">
		{#if selectedCell}
			<div class="rounded-lg border bg-muted/20 p-3 text-xs">
				<p class="font-semibold text-muted-foreground">Configuration</p>
				<p class="mt-1">
					A {analyzerResult?.axes.indicatorA.kind.toUpperCase()} {selectedCell.aPeriod}
					vs B {analyzerResult?.axes.indicatorB.kind.toUpperCase()} {selectedCell.bPeriod}
				</p>
				{#if selectedCell.takeProfit !== null || selectedCell.stopLoss !== null}
					<p class="mt-1">
						TP {selectedCell.takeProfit ?? "n/a"}% / SL {selectedCell.stopLoss ?? "n/a"}%
					</p>
				{/if}
			</div>
			<Table.Root>
				<Table.Body>
					<Table.Row>
						<Table.Cell class="text-muted-foreground">Fitness</Table.Cell>
						<Table.Cell class="text-right font-medium">
							{formatMetricValue(selectedCell.metrics.fitness, "fitness")}
						</Table.Cell>
					</Table.Row>
					<Table.Row>
						<Table.Cell class="text-muted-foreground">Net PnL</Table.Cell>
						<Table.Cell class="text-right">
							{formatMetricValue(selectedCell.metrics.netPnl, "netPnl")}
						</Table.Cell>
					</Table.Row>
					<Table.Row>
						<Table.Cell class="text-muted-foreground">Max Drawdown</Table.Cell>
						<Table.Cell class="text-right">
							{formatMetricValue(selectedCell.metrics.maxDrawdown, "maxDrawdown")}
						</Table.Cell>
					</Table.Row>
					<Table.Row>
						<Table.Cell class="text-muted-foreground">Sortino</Table.Cell>
						<Table.Cell class="text-right">
							{formatMetricValue(selectedCell.metrics.sortino, "sortino")}
						</Table.Cell>
					</Table.Row>
					<Table.Row>
						<Table.Cell class="text-muted-foreground">Sharpe</Table.Cell>
						<Table.Cell class="text-right">
							{formatMetricValue(selectedCell.metrics.sharpe, "sharpe")}
						</Table.Cell>
					</Table.Row>
					<Table.Row>
						<Table.Cell class="text-muted-foreground">Trades</Table.Cell>
						<Table.Cell class="text-right">
							{formatMetricValue(selectedCell.metrics.trades, "trades")}
						</Table.Cell>
					</Table.Row>
					<Table.Row>
						<Table.Cell class="text-muted-foreground">Win Rate</Table.Cell>
						<Table.Cell class="text-right">
							{formatMetricValue(selectedCell.metrics.winRate, "winRate")}
						</Table.Cell>
					</Table.Row>
				</Table.Body>
			</Table.Root>
		{:else}
			<div class="rounded-lg border border-dashed px-4 py-10 text-center text-sm text-muted-foreground">
				Click a heatmap cell to inspect metrics.
			</div>
		{/if}
	</Card.Content>
</Card.Root>
