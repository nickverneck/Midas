<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import * as Table from "$lib/components/ui/table";
	import { Badge } from "$lib/components/ui/badge";
	import { Separator } from "$lib/components/ui/separator";
	import type { BacktestMetrics, BacktestResult } from "../types";

	type Props = {
		activeMetrics: BacktestMetrics;
		runResult: BacktestResult | null;
	};

	let { activeMetrics, runResult }: Props = $props();
</script>

<Card.Root>
	<Card.Header class="flex flex-row items-start justify-between">
		<div>
			<Card.Title>Performance Snapshot</Card.Title>
			<Card.Description>High-level stats for the latest run.</Card.Description>
		</div>
		{#if !runResult}
			<Badge variant="outline">Demo</Badge>
		{/if}
	</Card.Header>
	<Card.Content>
		<Table.Root>
			<Table.Body>
				<Table.Row>
					<Table.Cell class="text-muted-foreground">Net PnL</Table.Cell>
					<Table.Cell class="text-right font-medium">${activeMetrics.net_pnl.toFixed(2)}</Table.Cell>
				</Table.Row>
				<Table.Row>
					<Table.Cell class="text-muted-foreground">Ending Equity</Table.Cell>
					<Table.Cell class="text-right font-medium">${activeMetrics.ending_equity.toFixed(2)}</Table.Cell>
				</Table.Row>
				<Table.Row>
					<Table.Cell class="text-muted-foreground">Sharpe</Table.Cell>
					<Table.Cell class="text-right">{activeMetrics.sharpe.toFixed(2)}</Table.Cell>
				</Table.Row>
				<Table.Row>
					<Table.Cell class="text-muted-foreground">Max Drawdown</Table.Cell>
					<Table.Cell class="text-right">${activeMetrics.max_drawdown.toFixed(2)}</Table.Cell>
				</Table.Row>
				<Table.Row>
					<Table.Cell class="text-muted-foreground">Profit Factor</Table.Cell>
					<Table.Cell class="text-right">{activeMetrics.profit_factor.toFixed(2)}</Table.Cell>
				</Table.Row>
				<Table.Row>
					<Table.Cell class="text-muted-foreground">Win Rate</Table.Cell>
					<Table.Cell class="text-right">{(activeMetrics.win_rate * 100).toFixed(1)}%</Table.Cell>
				</Table.Row>
				<Table.Row>
					<Table.Cell class="text-muted-foreground">Max Consec Losses</Table.Cell>
					<Table.Cell class="text-right">{activeMetrics.max_consecutive_losses}</Table.Cell>
				</Table.Row>
				<Table.Row>
					<Table.Cell class="text-muted-foreground">Steps</Table.Cell>
					<Table.Cell class="text-right">{activeMetrics.steps.toLocaleString()}</Table.Cell>
				</Table.Row>
			</Table.Body>
		</Table.Root>
		<Separator class="my-4" />
		<div class="space-y-2 text-xs text-muted-foreground">
			<p>Baseline comparison (GA/RL) will appear once metrics are wired in.</p>
			<div class="flex gap-2">
				<Badge variant="outline">GA latest</Badge>
				<Badge variant="outline">RL latest</Badge>
			</div>
		</div>
	</Card.Content>
</Card.Root>
