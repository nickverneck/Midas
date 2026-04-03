<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import * as Table from "$lib/components/ui/table";
	import { formatNum } from "../analytics";
	import type { LogRow } from "../types";

	type Props = {
		logs: LogRow[];
		pagedLogs: LogRow[];
		sourceLabel: string;
		metricLabel: string;
		pnlKey: string;
		pnlRealizedKey: string;
		pnlTotalKey: string;
		metricKey: string;
		drawdownKey: string;
		wMetricKey: string;
		logPage: number;
		maxLogPage: number;
	};

	let {
		logs,
		pagedLogs,
		sourceLabel,
		metricLabel,
		pnlKey,
		pnlRealizedKey,
		pnlTotalKey,
		metricKey,
		drawdownKey,
		wMetricKey,
		logPage = $bindable(),
		maxLogPage
	}: Props = $props();
</script>

<Card.Root>
	<Card.Header class="flex flex-row items-center justify-between pb-4">
		<div>
			<Card.Title>Raw Training Logs</Card.Title>
			<Card.Description>Scroll to view all {logs.length} data points</Card.Description>
		</div>
	</Card.Header>
	<Card.Content>
		<ScrollArea class="h-[600px] rounded-md border">
			<Table.Root>
				<Table.Header class="sticky top-0 z-10 border-b bg-background shadow-sm">
					<Table.Row>
						<Table.Head class="w-[80px]">Gen</Table.Head>
						<Table.Head class="w-[80px]">Idx</Table.Head>
						<Table.Head>Fitness</Table.Head>
						<Table.Head>{sourceLabel} Fitness PNL</Table.Head>
						<Table.Head>Realized PNL</Table.Head>
						<Table.Head>Total PNL</Table.Head>
						<Table.Head>{metricLabel}</Table.Head>
						<Table.Head>{sourceLabel} DD</Table.Head>
						<Table.Head class="hidden text-right md:table-cell">
							Weights (PNL/{metricLabel}/MDD)
						</Table.Head>
					</Table.Row>
				</Table.Header>
				<Table.Body>
					{#each pagedLogs as log}
						<Table.Row class="transition-colors hover:bg-muted/30">
							<Table.Cell>{log.gen}</Table.Cell>
							<Table.Cell>{log.idx}</Table.Cell>
							<Table.Cell class="font-medium text-blue-500">
								{formatNum(log.fitness)}
							</Table.Cell>
							<Table.Cell
								class={log[pnlKey] >= 0 ? "font-medium text-green-500" : "text-red-500"}
							>
								{formatNum(log[pnlKey])}
							</Table.Cell>
							<Table.Cell
								class={log[pnlRealizedKey] >= 0
									? "font-medium text-emerald-500"
									: "text-red-500"}
							>
								{formatNum(log[pnlRealizedKey])}
							</Table.Cell>
							<Table.Cell
								class={log[pnlTotalKey] >= 0 ? "font-medium text-teal-500" : "text-red-500"}
							>
								{formatNum(log[pnlTotalKey])}
							</Table.Cell>
							<Table.Cell>{formatNum(log[metricKey])}</Table.Cell>
							<Table.Cell class="text-red-400">{formatNum(log[drawdownKey])}%</Table.Cell>
							<Table.Cell class="hidden text-right font-mono text-[10px] text-muted-foreground md:table-cell">
								{formatNum(log.w_pnl, 2)} / {formatNum(log[wMetricKey], 2)} /
								{formatNum(log.w_mdd, 2)}
							</Table.Cell>
						</Table.Row>
					{/each}
				</Table.Body>
			</Table.Root>
		</ScrollArea>
		<div class="mt-4 flex items-center justify-between">
			<div class="text-xs text-muted-foreground">{logs.length.toLocaleString()} rows loaded</div>
			<div class="flex items-center gap-2">
				<button
					class="rounded-md border px-3 py-1 text-xs disabled:opacity-40"
					onclick={() => (logPage = Math.max(1, logPage - 1))}
					disabled={logPage === 1}
				>
					Prev
				</button>
				<div class="text-xs">
					Page {logPage} / {maxLogPage}
				</div>
				<button
					class="rounded-md border px-3 py-1 text-xs disabled:opacity-40"
					onclick={() => (logPage = Math.min(maxLogPage, logPage + 1))}
					disabled={logPage >= maxLogPage}
				>
					Next
				</button>
			</div>
		</div>
	</Card.Content>
</Card.Root>
