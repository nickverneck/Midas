<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { ScrollArea } from "$lib/components/ui/scroll-area";
	import * as Table from "$lib/components/ui/table";
	import {
		actionClass,
		formatNum,
		normalizeAction,
		resolveBehaviorClose,
		resolveBehaviorTimestamp
	} from "../analytics";
	import type { BehaviorDisplay } from "../types";

	type Props = {
		title: string;
		display: BehaviorDisplay;
		loading: boolean;
		loadingLabel: string;
		emptyMessage: string;
		onOpenChart: () => void;
	};

	let { title, display, loading, loadingLabel, emptyMessage, onOpenChart }: Props = $props();
</script>

<Card.Root>
	<Card.Header class="flex flex-wrap items-start justify-between gap-3">
		<div>
			<Card.Title>{title}</Card.Title>
			<Card.Description>
				Showing {display.rows.length.toLocaleString()} of {display.total.toLocaleString()} rows
			</Card.Description>
		</div>
		<button
			class="rounded-md border px-3 py-1 text-xs disabled:opacity-40"
			onclick={onOpenChart}
			disabled={display.rows.length === 0}
		>
			Open Candlestick
		</button>
	</Card.Header>
	<Card.Content>
		{#if loading}
			<div class="py-10 text-center text-muted-foreground">{loadingLabel}</div>
		{:else if display.rows.length === 0}
			<div class="py-10 text-center text-muted-foreground">{emptyMessage}</div>
		{:else}
			<ScrollArea class="h-[540px] rounded-md border">
				<Table.Root>
					<Table.Header class="sticky top-0 z-10 border-b bg-background shadow-sm">
						<Table.Row>
							<Table.Head>Idx</Table.Head>
							<Table.Head>Window</Table.Head>
							<Table.Head>Time</Table.Head>
							<Table.Head class="text-right">Close</Table.Head>
							<Table.Head>Action</Table.Head>
							<Table.Head class="text-right">Pos</Table.Head>
							<Table.Head class="text-right">PNL</Table.Head>
							<Table.Head class="text-right">Realized</Table.Head>
							<Table.Head class="text-right">Reward</Table.Head>
							<Table.Head class="text-right">Equity</Table.Head>
						</Table.Row>
					</Table.Header>
					<Table.Body>
						{#each display.rows as entry}
							{@const action = normalizeAction(entry.row.action)}
							<Table.Row class="transition-colors hover:bg-muted/30">
								<Table.Cell>{entry.row.data_idx ?? entry.row.idx}</Table.Cell>
								<Table.Cell>{entry.row.window ?? entry.row.window_idx ?? "—"}</Table.Cell>
								<Table.Cell class="font-mono text-[11px]">
									{resolveBehaviorTimestamp(entry.row, entry.data)}
								</Table.Cell>
								<Table.Cell class="text-right">
									{formatNum(resolveBehaviorClose(entry.row, entry.data), 4)}
								</Table.Cell>
								<Table.Cell class={actionClass(action)}>{action || "—"}</Table.Cell>
								<Table.Cell class="text-right">
									{entry.row.position_after ?? entry.row.position ?? "—"}
								</Table.Cell>
								<Table.Cell class="text-right">{formatNum(entry.row.pnl_change)}</Table.Cell>
								<Table.Cell class="text-right">
									{formatNum(entry.row.realized_pnl_change)}
								</Table.Cell>
								<Table.Cell class="text-right">{formatNum(entry.row.reward)}</Table.Cell>
								<Table.Cell class="text-right">{formatNum(entry.row.equity_after)}</Table.Cell>
							</Table.Row>
						{/each}
					</Table.Body>
				</Table.Root>
			</ScrollArea>
		{/if}
	</Card.Content>
</Card.Root>
