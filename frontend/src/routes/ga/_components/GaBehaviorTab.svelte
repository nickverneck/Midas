<script lang="ts">
	import { AlertCircle } from "lucide-svelte";
	import * as Alert from "$lib/components/ui/alert";
	import * as Card from "$lib/components/ui/card";
	import { formatBehaviorLabel } from "../analytics";
	import type {
		BehaviorDisplay,
		BehaviorFile,
		BehaviorFilter
	} from "../types";
	import GaBehaviorTableCard from "./GaBehaviorTableCard.svelte";

	type Props = {
		activeLogDir: string;
		behaviorListLoading: boolean;
		behaviorError: string;
		trainBehaviorFiles: BehaviorFile[];
		valBehaviorFiles: BehaviorFile[];
		selectedTrainBehavior: string;
		selectedValBehavior: string;
		trainBehaviorLoading: boolean;
		valBehaviorLoading: boolean;
		trainDataLoading: boolean;
		valDataLoading: boolean;
		trainBehaviorRowCount: number;
		valBehaviorRowCount: number;
		trainDataRowCount: number;
		valDataRowCount: number;
		behaviorFilter: BehaviorFilter;
		behaviorRowLimit: number;
		trainBehaviorDisplay: BehaviorDisplay;
		valBehaviorDisplay: BehaviorDisplay;
		onRefresh: () => void;
		onLoadTrain: () => void;
		onLoadVal: () => void;
		onReloadParquet: () => void;
		onOpenTrainChart: () => void;
		onOpenValChart: () => void;
	};

	let {
		activeLogDir,
		behaviorListLoading,
		behaviorError,
		trainBehaviorFiles,
		valBehaviorFiles,
		selectedTrainBehavior = $bindable(),
		selectedValBehavior = $bindable(),
		trainBehaviorLoading,
		valBehaviorLoading,
		trainDataLoading,
		valDataLoading,
		trainBehaviorRowCount,
		valBehaviorRowCount,
		trainDataRowCount,
		valDataRowCount,
		behaviorFilter = $bindable(),
		behaviorRowLimit = $bindable(),
		trainBehaviorDisplay,
		valBehaviorDisplay,
		onRefresh,
		onLoadTrain,
		onLoadVal,
		onReloadParquet,
		onOpenTrainChart,
		onOpenValChart
	}: Props = $props();
</script>

<Card.Root>
	<Card.Header class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
		<div>
			<Card.Title>Behavior Trace Compare</Card.Title>
			<Card.Description>
				Load the best-gen behavior CSVs and compare actions against SPY parquet bars.
			</Card.Description>
		</div>
		<div class="flex flex-wrap items-center gap-2 text-xs">
			<span class="text-muted-foreground">Run dir</span>
			<span class="rounded-md border bg-muted/40 px-2 py-1">{activeLogDir}</span>
			<button
				class="rounded-md border px-3 py-1 text-xs disabled:opacity-40"
				onclick={onRefresh}
				disabled={behaviorListLoading}
			>
				{behaviorListLoading ? "Loading..." : "Refresh CSVs"}
			</button>
		</div>
	</Card.Header>
	<Card.Content class="space-y-4">
		{#if behaviorError}
			<Alert.Root variant="destructive">
				<AlertCircle class="h-4 w-4" />
				<Alert.Title>Behavior Load Error</Alert.Title>
				<Alert.Description>{behaviorError}</Alert.Description>
			</Alert.Root>
		{/if}

		<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
			<div class="space-y-2">
				<div class="text-xs text-muted-foreground">Train behavior CSV</div>
				<div class="flex items-center gap-2">
					<select
						bind:value={selectedTrainBehavior}
						class="flex-1 rounded-md border bg-background px-2 py-1 text-xs"
					>
						<option value="">Select train CSV</option>
						{#each trainBehaviorFiles as file}
							<option value={file.name}>{formatBehaviorLabel(file)}</option>
						{/each}
					</select>
					<button
						class="rounded-md border px-3 py-1 text-xs disabled:opacity-40"
						onclick={onLoadTrain}
						disabled={!selectedTrainBehavior || trainBehaviorLoading}
					>
						{trainBehaviorLoading ? "Loading..." : "Load"}
					</button>
				</div>
				<div class="text-xs text-muted-foreground">
					Rows: {trainBehaviorRowCount.toLocaleString()} | Parquet:
					{trainDataRowCount.toLocaleString()}
				</div>
			</div>

			<div class="space-y-2">
				<div class="text-xs text-muted-foreground">Eval behavior CSV</div>
				<div class="flex items-center gap-2">
					<select
						bind:value={selectedValBehavior}
						class="flex-1 rounded-md border bg-background px-2 py-1 text-xs"
					>
						<option value="">Select eval CSV</option>
						{#each valBehaviorFiles as file}
							<option value={file.name}>{formatBehaviorLabel(file)}</option>
						{/each}
					</select>
					<button
						class="rounded-md border px-3 py-1 text-xs disabled:opacity-40"
						onclick={onLoadVal}
						disabled={!selectedValBehavior || valBehaviorLoading}
					>
						{valBehaviorLoading ? "Loading..." : "Load"}
					</button>
				</div>
				<div class="text-xs text-muted-foreground">
					Rows: {valBehaviorRowCount.toLocaleString()} | Parquet:
					{valDataRowCount.toLocaleString()}
				</div>
			</div>
		</div>

		<div class="flex flex-wrap items-center gap-3 text-xs">
			<span class="text-muted-foreground">Action filter</span>
			<select
				bind:value={behaviorFilter}
				class="rounded-md border bg-background px-2 py-1 text-xs"
			>
				<option value="trades">Trades only</option>
				<option value="all">All actions</option>
				<option value="buy">Buy only</option>
				<option value="sell">Sell only</option>
				<option value="hold">Hold only</option>
				<option value="revert">Revert only</option>
			</select>
			<span class="text-muted-foreground">Row limit</span>
			<input
				type="number"
				min="0"
				step="100"
				bind:value={behaviorRowLimit}
				class="w-24 rounded-md border bg-background px-2 py-1 text-xs"
			/>
			<button
				class="rounded-md border px-3 py-1 text-xs disabled:opacity-40"
				onclick={onReloadParquet}
				disabled={trainDataLoading || valDataLoading}
			>
				Reload Parquet
			</button>
		</div>
	</Card.Content>
</Card.Root>

<div class="grid grid-cols-1 gap-6 xl:grid-cols-2">
	<GaBehaviorTableCard
		title="Train Behavior"
		display={trainBehaviorDisplay}
		loading={trainBehaviorLoading || trainDataLoading}
		loadingLabel="Loading train behavior..."
		emptyMessage="Select a train behavior CSV to view actions."
		onOpenChart={onOpenTrainChart}
	/>

	<GaBehaviorTableCard
		title="Eval Behavior"
		display={valBehaviorDisplay}
		loading={valBehaviorLoading || valDataLoading}
		loadingLabel="Loading eval behavior..."
		emptyMessage="Select an eval behavior CSV to view actions."
		onOpenChart={onOpenValChart}
	/>
</div>
