<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Button } from "$lib/components/ui/button";
	import { Play } from "lucide-svelte";
	import type { BacktestResult } from "../types";

	type Props = {
		running: boolean;
		canRun: boolean;
		validationErrors: string[];
		runError: string;
		runResult: BacktestResult | null;
		onRun: () => void | Promise<void>;
	};

	let { running, canRun, validationErrors, runError, runResult, onRun }: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title class="flex items-center gap-2">
			<Play size={18} />
			Run Backtest
		</Card.Title>
		<Card.Description>Execute the Lua script against the selected dataset.</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-3">
		<Button class="w-full" onclick={onRun} disabled={!canRun}>
			{running ? "Running..." : "Run Script"}
		</Button>
		{#if validationErrors.length > 0}
			<div class="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
				<p class="font-medium">Fix before running:</p>
				<ul class="mt-1 space-y-1">
					{#each validationErrors as err}
						<li>• {err}</li>
					{/each}
				</ul>
			</div>
		{/if}
		{#if runError}
			<p class="text-xs text-destructive">{runError}</p>
		{/if}
		<p class="text-xs text-muted-foreground">
			{#if runResult}
				Latest run loaded. Adjust inputs and rerun to compare changes.
			{:else}
				Run a backtest to populate metrics and charts.
			{/if}
		</p>
	</Card.Content>
</Card.Root>
