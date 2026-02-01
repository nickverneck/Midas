<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Button } from "$lib/components/ui/button";
	import { Play } from "lucide-svelte";
	import type { AnalyzerResult } from "../types";

	type Props = {
		analyzerRunning: boolean;
		analyzerCanRun: boolean;
		analyzerCombos: number;
		analyzerResult: AnalyzerResult | null;
		analyzerValidationErrors: string[];
		analyzerError: string;
		onRun: () => void | Promise<void>;
	};

	let {
		analyzerRunning,
		analyzerCanRun,
		analyzerCombos,
		analyzerResult,
		analyzerValidationErrors,
		analyzerError,
		onRun
	}: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title class="flex items-center gap-2">
			<Play size={18} />
			Run Analyzer
		</Card.Title>
		<Card.Description>Launch the parameter sweep in Rust.</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-3">
		<Button class="w-full" onclick={onRun} disabled={!analyzerCanRun}>
			{analyzerRunning ? "Running..." : "Run Sweep"}
		</Button>
		<div class="text-xs text-muted-foreground">
			<p>Combinations: {analyzerCombos.toLocaleString()}</p>
			{#if analyzerResult}
				<p>
					Last run:
					{(analyzerResult.axes.totalCombinations ?? analyzerResult.axes.total_combinations)?.toLocaleString() ??
						"n/a"}
					combos.
				</p>
			{/if}
		</div>
		{#if analyzerValidationErrors.length > 0}
			<div class="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
				<p class="font-medium">Fix before running:</p>
				<ul class="mt-1 space-y-1">
					{#each analyzerValidationErrors as err}
						<li>• {err}</li>
					{/each}
				</ul>
			</div>
		{/if}
		{#if analyzerError}
			<p class="text-xs text-destructive">{analyzerError}</p>
		{/if}
	</Card.Content>
</Card.Root>
