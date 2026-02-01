<script lang="ts">
	import * as Card from "$lib/components/ui/card";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { Separator } from "$lib/components/ui/separator";
	import { Gauge } from "lucide-svelte";
	import type { AnalyzerConfig } from "../types";

	type Props = {
		analyzer: AnalyzerConfig;
	};

	let { analyzer }: Props = $props();
</script>

<Card.Root>
	<Card.Header>
		<Card.Title class="flex items-center gap-2">
			<Gauge size={18} />
			Exit Ranges
		</Card.Title>
		<Card.Description>Add optional take-profit or stop-loss sweeps (percent).</Card.Description>
	</Card.Header>
	<Card.Content class="space-y-5">
		<div class="space-y-3">
			<div class="flex items-center gap-2">
				<input
					type="checkbox"
					class="h-4 w-4 rounded border-muted-foreground"
					bind:checked={analyzer.takeProfit.enabled}
				/>
				<Label>Take Profit (%)</Label>
			</div>
			{#if analyzer.takeProfit.enabled}
				<div class="grid gap-2 sm:grid-cols-3">
					<div class="space-y-1">
						<Label class="text-xs">Start</Label>
						<Input type="number" bind:value={analyzer.takeProfit.start} min="0" step="0.1" />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">End</Label>
						<Input type="number" bind:value={analyzer.takeProfit.end} min="0" step="0.1" />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">Step</Label>
						<Input type="number" bind:value={analyzer.takeProfit.step} min="0.1" step="0.1" />
					</div>
				</div>
			{/if}
		</div>
		<Separator />
		<div class="space-y-3">
			<div class="flex items-center gap-2">
				<input
					type="checkbox"
					class="h-4 w-4 rounded border-muted-foreground"
					bind:checked={analyzer.stopLoss.enabled}
				/>
				<Label>Stop Loss (%)</Label>
			</div>
			{#if analyzer.stopLoss.enabled}
				<div class="grid gap-2 sm:grid-cols-3">
					<div class="space-y-1">
						<Label class="text-xs">Start</Label>
						<Input type="number" bind:value={analyzer.stopLoss.start} min="0" step="0.1" />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">End</Label>
						<Input type="number" bind:value={analyzer.stopLoss.end} min="0" step="0.1" />
					</div>
					<div class="space-y-1">
						<Label class="text-xs">Step</Label>
						<Input type="number" bind:value={analyzer.stopLoss.step} min="0.1" step="0.1" />
					</div>
				</div>
			{/if}
		</div>
	</Card.Content>
</Card.Root>
