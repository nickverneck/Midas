<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import type { FitnessWeights } from "../types";

	type Props = {
		logDir: string;
		fitnessWeights: FitnessWeights;
		loading: boolean;
		onReload: () => void;
		onBrowse: () => void;
	};

	let {
		logDir = $bindable(),
		fitnessWeights = $bindable(),
		loading,
		onReload,
		onBrowse
	}: Props = $props();
</script>

<div class="flex flex-wrap items-center justify-between gap-4">
	<div>
		<h1 class="text-4xl font-bold tracking-tight">RL Analytics</h1>
		<p class="text-sm text-muted-foreground">RL training metrics from Rust runs.</p>
	</div>
	<div class="flex flex-wrap items-center gap-3">
		<div class="flex items-center gap-2">
			<Input class="w-56" placeholder="runs_rl" bind:value={logDir} />
			<Button onclick={onReload} disabled={loading}>
				{loading ? "Loading..." : "Reload"}
			</Button>
			<Button variant="outline" onclick={onBrowse}>Browse</Button>
		</div>
		<div class="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
			<span class="font-medium uppercase tracking-wide">Fitness weights</span>
			<div class="flex items-center gap-1">
				<span>w_pnl</span>
				<Input
					class="h-8 w-20 text-xs"
					type="number"
					step="0.01"
					aria-label="Fitness weight PnL"
					bind:value={fitnessWeights.pnl}
				/>
			</div>
			<div class="flex items-center gap-1">
				<span>w_sortino</span>
				<Input
					class="h-8 w-20 text-xs"
					type="number"
					step="0.01"
					aria-label="Fitness weight Sortino"
					bind:value={fitnessWeights.sortino}
				/>
			</div>
			<div class="flex items-center gap-1">
				<span>w_mdd</span>
				<Input
					class="h-8 w-20 text-xs"
					type="number"
					step="0.01"
					aria-label="Fitness weight MDD"
					bind:value={fitnessWeights.mdd}
				/>
			</div>
		</div>
	</div>
</div>
