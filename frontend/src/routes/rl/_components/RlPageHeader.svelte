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

<div class="flex min-w-0 flex-wrap items-center justify-between gap-4">
	<div class="min-w-0">
		<h1 class="break-words text-3xl font-bold tracking-tight sm:text-4xl">RL Analytics</h1>
		<p class="text-sm text-muted-foreground">RL training metrics from Rust runs.</p>
	</div>
	<div class="flex w-full min-w-0 flex-col gap-3 sm:w-auto sm:flex-row sm:flex-wrap sm:items-center sm:gap-5">
		<div class="flex w-full min-w-0 flex-wrap items-center gap-2 sm:w-auto">
			<Input class="w-full min-w-0 sm:w-56" placeholder="runs_rl" bind:value={logDir} />
			<Button onclick={onReload} disabled={loading}>
				{loading ? "Loading..." : "Reload"}
			</Button>
			<Button variant="outline" onclick={onBrowse}>Browse</Button>
		</div>
		<div class="grid w-full min-w-0 gap-2 text-xs text-muted-foreground sm:flex sm:w-auto sm:flex-wrap sm:items-center sm:gap-3 sm:border-l sm:border-border sm:pl-4">
			<span class="whitespace-nowrap font-medium uppercase tracking-wide">Fitness weights</span>
			<div class="grid min-w-0 grid-cols-[minmax(0,1fr)_5rem] items-center gap-2 sm:flex sm:items-center sm:gap-2">
				<span>w_pnl</span>
				<Input
					class="h-8 w-full min-w-0 text-xs sm:w-20"
					type="number"
					step="0.01"
					aria-label="Fitness weight PnL"
					bind:value={fitnessWeights.pnl}
				/>
			</div>
			<div class="grid min-w-0 grid-cols-[minmax(0,1fr)_5rem] items-center gap-2 sm:flex sm:items-center sm:gap-2">
				<span>w_sortino</span>
				<Input
					class="h-8 w-full min-w-0 text-xs sm:w-20"
					type="number"
					step="0.01"
					aria-label="Fitness weight Sortino"
					bind:value={fitnessWeights.sortino}
				/>
			</div>
			<div class="grid min-w-0 grid-cols-[minmax(0,1fr)_5rem] items-center gap-2 sm:flex sm:items-center sm:gap-2">
				<span>w_mdd</span>
				<Input
					class="h-8 w-full min-w-0 text-xs sm:w-20"
					type="number"
					step="0.01"
					aria-label="Fitness weight MDD"
					bind:value={fitnessWeights.mdd}
				/>
			</div>
		</div>
	</div>
</div>
