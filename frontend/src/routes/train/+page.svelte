<script lang="ts">
	import { BrainCircuit, Network, Sparkles } from 'lucide-svelte';
	import RlvrTrainingPanel from '$lib/components/training/RlvrTrainingPanel.svelte';
	import SupervisedTrainingPanel from '$lib/components/training/SupervisedTrainingPanel.svelte';
	import LegacyTrainingPanel from './_components/LegacyTrainingPanel.svelte';

	type WorkspaceMode = 'legacy' | 'supervised' | 'rlvr';
	let workspaceMode = $state<WorkspaceMode>('legacy');
	let legacyParamsCollapsed = $state(false);
</script>

<svelte:head>
	<title>Training · Midas</title>
	<meta name="description" content="Run GA, RL, and supervised Midas training workflows." />
</svelte:head>

<div class="min-h-screen bg-background">
	<header
		class={`flex w-full flex-wrap items-center justify-between gap-3 px-3 py-3 sm:px-5 ${
			workspaceMode === 'legacy'
				? legacyParamsCollapsed
					? 'lg:ml-[120px] lg:w-[calc(100%-120px)] lg:max-w-none'
					: 'lg:ml-[360px] lg:w-[calc(100%-360px)] lg:max-w-none'
				: 'mx-auto max-w-7xl'
		}`}
	>
		<div>
			<div class="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Midas training</div>
			<p class="mt-0.5 text-xs text-muted-foreground">Choose legacy GA/RL, supervised events, or the exact-reward RLVR gate.</p>
		</div>
		<nav class="flex rounded-lg border bg-card p-1 shadow-xs" aria-label="Training workflow">
			<button
				type="button"
				onclick={() => (workspaceMode = 'legacy')}
				aria-pressed={workspaceMode === 'legacy'}
				class:bg-foreground={workspaceMode === 'legacy'}
				class:text-background={workspaceMode === 'legacy'}
				class="inline-flex min-h-10 items-center gap-1.5 rounded-md px-3 text-xs font-semibold transition-colors hover:bg-muted"
			>
				<Network size={14} /> GA / RL
			</button>
			<button
				type="button"
				onclick={() => (workspaceMode = 'supervised')}
				aria-pressed={workspaceMode === 'supervised'}
				class:bg-foreground={workspaceMode === 'supervised'}
				class:text-background={workspaceMode === 'supervised'}
				class="inline-flex min-h-10 items-center gap-1.5 rounded-md px-3 text-xs font-semibold transition-colors hover:bg-muted"
			>
				<BrainCircuit size={14} /> Supervised
			</button>
			<button
				type="button"
				onclick={() => (workspaceMode = 'rlvr')}
				aria-pressed={workspaceMode === 'rlvr'}
				class:bg-foreground={workspaceMode === 'rlvr'}
				class:text-background={workspaceMode === 'rlvr'}
				class="inline-flex min-h-10 items-center gap-1.5 rounded-md px-3 text-xs font-semibold transition-colors hover:bg-muted"
			>
				<Sparkles size={14} /> RLVR
			</button>
		</nav>
	</header>

	{#if workspaceMode === 'supervised'}
		<SupervisedTrainingPanel />
	{:else if workspaceMode === 'rlvr'}
		<RlvrTrainingPanel />
	{:else}
		<LegacyTrainingPanel bind:paramsCollapsed={legacyParamsCollapsed} />
	{/if}
</div>
