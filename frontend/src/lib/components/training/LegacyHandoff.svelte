<script lang="ts">
	import { ArrowRight, BarChart3, Network, Workflow } from 'lucide-svelte';
	import type { TrainingAlgorithm } from './types';

	type Props = { algorithm: Exclude<TrainingAlgorithm, 'supervised'> };
	let { algorithm }: Props = $props();
	const isGa = $derived(algorithm === 'ga');
</script>

<section class="grid gap-4 rounded-xl border bg-card p-5 shadow-xs md:grid-cols-[1fr_auto] md:items-center">
	<div class="flex items-start gap-3">
		<div class="grid size-10 shrink-0 place-items-center rounded-lg bg-muted">{#if isGa}<Network size={20} />{:else}<Workflow size={20} />{/if}</div>
		<div>
			<h2 class="font-semibold">{isGa ? 'Genetic algorithm' : 'Reinforcement learning'} remains in the legacy runner</h2>
			<p class="mt-1 max-w-2xl text-sm leading-6 text-muted-foreground">This compact training workspace does not embed the old parameter wall. Existing runs, charts, and analysis stay available in their dedicated view.</p>
			<div class="mt-3 rounded-lg bg-muted/60 px-3 py-2 text-xs"><b>Small config:</b> backend + dataset split + iterations belong in the legacy runner; review results here as analysis, not as a second training form.</div>
		</div>
	</div>
	<a href={isGa ? '/ga' : '/rl'} class="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-foreground px-4 text-sm font-semibold text-background"><BarChart3 size={16} /> Open {isGa ? 'GA' : 'RL'} charts <ArrowRight size={15} /></a>
</section>
