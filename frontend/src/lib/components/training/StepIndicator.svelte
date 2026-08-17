<script lang="ts">
	import { Check } from 'lucide-svelte';

	type Props = { step: 1 | 2 | 3 };
	let { step }: Props = $props();
	const steps = ['Define', 'Prepare', 'Train'];
</script>

<ol class="grid grid-cols-3" aria-label="Training progress">
	{#each steps as label, index}
		<li class="relative flex items-center gap-2 text-xs font-medium sm:text-sm">
			{#if index > 0}
				<span class="absolute right-1/2 left-[-50%] top-3 h-px bg-border" aria-hidden="true"></span>
			{/if}
			<span
				class:!border-emerald-600={index + 1 < step}
				class:!bg-emerald-600={index + 1 < step}
				class:!text-white={index + 1 < step}
				class:!border-foreground={index + 1 === step}
				class="relative z-10 grid size-6 shrink-0 place-items-center rounded-full border bg-background text-[11px]"
			>
				{#if index + 1 < step}<Check size={13} strokeWidth={3} />{:else}{index + 1}{/if}
			</span>
			<span class:text-muted-foreground={index + 1 > step}>{label}</span>
		</li>
	{/each}
</ol>
