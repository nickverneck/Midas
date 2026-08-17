<script lang="ts">
	import { page } from '$app/stores';
	import { BarChart3, ChevronDown, Menu, X } from 'lucide-svelte';
	import favicon from '$lib/assets/favicon.svg';
	import '../app.css';

	let { children } = $props();
	let mobileOpen = $state(false);
	const path = $derived($page.url.pathname);
	const trainingActive = $derived(path === '/' || path.startsWith('/train'));
	const chartsActive = $derived(path.startsWith('/ga') || path.startsWith('/rl'));
	const backtestActive = $derived(path.startsWith('/backtest'));
	const navClass = 'rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-muted hover:text-foreground';
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>

<div class="min-h-screen bg-background">
	<nav class="sticky top-0 z-40 border-b bg-background/90 backdrop-blur" aria-label="Primary navigation">
		<div class="mx-auto flex h-14 max-w-6xl items-center justify-between px-3 sm:px-6">
			<a href="/" class="flex items-center gap-2 font-semibold tracking-tight" aria-label="Midas home">
				<span class="grid size-7 place-items-center rounded-md bg-foreground text-background"><BarChart3 size={15} /></span>
				<span>Midas</span>
			</a>

			<div class="hidden items-center gap-1 sm:flex">
				<a href="/train" class={`${navClass} ${trainingActive ? 'bg-muted text-foreground' : 'text-muted-foreground'}`} aria-current={trainingActive ? 'page' : undefined}>Training</a>
				<details class="group relative">
					<summary class={`${navClass} flex cursor-pointer list-none items-center gap-1 marker:hidden ${chartsActive ? 'bg-muted text-foreground' : 'text-muted-foreground'}`}>Charts <ChevronDown size={14} class="transition-transform group-open:rotate-180" /></summary>
					<div class="absolute left-0 mt-1 w-52 rounded-lg border bg-background p-1 shadow-lg">
						<a href="/ga" class="block rounded-md px-3 py-2 text-sm hover:bg-muted"><b class="block">GA charts</b><span class="text-xs text-muted-foreground">Runs & evolution analysis</span></a>
						<a href="/rl" class="block rounded-md px-3 py-2 text-sm hover:bg-muted"><b class="block">RL charts</b><span class="text-xs text-muted-foreground">Policy & reward analysis</span></a>
					</div>
				</details>
				<a href="/backtest" class={`${navClass} ${backtestActive ? 'bg-muted text-foreground' : 'text-muted-foreground'}`} aria-current={backtestActive ? 'page' : undefined}>Backtest</a>
			</div>

			<button type="button" class="grid size-11 place-items-center rounded-md border sm:hidden" onclick={() => (mobileOpen = !mobileOpen)} aria-expanded={mobileOpen} aria-controls="mobile-navigation" aria-label={mobileOpen ? 'Close menu' : 'Open menu'}>
				{#if mobileOpen}<X size={18} />{:else}<Menu size={18} />{/if}
			</button>
		</div>

		{#if mobileOpen}
			<div id="mobile-navigation" class="border-t bg-background p-2 sm:hidden">
				<a href="/train" onclick={() => (mobileOpen = false)} class={`block min-h-11 rounded-md px-3 py-3 text-sm font-medium ${trainingActive ? 'bg-muted' : ''}`}>Training</a>
				<div class="mt-1 px-3 pt-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Charts & analysis</div>
				<div class="grid grid-cols-2 gap-1 p-1">
					<a href="/ga" onclick={() => (mobileOpen = false)} class={`min-h-11 rounded-md px-3 py-3 text-sm ${path.startsWith('/ga') ? 'bg-muted font-medium' : ''}`}>GA charts</a>
					<a href="/rl" onclick={() => (mobileOpen = false)} class={`min-h-11 rounded-md px-3 py-3 text-sm ${path.startsWith('/rl') ? 'bg-muted font-medium' : ''}`}>RL charts</a>
				</div>
				<a href="/backtest" onclick={() => (mobileOpen = false)} class={`block min-h-11 rounded-md px-3 py-3 text-sm font-medium ${backtestActive ? 'bg-muted' : ''}`}>Backtest</a>
			</div>
		{/if}
	</nav>
	{@render children()}
</div>
