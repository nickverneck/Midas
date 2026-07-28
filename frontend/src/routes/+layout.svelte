<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import '../app.css';
	import { page } from '$app/stores';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<div class="min-h-screen bg-background">
	<nav class="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
		<div class="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3 text-sm">
			{#if $page.url.pathname === '/' || $page.url.pathname === '/train'}
				<a href="/" class="font-semibold text-foreground" aria-current="page">Train</a>
			{:else}
				<a href="/" class="text-muted-foreground hover:text-foreground">Train</a>
			{/if}
			{#if $page.url.pathname === '/backtest'}
				<a href="/backtest" class="font-semibold text-foreground" aria-current="page">Backtest</a>
			{:else}
				<a href="/backtest" class="text-muted-foreground hover:text-foreground">Backtest</a>
			{/if}
			{#if $page.url.pathname === '/ga'}
				<a href="/ga" class="font-semibold text-foreground" aria-current="page">GA Analytics</a>
			{:else}
				<a href="/ga" class="text-muted-foreground hover:text-foreground">GA Analytics</a>
			{/if}
			{#if $page.url.pathname === '/rl'}
				<a href="/rl" class="font-semibold text-foreground" aria-current="page">RL Analytics</a>
			{:else}
				<a href="/rl" class="text-muted-foreground hover:text-foreground">RL Analytics</a>
			{/if}
		</div>
	</nav>
	{@render children()}
</div>
