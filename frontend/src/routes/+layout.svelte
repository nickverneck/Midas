<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import '../app.css';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<div class="min-h-screen bg-background">
	<nav class="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
		<div class="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3 text-sm">
			<a href="/" class="font-semibold text-foreground">GA Analytics</a>
			<a href="/rl" class="text-muted-foreground hover:text-foreground">RL Analytics</a>
			<a href="/train" class="text-muted-foreground hover:text-foreground">Train</a>
		</div>
	</nav>
	{@render children()}
</div>
