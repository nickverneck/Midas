<script lang="ts">
	import { TrendingUp, ListFilter } from "lucide-svelte";
	import { Input } from "$lib/components/ui/input";

	type Props = {
		logDir: string;
		loading?: boolean;
		onReload: () => void;
	};

	let { logDir = $bindable(), loading = false, onReload }: Props = $props();
</script>

<div class="flex flex-wrap items-center justify-between gap-4 rounded-xl border bg-card p-6 shadow-sm">
	<div>
		<h1
			class="bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-4xl font-bold tracking-tight text-transparent"
		>
			GA Analytics
		</h1>
		<p class="mt-1 text-muted-foreground">Genetic algorithm training performance & evolution.</p>
	</div>
	<div class="flex flex-wrap items-center justify-end gap-3">
		<div class="flex items-center gap-2">
			<span class="text-xs text-muted-foreground">Log folder</span>
			<Input
				class="h-9 w-48"
				placeholder="runs_ga"
				bind:value={logDir}
				aria-label="Log folder for charts"
			/>
		</div>
		<button
			onclick={onReload}
			disabled={loading}
			class="flex items-center gap-2 rounded-lg border bg-secondary px-5 py-2.5 font-semibold text-secondary-foreground transition-all hover:bg-secondary/80 disabled:opacity-40"
		>
			<ListFilter size={18} />
			{loading ? "Loading..." : "Load Logs"}
		</button>
		<a
			href="/"
			class="flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 font-semibold text-primary-foreground shadow-sm transition-all hover:bg-primary/90"
		>
			<TrendingUp size={18} />
			New Run
		</a>
	</div>
</div>
