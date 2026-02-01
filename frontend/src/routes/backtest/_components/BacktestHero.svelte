<script lang="ts">
	import { Badge } from "$lib/components/ui/badge";
</script>

<section class="relative overflow-hidden rounded-2xl border bg-gradient-to-br from-slate-50 via-white to-slate-50 p-8">
	<div class="flex flex-wrap items-center gap-2">
		<Badge variant="secondary">Lua 5.4</Badge>
		<Badge variant="secondary">Local Parquet</Badge>
		<Badge variant="secondary">Safe Runtime</Badge>
	</div>
	<div class="mt-4 space-y-2">
		<h1 class="text-3xl font-semibold tracking-tight">Scripted Backtests</h1>
		<p class="max-w-2xl text-sm text-muted-foreground">
			Design and iterate on Lua strategies with full transparency into signals, trades, and
			performance. Compare scripts against trained models on the same dataset and environment.
		</p>
	</div>
	<div class="absolute -right-16 -top-16 h-52 w-52 rounded-full bg-primary/10 blur-3xl"></div>
	<div class="absolute -bottom-20 -left-8 h-48 w-48 rounded-full bg-secondary/40 blur-3xl"></div>
</section>
