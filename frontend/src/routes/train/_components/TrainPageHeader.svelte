<script lang="ts">
	import type { TrainMode } from "../types";

	type Props = {
		trainMode: TrainMode;
	};

	let { trainMode }: Props = $props();
</script>

<div class="flex items-center gap-4">
	<a href="/ga" class="text-sm text-muted-foreground hover:text-foreground">← Back to GA Analytics</a>
	<h1 class="text-4xl font-bold tracking-tight">Train {trainMode === "ga" ? "GA" : "RL"} Model</h1>
</div>
