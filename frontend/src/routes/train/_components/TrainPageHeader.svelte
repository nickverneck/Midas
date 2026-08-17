<script lang="ts">
	import type { TrainMode } from "../types";

	type Props = {
		trainMode: TrainMode;
	};

	let { trainMode }: Props = $props();
</script>

<div id="training-page-header" class="flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:gap-4">
	<a href="/ga" class="text-sm text-muted-foreground hover:text-foreground">← Back to GA Analytics</a>
	<h1 class="text-3xl font-bold leading-tight tracking-tight sm:text-4xl">
		Train {trainMode === "ga" ? "GA" : "RL"} Model
	</h1>
</div>
