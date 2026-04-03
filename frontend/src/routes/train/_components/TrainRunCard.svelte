<script lang="ts">
	import { Badge } from "$lib/components/ui/badge";
	import { Button } from "$lib/components/ui/button";
	import { Label } from "$lib/components/ui/label";
	import type { StartChoice } from "../types";

	type Props = {
		training: boolean;
		startChoice: StartChoice | null;
		liveLogUpdates: boolean;
		canStartTraining: boolean;
		runLabel: string;
		runTitle: string;
		trimmedCheckpoint: string;
		onToggleTraining: () => void;
	};

	let {
		training,
		startChoice,
		liveLogUpdates = $bindable(),
		canStartTraining,
		runLabel,
		runTitle,
		trimmedCheckpoint,
		onToggleTraining
	}: Props = $props();
</script>

<div class="space-y-3 rounded-lg border bg-card/50 p-4">
	<div class="flex items-center justify-between">
		<div>
			<div class="text-xs uppercase tracking-wide text-muted-foreground">Step 3</div>
			<div class="text-sm font-semibold">Run training</div>
		</div>
		{#if training}
			<Badge variant="outline" class="animate-pulse">Active</Badge>
		{/if}
	</div>
	<div class="flex items-center gap-2 text-sm">
		<input
			id="live-log-updates"
			type="checkbox"
			class="h-4 w-4 rounded border-input"
			bind:checked={liveLogUpdates}
		/>
		<Label for="live-log-updates">Live log updates</Label>
	</div>
	<p class="text-xs text-muted-foreground">Disable to load logs only after training completes.</p>
	<Button
		variant={training ? "destructive" : "default"}
		onclick={onToggleTraining}
		class="w-full"
		title={runTitle}
		disabled={!training && !canStartTraining}
	>
		{runLabel}
	</Button>
	{#if startChoice === "resume" && !trimmedCheckpoint}
		<div class="text-xs text-muted-foreground">Add a checkpoint path to resume training.</div>
	{/if}
</div>
