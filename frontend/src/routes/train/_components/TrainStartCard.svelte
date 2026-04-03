<script lang="ts">
	import { Badge } from "$lib/components/ui/badge";
	import { Button } from "$lib/components/ui/button";
	import { Input } from "$lib/components/ui/input";
	import { Label } from "$lib/components/ui/label";
	import { nativeSelectClass } from "../constants";
	import type { BuildProfile, StartChoice, TrainMode } from "../types";

	type Props = {
		startChoice: StartChoice | null;
		training: boolean;
		diagnosticsLoading: boolean;
		buildProfile: BuildProfile;
		checkpointPath: string;
		trainMode: TrainMode;
		onResetTrainingSetup: () => void;
		onSelectStartChoice: (choice: StartChoice) => void;
		onRunDiagnostics: () => void;
		onBrowseCheckpoint: () => void;
	};

	let {
		startChoice,
		training,
		diagnosticsLoading,
		buildProfile = $bindable(),
		checkpointPath = $bindable(),
		trainMode,
		onResetTrainingSetup,
		onSelectStartChoice,
		onRunDiagnostics,
		onBrowseCheckpoint
	}: Props = $props();
</script>

<div class="space-y-3 rounded-lg border bg-card/50 p-4">
	<div class="flex items-start justify-between gap-3">
		<div>
			<div class="text-xs uppercase tracking-wide text-muted-foreground">Step 1</div>
			<div class="text-sm font-semibold">Choose a starting point</div>
		</div>
		{#if startChoice}
			<Button variant="ghost" size="sm" onclick={onResetTrainingSetup} disabled={training}>
				New Training
			</Button>
		{/if}
	</div>

	{#if !startChoice}
		<p class="text-xs text-muted-foreground">
			Do you want to start fresh or continue from a checkpoint?
		</p>
		<div class="grid gap-2">
			<Button onclick={() => onSelectStartChoice("new")} disabled={training}>New Training</Button>
			<Button
				variant="outline"
				onclick={() => onSelectStartChoice("resume")}
				disabled={training}
			>
				Continue from Checkpoint
			</Button>
			<Button variant="outline" onclick={onRunDiagnostics} disabled={diagnosticsLoading}>
				{diagnosticsLoading ? "Running diagnostics..." : "Run libtorch + MLX diagnostics"}
			</Button>
		</div>
	{:else}
		<div class="flex items-center gap-2 text-sm">
			<span class="text-muted-foreground">Selected:</span>
			<Badge variant="secondary">
				{startChoice === "resume" ? "Resume from Checkpoint" : "New Training"}
			</Badge>
		</div>
		<div class="grid gap-2">
			<Label for="build-profile">Build Profile</Label>
			<select
				id="build-profile"
				bind:value={buildProfile}
				disabled={training}
				class={nativeSelectClass}
			>
				<option value="debug">Debug</option>
				<option value="release">Release</option>
			</select>
			<div class="text-xs text-muted-foreground">
				`release` uses optimized Rust binaries and is the right choice for backend speed comparisons. `debug` compiles faster while you are iterating.
			</div>
		</div>
		<Button variant="outline" onclick={onRunDiagnostics} disabled={diagnosticsLoading}>
			{diagnosticsLoading ? "Running diagnostics..." : "Run libtorch + MLX diagnostics"}
		</Button>
		{#if startChoice === "resume"}
			<div class="grid gap-2">
				<Label for="checkpoint-path">Checkpoint Path</Label>
				<div class="flex flex-col gap-2">
					<Input
						id="checkpoint-path"
						type="text"
						bind:value={checkpointPath}
						placeholder={trainMode === "ga" ? "runs_ga/checkpoint_gen4.bin" : "runs_rl/checkpoint_epoch4.pt"}
					/>
					<Button type="button" variant="outline" onclick={onBrowseCheckpoint}>Browse</Button>
				</div>
				<div class="text-xs text-muted-foreground">
					GA checkpoints still use `.bin`. Policy artifacts are backend-specific: `libtorch` writes `.pt`, Candle GA/RL writes `.safetensors`, and Burn GA writes portable JSON today.
				</div>
			</div>
		{/if}
	{/if}
</div>
