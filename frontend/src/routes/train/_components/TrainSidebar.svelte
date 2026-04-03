<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import TrainConfigCard from "./TrainConfigCard.svelte";
	import TrainRunCard from "./TrainRunCard.svelte";
	import TrainStartCard from "./TrainStartCard.svelte";
	import type {
		BuildProfile,
		DataMode,
		FuturesPresetKey,
		GaParams,
		ParquetKey,
		RlAlgorithm,
		RlParams,
		StartChoice,
		TrainMode
	} from "../types";

	type Props = {
		paramsCollapsed: boolean;
		training: boolean;
		startChoice: StartChoice | null;
		buildProfile: BuildProfile;
		checkpointPath: string;
		trainMode: TrainMode;
		rlAlgorithm: RlAlgorithm;
		gaDataMode: DataMode;
		rlDataMode: DataMode;
		gaParams: GaParams;
		rlParams: RlParams;
		diagnosticsLoading: boolean;
		liveLogUpdates: boolean;
		canStartTraining: boolean;
		runLabel: string;
		runTitle: string;
		trimmedCheckpoint: string;
		collapsedLabel: string;
		collapsedTitle: string;
		onResetTrainingSetup: () => void;
		onSelectStartChoice: (choice: StartChoice) => void;
		onRunDiagnostics: () => void;
		onBrowseCheckpoint: () => void;
		onBrowseParquet: (mode: TrainMode, key: ParquetKey) => void;
		onApplyFuturesPreset: (mode: TrainMode, presetKey: FuturesPresetKey) => void;
		onToggleTraining: () => void;
		onCollapsedAction: () => void;
		onSubmitGa: () => void;
		onSubmitRl: () => void;
	};

	let {
		paramsCollapsed = $bindable(),
		training,
		startChoice,
		buildProfile = $bindable(),
		checkpointPath = $bindable(),
		trainMode = $bindable(),
		rlAlgorithm = $bindable(),
		gaDataMode = $bindable(),
		rlDataMode = $bindable(),
		gaParams,
		rlParams,
		diagnosticsLoading,
		liveLogUpdates = $bindable(),
		canStartTraining,
		runLabel,
		runTitle,
		trimmedCheckpoint,
		collapsedLabel,
		collapsedTitle,
		onResetTrainingSetup,
		onSelectStartChoice,
		onRunDiagnostics,
		onBrowseCheckpoint,
		onBrowseParquet,
		onApplyFuturesPreset,
		onToggleTraining,
		onCollapsedAction,
		onSubmitGa,
		onSubmitRl
	}: Props = $props();
</script>

<aside
	class={`bg-card border-r shadow-sm transition-[width] duration-200 ease-out w-full lg:fixed lg:inset-y-0 lg:left-0 lg:pt-14 relative ${
		paramsCollapsed ? "lg:w-[120px]" : "lg:w-[360px]"
	}`}
>
	{#if paramsCollapsed}
		<button
			type="button"
			class="absolute inset-0 z-0 cursor-pointer appearance-none border-0 bg-transparent p-0"
			aria-label="Expand training parameters"
			title="Expand parameters"
			onclick={() => (paramsCollapsed = false)}
		></button>
	{/if}
	<div class="relative z-10 flex h-full flex-col">
		{#if !paramsCollapsed}
			<div class="flex items-center justify-between border-b px-6 py-4">
				<div class="text-sm font-semibold tracking-tight">Training Controls</div>
				<Button variant="ghost" size="sm" onclick={() => (paramsCollapsed = true)}>
					Collapse
				</Button>
			</div>
		{/if}

		{#if paramsCollapsed}
			<div class="px-4 py-4">
				<Button
					variant={training ? "destructive" : "secondary"}
					onclick={onCollapsedAction}
					class="w-full"
					title={collapsedTitle}
				>
					{collapsedLabel}
				</Button>
			</div>
		{:else}
			<div class="flex-1 overflow-y-auto px-6 pb-6">
				<div class="space-y-6">
					<TrainStartCard
						{startChoice}
						{training}
						{diagnosticsLoading}
						bind:buildProfile
						bind:checkpointPath
						{trainMode}
						onResetTrainingSetup={onResetTrainingSetup}
						onSelectStartChoice={onSelectStartChoice}
						onRunDiagnostics={onRunDiagnostics}
						onBrowseCheckpoint={onBrowseCheckpoint}
					/>

					{#if startChoice}
						<TrainConfigCard
							bind:trainMode
							bind:rlAlgorithm
							bind:gaDataMode
							bind:rlDataMode
							{gaParams}
							{rlParams}
							onBrowseParquet={onBrowseParquet}
							onApplyFuturesPreset={onApplyFuturesPreset}
							onSubmitGa={onSubmitGa}
							onSubmitRl={onSubmitRl}
						/>

						<TrainRunCard
							{training}
							{startChoice}
							bind:liveLogUpdates
							{canStartTraining}
							{runLabel}
							{runTitle}
							{trimmedCheckpoint}
							onToggleTraining={onToggleTraining}
						/>
					{/if}
				</div>
			</div>
		{/if}
	</div>
</aside>
