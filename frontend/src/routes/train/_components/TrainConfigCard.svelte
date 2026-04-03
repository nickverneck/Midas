<script lang="ts">
	import { Badge } from "$lib/components/ui/badge";
	import * as Tabs from "$lib/components/ui/tabs";
	import GaTrainingForm from "./GaTrainingForm.svelte";
	import RlTrainingForm from "./RlTrainingForm.svelte";
	import type {
		DataMode,
		FuturesPresetKey,
		GaParams,
		ParquetKey,
		RlAlgorithm,
		RlParams,
		TrainMode
	} from "../types";

	type Props = {
		trainMode: TrainMode;
		rlAlgorithm: RlAlgorithm;
		gaDataMode: DataMode;
		rlDataMode: DataMode;
		gaParams: GaParams;
		rlParams: RlParams;
		onBrowseParquet: (mode: TrainMode, key: ParquetKey) => void;
		onApplyFuturesPreset: (mode: TrainMode, presetKey: FuturesPresetKey) => void;
		onSubmitGa: () => void;
		onSubmitRl: () => void;
	};

	let {
		trainMode = $bindable(),
		rlAlgorithm = $bindable(),
		gaDataMode = $bindable(),
		rlDataMode = $bindable(),
		gaParams,
		rlParams,
		onBrowseParquet,
		onApplyFuturesPreset,
		onSubmitGa,
		onSubmitRl
	}: Props = $props();

	const handleGaSubmit = (event: SubmitEvent) => {
		event.preventDefault();
		onSubmitGa();
	};

	const handleRlSubmit = (event: SubmitEvent) => {
		event.preventDefault();
		onSubmitRl();
	};
</script>

<div class="space-y-4 rounded-lg border bg-card/50 p-4">
	<div class="flex items-center justify-between">
		<div>
			<div class="text-xs uppercase tracking-wide text-muted-foreground">Step 2</div>
			<div class="text-sm font-semibold">Configure your run</div>
		</div>
		<Badge variant="outline">{trainMode === "ga" ? "GA" : "RL"}</Badge>
	</div>

	<Tabs.Root bind:value={trainMode} class="w-full">
		<Tabs.List class="mb-4 grid w-full grid-cols-2">
			<Tabs.Trigger value="ga">GA (Primary)</Tabs.Trigger>
			<Tabs.Trigger value="rl">RL (PPO/GRPO)</Tabs.Trigger>
		</Tabs.List>

		<Tabs.Content value="ga">
			<form class="space-y-4" onsubmit={handleGaSubmit}>
				<GaTrainingForm
					params={gaParams}
					bind:dataMode={gaDataMode}
					onBrowseParquet={(key) => onBrowseParquet("ga", key)}
					onApplyFuturesPreset={(presetKey) => onApplyFuturesPreset("ga", presetKey)}
				/>
			</form>
		</Tabs.Content>

		<Tabs.Content value="rl">
			<form class="space-y-4" onsubmit={handleRlSubmit}>
				<RlTrainingForm
					params={rlParams}
					bind:dataMode={rlDataMode}
					bind:rlAlgorithm
					onBrowseParquet={(key) => onBrowseParquet("rl", key)}
					onApplyFuturesPreset={(presetKey) => onApplyFuturesPreset("rl", presetKey)}
				/>
			</form>
		</Tabs.Content>
	</Tabs.Root>
</div>
