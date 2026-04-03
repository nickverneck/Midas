<script lang="ts">
	import type { ChartConfiguration } from "chart.js";

	import BacktestEnvironmentCard from "./BacktestEnvironmentCard.svelte";
	import BacktestHero from "./BacktestHero.svelte";
	import DatasetCard from "./DatasetCard.svelte";
	import EquityCurveCard from "./EquityCurveCard.svelte";
	import PerformanceSnapshotCard from "./PerformanceSnapshotCard.svelte";
	import RunBacktestCard from "./RunBacktestCard.svelte";
	import ScriptApiCard from "./ScriptApiCard.svelte";
	import ScriptEditorCard from "./ScriptEditorCard.svelte";
	import type {
		BacktestEnv,
		BacktestLimits,
		BacktestMetrics,
		BacktestResult,
		DatasetMode
	} from "../types";

	type ValidationState = {
		invalidInitialBalance: boolean;
		invalidMaxPosition: boolean;
		invalidCommission: boolean;
		invalidSlippage: boolean;
		invalidMargin: boolean;
		invalidContractMultiplier: boolean;
		invalidMemory: boolean;
		invalidInstructionLimit: boolean;
		invalidInstructionInterval: boolean;
	};

	type Props = {
		running: boolean;
		canRun: boolean;
		validationErrors: string[];
		runError: string;
		runResult: BacktestResult | null;
		onRun: () => void | Promise<void>;
		scriptText: string;
		sampleScript: string;
		scriptInvalid: boolean;
		datasetMode: DatasetMode;
		datasetPath: string;
		datasetPathInvalid: boolean;
		onBrowseDataset: () => void;
		env: BacktestEnv;
		limits: BacktestLimits;
		validation: ValidationState;
		activeMetrics: BacktestMetrics;
		equityChartData: ChartConfiguration["data"];
		chartOptions: ChartConfiguration["options"];
	};

	let {
		running,
		canRun,
		validationErrors,
		runError,
		runResult,
		onRun,
		scriptText = $bindable(),
		sampleScript,
		scriptInvalid,
		datasetMode = $bindable(),
		datasetPath = $bindable(),
		datasetPathInvalid,
		onBrowseDataset,
		env,
		limits,
		validation,
		activeMetrics,
		equityChartData,
		chartOptions
	}: Props = $props();
</script>

<section class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
	<div class="lg:col-start-2">
		<RunBacktestCard
			{running}
			{canRun}
			{validationErrors}
			{runError}
			{runResult}
			{onRun}
		/>
	</div>
	<div class="lg:col-start-1 lg:row-start-1">
		<BacktestHero />
	</div>
</section>

<section class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
	<ScriptEditorCard bind:scriptText {sampleScript} {scriptInvalid} />
	<div class="space-y-6">
		<DatasetCard
			title="Dataset"
			description="Select the parquet slice to evaluate."
			bind:datasetMode
			bind:datasetPath
			{datasetPathInvalid}
			onBrowse={onBrowseDataset}
		/>
		<BacktestEnvironmentCard
			{env}
			{limits}
			invalidInitialBalance={validation.invalidInitialBalance}
			invalidMaxPosition={validation.invalidMaxPosition}
			invalidCommission={validation.invalidCommission}
			invalidSlippage={validation.invalidSlippage}
			invalidMargin={validation.invalidMargin}
			invalidContractMultiplier={validation.invalidContractMultiplier}
			invalidMemory={validation.invalidMemory}
			invalidInstructionLimit={validation.invalidInstructionLimit}
			invalidInstructionInterval={validation.invalidInstructionInterval}
		/>
	</div>
</section>

<section class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
	<EquityCurveCard {runResult} {equityChartData} {chartOptions} />
	<div class="space-y-6">
		<PerformanceSnapshotCard {activeMetrics} {runResult} />
		<ScriptApiCard />
	</div>
</section>
