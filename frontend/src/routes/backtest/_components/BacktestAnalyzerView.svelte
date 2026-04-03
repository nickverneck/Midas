<script lang="ts">
	import AnalyzerEnvironmentCard from "./AnalyzerEnvironmentCard.svelte";
	import AnalyzerHeader from "./AnalyzerHeader.svelte";
	import DatasetCard from "./DatasetCard.svelte";
	import ExitRangesCard from "./ExitRangesCard.svelte";
	import HeatmapCard from "./HeatmapCard.svelte";
	import RunAnalyzerCard from "./RunAnalyzerCard.svelte";
	import SelectionDetailsCard from "./SelectionDetailsCard.svelte";
	import SignalBuilderCard from "./SignalBuilderCard.svelte";
	import type {
		AnalyzerCell,
		AnalyzerConfig,
		AnalyzerEnv,
		AnalyzerMetrics,
		AnalyzerResult,
		DatasetMode,
		NumericInput
	} from "../types";

	type ValidationState = {
		invalidAnalyzerInitialBalance: boolean;
		invalidAnalyzerMaxPosition: boolean;
		invalidAnalyzerCommission: boolean;
		invalidAnalyzerSlippage: boolean;
		invalidAnalyzerMargin: boolean;
		invalidAnalyzerContractMultiplier: boolean;
	};

	type HeatmapMetricOption = { value: keyof AnalyzerMetrics; label: string };

	type Props = {
		analyzerRunning: boolean;
		analyzerCanRun: boolean;
		analyzerCombos: number;
		analyzerResult: AnalyzerResult | null;
		analyzerValidationErrors: string[];
		analyzerError: string;
		onRun: () => void | Promise<void>;
		analyzer: AnalyzerConfig;
		datasetMode: DatasetMode;
		datasetPath: string;
		datasetPathInvalid: boolean;
		onBrowseDataset: () => void;
		analyzerEnv: AnalyzerEnv;
		validation: ValidationState;
		heatmapMetric: keyof AnalyzerMetrics;
		heatmapMetricOptions: HeatmapMetricOption[];
		selectedTakeProfit: string | undefined;
		selectedStopLoss: string | undefined;
		heatmapRangeMin: NumericInput;
		heatmapRangeMax: NumericInput;
		heatmapRangePlaceholderMin: string;
		heatmapRangePlaceholderMax: string;
		analyzerAxisA: number[];
		analyzerAxisB: number[];
		findCell: (aPeriod: number, bPeriod: number) => AnalyzerCell | undefined;
		heatmapColor: (metrics: AnalyzerMetrics) => string;
		metricValue: (metrics: AnalyzerMetrics, metric: keyof AnalyzerMetrics) => number;
		formatMetricValue: (value: number, metric: keyof AnalyzerMetrics) => string;
		isHeatmapValueInRange: (metrics: AnalyzerMetrics) => boolean;
		selectedCell: AnalyzerCell | null;
	};

	let {
		analyzerRunning,
		analyzerCanRun,
		analyzerCombos,
		analyzerResult,
		analyzerValidationErrors,
		analyzerError,
		onRun,
		analyzer,
		datasetMode = $bindable(),
		datasetPath = $bindable(),
		datasetPathInvalid,
		onBrowseDataset,
		analyzerEnv,
		validation,
		heatmapMetric = $bindable(),
		heatmapMetricOptions,
		selectedTakeProfit = $bindable(),
		selectedStopLoss = $bindable(),
		heatmapRangeMin = $bindable(),
		heatmapRangeMax = $bindable(),
		heatmapRangePlaceholderMin,
		heatmapRangePlaceholderMax,
		analyzerAxisA,
		analyzerAxisB,
		findCell,
		heatmapColor,
		metricValue,
		formatMetricValue,
		isHeatmapValueInRange,
		selectedCell = $bindable()
	}: Props = $props();
</script>

<section class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
	<div class="lg:col-start-2">
		<RunAnalyzerCard
			{analyzerRunning}
			{analyzerCanRun}
			{analyzerCombos}
			{analyzerResult}
			{analyzerValidationErrors}
			{analyzerError}
			onRun={onRun}
		/>
	</div>
	<div class="lg:col-start-1 lg:row-start-1">
		<AnalyzerHeader />
	</div>
</section>

<section class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
	<div class="space-y-6">
		<SignalBuilderCard {analyzer} />
		<ExitRangesCard {analyzer} />
	</div>
	<div class="space-y-6">
		<DatasetCard
			title="Analyzer Dataset"
			description="Select the parquet slice to evaluate."
			bind:datasetMode
			bind:datasetPath
			datasetPathInvalid={datasetPathInvalid}
			onBrowse={onBrowseDataset}
		/>
		<AnalyzerEnvironmentCard
			{analyzerEnv}
			invalidAnalyzerInitialBalance={validation.invalidAnalyzerInitialBalance}
			invalidAnalyzerMaxPosition={validation.invalidAnalyzerMaxPosition}
			invalidAnalyzerCommission={validation.invalidAnalyzerCommission}
			invalidAnalyzerSlippage={validation.invalidAnalyzerSlippage}
			invalidAnalyzerMargin={validation.invalidAnalyzerMargin}
			invalidAnalyzerContractMultiplier={validation.invalidAnalyzerContractMultiplier}
		/>
	</div>
</section>

<section class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
	<HeatmapCard
		{analyzerResult}
		bind:heatmapMetric
		{heatmapMetricOptions}
		bind:selectedTakeProfit
		bind:selectedStopLoss
		bind:heatmapRangeMin
		bind:heatmapRangeMax
		heatmapRangePlaceholderMin={heatmapRangePlaceholderMin}
		heatmapRangePlaceholderMax={heatmapRangePlaceholderMax}
		{analyzerAxisA}
		{analyzerAxisB}
		{findCell}
		{heatmapColor}
		{metricValue}
		{formatMetricValue}
		{isHeatmapValueInRange}
		bind:selectedCell
	/>
	<SelectionDetailsCard {selectedCell} {analyzerResult} {formatMetricValue} />
</section>
