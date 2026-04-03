<script lang="ts">
	import BacktestAnalyzerView from "./_components/BacktestAnalyzerView.svelte";
	import BacktestFileBrowserModal from "./_components/BacktestFileBrowserModal.svelte";
	import BacktestScriptView from "./_components/BacktestScriptView.svelte";
	import SideMenu from "./_components/SideMenu.svelte";
	import {
		ANALYZER_MAX_COMBOS,
		axisValueKey,
		buildEquityChartData,
		buildFileBrowserUrl,
		countRange,
		createDefaultAnalyzerConfig,
		createDefaultAnalyzerEnv,
		createDefaultBacktestEnv,
		createDefaultBacktestLimits,
		demoEquitySeries,
		demoMetrics,
		equityChartOptions,
		formatMetricValue,
		formatRangePlaceholder,
		heatmapColorForValue,
		heatmapMetricOptions,
		metricColorValue,
		metricValue,
		normalizeExtensions,
		normalizeSweepParam,
		presetDatasetPath,
		sampleScript,
		toHeatmapRangeValue,
		toInteger,
		toNumber,
		validateAnalyzerIndicator
	} from "./backtest";

	import type {
		AnalyzerCell,
		AnalyzerMetrics,
		AnalyzerResult,
		BacktestView,
		BacktestResult,
		DatasetMode,
		DatasetPickerTarget,
		FileEntry,
		NumericInput
	} from "./types";

	let activeView = $state<BacktestView>("script");
	let menuCollapsed = $state(false);

	let filePickerTarget: DatasetPickerTarget | null = null;
	let fileBrowserOpen = $state(false);
	let fileBrowserDir = $state("");
	let fileBrowserParent = $state<string | null>(null);
	let fileBrowserEntries = $state<FileEntry[]>([]);
	let fileBrowserLoading = $state(false);
	let fileBrowserError = $state("");
	let fileBrowserToken = 0;
	let fileBrowserTitle = $state("Select File");
	let fileBrowserExtensions = $state<string[]>(["parquet"]);

	let datasetMode = $state<DatasetMode>("train");
	let datasetPath = $state("data/train/SPY0.parquet");

	let scriptText = $state(sampleScript);

	let runResult = $state<BacktestResult | null>(null);
	let running = $state(false);
	let runError = $state("");

	let analyzerResult = $state<AnalyzerResult | null>(null);
	let analyzerRunning = $state(false);
	let analyzerError = $state("");

	let analyzerDatasetMode = $state<DatasetMode>("train");
	let analyzerDatasetPath = $state("data/train/SPY0.parquet");

	let analyzer = $state(createDefaultAnalyzerConfig());

	let analyzerEnv = $state(createDefaultAnalyzerEnv());

	let heatmapMetric = $state<keyof AnalyzerMetrics>("fitness");
	let heatmapRangeMin = $state<NumericInput>("");
	let heatmapRangeMax = $state<NumericInput>("");
	let selectedTakeProfit = $state<string | undefined>(undefined);
	let selectedStopLoss = $state<string | undefined>(undefined);
	let selectedCell = $state<AnalyzerCell | null>(null);

	let env = $state(createDefaultBacktestEnv());

	let limits = $state(createDefaultBacktestLimits());

	const loadFileBrowser = async (dir: string, extensions: string[]) => {
		const token = ++fileBrowserToken;
		fileBrowserLoading = true;
		fileBrowserError = "";
		try {
			const res = await fetch(buildFileBrowserUrl(dir, extensions));
			if (!res.ok) {
				const errPayload = await res.json().catch(() => null);
				throw new Error(errPayload?.error || `Failed to load files (${res.status})`);
			}
			const payload = await res.json();
			if (token !== fileBrowserToken) return false;
			fileBrowserDir = typeof payload.dir === "string" ? payload.dir : dir;
			fileBrowserParent = typeof payload.parent === "string" ? payload.parent : null;
			fileBrowserEntries = Array.isArray(payload.entries) ? payload.entries : [];
			return true;
		} catch (err) {
			if (token === fileBrowserToken) {
				fileBrowserError = err instanceof Error ? err.message : String(err);
				fileBrowserEntries = [];
				fileBrowserParent = null;
			}
			return false;
		} finally {
			if (token === fileBrowserToken) {
				fileBrowserLoading = false;
			}
		}
	};

	const openFileBrowser = async (
		target: DatasetPickerTarget,
		title: string,
		extensions: string[],
		preferredDir: string
	) => {
		filePickerTarget = target;
		fileBrowserTitle = title;
		fileBrowserExtensions = normalizeExtensions(extensions);
		fileBrowserOpen = true;
		fileBrowserDir = "";
		fileBrowserParent = null;
		fileBrowserEntries = [];
		const ok = await loadFileBrowser(preferredDir, fileBrowserExtensions);
		if (!ok && preferredDir !== "") {
			await loadFileBrowser("", fileBrowserExtensions);
		}
	};

	const openDatasetPicker = async (target: DatasetPickerTarget) => {
		await openFileBrowser(target, "Select Parquet File", ["parquet"], "data");
	};

	const closeFileBrowser = () => {
		fileBrowserOpen = false;
		fileBrowserError = "";
		filePickerTarget = null;
	};

	const handleFileEntry = (entry: FileEntry) => {
		if (entry.kind === "dir") {
			void loadFileBrowser(entry.path, fileBrowserExtensions);
			return;
		}
		if (!filePickerTarget) return;
		if (filePickerTarget === "script") {
			datasetMode = "custom";
			datasetPath = entry.path;
		} else {
			analyzerDatasetMode = "custom";
			analyzerDatasetPath = entry.path;
		}
		closeFileBrowser();
	};

	const handleFileBrowserUp = () => {
		if (fileBrowserParent === null) return;
		void loadFileBrowser(fileBrowserParent, fileBrowserExtensions);
	};

	let numericEnv = $derived.by(() => ({
		initialBalance: toNumber(env.initialBalance),
		maxPosition: toNumber(env.maxPosition),
		commission: toNumber(env.commission),
		slippage: toNumber(env.slippage),
		marginPerContract: toNumber(env.marginPerContract),
		contractMultiplier: toNumber(env.contractMultiplier)
	}));

	let numericLimits = $derived.by(() => ({
		memoryMb: toNumber(limits.memoryMb),
		instructionLimit: toNumber(limits.instructionLimit),
		instructionInterval: toNumber(limits.instructionInterval)
	}));

	let scriptInvalid = $derived.by(() => scriptText.trim().length === 0);
	let datasetPathInvalid = $derived.by(() => {
		if (datasetMode !== "custom") return false;
		const trimmed = datasetPath.trim().toLowerCase();
		if (!trimmed) return true;
		return !trimmed.endsWith(".parquet");
	});

	let invalidInitialBalance = $derived.by(
		() => numericEnv.initialBalance === null || numericEnv.initialBalance <= 0
	);
	let invalidMaxPosition = $derived.by(
		() => numericEnv.maxPosition === null || numericEnv.maxPosition < 0
	);
	let invalidCommission = $derived.by(
		() => numericEnv.commission === null || numericEnv.commission < 0
	);
	let invalidSlippage = $derived.by(
		() => numericEnv.slippage === null || numericEnv.slippage < 0
	);
	let invalidMargin = $derived.by(
		() => numericEnv.marginPerContract === null || numericEnv.marginPerContract < 0
	);
	let invalidContractMultiplier = $derived.by(
		() => numericEnv.contractMultiplier === null || numericEnv.contractMultiplier <= 0
	);
	let invalidMemory = $derived.by(() => numericLimits.memoryMb === null || numericLimits.memoryMb < 0);
	let invalidInstructionLimit = $derived.by(
		() => numericLimits.instructionLimit === null || numericLimits.instructionLimit < 0
	);
	let invalidInstructionInterval = $derived.by(
		() => numericLimits.instructionInterval === null || numericLimits.instructionInterval <= 0
	);

	let validationErrors = $derived.by(() => {
		const errors: string[] = [];
		if (scriptInvalid) errors.push("Script is required.");
		if (datasetPathInvalid) {
			errors.push("Custom dataset path must be a .parquet file.");
		}
		if (invalidInitialBalance) errors.push("Initial balance must be greater than 0.");
		if (invalidMaxPosition) errors.push("Max position must be 0 or higher.");
		if (invalidCommission) errors.push("Commission must be 0 or higher.");
		if (invalidSlippage) errors.push("Slippage must be 0 or higher.");
		if (invalidMargin) errors.push("Margin per contract must be 0 or higher.");
		if (invalidContractMultiplier) errors.push("Contract multiplier must be greater than 0.");
		if (invalidMemory) errors.push("Memory limit must be 0 or higher.");
		if (invalidInstructionLimit) errors.push("Instruction limit must be 0 or higher.");
		if (invalidInstructionInterval) errors.push("Instruction interval must be greater than 0.");
		return errors;
	});

	let canRun = $derived.by(() => !running && validationErrors.length === 0);

	let numericAnalyzerEnv = $derived.by(() => ({
		initialBalance: toNumber(analyzerEnv.initialBalance),
		maxPosition: toNumber(analyzerEnv.maxPosition),
		commission: toNumber(analyzerEnv.commission),
		slippage: toNumber(analyzerEnv.slippage),
		marginPerContract: toNumber(analyzerEnv.marginPerContract),
		contractMultiplier: toNumber(analyzerEnv.contractMultiplier)
	}));

	let analyzerIndicatorA = $derived.by(() => ({
		start: toNumber(analyzer.indicatorA.start),
		end: toNumber(analyzer.indicatorA.end),
		step: toNumber(analyzer.indicatorA.step),
		period: toInteger(analyzer.indicatorA.period),
		kamaFast: toInteger(analyzer.indicatorA.kamaFast),
		kamaSlow: toInteger(analyzer.indicatorA.kamaSlow),
		almaOffset: toNumber(analyzer.indicatorA.almaOffset),
		almaSigma: toNumber(analyzer.indicatorA.almaSigma)
	}));

	let analyzerIndicatorB = $derived.by(() => ({
		start: toNumber(analyzer.indicatorB.start),
		end: toNumber(analyzer.indicatorB.end),
		step: toNumber(analyzer.indicatorB.step),
		period: toInteger(analyzer.indicatorB.period),
		kamaFast: toInteger(analyzer.indicatorB.kamaFast),
		kamaSlow: toInteger(analyzer.indicatorB.kamaSlow),
		almaOffset: toNumber(analyzer.indicatorB.almaOffset),
		almaSigma: toNumber(analyzer.indicatorB.almaSigma)
	}));

	let analyzerTakeProfit = $derived.by(() => ({
		start: toNumber(analyzer.takeProfit.start),
		end: toNumber(analyzer.takeProfit.end),
		step: toNumber(analyzer.takeProfit.step)
	}));

	let analyzerStopLoss = $derived.by(() => ({
		start: toNumber(analyzer.stopLoss.start),
		end: toNumber(analyzer.stopLoss.end),
		step: toNumber(analyzer.stopLoss.step)
	}));

	let analyzerDatasetPathInvalid = $derived.by(() => {
		if (analyzerDatasetMode !== "custom") return false;
		const trimmed = analyzerDatasetPath.trim().toLowerCase();
		if (!trimmed) return true;
		return !trimmed.endsWith(".parquet");
	});

	let analyzerIndicatorAErrors = $derived.by(() =>
		validateAnalyzerIndicator(
			"A",
			analyzer.indicatorA.kind,
			analyzer.indicatorA.sweepParam,
			analyzerIndicatorA
		)
	);

	let analyzerIndicatorBErrors = $derived.by(() =>
		validateAnalyzerIndicator(
			"B",
			analyzer.indicatorB.kind,
			analyzer.indicatorB.sweepParam,
			analyzerIndicatorB
		)
	);

	let invalidAnalyzerTakeProfit = $derived.by(() => {
		if (!analyzer.takeProfit.enabled) return false;
		const { start, end, step } = analyzerTakeProfit;
		if (start === null || end === null || step === null) return true;
		if (start < 0 || step <= 0) return true;
		return end < start;
	});

	let invalidAnalyzerStopLoss = $derived.by(() => {
		if (!analyzer.stopLoss.enabled) return false;
		const { start, end, step } = analyzerStopLoss;
		if (start === null || end === null || step === null) return true;
		if (start < 0 || step <= 0) return true;
		return end < start;
	});

	let invalidAnalyzerInitialBalance = $derived.by(
		() => numericAnalyzerEnv.initialBalance === null || numericAnalyzerEnv.initialBalance <= 0
	);
	let invalidAnalyzerMaxPosition = $derived.by(
		() => numericAnalyzerEnv.maxPosition === null || numericAnalyzerEnv.maxPosition < 0
	);
	let invalidAnalyzerCommission = $derived.by(
		() => numericAnalyzerEnv.commission === null || numericAnalyzerEnv.commission < 0
	);
	let invalidAnalyzerSlippage = $derived.by(
		() => numericAnalyzerEnv.slippage === null || numericAnalyzerEnv.slippage < 0
	);
	let invalidAnalyzerMargin = $derived.by(
		() => numericAnalyzerEnv.marginPerContract === null || numericAnalyzerEnv.marginPerContract < 0
	);
	let invalidAnalyzerContractMultiplier = $derived.by(
		() => numericAnalyzerEnv.contractMultiplier === null || numericAnalyzerEnv.contractMultiplier <= 0
	);

	let analyzerCombos = $derived.by(() => {
		const countA = countRange(analyzerIndicatorA.start, analyzerIndicatorA.end, analyzerIndicatorA.step);
		const countB = countRange(analyzerIndicatorB.start, analyzerIndicatorB.end, analyzerIndicatorB.step);
		const countTp = analyzer.takeProfit.enabled
			? countRange(analyzerTakeProfit.start, analyzerTakeProfit.end, analyzerTakeProfit.step)
			: 1;
		const countSl = analyzer.stopLoss.enabled
			? countRange(analyzerStopLoss.start, analyzerStopLoss.end, analyzerStopLoss.step)
			: 1;
		return countA * countB * countTp * countSl;
	});

	let analyzerValidationErrors = $derived.by(() => {
		const errors: string[] = [];
		if (analyzerDatasetPathInvalid) errors.push("Custom dataset path must be a .parquet file.");
		errors.push(...analyzerIndicatorAErrors, ...analyzerIndicatorBErrors);
		if (invalidAnalyzerTakeProfit) errors.push("Take profit range must be valid.");
		if (invalidAnalyzerStopLoss) errors.push("Stop loss range must be valid.");
		if (invalidAnalyzerInitialBalance) errors.push("Initial balance must be greater than 0.");
		if (invalidAnalyzerMaxPosition) errors.push("Max position must be 0 or higher.");
		if (invalidAnalyzerCommission) errors.push("Commission must be 0 or higher.");
		if (invalidAnalyzerSlippage) errors.push("Slippage must be 0 or higher.");
		if (invalidAnalyzerMargin) errors.push("Margin per contract must be 0 or higher.");
		if (invalidAnalyzerContractMultiplier) {
			errors.push("Contract multiplier must be greater than 0.");
		}
		if (analyzerCombos > ANALYZER_MAX_COMBOS) {
			errors.push(`Combination count exceeds ${ANALYZER_MAX_COMBOS.toLocaleString()}.`);
		}
		return errors;
	});

	let analyzerCanRun = $derived.by(() => !analyzerRunning && analyzerValidationErrors.length === 0);

	$effect(() => {
		const presetPath = presetDatasetPath(datasetMode);
		if (presetPath) {
			datasetPath = presetPath;
		}
	});

	$effect(() => {
		const presetPath = presetDatasetPath(analyzerDatasetMode);
		if (presetPath) {
			analyzerDatasetPath = presetPath;
		}
	});

	$effect(() => {
		const normalized = normalizeSweepParam(analyzer.indicatorA.kind, analyzer.indicatorA.sweepParam);
		if (analyzer.indicatorA.sweepParam !== normalized) {
			analyzer.indicatorA.sweepParam = normalized;
		}
	});

	$effect(() => {
		const normalized = normalizeSweepParam(analyzer.indicatorB.kind, analyzer.indicatorB.sweepParam);
		if (analyzer.indicatorB.sweepParam !== normalized) {
			analyzer.indicatorB.sweepParam = normalized;
		}
	});

	let equitySeries = $derived.by(() => runResult?.equity_curve ?? demoEquitySeries);
	let activeMetrics = $derived.by(() => runResult?.metrics ?? demoMetrics);

	let equityChartData = $derived.by(() => buildEquityChartData(equitySeries, !runResult));

	let analyzerAxisA = $derived.by(() => analyzerResult?.axes.indicatorA.periods ?? []);
	let analyzerAxisB = $derived.by(() => analyzerResult?.axes.indicatorB.periods ?? []);

	let heatmapRange = $derived.by(() => ({
		min: toHeatmapRangeValue(heatmapRangeMin, heatmapMetric),
		max: toHeatmapRangeValue(heatmapRangeMax, heatmapMetric)
	}));

	let filteredAnalyzerCells = $derived.by(() => {
		if (!analyzerResult) return [];
		const tpValue = selectedTakeProfit === undefined ? null : Number(selectedTakeProfit);
		const slValue = selectedStopLoss === undefined ? null : Number(selectedStopLoss);
		return analyzerResult.results.filter((cell) => {
			const tpMatch = tpValue === null ? cell.takeProfit === null : cell.takeProfit === tpValue;
			const slMatch = slValue === null ? cell.stopLoss === null : cell.stopLoss === slValue;
			return tpMatch && slMatch;
		});
	});

	const isHeatmapValueInRange = (metrics: AnalyzerMetrics) => {
		const { min, max } = heatmapRange;
		if (min === null && max === null) return true;
		const value = metricValue(metrics, heatmapMetric);
		if (!Number.isFinite(value)) return false;
		if (min !== null && value < min) return false;
		if (max !== null && value > max) return false;
		return true;
	};

	let analyzerCellMap = $derived.by(() => {
		const map = new Map<string, AnalyzerCell>();
		for (const cell of filteredAnalyzerCells) {
			map.set(`${axisValueKey(cell.aPeriod)}-${axisValueKey(cell.bPeriod)}`, cell);
		}
		return map;
	});

	let heatmapMetricDomain = $derived.by(() => {
		const values = filteredAnalyzerCells
			.map((cell) => metricValue(cell.metrics, heatmapMetric))
			.filter((value) => Number.isFinite(value));
		if (values.length === 0) return { min: 0, max: 1 };
		const min = Math.min(...values);
		const max = Math.max(...values);
		if (min === max) return { min, max: min + 1 };
		return { min, max };
	});

	let heatmapRangePlaceholder = $derived.by(() => ({
		min: formatRangePlaceholder(heatmapMetricDomain.min, heatmapMetric),
		max: formatRangePlaceholder(heatmapMetricDomain.max, heatmapMetric)
	}));

	let heatmapDomain = $derived.by(() => {
		const values = filteredAnalyzerCells
			.filter((cell) => isHeatmapValueInRange(cell.metrics))
			.map((cell) => metricColorValue(cell.metrics, heatmapMetric))
			.filter((value) => Number.isFinite(value));
		if (values.length === 0) return { min: 0, max: 1 };
		const min = Math.min(...values);
		const max = Math.max(...values);
		if (min === max) return { min, max: min + 1 };
		return { min, max };
	});

	const heatmapColor = (metrics: AnalyzerMetrics) => {
		const value = metricColorValue(metrics, heatmapMetric);
		return heatmapColorForValue(value, heatmapDomain);
	};

	const findCell = (aPeriod: number, bPeriod: number) => {
		return analyzerCellMap.get(`${axisValueKey(aPeriod)}-${axisValueKey(bPeriod)}`);
	};

	$effect(() => {
		if (!selectedCell) return;
		const tpValue = selectedTakeProfit === undefined ? null : Number(selectedTakeProfit);
		const slValue = selectedStopLoss === undefined ? null : Number(selectedStopLoss);
		const matchesCurrentTp = tpValue === null ? true : selectedCell.takeProfit === tpValue;
		const matchesCurrentSl = slValue === null ? true : selectedCell.stopLoss === slValue;
		if (!matchesCurrentTp || !matchesCurrentSl) {
			return;
		}
		const key = `${axisValueKey(selectedCell.aPeriod)}-${axisValueKey(selectedCell.bPeriod)}`;
		if (!analyzerCellMap.has(key)) {
			selectedCell = null;
		}
	});

	const runBacktest = async () => {
		if (validationErrors.length > 0) {
			runError = "Fix validation errors before running.";
			return;
		}
		running = true;
		runError = "";
		try {
			const payload = {
				dataset: datasetMode === "custom" ? null : datasetMode,
				path: datasetMode === "custom" ? datasetPath : null,
				script: scriptText,
				env: {
					initialBalance: numericEnv.initialBalance,
					maxPosition: numericEnv.maxPosition,
					commission: numericEnv.commission,
					slippage: numericEnv.slippage,
					marginPerContract: numericEnv.marginPerContract,
					marginMode: env.marginMode,
					contractMultiplier: numericEnv.contractMultiplier,
					enforceMargin: env.enforceMargin
				},
				limits: {
					memoryMb: numericLimits.memoryMb,
					instructionLimit: numericLimits.instructionLimit,
					instructionInterval: numericLimits.instructionInterval
				}
			};

			const res = await fetch("/api/backtest", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(payload)
			});
			const data = await res.json();
			if (!res.ok) {
				throw new Error(data?.error || `Backtest failed (${res.status})`);
			}
			runResult = data as BacktestResult;
		} catch (err) {
			runError = err instanceof Error ? err.message : String(err);
		} finally {
			running = false;
		}
	};

	const runAnalyzer = async () => {
		if (analyzerValidationErrors.length > 0) {
			analyzerError = "Fix validation errors before running.";
			return;
		}
		analyzerRunning = true;
		analyzerError = "";
		try {
			const payload = {
				dataset: analyzerDatasetMode === "custom" ? null : analyzerDatasetMode,
				path: analyzerDatasetMode === "custom" ? analyzerDatasetPath : null,
				signal: {
					indicatorA: {
						kind: analyzer.indicatorA.kind,
						sweepParam: analyzer.indicatorA.sweepParam,
						range: {
							start: analyzerIndicatorA.start,
							end: analyzerIndicatorA.end,
							step: analyzerIndicatorA.step
						},
						params: {
							period: analyzerIndicatorA.period,
							fast: analyzerIndicatorA.kamaFast,
							slow: analyzerIndicatorA.kamaSlow,
							offset: analyzerIndicatorA.almaOffset,
							sigma: analyzerIndicatorA.almaSigma
						}
					},
					indicatorB: {
						kind: analyzer.indicatorB.kind,
						sweepParam: analyzer.indicatorB.sweepParam,
						range: {
							start: analyzerIndicatorB.start,
							end: analyzerIndicatorB.end,
							step: analyzerIndicatorB.step
						},
						params: {
							period: analyzerIndicatorB.period,
							fast: analyzerIndicatorB.kamaFast,
							slow: analyzerIndicatorB.kamaSlow,
							offset: analyzerIndicatorB.almaOffset,
							sigma: analyzerIndicatorB.almaSigma
						}
					},
					buyAction: analyzer.buyAction,
					sellAction: analyzer.sellAction
				},
				takeProfit: analyzer.takeProfit.enabled
					? {
							start: analyzerTakeProfit.start,
							end: analyzerTakeProfit.end,
							step: analyzerTakeProfit.step
						}
					: null,
				stopLoss: analyzer.stopLoss.enabled
					? {
							start: analyzerStopLoss.start,
							end: analyzerStopLoss.end,
							step: analyzerStopLoss.step
						}
					: null,
				env: {
					initialBalance: numericAnalyzerEnv.initialBalance,
					maxPosition: numericAnalyzerEnv.maxPosition,
					commission: numericAnalyzerEnv.commission,
					slippage: numericAnalyzerEnv.slippage,
					marginPerContract: numericAnalyzerEnv.marginPerContract,
					marginMode: analyzerEnv.marginMode,
					contractMultiplier: numericAnalyzerEnv.contractMultiplier,
					enforceMargin: analyzerEnv.enforceMargin
				},
				maxCombinations: ANALYZER_MAX_COMBOS
			};

			const res = await fetch("/api/strategy-analyzer", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(payload)
			});
			const data = await res.json();
			if (!res.ok) {
				throw new Error(data?.error || `Analyzer failed (${res.status})`);
			}
			analyzerResult = data as AnalyzerResult;
		} catch (err) {
			analyzerError = err instanceof Error ? err.message : String(err);
		} finally {
			analyzerRunning = false;
		}
	};

	$effect(() => {
		if (!analyzerResult) {
			selectedTakeProfit = undefined;
			selectedStopLoss = undefined;
			selectedCell = null;
			return;
		}
		selectedTakeProfit =
			analyzerResult.axes.takeProfitValues?.[0] !== undefined
				? String(analyzerResult.axes.takeProfitValues[0])
				: undefined;
		selectedStopLoss =
			analyzerResult.axes.stopLossValues?.[0] !== undefined
				? String(analyzerResult.axes.stopLossValues[0])
				: undefined;
		selectedCell = null;
	});
</script>

<div class="min-h-screen bg-background">
	<SideMenu bind:activeView={activeView} bind:collapsed={menuCollapsed} />

	<main
		class={`p-6 lg:p-8 space-y-6 lg:space-y-8 mx-auto max-w-3xl lg:mx-0 lg:max-w-none ${
			menuCollapsed ? "lg:ml-[120px]" : "lg:ml-[320px]"
		}`}
	>
		{#if activeView === "script"}
			<BacktestScriptView
				{running}
				{canRun}
				{validationErrors}
				{runError}
				{runResult}
				onRun={runBacktest}
				bind:scriptText
				{sampleScript}
				{scriptInvalid}
				bind:datasetMode
				bind:datasetPath
				{datasetPathInvalid}
				onBrowseDataset={() => void openDatasetPicker("script")}
				{env}
				{limits}
				validation={{
					invalidInitialBalance,
					invalidMaxPosition,
					invalidCommission,
					invalidSlippage,
					invalidMargin,
					invalidContractMultiplier,
					invalidMemory,
					invalidInstructionLimit,
					invalidInstructionInterval
				}}
				{activeMetrics}
				{equityChartData}
				chartOptions={equityChartOptions}
			/>
		{:else}
			<BacktestAnalyzerView
				{analyzerRunning}
				{analyzerCanRun}
				{analyzerCombos}
				{analyzerResult}
				{analyzerValidationErrors}
				{analyzerError}
				onRun={runAnalyzer}
				{analyzer}
				bind:datasetMode={analyzerDatasetMode}
				bind:datasetPath={analyzerDatasetPath}
				datasetPathInvalid={analyzerDatasetPathInvalid}
				onBrowseDataset={() => void openDatasetPicker("analyzer")}
				{analyzerEnv}
				validation={{
					invalidAnalyzerInitialBalance,
					invalidAnalyzerMaxPosition,
					invalidAnalyzerCommission,
					invalidAnalyzerSlippage,
					invalidAnalyzerMargin,
					invalidAnalyzerContractMultiplier
				}}
				bind:heatmapMetric
				{heatmapMetricOptions}
				bind:selectedTakeProfit
				bind:selectedStopLoss
				bind:heatmapRangeMin
				bind:heatmapRangeMax
				heatmapRangePlaceholderMin={heatmapRangePlaceholder.min}
				heatmapRangePlaceholderMax={heatmapRangePlaceholder.max}
				{analyzerAxisA}
				{analyzerAxisB}
				{findCell}
				{heatmapColor}
				{metricValue}
				{formatMetricValue}
				{isHeatmapValueInRange}
				bind:selectedCell
			/>
		{/if}
	</main>
</div>

<BacktestFileBrowserModal
	open={fileBrowserOpen}
	title={fileBrowserTitle}
	dir={fileBrowserDir}
	extensions={fileBrowserExtensions}
	parent={fileBrowserParent}
	entries={fileBrowserEntries}
	loading={fileBrowserLoading}
	error={fileBrowserError}
	emptyMessage="No parquet files found here."
	onClose={closeFileBrowser}
	onUp={handleFileBrowserUp}
	onSelect={handleFileEntry}
/>
