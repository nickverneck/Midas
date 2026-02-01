<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import AnalyzerEnvironmentCard from "./_components/AnalyzerEnvironmentCard.svelte";
	import AnalyzerHeader from "./_components/AnalyzerHeader.svelte";
	import BacktestEnvironmentCard from "./_components/BacktestEnvironmentCard.svelte";
	import BacktestHero from "./_components/BacktestHero.svelte";
	import DatasetCard from "./_components/DatasetCard.svelte";
	import EquityCurveCard from "./_components/EquityCurveCard.svelte";
	import ExitRangesCard from "./_components/ExitRangesCard.svelte";
	import HeatmapCard from "./_components/HeatmapCard.svelte";
	import PerformanceSnapshotCard from "./_components/PerformanceSnapshotCard.svelte";
	import RunAnalyzerCard from "./_components/RunAnalyzerCard.svelte";
	import RunBacktestCard from "./_components/RunBacktestCard.svelte";
	import SideMenu from "./_components/SideMenu.svelte";
	import ScriptApiCard from "./_components/ScriptApiCard.svelte";
	import ScriptEditorCard from "./_components/ScriptEditorCard.svelte";
	import SelectionDetailsCard from "./_components/SelectionDetailsCard.svelte";
	import SignalBuilderCard from "./_components/SignalBuilderCard.svelte";

	import type {
		AnalyzerCell,
		AnalyzerConfig,
		AnalyzerEnv,
		AnalyzerMetrics,
		AnalyzerResult,
		BacktestView,
		BacktestEnv,
		BacktestLimits,
		BacktestResult,
		CrossAction,
		DatasetMode,
		IndicatorKind,
		NumericInput
	} from "./types";

	type FileEntry = {
		name: string;
		path: string;
		kind: "file" | "dir";
	};

	type DatasetPickerTarget = "script" | "analyzer";

	const sampleScript = `-- Example: EMA cross strategy
-- Return one of: "buy", "sell", "revert", "hold"

local fast = 10
local slow = 30

function on_bar(ctx, bar)
  local fast_ema = bar["ema_" .. fast]
  local slow_ema = bar["ema_" .. slow]

  if fast_ema == nil or slow_ema == nil then
    return "hold"
  end

  if fast_ema ~= fast_ema or slow_ema ~= slow_ema then
    return "hold"
  end

  if fast_ema > slow_ema then
    if ctx.position <= 0 then
      return ctx.position == 0 and "buy" or "revert"
    end
  else
    if ctx.position >= 0 then
      return ctx.position == 0 and "sell" or "revert"
    end
  end

  return "hold"
end
`;

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

	let analyzer = $state<AnalyzerConfig>({
		indicatorA: { kind: "ema" as IndicatorKind, start: 5, end: 20, step: 1 },
		indicatorB: { kind: "ema" as IndicatorKind, start: 5, end: 20, step: 1 },
		buyAction: "crossover" as CrossAction,
		sellAction: "crossunder" as CrossAction,
		takeProfit: { enabled: false, start: 0.5, end: 2.0, step: 0.5 },
		stopLoss: { enabled: false, start: 0.5, end: 2.0, step: 0.5 }
	});

	let analyzerEnv = $state<AnalyzerEnv>({
		initialBalance: 10_000,
		maxPosition: 1,
		commission: 1.6,
		slippage: 0.25,
		marginPerContract: 50,
		marginMode: "per-contract",
		contractMultiplier: 1.0,
		enforceMargin: true
	});

	let heatmapMetric = $state<keyof AnalyzerMetrics>("fitness");
	let heatmapRangeMin = $state<NumericInput>("");
	let heatmapRangeMax = $state<NumericInput>("");
	let selectedTakeProfit = $state<string | null>(null);
	let selectedStopLoss = $state<string | null>(null);
	let selectedCell = $state<AnalyzerCell | null>(null);

	const toNumber = (value: unknown) => {
		if (value === null || value === undefined || value === "") return null;
		const num = Number(value);
		return Number.isFinite(num) ? num : null;
	};

	const toInteger = (value: unknown) => {
		const num = toNumber(value);
		if (num === null) return null;
		return Number.isInteger(num) ? num : null;
	};

	const toHeatmapRangeValue = (value: unknown, metric: keyof AnalyzerMetrics) => {
		const num = toNumber(value);
		if (num === null) return null;
		if (metric === "winRate") return num / 100;
		return num;
	};

	const countRange = (start: number | null, end: number | null, step: number | null) => {
		if (start === null || end === null || step === null) return 0;
		if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(step)) return 0;
		if (step <= 0 || end < start) return 0;
		return Math.floor((end - start) / step) + 1;
	};

	const normalizeExtensions = (extensions: string[]) =>
		extensions.map((ext) => ext.replace(/^\./, "").toLowerCase());

	const buildFileBrowserUrl = (dir: string, extensions: string[]) => {
		const params = new URLSearchParams();
		if (dir.trim() !== "") {
			params.set("dir", dir);
		}
		const normalized = normalizeExtensions(extensions);
		if (normalized.length > 0) {
			params.set("ext", normalized.join(","));
		}
		const query = params.toString();
		return query ? `/api/files?${query}` : "/api/files";
	};

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

	let env = $state<BacktestEnv>({
		initialBalance: 10_000,
		maxPosition: 1,
		commission: 1.6,
		slippage: 0.25,
		marginPerContract: 50,
		marginMode: "per-contract",
		contractMultiplier: 1.0,
		enforceMargin: true
	});

	let limits = $state<BacktestLimits>({
		memoryMb: 64,
		instructionLimit: 5_000_000,
		instructionInterval: 10_000
	});

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

	const ANALYZER_MAX_COMBOS = 20_000;

	let numericAnalyzerEnv = $derived.by(() => ({
		initialBalance: toNumber(analyzerEnv.initialBalance),
		maxPosition: toNumber(analyzerEnv.maxPosition),
		commission: toNumber(analyzerEnv.commission),
		slippage: toNumber(analyzerEnv.slippage),
		marginPerContract: toNumber(analyzerEnv.marginPerContract),
		contractMultiplier: toNumber(analyzerEnv.contractMultiplier)
	}));

	let analyzerIndicatorA = $derived.by(() => ({
		start: toInteger(analyzer.indicatorA.start),
		end: toInteger(analyzer.indicatorA.end),
		step: toInteger(analyzer.indicatorA.step)
	}));

	let analyzerIndicatorB = $derived.by(() => ({
		start: toInteger(analyzer.indicatorB.start),
		end: toInteger(analyzer.indicatorB.end),
		step: toInteger(analyzer.indicatorB.step)
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

	let invalidAnalyzerIndicatorA = $derived.by(() => {
		const { start, end, step } = analyzerIndicatorA;
		if (start === null || end === null || step === null) return true;
		if (start <= 0 || step <= 0) return true;
		return end < start;
	});

	let invalidAnalyzerIndicatorB = $derived.by(() => {
		const { start, end, step } = analyzerIndicatorB;
		if (start === null || end === null || step === null) return true;
		if (start <= 0 || step <= 0) return true;
		return end < start;
	});

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
		if (invalidAnalyzerIndicatorA) errors.push("Indicator A range must be valid (start/end/step).");
		if (invalidAnalyzerIndicatorB) errors.push("Indicator B range must be valid (start/end/step).");
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
		if (datasetMode === "train") {
			datasetPath = "data/train/SPY0.parquet";
		} else if (datasetMode === "val") {
			datasetPath = "data/val/SPY.parquet";
		}
	});

	$effect(() => {
		if (analyzerDatasetMode === "train") {
			analyzerDatasetPath = "data/train/SPY0.parquet";
		} else if (analyzerDatasetMode === "val") {
			analyzerDatasetPath = "data/val/SPY.parquet";
		}
	});

	const demoEquitySeries = Array.from({ length: 140 }, (_, i) => {
		const drift = i * 3.2;
		const wave = Math.sin(i / 9) * 40;
		return 10_000 + drift + wave;
	});

	const demoBaselineSeries = demoEquitySeries.map((v, i) => v - 120 + Math.cos(i / 11) * 25);

	const demoMetrics = {
		total_reward: 0,
		net_pnl: 412.36,
		ending_equity: 10_412.36,
		sharpe: 1.48,
		max_drawdown: 132.4,
		profit_factor: 1.32,
		win_rate: 0.54,
		max_consecutive_losses: 4,
		steps: 13_892
	};

	let equitySeries = $derived.by(() => runResult?.equity_curve ?? demoEquitySeries);
	let activeMetrics = $derived.by(() => runResult?.metrics ?? demoMetrics);

	let equityChartData = $derived.by(() => {
		const datasets = [
			{
				label: "Script",
				data: equitySeries,
				borderColor: "var(--color-chart-2)",
				backgroundColor: "rgba(59, 130, 246, 0.12)",
				fill: true,
				borderWidth: 2,
				tension: 0.25
			}
		];

		if (!runResult) {
			datasets.push({
				label: "Model",
				data: demoBaselineSeries,
				borderColor: "var(--color-chart-4)",
				backgroundColor: "transparent",
				borderDash: [6, 4],
				borderWidth: 2,
				tension: 0.2
			});
		}

		return {
			labels: equitySeries.map((_, i) => i + 1),
			datasets
		};
	});

	const chartOptions = {
		plugins: {
			legend: {
				display: true,
				position: "bottom"
			}
		},
		scales: {
			x: {
				display: false
			},
			y: {
				ticks: {
					callback: (value: number) => `$${value.toLocaleString()}`
				}
			}
		}
	};

	const metricValue = (metrics: AnalyzerMetrics, metric: keyof AnalyzerMetrics) => {
		switch (metric) {
			case "netPnl":
				return metrics.netPnl;
			case "endingEquity":
				return metrics.endingEquity;
			case "sharpe":
				return metrics.sharpe;
			case "sortino":
				return metrics.sortino;
			case "maxDrawdown":
				return metrics.maxDrawdown;
			case "profitFactor":
				return metrics.profitFactor;
			case "winRate":
				return metrics.winRate;
			case "trades":
				return metrics.trades;
			case "fitness":
				return metrics.fitness;
			case "totalReward":
				return metrics.totalReward;
			case "steps":
				return metrics.steps;
			default:
				return metrics.fitness;
		}
	};

	const heatmapMetricOptions: { value: keyof AnalyzerMetrics; label: string }[] = [
		{ value: "fitness", label: "Fitness" },
		{ value: "netPnl", label: "Net PnL" },
		{ value: "maxDrawdown", label: "Max Drawdown" },
		{ value: "sortino", label: "Sortino" },
		{ value: "sharpe", label: "Sharpe" },
		{ value: "profitFactor", label: "Profit Factor" },
		{ value: "winRate", label: "Win Rate" },
		{ value: "trades", label: "Trades" }
	];

	const metricColorValue = (metrics: AnalyzerMetrics, metric: keyof AnalyzerMetrics) => {
		const value = metricValue(metrics, metric);
		if (metric === "maxDrawdown") return -value;
		return value;
	};

	const formatMetricValue = (value: number, metric: keyof AnalyzerMetrics) => {
		if (!Number.isFinite(value)) return "n/a";
		if (metric === "winRate") return `${(value * 100).toFixed(1)}%`;
		if (metric === "trades" || metric === "steps") return value.toLocaleString();
		return value.toFixed(2);
	};

	let analyzerAxisA = $derived.by(() => analyzerResult?.axes.indicatorA.periods ?? []);
	let analyzerAxisB = $derived.by(() => analyzerResult?.axes.indicatorB.periods ?? []);

	let heatmapRange = $derived.by(() => ({
		min: toHeatmapRangeValue(heatmapRangeMin, heatmapMetric),
		max: toHeatmapRangeValue(heatmapRangeMax, heatmapMetric)
	}));

	let filteredAnalyzerCells = $derived.by(() => {
		if (!analyzerResult) return [];
		const tpValue = selectedTakeProfit === null ? null : Number(selectedTakeProfit);
		const slValue = selectedStopLoss === null ? null : Number(selectedStopLoss);
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
			map.set(`${cell.aPeriod}-${cell.bPeriod}`, cell);
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

	const formatRangePlaceholder = (value: number, metric: keyof AnalyzerMetrics) => {
		if (!Number.isFinite(value)) return "";
		if (metric === "winRate") return (value * 100).toFixed(1);
		if (metric === "trades" || metric === "steps") return Math.round(value).toString();
		return value.toFixed(2);
	};

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
		if (!Number.isFinite(value)) return "hsl(0, 0%, 70%)";
		const { min, max } = heatmapDomain;
		const t = Math.min(1, Math.max(0, (value - min) / (max - min)));
		const hue = 20 + 120 * t;
		const lightness = 38 + 22 * t;
		return `hsl(${hue}, 70%, ${lightness}%)`;
	};

	const findCell = (aPeriod: number, bPeriod: number) => {
		return analyzerCellMap.get(`${aPeriod}-${bPeriod}`);
	};

	$effect(() => {
		if (!selectedCell) return;
		const key = `${selectedCell.aPeriod}-${selectedCell.bPeriod}`;
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
						range: {
							start: analyzerIndicatorA.start,
							end: analyzerIndicatorA.end,
							step: analyzerIndicatorA.step
						}
					},
					indicatorB: {
						kind: analyzer.indicatorB.kind,
						range: {
							start: analyzerIndicatorB.start,
							end: analyzerIndicatorB.end,
							step: analyzerIndicatorB.step
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
			selectedTakeProfit = null;
			selectedStopLoss = null;
			selectedCell = null;
			return;
		}
		selectedTakeProfit =
			analyzerResult.axes.takeProfitValues?.[0] !== undefined
				? String(analyzerResult.axes.takeProfitValues[0])
				: null;
		selectedStopLoss =
			analyzerResult.axes.stopLossValues?.[0] !== undefined
				? String(analyzerResult.axes.stopLossValues[0])
				: null;
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
			<section class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
				<div class="lg:col-start-2">
					<RunBacktestCard
						{running}
						{canRun}
						{validationErrors}
						{runError}
						{runResult}
						onRun={runBacktest}
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
						onBrowse={() => void openDatasetPicker("script")}
					/>
					<BacktestEnvironmentCard
						{env}
						{limits}
						{invalidInitialBalance}
						{invalidMaxPosition}
						{invalidCommission}
						{invalidSlippage}
						{invalidMargin}
						{invalidContractMultiplier}
						{invalidMemory}
						{invalidInstructionLimit}
						{invalidInstructionInterval}
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
		{:else}
			<section class="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
				<div class="lg:col-start-2">
					<RunAnalyzerCard
						{analyzerRunning}
						{analyzerCanRun}
						{analyzerCombos}
						{analyzerResult}
						{analyzerValidationErrors}
						{analyzerError}
						onRun={runAnalyzer}
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
						bind:datasetMode={analyzerDatasetMode}
						bind:datasetPath={analyzerDatasetPath}
						datasetPathInvalid={analyzerDatasetPathInvalid}
						onBrowse={() => void openDatasetPicker("analyzer")}
					/>
					<AnalyzerEnvironmentCard
						{analyzerEnv}
						{invalidAnalyzerInitialBalance}
						{invalidAnalyzerMaxPosition}
						{invalidAnalyzerCommission}
						{invalidAnalyzerSlippage}
						{invalidAnalyzerMargin}
						{invalidAnalyzerContractMultiplier}
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
				<SelectionDetailsCard {selectedCell} {analyzerResult} {formatMetricValue} />
			</section>
		{/if}
	</main>
</div>

{#if fileBrowserOpen}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center p-4"
		role="dialog"
		aria-modal="true"
		aria-label="Select parquet file"
	>
		<button
			type="button"
			class="absolute inset-0 bg-black/40"
			onclick={closeFileBrowser}
			aria-label="Close file picker"
		></button>
		<div class="relative z-10 w-full max-w-2xl rounded-xl border bg-background p-4 shadow-lg">
			<div class="flex items-start justify-between gap-4">
				<div>
					<div class="text-xs uppercase tracking-wide text-muted-foreground">{fileBrowserTitle}</div>
					<div class="text-sm font-semibold">{fileBrowserDir || "Project root"}</div>
					<div class="text-xs text-muted-foreground">
						Allowed: {fileBrowserExtensions.map((ext) => `.${ext}`).join(", ")}
					</div>
				</div>
				<Button type="button" variant="ghost" size="sm" onclick={closeFileBrowser}>
					Close
				</Button>
			</div>
			<div class="mt-3 flex items-center gap-2">
				<Button type="button" variant="outline" size="sm" onclick={handleFileBrowserUp} disabled={fileBrowserParent === null}>
					Up
				</Button>
				<div class="text-xs text-muted-foreground truncate">
					{fileBrowserDir ? `/${fileBrowserDir}` : "/"}
				</div>
			</div>

			{#if fileBrowserError}
				<div class="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
					{fileBrowserError}
				</div>
			{/if}

			<div class="mt-3 max-h-[360px] overflow-auto rounded-lg border">
				{#if fileBrowserLoading}
					<div class="px-4 py-6 text-sm text-muted-foreground">Loading files...</div>
				{:else if fileBrowserEntries.length === 0}
					<div class="px-4 py-6 text-sm text-muted-foreground">No parquet files found here.</div>
				{:else}
					<div class="divide-y">
						{#each fileBrowserEntries as entry}
							<button
								type="button"
								class="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-muted/50"
								onclick={() => handleFileEntry(entry)}
							>
								<span class="text-[11px] font-semibold uppercase text-muted-foreground">
									{entry.kind === "dir" ? "Dir" : "File"}
								</span>
								<span class="text-sm">{entry.name}</span>
							</button>
						{/each}
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}
