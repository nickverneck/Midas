import type { ChartConfiguration, ChartDataset } from "chart.js";

import type {
	AnalyzerConfig,
	AnalyzerEnv,
	AnalyzerMetrics,
	BacktestEnv,
	BacktestLimits,
	BacktestMetrics,
	DatasetMode,
	IndicatorKind,
	IndicatorSweepParam
} from "./types";

export const sampleScript = `-- Example: EMA cross strategy
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

export const createDefaultBacktestEnv = (): BacktestEnv => ({
	initialBalance: 10_000,
	maxPosition: 1,
	commission: 1.6,
	slippage: 0.25,
	marginPerContract: 50,
	marginMode: "per-contract",
	contractMultiplier: 1.0,
	enforceMargin: true
});

export const createDefaultBacktestLimits = (): BacktestLimits => ({
	memoryMb: 64,
	instructionLimit: 5_000_000,
	instructionInterval: 10_000
});

export const createDefaultAnalyzerConfig = (): AnalyzerConfig => ({
	indicatorA: {
		kind: "ema",
		sweepParam: "period",
		period: 10,
		start: 5,
		end: 20,
		step: 1,
		kamaFast: 2,
		kamaSlow: 30,
		almaOffset: 0.85,
		almaSigma: 6
	},
	indicatorB: {
		kind: "ema",
		sweepParam: "period",
		period: 10,
		start: 5,
		end: 20,
		step: 1,
		kamaFast: 2,
		kamaSlow: 30,
		almaOffset: 0.85,
		almaSigma: 6
	},
	buyAction: "crossover",
	sellAction: "crossunder",
	takeProfit: { enabled: false, start: 0.5, end: 2.0, step: 0.5 },
	stopLoss: { enabled: false, start: 0.5, end: 2.0, step: 0.5 }
});

export const createDefaultAnalyzerEnv = (): AnalyzerEnv => ({
	initialBalance: 10_000,
	maxPosition: 1,
	commission: 1.6,
	slippage: 0.25,
	marginPerContract: 50,
	marginMode: "per-contract",
	contractMultiplier: 1.0,
	enforceMargin: true
});

const presetDatasetPaths = {
	train: "data/train/SPY0.parquet",
	val: "data/val/SPY.parquet"
} satisfies Record<Exclude<DatasetMode, "custom">, string>;

export const presetDatasetPath = (mode: DatasetMode) => {
	if (mode === "custom") return null;
	return presetDatasetPaths[mode];
};

export const ANALYZER_MAX_COMBOS = 20_000;

export const demoEquitySeries = Array.from({ length: 140 }, (_, i) => {
	const drift = i * 3.2;
	const wave = Math.sin(i / 9) * 40;
	return 10_000 + drift + wave;
});

export const demoBaselineSeries = demoEquitySeries.map(
	(value, index) => value - 120 + Math.cos(index / 11) * 25
);

export const demoMetrics: BacktestMetrics = {
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

export const buildEquityChartData = (equitySeries: number[], includeModelSeries: boolean) => {
	const datasets: ChartDataset<"line", number[]>[] = [
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

	if (includeModelSeries) {
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
		labels: equitySeries.map((_, index) => index + 1),
		datasets
	};
};

export const equityChartOptions = {
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
				callback: (value: string | number) => {
					const numericValue = Number(value);
					return Number.isFinite(numericValue)
						? `$${numericValue.toLocaleString()}`
						: String(value);
				}
			}
		}
	}
} satisfies ChartConfiguration<"line", number[], number>["options"];

export const heatmapMetricOptions: { value: keyof AnalyzerMetrics; label: string }[] = [
	{ value: "fitness", label: "Fitness" },
	{ value: "netPnl", label: "Net PnL" },
	{ value: "maxDrawdown", label: "Max Drawdown" },
	{ value: "sortino", label: "Sortino" },
	{ value: "sharpe", label: "Sharpe" },
	{ value: "profitFactor", label: "Profit Factor" },
	{ value: "winRate", label: "Win Rate" },
	{ value: "trades", label: "Trades" }
];

export const toNumber = (value: unknown) => {
	if (value === null || value === undefined || value === "") return null;
	const num = Number(value);
	return Number.isFinite(num) ? num : null;
};

export const toInteger = (value: unknown) => {
	const num = toNumber(value);
	if (num === null) return null;
	return Number.isInteger(num) ? num : null;
};

export const toHeatmapRangeValue = (value: unknown, metric: keyof AnalyzerMetrics) => {
	const num = toNumber(value);
	if (num === null) return null;
	if (metric === "winRate") return num / 100;
	return num;
};

export const indicatorSweepParams = (kind: IndicatorKind): IndicatorSweepParam[] => {
	switch (kind) {
		case "kama":
			return ["period", "fast", "slow"];
		case "alma":
			return ["period", "offset", "sigma"];
		default:
			return ["period"];
	}
};

export const normalizeSweepParam = (
	kind: IndicatorKind,
	param: IndicatorSweepParam | undefined
): IndicatorSweepParam => {
	const allowed = indicatorSweepParams(kind);
	if (param && allowed.includes(param)) return param;
	return allowed[0];
};

export const isIntegerSweepParam = (param: IndicatorSweepParam) =>
	param === "period" || param === "fast" || param === "slow";

export const countRange = (start: number | null, end: number | null, step: number | null) => {
	if (start === null || end === null || step === null) return 0;
	if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(step)) return 0;
	if (step <= 0 || end < start) return 0;
	return Math.floor((end - start) / step) + 1;
};

export const normalizeExtensions = (extensions: string[]) =>
	extensions.map((ext) => ext.replace(/^\./, "").toLowerCase());

export const buildFileBrowserUrl = (dir: string, extensions: string[]) => {
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

export type ParsedIndicator = {
	start: number | null;
	end: number | null;
	step: number | null;
	period: number | null;
	kamaFast: number | null;
	kamaSlow: number | null;
	almaOffset: number | null;
	almaSigma: number | null;
};

export const validateAnalyzerIndicator = (
	label: "A" | "B",
	kind: IndicatorKind,
	sweepParam: IndicatorSweepParam,
	parsed: ParsedIndicator
) => {
	const errors: string[] = [];
	const allowedSweepParams = indicatorSweepParams(kind);
	if (!allowedSweepParams.includes(sweepParam)) {
		errors.push(`Indicator ${label} sweep parameter is not valid for ${kind.toUpperCase()}.`);
		return errors;
	}

	const { start, end, step } = parsed;
	if (start === null || end === null || step === null) {
		errors.push(`Indicator ${label} sweep range is incomplete.`);
		return errors;
	}
	if (step <= 0 || end < start) {
		errors.push(`Indicator ${label} sweep range must satisfy end >= start and step > 0.`);
		return errors;
	}
	if (isIntegerSweepParam(sweepParam)) {
		if (!Number.isInteger(start) || !Number.isInteger(end) || !Number.isInteger(step)) {
			errors.push(`Indicator ${label} ${sweepParam} sweep values must be integers.`);
		}
		if (start <= 0) {
			errors.push(`Indicator ${label} ${sweepParam} sweep values must be greater than 0.`);
		}
	} else if (sweepParam === "offset") {
		if (start < 0 || end > 1) {
			errors.push(`Indicator ${label} offset sweep must stay within [0, 1].`);
		}
	} else if (sweepParam === "sigma") {
		if (start <= 0) {
			errors.push(`Indicator ${label} sigma sweep values must be greater than 0.`);
		}
	}

	if (kind !== "price" && sweepParam !== "period") {
		if (parsed.period === null || !Number.isInteger(parsed.period) || parsed.period <= 0) {
			errors.push(`Indicator ${label} fixed period must be an integer greater than 0.`);
		}
	}

	if (kind === "kama") {
		if (sweepParam !== "fast") {
			if (parsed.kamaFast === null || !Number.isInteger(parsed.kamaFast) || parsed.kamaFast <= 0) {
				errors.push(`Indicator ${label} fixed KAMA fast must be an integer greater than 0.`);
			}
		}
		if (sweepParam !== "slow") {
			if (parsed.kamaSlow === null || !Number.isInteger(parsed.kamaSlow) || parsed.kamaSlow <= 0) {
				errors.push(`Indicator ${label} fixed KAMA slow must be an integer greater than 0.`);
			}
		}

		const fastMax = sweepParam === "fast" ? end : parsed.kamaFast;
		const slowMin = sweepParam === "slow" ? start : parsed.kamaSlow;
		if (fastMax !== null && slowMin !== null && slowMin <= fastMax) {
			errors.push(`Indicator ${label} KAMA requires slow > fast across the sweep.`);
		}
	}

	if (kind === "alma") {
		if (sweepParam !== "offset") {
			if (parsed.almaOffset === null || parsed.almaOffset < 0 || parsed.almaOffset > 1) {
				errors.push(`Indicator ${label} fixed ALMA offset must be within [0, 1].`);
			}
		}
		if (sweepParam !== "sigma") {
			if (parsed.almaSigma === null || parsed.almaSigma <= 0) {
				errors.push(`Indicator ${label} fixed ALMA sigma must be greater than 0.`);
			}
		}
	}

	return errors;
};

export const metricValue = (metrics: AnalyzerMetrics, metric: keyof AnalyzerMetrics) => {
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

export const metricColorValue = (metrics: AnalyzerMetrics, metric: keyof AnalyzerMetrics) => {
	const value = metricValue(metrics, metric);
	if (metric === "maxDrawdown") return -value;
	return value;
};

export const formatMetricValue = (value: number, metric: keyof AnalyzerMetrics) => {
	if (!Number.isFinite(value)) return "n/a";
	if (metric === "winRate") return `${(value * 100).toFixed(1)}%`;
	if (metric === "trades" || metric === "steps") return value.toLocaleString();
	return value.toFixed(2);
};

export const axisValueKey = (value: number) => (Number.isFinite(value) ? value.toFixed(6) : "nan");

export const formatRangePlaceholder = (value: number, metric: keyof AnalyzerMetrics) => {
	if (!Number.isFinite(value)) return "";
	if (metric === "winRate") return (value * 100).toFixed(1);
	if (metric === "trades" || metric === "steps") return Math.round(value).toString();
	return value.toFixed(2);
};

export const heatmapColorForValue = (value: number, domain: { min: number; max: number }) => {
	if (!Number.isFinite(value)) return "hsl(0, 0%, 70%)";
	const t = Math.min(1, Math.max(0, (value - domain.min) / (domain.max - domain.min)));
	const hue = 20 + 120 * t;
	const lightness = 38 + 22 * t;
	return `hsl(${hue}, 70%, ${lightness}%)`;
};
