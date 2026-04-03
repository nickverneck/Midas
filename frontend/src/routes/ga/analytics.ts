import type { ChartConfiguration } from "chart.js";
import Papa from "papaparse";
import type {
	ActiveChartMeta,
	BehaviorDisplay,
	BehaviorFile,
	BehaviorFilter,
	BehaviorRow,
	ChartTab,
	FilteredSummary,
	GenMember,
	GenSummary,
	IssueItem,
	IssueState,
	KeySet,
	LogRow,
	ParquetRow,
	PlateauResult,
	PopulationFilter
} from "./types";

export const DEFAULT_LOG_DIR = "runs_ga";
export const LOG_CHUNK = 1000;
export const PAGE_SIZE = 200;
export const MIN_ZOOM_WINDOW = 50;
export const ZOOM_STEP = 1.5;

export const TRAIN_KEYS: KeySet = {
	pnlKey: "train_fitness_pnl",
	pnlRealizedKey: "train_pnl_realized",
	pnlTotalKey: "train_pnl_total",
	metricKey: "train_sortino",
	drawdownKey: "train_drawdown",
	retKey: "train_ret_mean"
};

export const EVAL_KEYS: KeySet = {
	pnlKey: "eval_fitness_pnl",
	pnlRealizedKey: "eval_pnl_realized",
	pnlTotalKey: "eval_pnl_total",
	metricKey: "eval_sortino",
	drawdownKey: "eval_drawdown",
	retKey: "eval_ret_mean"
};

export const EVAL_PROBE_KEYS = [
	"eval_fitness_pnl",
	"eval_pnl_realized",
	"eval_pnl_total",
	"eval_sortino",
	"eval_drawdown",
	"eval_ret_mean"
] as const;

export const createEmptyIssueState = (): IssueState => ({
	highFitnessCount: 0,
	highFitnessItems: [],
	cappedCount: 0,
	cappedItems: [],
	zeroDdCount: 0,
	zeroDdItems: [],
	retCount: 0,
	retNegativeCount: 0
});

export const toNumber = (value: unknown): number | null => {
	if (typeof value === "number") return Number.isFinite(value) ? value : null;
	if (typeof value === "string" && value.trim() !== "") {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : null;
	}
	return null;
};

export const formatNum = (value: unknown, digits = 4) => {
	const num = toNumber(value);
	return num === null ? "—" : num.toFixed(digits);
};

export const avgOrNull = (sum: number, count: number) => (count > 0 ? sum / count : null);

export const finiteValues = (values: Array<number | null>) =>
	values.filter((value): value is number => value !== null && Number.isFinite(value));

export const meanValue = (values: number[]) =>
	values.length ? values.reduce((acc, value) => acc + value, 0) / values.length : 0;

export const maxValue = (values: number[]) => (values.length ? Math.max(...values) : 0);
export const minValue = (values: number[]) => (values.length ? Math.min(...values) : 0);

export const normalizeLogDir = (value: string) => value.trim() || DEFAULT_LOG_DIR;

export const buildLogsUrl = (offset: number, dir: string) => {
	const params = new URLSearchParams({
		limit: String(LOG_CHUNK),
		offset: String(offset),
		dir,
		log: "ga"
	});
	return `/api/logs?${params.toString()}`;
};

export const percentileValue = (sortedValues: number[], percentile: number) => {
	if (sortedValues.length === 0) return null;

	const index = (sortedValues.length - 1) * percentile;
	const lower = Math.floor(index);
	const upper = Math.ceil(index);
	if (lower === upper) return sortedValues[lower];

	const weight = index - lower;
	return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
};

export const selectMembers = (members: GenMember[], populationFilter: PopulationFilter) => {
	if (populationFilter === "all") return members;

	const ranked = members
		.filter((member) => member.fitness !== null)
		.sort(
			(a, b) =>
				(b.fitness ?? Number.NEGATIVE_INFINITY) - (a.fitness ?? Number.NEGATIVE_INFINITY)
		);

	if (ranked.length === 0) return members;

	let limit = ranked.length;
	if (populationFilter === "top5") limit = Math.min(5, ranked.length);
	if (populationFilter === "top10") limit = Math.min(10, ranked.length);
	if (populationFilter === "top20p") limit = Math.max(1, Math.round(ranked.length * 0.2));
	return ranked.slice(0, limit);
};

export const parseBehaviorFile = (name: string): BehaviorFile => {
	const lower = name.toLowerCase();
	let split: BehaviorFile["split"] = "unknown";

	if (lower.startsWith("train_") || lower.includes("train")) {
		split = "train";
	} else if (lower.startsWith("val_") || lower.includes("eval") || lower.includes("val")) {
		split = "val";
	}

	const genMatch = lower.match(/gen(\d+)/);
	const idxMatch = lower.match(/idx(\d+)/);

	return {
		name,
		split,
		gen: genMatch ? Number(genMatch[1]) : null,
		idx: idxMatch ? Number(idxMatch[1]) : null
	};
};

export const formatBehaviorLabel = (file: BehaviorFile) => {
	if (file.gen !== null && file.idx !== null) {
		const splitLabel =
			file.split === "train" ? "Train" : file.split === "val" ? "Eval" : "Run";
		return `${splitLabel} Gen ${file.gen} (Idx ${file.idx})`;
	}
	return file.name;
};

export const normalizeAction = (value: unknown) =>
	typeof value === "string" ? value.trim().toLowerCase() : "";

export const toIndex = (value: unknown): number | null => {
	if (typeof value === "number" && Number.isFinite(value)) return value;
	if (typeof value === "string" && value.trim() !== "") {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : null;
	}
	return null;
};

export const matchesBehaviorFilter = (action: string, behaviorFilter: BehaviorFilter) => {
	if (behaviorFilter === "all") return true;
	if (behaviorFilter === "trades") {
		return action === "buy" || action === "sell" || action === "revert";
	}
	return action === behaviorFilter;
};

export const formatTimestamp = (value: unknown) => {
	if (value === null || value === undefined) return "—";
	if (typeof value === "string" && value.trim() !== "") return value;

	const num = toNumber(value);
	if (num === null) return "—";

	let ms = num;
	if (num > 1e14) {
		ms = Math.round(num / 1e6);
	} else if (num > 1e12) {
		ms = num;
	} else if (num > 1e9) {
		ms = num * 1000;
	}

	const date = new Date(ms);
	if (Number.isNaN(date.getTime())) return String(value);
	return date.toISOString().replace("T", " ").replace("Z", "");
};

export const parseCsvRows = <T,>(text: string): T[] => {
	const parsed = Papa.parse(text, {
		header: true,
		dynamicTyping: true,
		skipEmptyLines: true
	});

	return (parsed.data as T[]).filter((row) => row && typeof row === "object");
};

export const buildBehaviorDisplay = (
	rows: BehaviorRow[],
	dataMap: Map<number, ParquetRow>,
	behaviorFilter: BehaviorFilter,
	behaviorRowLimit: number
): BehaviorDisplay => {
	const filtered = rows.filter((row) =>
		matchesBehaviorFilter(normalizeAction(row.action), behaviorFilter)
	);
	const limited = behaviorRowLimit > 0 ? filtered.slice(0, behaviorRowLimit) : filtered;
	const display = limited.map((row) => {
		const idx = toIndex(row.data_idx ?? row.row_idx ?? row.idx);
		const data = idx === null ? null : dataMap.get(idx) ?? null;
		return { row, data };
	});

	return { total: filtered.length, rows: display };
};

export const resolveBehaviorTimestamp = (row: BehaviorRow, data: ParquetRow | null) =>
	formatTimestamp(
		data?.date ??
			data?.datetime ??
			row.datetime_ns ??
			row.date ??
			data?.datetime_ns ??
			data?.timestamp
	);

export const resolveBehaviorClose = (row: BehaviorRow, data: ParquetRow | null) =>
	data?.close ?? row.close ?? row.price ?? null;

export const actionClass = (action: string) => {
	switch (action) {
		case "buy":
			return "text-emerald-500 font-semibold";
		case "sell":
			return "text-rose-500 font-semibold";
		case "revert":
			return "text-orange-500 font-semibold";
		case "hold":
		default:
			return "text-muted-foreground";
	}
};

export const detectEval = (rows: LogRow[]) => {
	if (rows.length === 0) return false;
	const first = rows[0];
	if (!first || typeof first !== "object") return false;
	return EVAL_PROBE_KEYS.some((key) => key in first);
};

const formatIssueId = (row: LogRow) => {
	const genLabel = row.gen ?? "—";
	const idxLabel = row.idx ?? "—";
	return `Gen ${genLabel}, Idx ${idxLabel}`;
};

export const applyLogChunk = (
	genMembers: Map<number, GenMember[]>,
	bestFitness: number,
	issueState: IssueState,
	rows: LogRow[],
	hasEval: boolean
) => {
	if (rows.length === 0) {
		return { genMembers, bestFitness, issueState };
	}

	const keys = hasEval ? EVAL_KEYS : TRAIN_KEYS;
	const membersMap = new Map(genMembers);
	let nextBestFitness = bestFitness;
	const nextIssue: IssueState = {
		highFitnessCount: issueState.highFitnessCount,
		highFitnessItems: [...issueState.highFitnessItems],
		cappedCount: issueState.cappedCount,
		cappedItems: [...issueState.cappedItems],
		zeroDdCount: issueState.zeroDdCount,
		zeroDdItems: [...issueState.zeroDdItems],
		retCount: issueState.retCount,
		retNegativeCount: issueState.retNegativeCount
	};

	for (const row of rows) {
		if (!row || typeof row !== "object") continue;

		const fitness = toNumber(row.fitness);
		const pnl = toNumber(row[keys.pnlKey]);
		const realized = toNumber(row[keys.pnlRealizedKey]);
		const trainRealized = toNumber(row.train_pnl_realized);
		const evalRealized = toNumber(row.eval_pnl_realized);
		const total = toNumber(row[keys.pnlTotalKey]);
		const metric = toNumber(row[keys.metricKey]);
		const drawdown = toNumber(row[keys.drawdownKey]);
		const ret = toNumber(row[keys.retKey]);
		const genValue = toNumber(row.gen);
		const idxValue = toNumber(row.idx);
		const evalFitness = toNumber(row.eval_fitness);
		const selectionFitness = toNumber(row.selection_fitness);

		if (genValue !== null) {
			const member: GenMember = {
				idx: idxValue,
				fitness,
				evalFitness,
				selectionFitness,
				pnl,
				realized,
				total,
				metric,
				drawdown,
				trainRealized,
				evalRealized
			};

			const list = membersMap.get(genValue);
			if (list) {
				list.push(member);
			} else {
				membersMap.set(genValue, [member]);
			}
		}

		if (fitness !== null) {
			nextBestFitness = Math.max(nextBestFitness, fitness);
		}

		const issueId = formatIssueId(row);
		if (fitness !== null && fitness > 1000) {
			nextIssue.highFitnessCount += 1;
			if (nextIssue.highFitnessItems.length < 5) {
				nextIssue.highFitnessItems.push(`${issueId}: ${fitness.toFixed(2)}`);
			}
		}

		if (metric !== null && metric >= 50.0) {
			nextIssue.cappedCount += 1;
			if (nextIssue.cappedItems.length < 5) {
				nextIssue.cappedItems.push(issueId);
			}
		}

		if (drawdown !== null && pnl !== null && drawdown === 0 && pnl !== 0) {
			nextIssue.zeroDdCount += 1;
			if (nextIssue.zeroDdItems.length < 5) {
				nextIssue.zeroDdItems.push(issueId);
			}
		}

		if (ret !== null) {
			nextIssue.retCount += 1;
			if (ret < 0) nextIssue.retNegativeCount += 1;
		}
	}

	return {
		genMembers: membersMap,
		bestFitness: nextBestFitness,
		issueState: nextIssue
	};
};

export const buildGenData = (
	genMembers: Map<number, GenMember[]>,
	populationFilter: PopulationFilter
): GenSummary[] => {
	if (genMembers.size === 0) return [];

	return Array.from(genMembers.entries())
		.sort(([genA], [genB]) => genA - genB)
		.map(([gen, members]) => {
			const selected = selectMembers(members, populationFilter);
			const fitnessValues = finiteValues(selected.map((member) => member.fitness));
			const evalFitnessValues = finiteValues(selected.map((member) => member.evalFitness));
			const selectionFitnessValues = finiteValues(
				selected.map((member) => member.selectionFitness)
			);
			const pnlValues = finiteValues(selected.map((member) => member.pnl));
			const realizedValues = finiteValues(selected.map((member) => member.realized));
			const totalValues = finiteValues(selected.map((member) => member.total));
			const metricValues = finiteValues(selected.map((member) => member.metric));
			const drawdownValues = finiteValues(selected.map((member) => member.drawdown));
			const trainRealizedValues = finiteValues(selected.map((member) => member.trainRealized));
			const evalRealizedValues = finiteValues(selected.map((member) => member.evalRealized));
			const sortedFitness = [...fitnessValues].sort((a, b) => a - b);
			const metricSum = metricValues.reduce((acc, value) => acc + value, 0);
			const trainRealizedSum = trainRealizedValues.reduce((acc, value) => acc + value, 0);
			const evalRealizedSum = evalRealizedValues.reduce((acc, value) => acc + value, 0);

			return {
				gen,
				population: members.length,
				count: selected.length,
				bestFitness: maxValue(fitnessValues),
				avgFitness: meanValue(fitnessValues),
				p50Fitness: percentileValue(sortedFitness, 0.5),
				p90Fitness: percentileValue(sortedFitness, 0.9),
				bestEvalFitness: evalFitnessValues.length ? maxValue(evalFitnessValues) : null,
				avgEvalFitness: evalFitnessValues.length ? meanValue(evalFitnessValues) : null,
				bestSelectionFitness: selectionFitnessValues.length
					? maxValue(selectionFitnessValues)
					: null,
				avgSelectionFitness: selectionFitnessValues.length
					? meanValue(selectionFitnessValues)
					: null,
				bestPnl: maxValue(pnlValues),
				avgPnl: meanValue(pnlValues),
				bestRealizedPnl: maxValue(realizedValues),
				avgRealizedPnl: meanValue(realizedValues),
				bestTrainRealized: maxValue(trainRealizedValues),
				avgTrainRealized: meanValue(trainRealizedValues),
				bestEvalRealized: maxValue(evalRealizedValues),
				avgEvalRealized: meanValue(evalRealizedValues),
				bestTotalPnl: maxValue(totalValues),
				avgTotalPnl: meanValue(totalValues),
				bestMetric: maxValue(metricValues),
				avgMetric: meanValue(metricValues),
				metricSum,
				metricCount: metricValues.length,
				trainRealizedSum,
				trainRealizedCount: trainRealizedValues.length,
				evalRealizedSum,
				evalRealizedCount: evalRealizedValues.length,
				maxDrawdown: maxValue(drawdownValues),
				minDrawdown: minValue(drawdownValues),
				avgDrawdown: meanValue(drawdownValues)
			};
		});
};

export const buildFilteredSummary = (genData: GenSummary[]): FilteredSummary => {
	let metricSum = 0;
	let metricCount = 0;
	let trainSum = 0;
	let trainCount = 0;
	let trainBest = Number.NEGATIVE_INFINITY;
	let evalSum = 0;
	let evalCount = 0;
	let evalBest = Number.NEGATIVE_INFINITY;

	for (const generation of genData) {
		metricSum += generation.metricSum;
		metricCount += generation.metricCount;
		trainSum += generation.trainRealizedSum;
		trainCount += generation.trainRealizedCount;
		trainBest = Math.max(trainBest, generation.bestTrainRealized);
		evalSum += generation.evalRealizedSum;
		evalCount += generation.evalRealizedCount;
		evalBest = Math.max(evalBest, generation.bestEvalRealized);
	}

	return {
		metricSum,
		metricCount,
		trainSum,
		trainCount,
		trainBest,
		evalSum,
		evalCount,
		evalBest
	};
};

export const buildZoomRangeLabel = (
	genData: GenSummary[],
	zoomWindowSize: number,
	zoomStart: number
) => {
	if (genData.length === 0) return "No data";
	if (zoomWindowSize >= genData.length) return `All ${genData.length} generations`;

	const maxZoomStart = Math.max(0, genData.length - zoomWindowSize);
	const startIndex = Math.min(zoomStart, maxZoomStart);
	const endIndex = Math.min(startIndex + zoomWindowSize - 1, genData.length - 1);
	const startGen = genData[startIndex]?.gen ?? "—";
	const endGen = genData[endIndex]?.gen ?? "—";
	return `Gen ${startGen} to Gen ${endGen}`;
};

export const buildActiveChartMeta = (
	chartTab: ChartTab,
	sourceLabel: string,
	metricLabel: string
): ActiveChartMeta => {
	if (chartTab === "fitness") {
		return {
			title: "Fitness Evolution",
			description: "Maximum and average fitness progress per generation"
		};
	}

	if (chartTab === "performance") {
		return {
			title: "Performance Evolution",
			description: `Best ${sourceLabel} PNL, realized PNL, and ${metricLabel} across windows`
		};
	}

	if (chartTab === "drawdown") {
		return {
			title: "Drawdown Monitoring",
			description: "Worst and average drawdown per generation"
		};
	}

	if (chartTab === "frontier") {
		return {
			title: "Risk/Return Frontier",
			description: "Best realized PNL plotted against max drawdown"
		};
	}

	return {
		title: "Realized PNL Focus",
		description: "Best realized PNL for training and eval tracked independently"
	};
};

export const buildPlateauMetricLabel = (
	hasEvalRealized: boolean,
	hasTrainRealized: boolean
) => {
	if (hasEvalRealized) return "Best Eval Realized PNL";
	if (hasTrainRealized) return "Best Train Realized PNL";
	return "Best Fitness";
};

export const buildPlateauResult = (
	genData: GenSummary[],
	plateauWindow: number,
	plateauMinDelta: number,
	hasEvalRealized: boolean,
	hasTrainRealized: boolean
): PlateauResult => {
	if (genData.length === 0) {
		return { plateauGen: null, lastImprovementGen: null };
	}

	const windowSize = Math.max(1, Math.round(plateauWindow));
	const minDelta = Math.max(0, plateauMinDelta);
	let best = Number.NEGATIVE_INFINITY;
	let lastImprovementIndex: number | null = null;
	let plateauIndex: number | null = null;

	const pickValue = (generation: GenSummary) => {
		if (hasEvalRealized) return generation.bestEvalRealized;
		if (hasTrainRealized) return generation.bestTrainRealized;
		return generation.bestFitness;
	};

	for (let index = 0; index < genData.length; index += 1) {
		const value = pickValue(genData[index]);
		if (!Number.isFinite(value)) continue;

		if (best === Number.NEGATIVE_INFINITY) {
			best = value;
			lastImprovementIndex = index;
			continue;
		}

		if (value > best + minDelta) {
			best = value;
			lastImprovementIndex = index;
		}

		if (lastImprovementIndex !== null && index - lastImprovementIndex >= windowSize) {
			plateauIndex = index;
			break;
		}
	}

	return {
		plateauGen: plateauIndex !== null ? genData[plateauIndex]?.gen ?? null : null,
		lastImprovementGen:
			lastImprovementIndex !== null ? genData[lastImprovementIndex]?.gen ?? null : null
	};
};

export const buildFitnessChartData = (visibleGenData: GenSummary[]) => {
	const datasets = [
		{
			label: "Best Fitness (Train)",
			data: visibleGenData.map((generation) => generation.bestFitness),
			borderColor: "rgb(59, 130, 246)",
			backgroundColor: "rgba(59, 130, 246, 0.5)",
			tension: 0.1
		},
		{
			label: "Avg Fitness (Train)",
			data: visibleGenData.map((generation) => generation.avgFitness),
			borderColor: "rgb(147, 197, 253)",
			backgroundColor: "rgba(147, 197, 253, 0.2)",
			borderDash: [5, 5],
			tension: 0.1
		},
		{
			label: "P90 Fitness (Train)",
			data: visibleGenData.map((generation) => generation.p90Fitness),
			borderColor: "rgb(14, 116, 144)",
			backgroundColor: "rgba(14, 116, 144, 0.15)",
			borderDash: [4, 4],
			tension: 0.1
		},
		{
			label: "Median Fitness (Train)",
			data: visibleGenData.map((generation) => generation.p50Fitness),
			borderColor: "rgb(56, 189, 248)",
			backgroundColor: "rgba(56, 189, 248, 0.12)",
			borderDash: [2, 6],
			tension: 0.1
		}
	];

	const hasEvalFitness = visibleGenData.some((generation) => generation.bestEvalFitness !== null);
	if (hasEvalFitness) {
		datasets.push({
			label: "Best Fitness (Eval)",
			data: visibleGenData.map((generation) => generation.bestEvalFitness),
			borderColor: "rgb(244, 114, 182)",
			backgroundColor: "rgba(244, 114, 182, 0.25)",
			borderDash: [3, 5],
			tension: 0.1
		});
	}

	const hasSelectionFitness = visibleGenData.some(
		(generation) => generation.bestSelectionFitness !== null
	);
	if (hasSelectionFitness) {
		datasets.push({
			label: "Best Fitness (Selection)",
			data: visibleGenData.map((generation) => generation.bestSelectionFitness),
			borderColor: "rgb(251, 146, 60)",
			backgroundColor: "rgba(251, 146, 60, 0.2)",
			borderDash: [6, 3],
			tension: 0.1
		});
	}

	return {
		labels: visibleGenData.map((generation) => `Gen ${generation.gen}`),
		datasets
	};
};

export const buildPnlChartData = (
	visibleGenData: GenSummary[],
	sourceLabel: string,
	metricLabel: string
) => ({
	labels: visibleGenData.map((generation) => `Gen ${generation.gen}`),
	datasets: [
		{
			label: `Best ${sourceLabel} Fitness PNL`,
			data: visibleGenData.map((generation) => generation.bestPnl),
			borderColor: "rgb(34, 197, 94)",
			backgroundColor: "rgba(34, 197, 94, 0.5)",
			tension: 0.1
		},
		{
			label: `Best ${sourceLabel} Realized PNL`,
			data: visibleGenData.map((generation) => generation.bestRealizedPnl),
			borderColor: "rgb(16, 185, 129)",
			backgroundColor: "rgba(16, 185, 129, 0.35)",
			borderDash: [4, 4],
			tension: 0.1
		},
		{
			label: `Best ${metricLabel}`,
			data: visibleGenData.map((generation) => generation.bestMetric),
			borderColor: "rgb(168, 85, 247)",
			backgroundColor: "rgba(168, 85, 247, 0.5)",
			yAxisID: "y1",
			tension: 0.1
		}
	]
});

export const buildDrawdownChartData = (visibleGenData: GenSummary[], sourceLabel: string) => ({
	labels: visibleGenData.map((generation) => `Gen ${generation.gen}`),
	datasets: [
		{
			label: `Worst ${sourceLabel} Drawdown`,
			data: visibleGenData.map((generation) => generation.maxDrawdown),
			borderColor: "rgb(239, 68, 68)",
			backgroundColor: "rgba(239, 68, 68, 0.3)",
			tension: 0.1
		},
		{
			label: `Avg ${sourceLabel} Drawdown`,
			data: visibleGenData.map((generation) => generation.avgDrawdown),
			borderColor: "rgb(251, 146, 60)",
			backgroundColor: "rgba(251, 146, 60, 0.2)",
			borderDash: [4, 4],
			tension: 0.1
		}
	]
});

const preferredRealized = (member: GenMember) => {
	if (member.evalRealized !== null) return member.evalRealized;
	if (member.trainRealized !== null) return member.trainRealized;
	return member.realized;
};

export const buildFrontierChartData = (
	visibleGenData: GenSummary[],
	genMembers: Map<number, GenMember[]>,
	populationFilter: PopulationFilter
) => {
	if (visibleGenData.length === 0) {
		return { labels: [], datasets: [] };
	}

	const points = [];
	for (const generation of visibleGenData) {
		const members = genMembers.get(generation.gen);
		if (!members) continue;

		const selected = selectMembers(members, populationFilter);
		let bestPoint = null;
		for (const member of selected) {
			const realized = preferredRealized(member);
			const drawdown = member.drawdown;
			if (realized === null || drawdown === null) continue;

			if (!bestPoint || realized > bestPoint.y) {
				bestPoint = {
					x: drawdown,
					y: realized,
					gen: generation.gen,
					idx: member.idx
				};
			}
		}

		if (bestPoint) points.push(bestPoint);
	}

	return {
		labels: [],
		datasets: [
			{
				label: "Best Realized vs Drawdown",
				data: points,
				borderColor: "rgb(59, 130, 246)",
				backgroundColor: "rgba(59, 130, 246, 0.5)",
				showLine: false,
				pointRadius: 3
			}
		]
	};
};

export const buildFrontierOptions = (sourceLabel: string): ChartConfiguration["options"] => ({
	scales: {
		x: {
			type: "linear" as const,
			title: {
				display: true,
				text: `Max ${sourceLabel} Drawdown (%)`
			}
		},
		y: {
			title: {
				display: true,
				text: "Realized PNL"
			}
		}
	},
	plugins: {
		tooltip: {
			callbacks: {
				label: (ctx: any) => {
					const raw = ctx.raw as { x?: number; y?: number; gen?: number; idx?: number | null };
					const genLabel = raw?.gen ?? "—";
					const idxLabel = raw?.idx ?? "—";
					const drawdown = raw?.x ?? ctx.parsed?.x;
					const pnl = raw?.y ?? ctx.parsed?.y;

					return [
						`Gen ${genLabel} | Idx ${idxLabel}`,
						`Drawdown: ${drawdown ?? "—"}`,
						`Realized PNL: ${pnl ?? "—"}`
					];
				}
			}
		}
	}
});

export const buildRealizedChartData = (
	visibleGenData: GenSummary[],
	hasTrainRealized: boolean,
	hasEvalRealized: boolean
) => {
	const labels = visibleGenData.map((generation) => `Gen ${generation.gen}`);
	const datasets = [];

	if (hasTrainRealized) {
		datasets.push({
			label: "Best Train Realized PNL",
			data: visibleGenData.map((generation) => generation.bestTrainRealized),
			borderColor: "rgb(14, 116, 144)",
			backgroundColor: "rgba(14, 116, 144, 0.35)",
			tension: 0.1
		});
	}

	if (hasEvalRealized) {
		datasets.push({
			label: "Best Eval Realized PNL",
			data: visibleGenData.map((generation) => generation.bestEvalRealized),
			borderColor: "rgb(239, 68, 68)",
			backgroundColor: "rgba(239, 68, 68, 0.3)",
			tension: 0.1
		});
	}

	return { labels, datasets };
};

export const buildIssues = (
	logsLength: number,
	issueState: IssueState,
	metricLabel: string,
	sourceLabel: string,
	genData: GenSummary[]
): IssueItem[] => {
	const list: IssueItem[] = [];
	if (logsLength === 0) return list;

	if (issueState.highFitnessCount > 0) {
		list.push({
			type: "warning",
			title: "Fitness Outliers Detected",
			message: `${issueState.highFitnessCount} individuals have fitness > 1000. This may indicate logic errors or extreme lucky outliers corrupting selection.`,
			items: issueState.highFitnessItems
		});
	}

	if (issueState.cappedCount > 0) {
		list.push({
			type: "info",
			title: `${metricLabel} Cap Hit`,
			message: `${issueState.cappedCount} individuals hit the cap of 50.0 for ${metricLabel} (${sourceLabel}). Consider raising the cap if they are consistently flatlining at max.`,
			items: issueState.cappedItems
		});
	}

	if (issueState.zeroDdCount > 0) {
		list.push({
			type: "warning",
			title: "Zero Drawdown (Possible Fake Results)",
			message: `${issueState.zeroDdCount} individuals have 0% drawdown but non-zero PNL. This often indicates insufficient evaluation data or "one-hit wonder" trades.`,
			items: issueState.zeroDdItems
		});
	}

	if (issueState.retCount > 0 && issueState.retNegativeCount === issueState.retCount) {
		list.push({
			type: "destructive",
			title: "Universal Negative Returns",
			message: `Every single individual in the log has a negative return mean (${sourceLabel}). The strategy logic or features might be fundamentally flawed.`
		});
	}

	const counts = genData.map((generation) => generation.population);
	const uniqueCounts = new Set(counts);
	if (uniqueCounts.size > 1) {
		list.push({
			type: "info",
			title: "Inconsistent Population Sizes",
			message: `Population sizes vary across generations: ${Array.from(uniqueCounts).join(", ")}. Ensure this is intentional (e.g. elite pruning).`
		});
	}

	return list;
};

export const buildPagedLogs = (logs: LogRow[], logPage: number) => {
	if (logs.length === 0) return [];

	const total = logs.length;
	const end = total - (logPage - 1) * PAGE_SIZE;
	const start = Math.max(0, end - PAGE_SIZE);
	return logs.slice(start, end).reverse();
};
