import type { ChartConfiguration, ChartType } from "chart.js";

export type LogRow = Record<string, any>;
export type BehaviorRow = Record<string, any>;
export type ParquetRow = Record<string, any>;

export type BehaviorFile = {
	name: string;
	split: "train" | "val" | "unknown";
	gen: number | null;
	idx: number | null;
};

export type GenMember = {
	idx: number | null;
	fitness: number | null;
	evalFitness: number | null;
	selectionFitness: number | null;
	pnl: number | null;
	realized: number | null;
	total: number | null;
	metric: number | null;
	drawdown: number | null;
	trainRealized: number | null;
	evalRealized: number | null;
};

export type IssueState = {
	highFitnessCount: number;
	highFitnessItems: string[];
	cappedCount: number;
	cappedItems: string[];
	zeroDdCount: number;
	zeroDdItems: string[];
	retCount: number;
	retNegativeCount: number;
};

export type KeySet = {
	pnlKey: string;
	pnlRealizedKey: string;
	pnlTotalKey: string;
	metricKey: string;
	drawdownKey: string;
	retKey: string;
};

export type ChartTab = "fitness" | "performance" | "realized" | "drawdown" | "frontier";
export type PopulationFilter = "all" | "top5" | "top10" | "top20p";
export type MainTab = "overview" | "evolution" | "issues" | "data" | "behavior";
export type BehaviorFilter = "all" | "trades" | "buy" | "sell" | "hold" | "revert";

export type BehaviorDisplayEntry = {
	row: BehaviorRow;
	data: ParquetRow | null;
};

export type BehaviorDisplay = {
	total: number;
	rows: BehaviorDisplayEntry[];
};

export type GenSummary = {
	gen: number;
	population: number;
	count: number;
	bestFitness: number;
	avgFitness: number;
	p50Fitness: number | null;
	p90Fitness: number | null;
	bestEvalFitness: number | null;
	avgEvalFitness: number | null;
	bestSelectionFitness: number | null;
	avgSelectionFitness: number | null;
	bestPnl: number;
	avgPnl: number;
	bestRealizedPnl: number;
	avgRealizedPnl: number;
	bestTrainRealized: number;
	avgTrainRealized: number;
	bestEvalRealized: number;
	avgEvalRealized: number;
	bestTotalPnl: number;
	avgTotalPnl: number;
	bestMetric: number;
	avgMetric: number;
	metricSum: number;
	metricCount: number;
	trainRealizedSum: number;
	trainRealizedCount: number;
	evalRealizedSum: number;
	evalRealizedCount: number;
	maxDrawdown: number;
	minDrawdown: number;
	avgDrawdown: number;
};

export type FilteredSummary = {
	metricSum: number;
	metricCount: number;
	trainSum: number;
	trainCount: number;
	trainBest: number;
	evalSum: number;
	evalCount: number;
	evalBest: number;
};

export type ActiveChartMeta = {
	title: string;
	description: string;
};

export type PlateauResult = {
	plateauGen: number | null;
	lastImprovementGen: number | null;
};

export type IssueItem = {
	type: "info" | "warning" | "destructive";
	title: string;
	message: string;
	items?: string[];
};

export type GaChartConfig = {
	data: ChartConfiguration["data"];
	options?: ChartConfiguration["options"];
	type?: ChartType;
};
