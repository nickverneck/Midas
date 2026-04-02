import type { ChartConfiguration, ChartType } from "chart.js";

export type NumericInput = number | string;

export type ChartTab =
	| "fitness"
	| "performance"
	| "drawdown"
	| "frontier"
	| "loss"
	| "gradients"
	| "policy"
	| "probe";

export type FolderEntry = {
	name: string;
	path: string;
	kind: "dir" | "file";
	mtime?: number;
};

export type FitnessWeights = {
	pnl: NumericInput;
	sortino: NumericInput;
	mdd: NumericInput;
};

export type ResolvedFitnessWeights = {
	pnl: number;
	sortino: number;
	mdd: number;
};

export type RlPoint = {
	epoch: number;
	fitness: number | null;
	trainRet: number | null;
	trainPnl: number | null;
	trainRealizedPnl: number | null;
	trainSortino: number | null;
	trainDrawdown: number | null;
	trainCommission: number | null;
	trainSlippage: number | null;
	trainBuyFrac: number | null;
	trainSellFrac: number | null;
	trainHoldFrac: number | null;
	trainRevertFrac: number | null;
	trainMeanMaxProb: number | null;
	trainEntries: number | null;
	trainExits: number | null;
	trainFlips: number | null;
	trainAvgHold: number | null;
	evalRet: number | null;
	evalPnl: number | null;
	evalRealizedPnl: number | null;
	evalSortino: number | null;
	evalDrawdown: number | null;
	evalCommission: number | null;
	evalSlippage: number | null;
	evalBuyFrac: number | null;
	evalSellFrac: number | null;
	evalHoldFrac: number | null;
	evalRevertFrac: number | null;
	evalMeanMaxProb: number | null;
	evalEntries: number | null;
	evalExits: number | null;
	evalFlips: number | null;
	evalAvgHold: number | null;
	probeRet: number | null;
	probePnl: number | null;
	probeRealizedPnl: number | null;
	probeSortino: number | null;
	probeDrawdown: number | null;
	probeCommission: number | null;
	probeSlippage: number | null;
	probeBuyFrac: number | null;
	probeSellFrac: number | null;
	probeHoldFrac: number | null;
	probeRevertFrac: number | null;
	probeMeanMaxProb: number | null;
	probeEntries: number | null;
	probeExits: number | null;
	probeFlips: number | null;
	probeAvgHold: number | null;
	policyLoss: number | null;
	valueLoss: number | null;
	policyGradNorm: number | null;
	valueGradNorm: number | null;
	approxKl: number | null;
	klDiv: number | null;
	clipFrac: number | null;
	entropy: number | null;
	perplexity: number | null;
	totalLoss: number | null;
	algorithm: "ppo" | "grpo" | null;
};

export type RlChartConfig = {
	data: ChartConfiguration["data"];
	options: ChartConfiguration["options"];
	type?: ChartType;
	note?: string;
	emptyMessage?: string;
};

export type RlChartsViewModel = {
	fitness: RlChartConfig;
	performance: RlChartConfig;
	drawdown: RlChartConfig;
	frontier: RlChartConfig;
	loss: RlChartConfig;
	gradients: RlChartConfig;
	policy: RlChartConfig;
	probe: RlChartConfig;
};

export type RlSnapshotRow = {
	label: string;
	value: number | string | null;
	emphasis?: boolean;
};
