export type DatasetMode = "train" | "val" | "custom";
export type BacktestView = "script" | "analyzer";

export type BacktestMetrics = {
	total_reward: number;
	net_pnl: number;
	ending_equity: number;
	sharpe: number;
	max_drawdown: number;
	profit_factor: number;
	win_rate: number;
	max_consecutive_losses: number;
	steps: number;
};

export type BacktestResult = {
	metrics: BacktestMetrics;
	equity_curve: number[];
	actions?: { idx: number; action: string }[];
};

export type IndicatorKind = "sma" | "ema" | "hma";
export type CrossAction = "crossover" | "crossunder";

export type AnalyzerMetrics = {
	totalReward: number;
	netPnl: number;
	endingEquity: number;
	sharpe: number;
	sortino: number;
	maxDrawdown: number;
	profitFactor: number;
	winRate: number;
	trades: number;
	fitness: number;
	steps: number;
};

export type AnalyzerCell = {
	aPeriod: number;
	bPeriod: number;
	takeProfit: number | null;
	stopLoss: number | null;
	metrics: AnalyzerMetrics;
};

export type AnalyzerResult = {
	axes: {
		indicatorA: { kind: IndicatorKind; periods: number[] };
		indicatorB: { kind: IndicatorKind; periods: number[] };
		takeProfitValues: number[];
		stopLossValues: number[];
		total_combinations: number;
		totalCombinations?: number;
	};
	results: AnalyzerCell[];
};

export type NumericInput = number | string;

export type MarginMode = "per-contract" | "price";

export type BacktestEnv = {
	initialBalance: NumericInput;
	maxPosition: NumericInput;
	commission: NumericInput;
	slippage: NumericInput;
	marginPerContract: NumericInput;
	marginMode: MarginMode;
	contractMultiplier: NumericInput;
	enforceMargin: boolean;
};

export type BacktestLimits = {
	memoryMb: NumericInput;
	instructionLimit: NumericInput;
	instructionInterval: NumericInput;
};

export type AnalyzerRange = {
	start: NumericInput;
	end: NumericInput;
	step: NumericInput;
};

export type AnalyzerConfig = {
	indicatorA: { kind: IndicatorKind; start: NumericInput; end: NumericInput; step: NumericInput };
	indicatorB: { kind: IndicatorKind; start: NumericInput; end: NumericInput; step: NumericInput };
	buyAction: CrossAction;
	sellAction: CrossAction;
	takeProfit: { enabled: boolean; start: NumericInput; end: NumericInput; step: NumericInput };
	stopLoss: { enabled: boolean; start: NumericInput; end: NumericInput; step: NumericInput };
};

export type AnalyzerEnv = {
	initialBalance: NumericInput;
	maxPosition: NumericInput;
	commission: NumericInput;
	slippage: NumericInput;
	marginPerContract: NumericInput;
	marginMode: MarginMode;
	contractMultiplier: NumericInput;
	enforceMargin: boolean;
};
