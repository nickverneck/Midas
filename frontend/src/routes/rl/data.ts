import type { FitnessWeights, ResolvedFitnessWeights, RlPoint } from "./types";

const TRAIN_REALIZED_KEYS = ["train_realized_pnl", "train_pnl_realized", "train_realized"];
const EVAL_REALIZED_KEYS = ["eval_realized_pnl", "eval_pnl_realized", "eval_realized"];

export const DEFAULT_LOG_DIR = "runs_rl";

export const DEFAULT_FITNESS_WEIGHTS: FitnessWeights = {
	pnl: 1.0,
	sortino: 1.0,
	mdd: 0.5
};

export const toNumber = (value: unknown): number | null => {
	if (typeof value === "number") return Number.isFinite(value) ? value : null;
	if (typeof value === "string" && value.trim() !== "") {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : null;
	}
	return null;
};

const firstNumber = (row: Record<string, unknown>, keys: string[]) => {
	for (const key of keys) {
		const value = toNumber(row[key]);
		if (value !== null) return value;
	}
	return null;
};

export const normalizeLogDir = (value: string) => value.trim() || DEFAULT_LOG_DIR;

const parseRlRow = (row: Record<string, unknown>): RlPoint | null => {
	const epoch = toNumber(row.epoch);
	if (epoch === null) return null;

	const valueLoss = toNumber(row.value_loss);
	const approxKl = toNumber(row.approx_kl);
	const klDiv = toNumber(row.kl_div);
	const clipFrac = toNumber(row.clip_frac);
	const trainRealizedPnl = firstNumber(row, TRAIN_REALIZED_KEYS);
	const evalRealizedPnl = firstNumber(row, EVAL_REALIZED_KEYS);
	const entropy = toNumber(row.entropy);
	const algorithmName = typeof row.algorithm === "string" ? row.algorithm.trim().toLowerCase() : "";

	let algorithm: "ppo" | "grpo" | null = null;
	if (algorithmName === "ppo" || algorithmName === "grpo") {
		algorithm = algorithmName;
	} else if (valueLoss !== null) {
		algorithm = "ppo";
	} else if (klDiv !== null) {
		algorithm = "grpo";
	}

	const perplexity = toNumber(row.perplexity) ?? (entropy !== null ? Math.exp(entropy) : null);

	return {
		epoch,
		fitness: toNumber(row.fitness),
		trainRet: toNumber(row.train_ret_mean),
		trainPnl: toNumber(row.train_pnl),
		trainRealizedPnl,
		trainSortino: toNumber(row.train_sortino),
		trainDrawdown: toNumber(row.train_drawdown),
		trainCommission: toNumber(row.train_commission),
		trainSlippage: toNumber(row.train_slippage),
		trainBuyFrac: toNumber(row.train_buy_frac),
		trainSellFrac: toNumber(row.train_sell_frac),
		trainHoldFrac: toNumber(row.train_hold_frac),
		trainRevertFrac: toNumber(row.train_revert_frac),
		trainMeanMaxProb: toNumber(row.train_mean_max_prob),
		trainEntries: toNumber(row.train_entries),
		trainExits: toNumber(row.train_exits),
		trainFlips: toNumber(row.train_flips),
		trainAvgHold: toNumber(row.train_avg_hold),
		evalRet: toNumber(row.eval_ret_mean),
		evalPnl: toNumber(row.eval_pnl),
		evalRealizedPnl,
		evalSortino: toNumber(row.eval_sortino),
		evalDrawdown: toNumber(row.eval_drawdown),
		evalCommission: toNumber(row.eval_commission),
		evalSlippage: toNumber(row.eval_slippage),
		evalBuyFrac: toNumber(row.eval_buy_frac),
		evalSellFrac: toNumber(row.eval_sell_frac),
		evalHoldFrac: toNumber(row.eval_hold_frac),
		evalRevertFrac: toNumber(row.eval_revert_frac),
		evalMeanMaxProb: toNumber(row.eval_mean_max_prob),
		evalEntries: toNumber(row.eval_entries),
		evalExits: toNumber(row.eval_exits),
		evalFlips: toNumber(row.eval_flips),
		evalAvgHold: toNumber(row.eval_avg_hold),
		probeRet: toNumber(row.probe_ret_mean),
		probePnl: toNumber(row.probe_pnl),
		probeRealizedPnl: toNumber(row.probe_realized_pnl),
		probeSortino: toNumber(row.probe_sortino),
		probeDrawdown: toNumber(row.probe_drawdown),
		probeCommission: toNumber(row.probe_commission),
		probeSlippage: toNumber(row.probe_slippage),
		probeBuyFrac: toNumber(row.probe_buy_frac),
		probeSellFrac: toNumber(row.probe_sell_frac),
		probeHoldFrac: toNumber(row.probe_hold_frac),
		probeRevertFrac: toNumber(row.probe_revert_frac),
		probeMeanMaxProb: toNumber(row.probe_mean_max_prob),
		probeEntries: toNumber(row.probe_entries),
		probeExits: toNumber(row.probe_exits),
		probeFlips: toNumber(row.probe_flips),
		probeAvgHold: toNumber(row.probe_avg_hold),
		policyLoss: toNumber(row.policy_loss),
		valueLoss,
		policyGradNorm: toNumber(row.policy_grad_norm),
		valueGradNorm: toNumber(row.value_grad_norm),
		approxKl,
		klDiv,
		clipFrac,
		entropy,
		perplexity,
		totalLoss: toNumber(row.total_loss),
		algorithm
	};
};

export const mergeRlRows = (
	existing: Map<number, RlPoint>,
	rows: Array<Record<string, unknown>>
) => {
	if (rows.length === 0) return existing;

	const next = new Map(existing);
	for (const row of rows) {
		if (!row || typeof row !== "object") continue;
		const parsed = parseRlRow(row);
		if (parsed) {
			next.set(parsed.epoch, parsed);
		}
	}
	return next;
};

export const sortEpochData = (logMap: Map<number, RlPoint>) =>
	Array.from(logMap.values()).sort((a, b) => a.epoch - b.epoch);

export const getLatestPoint = (epochData: RlPoint[]) =>
	epochData.length > 0 ? epochData[epochData.length - 1] : null;

export const resolveFitnessWeights = (weights: FitnessWeights): ResolvedFitnessWeights => ({
	pnl: toNumber(weights.pnl) ?? 1,
	sortino: toNumber(weights.sortino) ?? 1,
	mdd: toNumber(weights.mdd) ?? 0.5
});

export const calcFitness = (
	pnl: number | null,
	sortino: number | null,
	drawdown: number | null,
	weights: ResolvedFitnessWeights
) => {
	if (pnl === null || sortino === null || drawdown === null) return null;
	return weights.pnl * pnl + weights.sortino * sortino - weights.mdd * drawdown;
};
