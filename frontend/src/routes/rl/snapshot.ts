import { calcFitness } from "./data";
import type { ResolvedFitnessWeights, RlPoint, RlSnapshotRow } from "./types";

export const buildSnapshotRows = (
	latest: RlPoint | null,
	weights: ResolvedFitnessWeights
): RlSnapshotRow[] => {
	if (!latest) return [];

	return [
		{ label: "Algorithm", value: latest.algorithm ?? "—", emphasis: true },
		{ label: "Epoch", value: latest.epoch, emphasis: true },
		{
			label: "Train Fitness",
			value: calcFitness(latest.trainPnl, latest.trainSortino, latest.trainDrawdown, weights),
			emphasis: true
		},
		{
			label: "Eval Fitness",
			value: calcFitness(latest.evalPnl, latest.evalSortino, latest.evalDrawdown, weights),
			emphasis: true
		},
		{ label: "Logged Fitness", value: latest.fitness, emphasis: true },
		{ label: "Eval PnL", value: latest.evalPnl },
		{ label: "Eval Realized PnL", value: latest.evalRealizedPnl },
		{ label: "Eval Sortino", value: latest.evalSortino },
		{ label: "Eval Drawdown", value: latest.evalDrawdown },
		{ label: "Eval Max Prob", value: latest.evalMeanMaxProb },
		{ label: "Probe PnL", value: latest.probePnl },
		{ label: "Probe Drawdown", value: latest.probeDrawdown },
		{ label: "Policy Loss", value: latest.policyLoss },
		{ label: "Value Loss", value: latest.valueLoss },
		{ label: "Policy Grad Norm", value: latest.policyGradNorm },
		{ label: "Value Grad Norm", value: latest.valueGradNorm },
		{ label: "Entropy", value: latest.entropy },
		{ label: "Perplexity", value: latest.perplexity },
		{ label: "Approx KL", value: latest.approxKl },
		{ label: "KL Div", value: latest.klDiv },
		{ label: "Clip Frac", value: latest.clipFrac },
		{
			label: "Probe Entries / Flips",
			value: `${latest.probeEntries ?? "—"} / ${latest.probeFlips ?? "—"}`
		}
	];
};
