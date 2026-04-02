import type { ChartConfiguration } from "chart.js";
import { calcFitness } from "./data";
import type { ResolvedFitnessWeights, RlChartsViewModel, RlPoint } from "./types";

const createEpochOptions = (): ChartConfiguration["options"] => ({
	scales: {
		x: {
			type: "linear" as const,
			title: { display: true, text: "Epoch" },
			ticks: { precision: 0 }
		}
	}
});

const toSeries = (epochData: RlPoint[], picker: (row: RlPoint) => number | null) =>
	epochData.map((row) => ({ x: row.epoch, y: picker(row) }));

const toFrontierSeries = (
	epochData: RlPoint[],
	drawdownPicker: (row: RlPoint) => number | null,
	realizedPicker: (row: RlPoint) => number | null,
	pnlPicker: (row: RlPoint) => number | null
) =>
	epochData.flatMap((row) => {
		const drawdown = drawdownPicker(row);
		const realizedPnl = realizedPicker(row);
		const pnl = pnlPicker(row);
		const value = realizedPnl ?? pnl;

		if (drawdown === null || value === null) return [];

		return [
			{
				x: drawdown,
				y: value,
				epoch: row.epoch,
				metricLabel: realizedPnl !== null ? "Realized PnL" : "PnL"
			}
		];
	});

export const buildRlCharts = (
	epochData: RlPoint[],
	weights: ResolvedFitnessWeights
): RlChartsViewModel => {
	const epochOptions = createEpochOptions();

	const hasTrainRealized = epochData.some((row) => row.trainRealizedPnl !== null);
	const hasEvalRealized = epochData.some((row) => row.evalRealizedPnl !== null);
	const realizedSeriesCount = Number(hasTrainRealized) + Number(hasEvalRealized);

	const frontierAxisLabel =
		realizedSeriesCount === 2
			? "Realized PnL"
			: realizedSeriesCount === 1
				? "Realized PnL / PnL"
				: "PnL";

	const hasTrainPnl = epochData.some((row) => row.trainPnl !== null);
	const hasEvalPnl = epochData.some((row) => row.evalPnl !== null);
	const trainFallback = hasTrainPnl && !hasTrainRealized;
	const evalFallback = hasEvalPnl && !hasEvalRealized;

	let frontierNote = "";
	if (trainFallback && evalFallback) {
		frontierNote =
			"This RL run does not expose realized-PnL columns, so the frontier uses the logged train and eval PnL values.";
	} else if (trainFallback) {
		frontierNote =
			"Train frontier points use logged PnL because this run does not include train realized-PnL columns.";
	} else if (evalFallback) {
		frontierNote =
			"Eval frontier points use logged PnL because this run does not include eval realized-PnL columns.";
	}

	const trainFrontierData = toFrontierSeries(
		epochData,
		(row) => row.trainDrawdown,
		(row) => row.trainRealizedPnl,
		(row) => row.trainPnl
	);
	const evalFrontierData = toFrontierSeries(
		epochData,
		(row) => row.evalDrawdown,
		(row) => row.evalRealizedPnl,
		(row) => row.evalPnl
	);

	const frontierDatasets = [];
	if (trainFrontierData.length > 0) {
		frontierDatasets.push({
			label: hasTrainRealized ? "Train Realized PnL vs Drawdown" : "Train PnL vs Drawdown",
			data: trainFrontierData,
			borderColor: "rgb(148, 163, 184)",
			backgroundColor: "rgba(148, 163, 184, 0.55)",
			showLine: false,
			pointRadius: 3
		});
	}
	if (evalFrontierData.length > 0) {
		frontierDatasets.push({
			label: hasEvalRealized ? "Eval Realized PnL vs Drawdown" : "Eval PnL vs Drawdown",
			data: evalFrontierData,
			borderColor: "rgb(34, 197, 94)",
			backgroundColor: "rgba(34, 197, 94, 0.45)",
			showLine: false,
			pointRadius: 3
		});
	}

	return {
		fitness: {
			data: {
				datasets: [
					{
						label: "Train Fitness",
						data: toSeries(epochData, (row) =>
							calcFitness(row.trainPnl, row.trainSortino, row.trainDrawdown, weights)
						),
						borderColor: "rgb(148, 163, 184)",
						backgroundColor: "rgba(148, 163, 184, 0.25)",
						tension: 0.1
					},
					{
						label: "Eval Fitness",
						data: toSeries(epochData, (row) =>
							calcFitness(row.evalPnl, row.evalSortino, row.evalDrawdown, weights)
						),
						borderColor: "rgb(34, 197, 94)",
						backgroundColor: "rgba(34, 197, 94, 0.3)",
						borderDash: [4, 4],
						tension: 0.1
					}
				]
			},
			options: epochOptions
		},
		performance: {
			data: {
				datasets: [
					{
						label: "Train PnL",
						data: toSeries(epochData, (row) => row.trainPnl),
						borderColor: "rgb(148, 163, 184)",
						backgroundColor: "rgba(148, 163, 184, 0.25)",
						tension: 0.1
					},
					{
						label: "Eval PnL",
						data: toSeries(epochData, (row) => row.evalPnl),
						borderColor: "rgb(34, 197, 94)",
						backgroundColor: "rgba(34, 197, 94, 0.3)",
						tension: 0.1
					},
					{
						label: "Eval Sortino",
						data: toSeries(epochData, (row) => row.evalSortino),
						borderColor: "rgb(168, 85, 247)",
						backgroundColor: "rgba(168, 85, 247, 0.2)",
						borderDash: [4, 4],
						yAxisID: "y1",
						tension: 0.1
					}
				]
			},
			options: epochOptions
		},
		drawdown: {
			data: {
				datasets: [
					{
						label: "Train Drawdown",
						data: toSeries(epochData, (row) => row.trainDrawdown),
						borderColor: "rgb(239, 68, 68)",
						backgroundColor: "rgba(239, 68, 68, 0.3)",
						tension: 0.1
					},
					{
						label: "Eval Drawdown",
						data: toSeries(epochData, (row) => row.evalDrawdown),
						borderColor: "rgb(248, 113, 113)",
						backgroundColor: "rgba(248, 113, 113, 0.2)",
						borderDash: [4, 4],
						tension: 0.1
					}
				]
			},
			options: epochOptions
		},
		frontier: {
			data: {
				labels: [],
				datasets: frontierDatasets
			},
			options: {
				scales: {
					x: {
						type: "linear" as const,
						title: { display: true, text: "Max Drawdown (%)" }
					},
					y: {
						title: { display: true, text: frontierAxisLabel }
					}
				},
				plugins: {
					tooltip: {
						callbacks: {
							label: (ctx: any) => {
								const raw = ctx.raw as {
									x?: number;
									y?: number;
									epoch?: number;
									metricLabel?: string;
								};
								const drawdown = raw?.x ?? ctx.parsed?.x;
								const pnl = raw?.y ?? ctx.parsed?.y;

								return [
									ctx.dataset.label ?? "Frontier Point",
									`Epoch: ${raw?.epoch ?? "—"}`,
									`Drawdown: ${drawdown ?? "—"}`,
									`${raw?.metricLabel ?? frontierAxisLabel}: ${pnl ?? "—"}`
								];
							}
						}
					}
				}
			},
			type: "scatter",
			note: frontierNote,
			emptyMessage: "Not enough drawdown and PnL data to build the frontier yet."
		},
		loss: {
			data: {
				datasets: [
					{
						label: "Policy Loss",
						data: toSeries(epochData, (row) => row.policyLoss),
						borderColor: "rgb(59, 130, 246)",
						backgroundColor: "rgba(59, 130, 246, 0.3)",
						tension: 0.1
					},
					{
						label: "Value Loss",
						data: toSeries(epochData, (row) => row.valueLoss),
						borderColor: "rgb(16, 185, 129)",
						backgroundColor: "rgba(16, 185, 129, 0.25)",
						tension: 0.1
					},
					{
						label: "Entropy",
						data: toSeries(epochData, (row) => row.entropy),
						borderColor: "rgb(251, 191, 36)",
						backgroundColor: "rgba(251, 191, 36, 0.2)",
						borderDash: [4, 4],
						yAxisID: "y1",
						tension: 0.1
					},
					{
						label: "Perplexity",
						data: toSeries(epochData, (row) => row.perplexity),
						borderColor: "rgb(249, 115, 22)",
						backgroundColor: "rgba(249, 115, 22, 0.18)",
						yAxisID: "y1",
						tension: 0.1
					},
					{
						label: "Approx KL",
						data: toSeries(epochData, (row) => row.approxKl),
						borderColor: "rgb(244, 63, 94)",
						backgroundColor: "rgba(244, 63, 94, 0.18)",
						yAxisID: "y2",
						tension: 0.1
					},
					{
						label: "KL Div",
						data: toSeries(epochData, (row) => row.klDiv),
						borderColor: "rgb(236, 72, 153)",
						backgroundColor: "rgba(236, 72, 153, 0.18)",
						borderDash: [4, 4],
						yAxisID: "y2",
						tension: 0.1
					},
					{
						label: "Clip Frac",
						data: toSeries(epochData, (row) => row.clipFrac),
						borderColor: "rgb(14, 165, 233)",
						backgroundColor: "rgba(14, 165, 233, 0.18)",
						borderDash: [2, 3],
						yAxisID: "y2",
						tension: 0.1
					}
				]
			},
			options: {
				scales: {
					x: {
						type: "linear" as const,
						title: { display: true, text: "Epoch" },
						ticks: { precision: 0 }
					},
					y: {
						title: { display: true, text: "Loss" }
					},
					y1: {
						position: "right" as const,
						grid: { drawOnChartArea: false },
						title: { display: true, text: "Entropy / Perplexity" }
					},
					y2: {
						position: "right" as const,
						grid: { drawOnChartArea: false },
						title: { display: true, text: "KL / Clip" }
					}
				}
			}
		},
		gradients: {
			data: {
				datasets: [
					{
						label: "Policy Grad Norm",
						data: toSeries(epochData, (row) => row.policyGradNorm),
						borderColor: "rgb(59, 130, 246)",
						backgroundColor: "rgba(59, 130, 246, 0.25)",
						tension: 0.1
					},
					{
						label: "Value Grad Norm",
						data: toSeries(epochData, (row) => row.valueGradNorm),
						borderColor: "rgb(16, 185, 129)",
						backgroundColor: "rgba(16, 185, 129, 0.22)",
						tension: 0.1
					}
				]
			},
			options: {
				scales: {
					x: {
						type: "linear" as const,
						title: { display: true, text: "Epoch" },
						ticks: { precision: 0 }
					},
					y: {
						title: { display: true, text: "L2 Gradient Norm" }
					}
				}
			}
		},
		policy: {
			data: {
				datasets: [
					{
						label: "Eval Buy %",
						data: toSeries(epochData, (row) => row.evalBuyFrac),
						borderColor: "rgb(34, 197, 94)",
						backgroundColor: "rgba(34, 197, 94, 0.2)",
						tension: 0.1
					},
					{
						label: "Eval Sell %",
						data: toSeries(epochData, (row) => row.evalSellFrac),
						borderColor: "rgb(239, 68, 68)",
						backgroundColor: "rgba(239, 68, 68, 0.2)",
						tension: 0.1
					},
					{
						label: "Eval Hold %",
						data: toSeries(epochData, (row) => row.evalHoldFrac),
						borderColor: "rgb(148, 163, 184)",
						backgroundColor: "rgba(148, 163, 184, 0.18)",
						tension: 0.1
					},
					{
						label: "Eval Revert %",
						data: toSeries(epochData, (row) => row.evalRevertFrac),
						borderColor: "rgb(168, 85, 247)",
						backgroundColor: "rgba(168, 85, 247, 0.16)",
						tension: 0.1
					},
					{
						label: "Eval Max Prob",
						data: toSeries(epochData, (row) => row.evalMeanMaxProb),
						borderColor: "rgb(245, 158, 11)",
						backgroundColor: "rgba(245, 158, 11, 0.18)",
						borderDash: [4, 4],
						tension: 0.1
					},
					{
						label: "Probe Max Prob",
						data: toSeries(epochData, (row) => row.probeMeanMaxProb),
						borderColor: "rgb(14, 165, 233)",
						backgroundColor: "rgba(14, 165, 233, 0.18)",
						borderDash: [2, 3],
						tension: 0.1
					}
				]
			},
			options: {
				scales: {
					x: {
						type: "linear" as const,
						title: { display: true, text: "Epoch" },
						ticks: { precision: 0 }
					},
					y: {
						min: 0,
						max: 1,
						title: { display: true, text: "Fraction / Probability" }
					}
				}
			}
		},
		probe: {
			data: {
				datasets: [
					{
						label: "Probe PnL",
						data: toSeries(epochData, (row) => row.probePnl),
						borderColor: "rgb(59, 130, 246)",
						backgroundColor: "rgba(59, 130, 246, 0.22)",
						tension: 0.1
					},
					{
						label: "Probe Realized PnL",
						data: toSeries(epochData, (row) => row.probeRealizedPnl),
						borderColor: "rgb(16, 185, 129)",
						backgroundColor: "rgba(16, 185, 129, 0.2)",
						tension: 0.1
					},
					{
						label: "Probe Drawdown",
						data: toSeries(epochData, (row) => row.probeDrawdown),
						borderColor: "rgb(239, 68, 68)",
						backgroundColor: "rgba(239, 68, 68, 0.18)",
						yAxisID: "y1",
						tension: 0.1
					},
					{
						label: "Probe Max Prob",
						data: toSeries(epochData, (row) => row.probeMeanMaxProb),
						borderColor: "rgb(245, 158, 11)",
						backgroundColor: "rgba(245, 158, 11, 0.18)",
						yAxisID: "y2",
						borderDash: [4, 4],
						tension: 0.1
					}
				]
			},
			options: {
				scales: {
					x: {
						type: "linear" as const,
						title: { display: true, text: "Epoch" },
						ticks: { precision: 0 }
					},
					y: {
						title: { display: true, text: "PnL" }
					},
					y1: {
						position: "right" as const,
						grid: { drawOnChartArea: false },
						title: { display: true, text: "Drawdown (%)" }
					},
					y2: {
						position: "right" as const,
						grid: { drawOnChartArea: false },
						min: 0,
						max: 1,
						title: { display: true, text: "Confidence" }
					}
				}
			}
		}
	};
};
