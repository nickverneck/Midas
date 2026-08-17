// @ts-nocheck
import { describe, expect, test } from "bun:test";
import {
	buildGaGenerationSeries,
	mergeGaLogRows,
	parseGaLogRow
} from "../src/routes/train/gaAnalytics";

describe("GA training analytics", () => {
	test("parses the current train/eval schema and prefers net objective PNL", () => {
		const point = parseGaLogRow({
			gen: "3",
			idx: "2",
			fitness: "12.5",
			eval_fitness: "8.25",
			selection_fitness: "9.1",
			train_net_objective_pnl: "120",
			train_total_net_equity_delta: "119",
			eval_net_objective_pnl: "80",
			eval_total_net_equity_delta: "79",
			train_net_realized_pnl_after_costs_and_penalties: "110",
			eval_net_realized_pnl_after_costs_and_penalties: "70"
		});

		expect(point).toEqual({
			gen: 3,
			idx: 2,
			fitness: 12.5,
			evalFitness: 8.25,
			selectionFitness: 9.1,
			trainPnl: 120,
			evalPnl: 80,
			trainRealizedPnl: 110,
			evalRealizedPnl: 70,
			trainTotalPnl: 119,
			evalTotalPnl: 79
		});
	});

	test("anchors eval and PNL to the candidate selected by selection_fitness", () => {
		const rows = [
			{ gen: 0, idx: 0, fitness: 15, selection_fitness: 5, eval_fitness: 4, train_net_objective_pnl: 150, eval_net_objective_pnl: 40 },
			{ gen: 0, idx: 1, fitness: 10, selection_fitness: 9, eval_fitness: 7, train_net_objective_pnl: 100, eval_net_objective_pnl: 70 },
			{ gen: 1, idx: 0, fitness: 5, eval_fitness: "", train_net_objective_pnl: 50, eval_net_objective_pnl: "" }
		];
		const map = mergeGaLogRows(new Map(), rows);
		const series = buildGaGenerationSeries(map.values());

		expect(series).toHaveLength(2);
		expect(series[0]).toMatchObject({
			gen: 0,
			anchorSource: "selection_fitness",
			trainFitness: 10,
			evalFitness: 7,
			trainPnl: 100,
			evalPnl: 70,
			fitnessGap: 3,
			pnlGap: 30,
			candidateCount: 2
		});
		expect(series[1]).toMatchObject({
			gen: 1,
			anchorSource: "train_fitness",
			trainFitness: 5,
			evalFitness: null,
			fitnessGap: null,
			candidateCount: 1
		});
	});

	test("falls back to the best train-fitness candidate for legacy rows", () => {
		const rows = [
			{ gen: 4, idx: 0, fitness: 4, eval_fitness: 2, train_fitness_pnl: 40, eval_fitness_pnl: 20 },
			{ gen: 4, idx: 1, fitness: 8, eval_fitness: 3, train_fitness_pnl: 80, eval_fitness_pnl: 30 }
		];
		const series = buildGaGenerationSeries(mergeGaLogRows(new Map(), rows).values());

		expect(series[0]).toMatchObject({
			anchorSource: "train_fitness",
			trainFitness: 8,
			evalFitness: 3,
			trainPnl: 80,
			evalPnl: 30,
			fitnessGap: 5
		});
	});

	test("handles train-only logs without inventing validation data", () => {
		const point = parseGaLogRow({ gen: 2, idx: 0, fitness: 3, train_pnl: 25 });
		expect(point).not.toBeNull();
		const series = buildGaGenerationSeries([point!]);
		expect(series[0].evalFitness).toBeNull();
		expect(series[0].evalPnl).toBeNull();
		expect(series[0].fitnessGap).toBeNull();
	});
});
