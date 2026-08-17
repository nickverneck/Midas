// @ts-nocheck
import { describe, expect, test } from "bun:test";
import {
	applyLogChunk,
	buildGenData,
	createEmptyIssueState,
	detectEval,
	resolveLogValue
} from "../src/routes/ga/analytics";

describe("GA dashboard analytics schema compatibility", () => {
	test("reads the current net-objective, realized, and total aliases", () => {
		const row = {
			gen: 3,
			idx: 1,
			fitness: 12,
			eval_fitness: 8,
			eval_net_objective_pnl: 80,
			eval_net_realized_pnl_after_costs_and_penalties: 70,
			eval_total_net_equity_delta: 79,
			train_net_objective_pnl: 120,
			train_net_realized_pnl_after_costs_and_penalties: 110,
			train_total_net_equity_delta: 119,
			eval_sortino: 1.5,
			eval_drawdown: 2,
			eval_ret_mean: 0.1
		};

		expect(detectEval([row])).toBe(true);
		expect(resolveLogValue(row, "eval_fitness_pnl")).toBe(80);
		expect(resolveLogValue(row, "eval_pnl_realized")).toBe(70);
		expect(resolveLogValue(row, "eval_pnl_total")).toBe(79);

		const result = applyLogChunk(
			new Map(),
			Number.NEGATIVE_INFINITY,
			createEmptyIssueState(),
			[row],
			true
		);
		const member = result.genMembers.get(3)?.[0];
		expect(member).toMatchObject({
			pnl: 80,
			realized: 70,
			total: 79,
			trainRealized: 110,
			evalRealized: 70
		});
	});

	test("continues to read the legacy dashboard aliases", () => {
		const row = {
			gen: 4,
			idx: 2,
			fitness: 9,
			eval_fitness: 6,
			eval_fitness_pnl: 60,
			eval_pnl_realized: 50,
			eval_pnl_total: 55,
			train_fitness_pnl: 90,
			train_pnl_realized: 75,
			train_pnl_total: 85,
			eval_sortino: 1,
			eval_drawdown: 3,
			eval_ret_mean: 0.2
		};

		expect(detectEval([row])).toBe(true);
		const result = applyLogChunk(
			new Map(),
			Number.NEGATIVE_INFINITY,
			createEmptyIssueState(),
			[row],
			true
		);
		const summary = buildGenData(result.genMembers, "all")[0];
		expect(summary).toMatchObject({
			bestPnl: 60,
			bestRealizedPnl: 50,
			bestTotalPnl: 55,
			bestEvalRealized: 50
		});
	});

	test("does not select eval for train-only rows with blank eval columns", () => {
		const row = {
			gen: 5,
			idx: 0,
			fitness: 14,
			eval_fitness: "",
			eval_net_objective_pnl: "",
			eval_net_realized_pnl_after_costs_and_penalties: "",
			eval_total_net_equity_delta: "",
			eval_fitness_pnl: "",
			eval_pnl_realized: "",
			eval_pnl_total: "",
			eval_pnl: "",
			eval_realized_pnl: "",
			eval_total_pnl: "",
			eval_realized: "",
			eval_total: "",
			eval_sortino: "",
			eval_drawdown: "",
			eval_ret_mean: "",
			train_net_objective_pnl: 140,
			train_net_realized_pnl_after_costs_and_penalties: 130,
			train_total_net_equity_delta: 139
		};

		expect(detectEval([row])).toBe(false);
	});
});
