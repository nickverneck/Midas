import type { GaGenerationPoint, GaLogPoint } from "./types";

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

/**
 * Parse both the current GA schema and the older aliases that may still exist
 * in a run directory. The current net-objective PNL fields are preferred.
 */
export const parseGaLogRow = (row: Record<string, unknown>): GaLogPoint | null => {
	if (!row || typeof row !== "object") return null;
	const gen = toNumber(row.gen);
	if (gen === null) return null;

	return {
		gen,
		idx: toNumber(row.idx),
		fitness: toNumber(row.fitness),
		evalFitness: toNumber(row.eval_fitness),
		selectionFitness: toNumber(row.selection_fitness),
		trainPnl: firstNumber(row, ["train_net_objective_pnl", "train_fitness_pnl", "train_pnl"]),
		evalPnl: firstNumber(row, ["eval_net_objective_pnl", "eval_fitness_pnl", "eval_pnl"]),
		trainRealizedPnl: firstNumber(row, [
			"train_net_realized_pnl_after_costs_and_penalties",
			"train_pnl_realized",
			"train_realized_pnl"
		]),
		evalRealizedPnl: firstNumber(row, [
			"eval_net_realized_pnl_after_costs_and_penalties",
			"eval_pnl_realized",
			"eval_realized_pnl"
		]),
		trainTotalPnl: firstNumber(row, ["train_total_net_equity_delta", "train_pnl_total"]),
		evalTotalPnl: firstNumber(row, ["eval_total_net_equity_delta", "eval_pnl_total"])
	};
};

export const gaLogPointKey = (point: GaLogPoint) =>
	`${point.gen}:${point.idx === null ? "unknown" : point.idx}`;

export const mergeGaLogRows = (
	existing: Map<string, GaLogPoint>,
	rows: Array<Record<string, unknown>>
) => {
	if (rows.length === 0) return existing;
	const next = new Map(existing);
	for (const row of rows) {
		const point = parseGaLogRow(row);
		if (point) next.set(gaLogPointKey(point), point);
	}
	return next;
};

const isFiniteNumber = (value: number | null): value is number =>
	value !== null && Number.isFinite(value);

const bestBy = (points: GaLogPoint[], selector: (point: GaLogPoint) => number | null) => {
	let best: GaLogPoint | null = null;
	for (const point of points) {
		const value = selector(point);
		if (!isFiniteNumber(value)) continue;
		if (!best || value > (selector(best) ?? Number.NEGATIVE_INFINITY)) best = point;
	}
	return best;
};

const gap = (train: number | null, evalValue: number | null) =>
	isFiniteNumber(train) && isFiniteNumber(evalValue) ? train - evalValue : null;

/**
 * Collapse candidate rows into one point per generation. The matched metrics
 * remain attached to the candidate the GA would actually select: the highest
 * selection_fitness when it is logged, otherwise the highest train fitness.
 */
export const buildGaGenerationSeries = (points: Iterable<GaLogPoint>): GaGenerationPoint[] => {
	const byGeneration = new Map<number, GaLogPoint[]>();
	for (const point of points) {
		const generation = byGeneration.get(point.gen) ?? [];
		generation.push(point);
		byGeneration.set(point.gen, generation);
	}

	return Array.from(byGeneration.entries())
		.sort(([a], [b]) => a - b)
		.map(([gen, candidates]) => {
			const selectionAnchor = bestBy(candidates, (point) => point.selectionFitness);
			const trainAnchor =
				bestBy(candidates, (point) => point.fitness) ??
				bestBy(candidates, (point) => point.trainPnl) ??
				candidates[0];
			const anchor = selectionAnchor ?? trainAnchor;
			const anchorSource = selectionAnchor ? "selection_fitness" : "train_fitness";

			return {
				gen,
				anchorSource,
				trainFitness: anchor?.fitness ?? null,
				evalFitness: anchor?.evalFitness ?? null,
				selectionFitness: anchor?.selectionFitness ?? null,
				trainPnl: anchor?.trainPnl ?? null,
				evalPnl: anchor?.evalPnl ?? null,
				trainRealizedPnl: anchor?.trainRealizedPnl ?? null,
				evalRealizedPnl: anchor?.evalRealizedPnl ?? null,
				trainTotalPnl: anchor?.trainTotalPnl ?? null,
				evalTotalPnl: anchor?.evalTotalPnl ?? null,
				fitnessGap: gap(anchor?.fitness ?? null, anchor?.evalFitness ?? null),
				pnlGap: gap(anchor?.trainPnl ?? null, anchor?.evalPnl ?? null),
				candidateCount: candidates.length
			};
		});
};

export const hasGaEvalMetrics = (series: GaGenerationPoint[]) =>
	series.some(
		(point) =>
			isFiniteNumber(point.evalFitness) ||
			isFiniteNumber(point.evalPnl) ||
			isFiniteNumber(point.evalRealizedPnl) ||
			isFiniteNumber(point.evalTotalPnl)
	);
