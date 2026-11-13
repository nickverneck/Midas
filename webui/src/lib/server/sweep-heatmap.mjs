const metricLabels = /** @type {Record<string, string>} */ ({
	gross_pnl: 'Gross P&L', net_pnl: 'Net P&L', fees: 'Fees', max_drawdown: 'Max drawdown',
	trade_count: 'Trades', fill_count: 'Fills'
});

/** @param {any} value */
const isScalar = value => value === null || ['string', 'number', 'boolean'].includes(typeof value);
/** @param {any} value */
const keyOf = value => JSON.stringify(value);

/** Build a truthful grid from native plan children and summary runs. No values are inferred. */
/** @param {{children: any[], spec?: {sweep_id?: string}}} plan @param {{runs: any[], sweep_id?: string, completed_count?: number, failed_count?: number, skipped_count?: number}} summary @param {string=} axisX @param {string=} axisY @param {string=} metric */
export function buildSweepHeatmap(plan, summary, axisX, axisY, metric) {
	if (!Array.isArray(plan?.children) || !Array.isArray(summary?.runs)) throw new Error('Native sweep plan or summary has an unsupported schema.');
	const parameters = new Map();
	for (const child of plan.children) {
		if (!child || typeof child !== 'object' || !child.parameter_values || typeof child.parameter_values !== 'object') continue;
		for (const [path, value] of Object.entries(child.parameter_values)) {
			if (!isScalar(value)) continue;
			if (!parameters.has(path)) parameters.set(path, new Map());
			parameters.get(path).set(keyOf(value), value);
		}
	}
	const axes = [...parameters.entries()].map(([path, values]) => ({ path, values: [...values.values()] }));
	if (!axisX || !axisY) { axisX ??= axes[0]?.path; axisY ??= axes.find(axis => axis.path !== axisX)?.path; }
	if (!axisX || !axisY || !parameters.has(axisX) || !parameters.has(axisY) || axisX === axisY) throw new Error('Choose two distinct sweep parameters.');
	const xPath = axisX;
	const yPath = axisY;
	const availableMetrics = Object.keys(metricLabels).filter(key => summary.runs.some(run => typeof run?.[key] === 'number' && Number.isFinite(run[key])));
	if (!metric || !availableMetrics.includes(metric)) metric = availableMetrics.includes('net_pnl') ? 'net_pnl' : availableMetrics[0];
	if (!metric) throw new Error('Sweep summary has no supported numeric metrics.');
	const metricKey = metric;
	const outcomes = new Map(summary.runs.filter(run => run && Number.isInteger(run.run_index)).map(run => [run.run_index, run]));
	const children = plan.children.filter(child => child?.parameter_values && isScalar(child.parameter_values[xPath]) && isScalar(child.parameter_values[yPath]));
	const xAxis = axes.find(axis => axis.path === xPath);
	const yAxis = axes.find(axis => axis.path === yPath);
	if (!xAxis || !yAxis) throw new Error('Choose two distinct sweep parameters.');
	const xs = xAxis.values;
	const ys = yAxis.values;
	const cells = ys.map(y => xs.map(x => {
		const matches = children.filter(child => keyOf(child.parameter_values[xPath]) === keyOf(x) && keyOf(child.parameter_values[yPath]) === keyOf(y));
		const candidates = matches.map(child => {
			const run = Number.isInteger(child.run_index) ? outcomes.get(child.run_index) : undefined;
			const status = !run ? 'missing_result' : run.skipped ? 'skipped' : run.status === 'completed' ? 'completed' : run.status || 'unknown';
			return {
				candidateId: child.run_id ?? run?.run_id ?? `run-index-${child.run_index}`,
				runIndex: child.run_index,
				status,
				value: status === 'completed' && Number.isFinite(run?.[metricKey]) ? run[metricKey] : null,
				error: run?.error ?? null,
				parameters: child.parameter_values
			};
		});
		return { x, y, planned: matches.length > 0, candidates };
	}));
	return {
		sweepId: summary.sweep_id ?? plan.spec?.sweep_id ?? null,
		axes, selectedAxes: { x: xPath, y: yPath },
		metrics: availableMetrics.map(key => ({ key, label: metricLabels[key] })), selectedMetric: metricKey,
		cells: ys.map((y, yi) => ({ y, cells: cells[yi] })),
		counts: { planned: plan.children.length, completed: summary.completed_count ?? summary.runs.filter(run => run.status === 'completed').length, failed: summary.failed_count ?? summary.runs.filter(run => run.status === 'failed').length, skipped: summary.skipped_count ?? summary.runs.filter(run => run.skipped).length }
	};
}
