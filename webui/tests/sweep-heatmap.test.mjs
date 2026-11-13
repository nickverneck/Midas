import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSweepHeatmap } from '../src/lib/server/sweep-heatmap.mjs';

// Shape matches native sweep-plan.json children and sweep-summary.json runs.
const plan = { spec: { sweep_id: 'native-example' }, children: [
	{ run_id: 'candidate-a', run_index: 0, parameter_values: { 'native_ema.fast_length': 10, 'native_ema.slow_length': 30, 'native_ema.inverted': false } },
	{ run_id: 'candidate-b', run_index: 1, parameter_values: { 'native_ema.fast_length': 20, 'native_ema.slow_length': 30, 'native_ema.inverted': false } },
	{ run_id: 'candidate-c', run_index: 2, parameter_values: { 'native_ema.fast_length': 10, 'native_ema.slow_length': 40, 'native_ema.inverted': false } },
	{ run_id: 'candidate-d', run_index: 3, parameter_values: { 'native_ema.fast_length': 20, 'native_ema.slow_length': 40, 'native_ema.inverted': false } },
	// A third varied parameter can legitimately produce multiple candidates at one coordinate.
	{ run_id: 'candidate-e', run_index: 4, parameter_values: { 'native_ema.fast_length': 10, 'native_ema.slow_length': 30, 'native_ema.inverted': true } }
] };
const summary = { sweep_id: 'native-example', completed_count: 2, failed_count: 1, skipped_count: 1, runs: [
	{ run_id: 'candidate-a', run_index: 0, status: 'completed', skipped: false, net_pnl: 125.5 },
	{ run_id: 'candidate-b', run_index: 1, status: 'failed', skipped: false, error: 'dataset unavailable', net_pnl: null },
	{ run_id: 'candidate-c', run_index: 2, status: 'completed', skipped: false, net_pnl: -20 },
	{ run_id: 'candidate-d', run_index: 3, status: 'skipped', skipped: true, net_pnl: null }
] };

test('native plan and summary become an exact, status-aware heatmap', () => {
	const grid = buildSweepHeatmap(plan, summary, 'native_ema.fast_length', 'native_ema.slow_length', 'net_pnl');
	assert.equal(grid.sweepId, 'native-example');
	assert.equal(grid.cells.length, 2);
	const [row30, row40] = grid.cells;
	assert.equal(row30.cells[0].candidates.length, 2);
	assert.deepEqual(row30.cells[0].candidates.map(candidate => candidate.candidateId), ['candidate-a', 'candidate-e']);
	assert.equal(row30.cells[0].candidates[0].value, 125.5);
	assert.equal(row30.cells[0].candidates[1].status, 'missing_result');
	assert.equal(row30.cells[1].candidates[0].status, 'failed');
	assert.equal(row30.cells[1].candidates[0].error, 'dataset unavailable');
	assert.equal(row40.cells[0].candidates[0].value, -20);
	assert.equal(row40.cells[1].candidates[0].status, 'skipped');
	assert.equal(grid.counts.planned, 5);
});

test('default axes come from child parameter_values and invalid axes are rejected', () => {
	const grid = buildSweepHeatmap(plan, summary);
	assert.equal(grid.selectedAxes.x, 'native_ema.fast_length');
	assert.equal(grid.selectedAxes.y, 'native_ema.slow_length');
	assert.equal(grid.selectedMetric, 'net_pnl');
	assert.throws(() => buildSweepHeatmap(plan, summary, 'native_ema.fast_length', 'native_ema.fast_length'), /two distinct/);
	assert.throws(() => buildSweepHeatmap(plan, summary, 'outside.path', 'native_ema.slow_length'), /two distinct/);
});

test('a coordinate absent from the native plan stays unplanned', () => {
	const sparse = { children: [plan.children[0], plan.children[3]] };
	const grid = buildSweepHeatmap(sparse, summary, 'native_ema.fast_length', 'native_ema.slow_length', 'net_pnl');
	assert.equal(grid.cells[0].cells[1].planned, false);
	assert.deepEqual(grid.cells[0].cells[1].candidates, []);
});
