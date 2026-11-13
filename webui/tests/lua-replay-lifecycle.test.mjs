import test from 'node:test';
import assert from 'node:assert/strict';
import { luaReplayFailureMessage, supportsReplayResults } from '../src/lib/server/lua-replay-lifecycle.mjs';

test('simulated successful Lua replay completes and exposes its artifacts', () => {
	const summary = { run_count: 1, completed_count: 1, failed_count: 0, skipped_count: 0, runs: [{ run_id: 'candidate-1', status: 'completed', result_path: '/job/output/result.json' }] };
	assert.equal(luaReplayFailureMessage(summary), undefined);
	assert.equal(supportsReplayResults('lua-run'), true);
});

test('zero-exit Lua replay with a failed child is surfaced as a failed job', () => {
	const summary = { runs: [{ run_id: 'candidate-1', status: 'failed', error: 'Lua callback exceeded instruction limit.' }] };
	assert.match(luaReplayFailureMessage(summary), /candidate-1.*failed.*instruction limit/);
	assert.equal(supportsReplayResults('lua-run'), true);
});

test('malformed or incomplete Lua summaries cannot be reported as successful', () => {
	assert.match(luaReplayFailureMessage(undefined), /valid sweep summary/);
	assert.match(luaReplayFailureMessage({ runs: [] }), /expected one replay run/);
	assert.match(luaReplayFailureMessage({ runs: [{ run_id: 'candidate-1', status: 'running' }] }), /status running/);
	assert.match(luaReplayFailureMessage({ run_count: 1, completed_count: 1, failed_count: 0, skipped_count: 0, runs: [{ run_id: 'candidate-1', status: 'completed' }] }), /without a result_path/);
	assert.match(luaReplayFailureMessage({ run_count: 1, completed_count: 0, failed_count: 0, skipped_count: 0, runs: [{ run_id: 'candidate-1', status: 'completed', result_path: '/job/output/result.json' }] }), /inconsistent run counts/);
	assert.equal(supportsReplayResults('lua-validate'), false);
	assert.equal(supportsReplayResults('run'), true);
});
