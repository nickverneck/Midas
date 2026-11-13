import test from 'node:test';
import assert from 'node:assert/strict';
import { assertNotCancelled, createActiveJobGate, runUnlessCancelled, startUnlessCancelled } from './strategy-job-controls.mjs';

test('active job slot remains reserved while asynchronous setup is in progress', async () => {
	const gate = createActiveJobGate();
	const releaseFirst = gate.reserve();
	assert.equal(typeof releaseFirst, 'function');
	assert.equal(gate.reserve(), undefined, 'a concurrent submission must be rejected');

	await Promise.resolve();
	assert.equal(gate.reserve(), undefined, 'the reservation must span async setup');
	releaseFirst();
	assert.equal(typeof gate.reserve(), 'function', 'a later submission can reserve the released slot');
});

test('a cancellation observed at a child-process boundary prevents the spawn', () => {
	let spawned = false;
	const job = { cancelled: true };
	assert.throws(() => startUnlessCancelled(job, () => { spawned = true; }), /cancelled/);
	assert.equal(spawned, false);
});

test('child-process boundary proceeds for a live job', () => {
	const job = { cancelled: false };
	assert.equal(startUnlessCancelled(job, () => 'started'), 'started');
});

test('cancellation during a successful child prevents the next orchestration step', async () => {
	const job = { cancelled: false };
	let nextStepStarted = false;
	await assert.rejects(runUnlessCancelled(job, async () => {
		job.cancelled = true;
		return 0;
	}), /cancelled/);
	try {
		startUnlessCancelled(job, () => { nextStepStarted = true; });
	} catch { /* A cancelled job must not launch its next child. */ }
	assert.equal(nextStepStarted, false);
});

test('cancellation is rejected at the completion boundary', () => {
	const job = { cancelled: true };
	assert.throws(() => assertNotCancelled(job), /cancelled/);
});
