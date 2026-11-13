import test from 'node:test';
import assert from 'node:assert/strict';
import { BridgeError } from '../src/lib/server/chart-cache.ts';
import { decodeSweepSpecId, isStrategyRankMetric, StrategyInputError } from '../src/lib/server/strategy-policy.ts';
import { isRecoverableJobRecord, rankingArtifactPath } from '../src/lib/server/strategy-artifacts.mjs';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';

test('sweep IDs decode only canonical relative catalog paths', () => {
	const safe = Buffer.from('replay-sweeps/example/specs/base.json').toString('base64url');
	assert.equal(decodeSweepSpecId(safe), 'replay-sweeps/example/specs/base.json');
	for (const value of ['../secret.json', '/tmp/secret.json', 'replay-sweeps/../secret.json', '']) {
		const id = Buffer.from(value).toString('base64url');
		assert.throws(() => decodeSweepSpecId(id), (error) => error instanceof StrategyInputError && error.status === 400);
	}
	assert.throws(() => decodeSweepSpecId('not-canonical!'), StrategyInputError);
});

test('ranking accepts only native ranking metric names', () => {
	assert.equal(isStrategyRankMetric('robustness'), true);
	assert.equal(isStrategyRankMetric('net-pnl'), true);
	for (const value of ['--output', '../file', '', null, 4]) assert.equal(isStrategyRankMetric(value), false);
});

test('rank artifact uses the TUI-discoverable conventional filename and is unique per job', () => {
	const output = '/repo/trader/.run/webui-strategy-jobs/run/output';
	const first = rankingArtifactPath(output, '11111111-1111-4111-8111-111111111111');
	const second = rankingArtifactPath(output, '22222222-2222-4222-8222-222222222222');
	assert.equal(first, resolve(output, '11111111-1111-4111-8111-111111111111/sweep-ranking.json'));
	assert.notEqual(first, second);
	assert.throws(() => rankingArtifactPath(output, '../escape'));
});

test('completed rank job with job-local spec survives persisted-record restart validation', async () => {
	const temp = await mkdtemp(resolve(tmpdir(), 'midas-strategy-restart-'));
	try {
		const jobsRoot = resolve(temp, 'webui-strategy-jobs');
		const rankJobId = '11111111-1111-4111-8111-111111111111';
		const sourceJobId = '22222222-2222-4222-8222-222222222222';
		const rankDir = resolve(jobsRoot, rankJobId);
		const sourceOutput = resolve(jobsRoot, sourceJobId, 'output');
		await mkdir(rankDir, { recursive: true });
		await mkdir(sourceOutput, { recursive: true });
		const record = { id: rankJobId, operation: 'rank', status: 'complete', specPath: resolve(rankDir, 'sweep.json'), outputDir: sourceOutput,
			result: { artifactPath: rankingArtifactPath(sourceOutput, rankJobId), sourceSummary: resolve(sourceOutput, 'sweep-summary.json'), sourcePlan: resolve(sourceOutput, 'sweep-plan.json') } };
		await writeFile(record.specPath, '{}');
		await writeFile(resolve(rankDir, 'job.json'), JSON.stringify(record));
		const reloaded = JSON.parse(await readFile(resolve(rankDir, 'job.json'), 'utf8'));
		assert.equal(isRecoverableJobRecord(reloaded, rankJobId, jobsRoot), true);
		assert.equal(reloaded.result.sourceSummary, resolve(sourceOutput, 'sweep-summary.json'));
		assert.equal(reloaded.result.sourcePlan, resolve(sourceOutput, 'sweep-plan.json'));
		assert.equal(reloaded.result.artifactPath, resolve(sourceOutput, `${rankJobId}/sweep-ranking.json`));
		assert.equal(isRecoverableJobRecord({ ...reloaded, specPath: '/tmp/original-spec.json' }, rankJobId, jobsRoot), false);
	} finally {
		await rm(temp, { recursive: true, force: true });
	}
});
