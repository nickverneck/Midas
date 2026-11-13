import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { canonicalCandidateArtifact, canonicalCandidateDirectory, makeStrategyCandidate, saveCandidateNoClobber } from '../src/lib/server/strategy-candidate.mjs';

const nativeLike = () => {
	const row = { rank: 2, run_id: 'ema-fast5-slow20', run_index: 1, fee_scenario: 'primary', parameter_values: { 'native_ema.fast_length': 5 }, overrides: { 'native_ema.fast_length': 5 }, metric_value: 123.5 };
	const strategy = { kind: 'native', native_strategy: 'ema_cross', order_qty: 1, native_hma: {}, native_ema: { fast_length: 5, slow_length: 20 } };
	return {
		row,
		ranking: { schema_version: 1, sweep_id: 'sweep-1', metric: 'net-pnl', source_summary: '/artifacts/sweep-summary.json', rows: [row] },
		plan: { schema_version: 1, spec: { sweep_id: 'sweep-1' }, children: [{ run_id: row.run_id, run_index: row.run_index, parameter_values: row.parameter_values, overrides: row.overrides, resolved_strategy: strategy }] }
	};
};

test('native ranking row and plan produce the exact TUI schema-1 envelope', () => {
	const { ranking, plan, row } = nativeLike();
	const result = makeStrategyCandidate(ranking, plan, row, { rankingArtifact: '/artifacts/rank/sweep-ranking.json', sourceSummary: '/artifacts/sweep-summary.json', sourcePlan: '/artifacts/sweep-plan.json' });
	assert.equal(result.schema_version, 1);
	assert.equal(result.provenance.run_id, row.run_id);
	assert.equal(result.provenance.metric, 'net-pnl');
	assert.deepEqual(result.execution_config, plan.children[0].resolved_strategy);
});

test('candidate mismatches and incompatible plan identity are rejected', () => {
	const { ranking, plan, row } = nativeLike();
	assert.throws(() => makeStrategyCandidate(ranking, { ...plan, spec: { sweep_id: 'other' } }, row, {}), /sweep IDs differ/);
	assert.throws(() => makeStrategyCandidate(ranking, { ...plan, children: [{ ...plan.children[0], overrides: { changed: true } }] }, row, {}), /parameters do not match/);
	assert.throws(() => makeStrategyCandidate(ranking, plan, { ...row, rank: 0 }, {}), /ranking artifact/);
	const malformedPlan = structuredClone(plan);
	delete malformedPlan.children[0].resolved_strategy.native_ema;
	assert.throws(() => makeStrategyCandidate(ranking, malformedPlan, row, {}), /missing required native strategy fields/);
	assert.throws(() => makeStrategyCandidate({ ...ranking, rows: [null] }, plan, row, {}), /unsupported schema/);
});

test('candidate save publishes a complete file without clobbering an existing import', async () => {
	const dir = await mkdtemp(join(tmpdir(), 'midas-candidates-'));
	try {
		const { ranking, plan, row } = nativeLike();
		const candidate = makeStrategyCandidate(ranking, plan, row, { rankingArtifact: '/artifacts/rank/sweep-ranking.json', sourceSummary: '/artifacts/sweep-summary.json', sourcePlan: '/artifacts/sweep-plan.json' });
		const [first, second] = await Promise.all([saveCandidateNoClobber(dir, candidate), saveCandidateNoClobber(dir, candidate)]);
		assert.notEqual(first, second);
		assert.deepEqual(JSON.parse(await readFile(first, 'utf8')), candidate);
		assert.deepEqual(JSON.parse(await readFile(second, 'utf8')), candidate);
		assert.equal((await readdir(dir)).filter(name => name.endsWith('.json')).length, 2);
	} finally { await rm(dir, { recursive: true, force: true }); }
});

test('canonical artifact and candidate destination checks reject symlink escapes', async () => {
	const root = await mkdtemp(join(tmpdir(), 'midas-managed-'));
	const outside = await mkdtemp(join(tmpdir(), 'midas-outside-'));
	try {
		const secret = join(outside, 'sweep-plan.json');
		await writeFile(secret, '{}');
		const fileLink = join(root, 'sweep-plan.json');
		const dirLink = join(root, 'candidates');
		await symlink(secret, fileLink);
		await symlink(outside, dirLink, 'dir');
		await assert.rejects(canonicalCandidateArtifact(fileLink, root), /escaped/);
		await assert.rejects(canonicalCandidateDirectory(dirLink, root), /outside/);
	} finally { await rm(root, { recursive: true, force: true }); await rm(outside, { recursive: true, force: true }); }
});
