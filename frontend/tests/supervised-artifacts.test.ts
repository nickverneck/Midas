// Bun supplies the runtime test types; the Svelte app's checker does not.
// Keep this server-only Bun test executable without adding a browser test
// dependency to the frontend bundle.
// @ts-nocheck
import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
	snapshotCompletionArtifactPair,
	verifyCompletionArtifactPair
} from '../src/lib/server/supervised';

const temporaryDirectories: string[] = [];

const makeMetaGateRun = () => {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), 'midas-artifact-test-'));
	temporaryDirectories.push(root);
	const policy = Buffer.from('{"schema_version":"meta-gate-policy-v1","weights":[1,2,3]}\n');
	const metrics = Buffer.from('{"schema_version":"meta-gate-metrics-v1","net_pnl":12.5}\n');
	const hash = (value: Buffer) => createHash('sha256').update(value).digest('hex');
	const manifest = Buffer.from(
		JSON.stringify({
			schema_version: 'meta-gate-artifacts-v1',
			status: 'complete',
			generation: 'test-generation',
			policy_file: 'policy.json',
			metrics_file: 'metrics.json',
			policy_sha256: hash(policy),
			metrics_sha256: hash(metrics)
		})
	);
	fs.writeFileSync(path.join(root, 'policy.json'), policy);
	fs.writeFileSync(path.join(root, 'metrics.json'), metrics);
	fs.writeFileSync(path.join(root, '.meta-gate-manifest.json'), manifest);
	return { root, policy, metrics, manifest };
};

const cleanup = () => {
	for (const directory of temporaryDirectories.splice(0)) {
		fs.rmSync(directory, { recursive: true, force: true });
	}
};

test('snapshots the verified bytes into a read-only sibling', () => {
	const source = makeMetaGateRun();
	try {
		const snapshot = snapshotCompletionArtifactPair({
			directory: source.root,
			manifestFile: '.meta-gate-manifest.json',
			expectedSchema: 'meta-gate-artifacts-v1',
			expectedStatus: 'complete',
			label: 'test meta-gate'
		});

		expect(snapshot.sourcePolicyPath).toBe(path.join(source.root, 'policy.json'));
		expect(snapshot.policyPath).not.toBe(snapshot.sourcePolicyPath);
		expect(fs.readFileSync(snapshot.policyPath)).toEqual(source.policy);
		expect(fs.readFileSync(snapshot.metricsPath)).toEqual(source.metrics);
		expect(fs.readFileSync(snapshot.manifestPath)).toEqual(source.manifest);
		expect(fs.statSync(snapshot.policyPath).mode & 0o777).toBe(0o400);
		expect(fs.statSync(snapshot.snapshotDirectory).mode & 0o777).toBe(0o500);

		const verifiedSnapshot = verifyCompletionArtifactPair({
			directory: snapshot.snapshotDirectory,
			manifestFile: '.meta-gate-manifest.json',
			expectedSchema: 'meta-gate-artifacts-v1',
			expectedStatus: 'complete',
			label: 'test snapshot'
		});
		expect(verifiedSnapshot.policyPath).toBe(snapshot.policyPath);
	} finally {
		cleanup();
	}
});

test('the snapshot is unaffected when the source artifact is replaced', () => {
	const source = makeMetaGateRun();
	try {
		const snapshot = snapshotCompletionArtifactPair({
			directory: source.root,
			manifestFile: '.meta-gate-manifest.json',
			expectedSchema: 'meta-gate-artifacts-v1',
			expectedStatus: 'complete',
			label: 'test meta-gate'
		});

		fs.writeFileSync(path.join(source.root, 'policy.json'), '{"attacker":true}\n');
		expect(fs.readFileSync(snapshot.policyPath)).toEqual(source.policy);
		expect(() =>
			verifyCompletionArtifactPair({
				directory: source.root,
				manifestFile: '.meta-gate-manifest.json',
				expectedSchema: 'meta-gate-artifacts-v1',
				expectedStatus: 'complete',
				label: 'replaced source'
			})
		).toThrow(/does not match its completion manifest/);
	} finally {
		cleanup();
	}
});

test('rejects a symlinked completion artifact', () => {
	if (process.platform === 'win32') return;
	const source = makeMetaGateRun();
	try {
		const outsideDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'midas-outside-'));
		temporaryDirectories.push(outsideDirectory);
		const outside = path.join(outsideDirectory, 'metrics.json');
		fs.writeFileSync(outside, source.metrics);
		fs.unlinkSync(path.join(source.root, 'metrics.json'));
		fs.symlinkSync(outside, path.join(source.root, 'metrics.json'));

		expect(() =>
			snapshotCompletionArtifactPair({
				directory: source.root,
				manifestFile: '.meta-gate-manifest.json',
				expectedSchema: 'meta-gate-artifacts-v1',
				expectedStatus: 'complete',
				label: 'symlinked source'
			})
		).toThrow(/not a regular artifact file|resolves outside/);
	} finally {
		cleanup();
	}
});

test('requires a complete meta-gate manifest status', () => {
	const source = makeMetaGateRun();
	try {
		const manifestPath = path.join(source.root, '.meta-gate-manifest.json');
		const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
		delete manifest.status;
		fs.writeFileSync(manifestPath, JSON.stringify(manifest));

		expect(() =>
			snapshotCompletionArtifactPair({
				directory: source.root,
				manifestFile: '.meta-gate-manifest.json',
				expectedSchema: 'meta-gate-artifacts-v1',
				expectedStatus: 'complete',
				label: 'missing-status meta-gate'
			})
		).toThrow(/not marked complete/);
	} finally {
		cleanup();
	}
});
