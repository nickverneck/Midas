import { mkdir, open, link, unlink, realpath, stat } from 'node:fs/promises';
import { relative, resolve, sep, isAbsolute } from 'node:path';
import { randomUUID } from 'node:crypto';

/** @typedef {{run_id: string, run_index: number, rank: number, fee_scenario: string, parameter_values: Record<string, unknown>, overrides: Record<string, unknown>, metric_value?: number|null}} RankingRow */
/** @typedef {{schema_version: number, sweep_id: string, metric: string, source_summary: string, rows: RankingRow[]}} RankingDocument */
/** @typedef {{schema_version: number, spec: {sweep_id: string}, children: Array<{run_id: string, run_index: number, parameter_values: Record<string, unknown>, overrides: Record<string, unknown>, resolved_strategy: Record<string, unknown>}>}} SweepPlan */
/** @typedef {{rankingArtifact: string, sourceSummary: string, sourcePlan: string}} CandidateProvenancePaths */
/** @typedef {{schema_version: 1, provenance: {ranking_artifact: string, source_summary: string, source_plan: string, sweep_id: string, run_id: string, rank: number, metric: string, metric_value: number|null, fee_scenario: string}, execution_config: Record<string, unknown>}} CandidateEnvelope */

export const MAX_CANDIDATE_ARTIFACT_BYTES = 16 * 1024 * 1024;

/** Canonicalize a server-selected artifact and reject symlink escapes. */
/** @param {string} path @param {string} root @returns {Promise<string>} */
export async function canonicalCandidateArtifact(path, root) {
	const [canonicalRoot, canonicalPath] = await Promise.all([realpath(root), realpath(path)]);
	const rel = relative(canonicalRoot, canonicalPath);
	if (rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw new Error('Native candidate artifact escaped its managed directory.');
	if (!(await stat(canonicalPath)).isFile()) throw new Error('Native candidate artifact is not a regular file.');
	return canonicalPath;
}

/** Resolve a candidate destination through symlinks and keep it below the managed root. */
/** @param {string} path @param {string} root @returns {Promise<string>} */
export async function canonicalCandidateDirectory(path, root) {
	const [canonicalRoot, canonicalPath] = await Promise.all([realpath(root), realpath(path)]);
	const rel = relative(canonicalRoot, canonicalPath);
	if (rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw new Error('Candidate destination is outside the managed strategy directory.');
	if (!(await stat(canonicalPath)).isDirectory()) throw new Error('Candidate destination is not a directory.');
	return canonicalPath;
}

/** Verify the ranking row against the native plan and return the exact TUI import envelope. */
/** @param {RankingDocument} ranking @param {SweepPlan} plan @param {RankingRow} row @param {CandidateProvenancePaths} provenance @returns {CandidateEnvelope} */
export function makeStrategyCandidate(ranking, plan, row, provenance) {
	if (ranking?.schema_version !== 1 || plan?.schema_version !== 1) throw new Error('Unsupported native ranking or sweep plan schema.');
	if (!Array.isArray(ranking.rows) || ranking.rows.length > 100 || ranking.rows.some(candidate => !candidate || typeof candidate !== 'object' || Array.isArray(candidate) || typeof candidate.run_id !== 'string' || !Number.isSafeInteger(candidate.run_index)) || !Array.isArray(plan.children) || plan.children.length > 10000) throw new Error('Native ranking or sweep plan has an unsupported schema.');
	if (!ranking.sweep_id || ranking.sweep_id !== plan.spec?.sweep_id) throw new Error('Ranking and plan sweep IDs differ.');
	const ranked = ranking.rows.find(candidate => candidate.run_id === row?.run_id && candidate.run_index === row?.run_index);
	if (!ranked || JSON.stringify(ranked) !== JSON.stringify(row)) throw new Error('Candidate row does not match the ranking artifact.');
	const child = plan.children.find(candidate => candidate.run_id === row.run_id);
	if (!child) throw new Error(`Candidate ${row.run_id} is absent from the source plan.`);
	if (child.run_index !== row.run_index) throw new Error('Candidate run index does not match source plan.');
	if (JSON.stringify(child.parameter_values) !== JSON.stringify(row.parameter_values) || JSON.stringify(child.overrides) !== JSON.stringify(row.overrides)) throw new Error('Candidate parameters do not match source plan.');
	if (!child.resolved_strategy || typeof child.resolved_strategy !== 'object' || Array.isArray(child.resolved_strategy)) throw new Error('Candidate has no resolved strategy configuration.');
	const config = child.resolved_strategy;
	if (typeof config.kind !== 'string' || typeof config.native_strategy !== 'string' || !config.native_hma || typeof config.native_hma !== 'object' || !config.native_ema || typeof config.native_ema !== 'object') throw new Error('Candidate execution configuration is missing required native strategy fields.');
	if (config.order_qty !== undefined && (typeof config.order_qty !== 'number' || !Number.isInteger(config.order_qty) || config.order_qty < 1)) throw new Error('Candidate order_qty must be a positive integer.');
	if (config.blockout_minutes_before_close !== undefined && (typeof config.blockout_minutes_before_close !== 'number' || !Number.isFinite(config.blockout_minutes_before_close) || config.blockout_minutes_before_close < 0)) throw new Error('Candidate blockout_minutes_before_close must be finite and non-negative.');
	if (typeof row.rank !== 'number' || !Number.isSafeInteger(row.rank) || row.rank < 1 || (row.metric_value !== null && row.metric_value !== undefined && (typeof row.metric_value !== 'number' || !Number.isFinite(row.metric_value)))) throw new Error('Candidate ranking metadata is invalid.');
	return {
		schema_version: 1,
		provenance: {
			ranking_artifact: provenance.rankingArtifact,
			source_summary: provenance.sourceSummary,
			source_plan: provenance.sourcePlan,
			sweep_id: ranking.sweep_id,
			run_id: row.run_id,
			rank: row.rank,
			metric: ranking.metric,
			metric_value: row.metric_value ?? null,
			fee_scenario: row.fee_scenario
		},
		execution_config: child.resolved_strategy
	};
}

/** Bounded artifact read via an already canonical, server-selected path. */
/** @param {string} path @returns {Promise<any>} */
export async function readCandidateJson(path) {
	const file = await open(path, 'r');
	try {
		const info = await file.stat();
		if (!info.isFile() || info.size > MAX_CANDIDATE_ARTIFACT_BYTES) throw new Error('Native candidate artifact is too large or is not a regular file.');
		const bytes = Buffer.alloc(info.size);
		let offset = 0;
		while (offset < bytes.length) { const { bytesRead } = await file.read(bytes, offset, bytes.length - offset, offset); if (!bytesRead) break; offset += bytesRead; }
		if (offset !== bytes.length) throw new Error('Native candidate artifact changed while being read.');
		return JSON.parse(bytes.toString('utf8'));
	} finally { await file.close(); }
}

/** Atomic, no-clobber write using an exclusive temp and hard-link publication. */
/** @param {string} directory @param {CandidateEnvelope} candidate @returns {Promise<string>} */
export async function saveCandidateNoClobber(directory, candidate) {
	await mkdir(directory, { recursive: true, mode: 0o700 });
	/** @param {unknown} value */ const safe = value => String(value).replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 100) || 'candidate';
	const stem = `${safe(candidate.provenance.sweep_id)}-${safe(candidate.provenance.run_id)}`;
	const bytes = Buffer.from(`${JSON.stringify(candidate, null, 2)}\n`);
	/** @type {string} */ let temporary = '';
	/** @type {import('node:fs/promises').FileHandle|undefined} */ let handle;
	for (let attempt = 0; attempt < 20; attempt++) {
		temporary = resolve(directory, `.${stem}.${process.pid}.${randomUUID()}.tmp`);
		try { handle = await open(temporary, 'wx', 0o600); break; } catch (error) { if (!isNodeError(error, 'EEXIST')) throw error; }
	}
	if (!handle) throw new Error('Unable to reserve a temporary candidate file.');
	try {
		await handle.writeFile(bytes); await handle.sync(); await handle.close(); handle = undefined;
		for (let suffix = 1; suffix < 10000; suffix++) {
			const path = resolve(directory, suffix === 1 ? `${stem}.json` : `${stem}-import${suffix}.json`);
			try { await link(temporary, path); return path; } catch (error) { if (!isNodeError(error, 'EEXIST')) throw error; }
		}
		throw new Error('Candidate import filename space is exhausted.');
	} finally { if (handle) await handle.close().catch(() => {}); await unlink(temporary).catch(() => {}); }
}

/** @param {unknown} error @param {string} code */
function isNodeError(error, code) { return typeof error === 'object' && error !== null && 'code' in error && error.code === code; }
