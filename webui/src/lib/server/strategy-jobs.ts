import { spawn, execFile as execFileCallback, type ChildProcess } from 'node:child_process';
import { promisify } from 'node:util';
import { tmpdir } from 'node:os';
import { randomUUID } from 'node:crypto';
import { mkdir, mkdtemp, readFile, readdir, realpath, rename, rm, stat, writeFile } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import type { RequestEvent } from '@sveltejs/kit';
import { assertLocalRequest, BridgeError } from './chart-cache';
import { decodeSweepSpecId, isStrategyRankMetric, StrategyInputError } from './strategy-policy';
import { assertNotCancelled, createActiveJobGate, runUnlessCancelled, startUnlessCancelled } from './strategy-job-controls.mjs';
import { readStrategyCsv, readStrategyText } from './strategy-csv.mjs';
import { isRecoverableJobRecord, isWithinDirectory, rankingArtifactPath } from './strategy-artifacts.mjs';
import { createLuaPrototypeSweep, validateLuaPrototypeSource } from './lua-prototype.mjs';
import { luaReplayFailureMessage, supportsReplayResults } from './lua-replay-lifecycle.mjs';
import { buildSweepHeatmap } from './sweep-heatmap.mjs';
import { createEditedSweep } from './strategy-sweep-editor.mjs';
import { canonicalCandidateArtifact, canonicalCandidateDirectory, makeStrategyCandidate, readCandidateJson, saveCandidateNoClobber } from './strategy-candidate.mjs';

const repo = resolve(process.cwd(), '..');
const traderDir = resolve(repo, 'trader');
const specsRoot = resolve(traderDir, '.run');
const jobsRoot = resolve(specsRoot, 'webui-strategy-jobs');
const createdSpecsRoot = resolve(specsRoot, 'webui-created');
const maxOutput = 96 * 1024;
const execFileAsync = promisify(execFileCallback);
const jobIds = /^[a-f0-9-]{36}$/;

export type StrategyJob = {
	id: string; operation: 'validate' | 'run' | 'rank' | 'lua-validate' | 'lua-run'; sweepId: string; status: 'queued' | 'running' | 'complete' | 'failed' | 'cancelled';
	createdAt: string; updatedAt: string; progress?: unknown; result?: unknown; error?: string; metric?: string;
};
type InternalJob = StrategyJob & { specPath: string; outputDir: string; child?: ChildProcess; cancelled?: boolean; stdout: string; stderr: string };
const jobs = new Map<string, InternalJob>();
const active = new Map<string, ChildProcess>();
const persistence = new Map<string, Promise<void>>();
const activeJobGate = createActiveJobGate();
let initialized: Promise<void> | undefined;

export function assertStrategyRequest(event: RequestEvent): void {
	assertLocalRequest(event);
}

async function initialize(): Promise<void> {
	if (initialized) return initialized;
	initialized = (async () => {
		await mkdir(jobsRoot, { recursive: true });
		for (const entry of await readdir(jobsRoot, { withFileTypes: true })) {
			if (!entry.isDirectory() || !jobIds.test(entry.name)) continue;
			try {
				const saved = JSON.parse(await readFile(resolve(jobsRoot, entry.name, 'job.json'), 'utf8')) as InternalJob;
				if (!isRecoverableJobRecord(saved, entry.name, jobsRoot)) continue;
				if (saved.status === 'running' || saved.status === 'queued') {
					saved.status = 'failed'; saved.error = 'The web server restarted before this job completed.'; saved.updatedAt = new Date().toISOString();
				}
				saved.child = undefined; saved.stdout = ''; saved.stderr = '';
				jobs.set(saved.id, saved);
				await persist(saved);
			} catch { /* Ignore malformed records; never interpret job files as paths. */ }
		}
	})();
	return initialized;
}

async function persist(job: InternalJob): Promise<void> {
	const previous = persistence.get(job.id) ?? Promise.resolve();
	const next = previous.catch(() => undefined).then(() => persistNow(job));
	persistence.set(job.id, next);
	try { await next; } finally { if (persistence.get(job.id) === next) persistence.delete(job.id); }
}

async function persistNow(job: InternalJob): Promise<void> {
	const dir = resolve(jobsRoot, job.id);
	await mkdir(dir, { recursive: true });
	const temp = resolve(dir, 'job.json.tmp');
	const { child: _child, stdout: _stdout, stderr: _stderr, ...publicAndPaths } = job;
	await writeFile(temp, JSON.stringify(publicAndPaths, null, 2), { mode: 0o600 });
	await rename(temp, resolve(dir, 'job.json'));
}

function publicJob(job: InternalJob): StrategyJob {
	const { specPath: _specPath, outputDir: _outputDir, child: _child, stdout: _stdout, stderr: _stderr, cancelled: _cancelled, ...value } = job;
	return value;
}

function contained(root: string, candidate: string): boolean {
	const rel = relative(root, candidate);
	return rel === '' || (!rel.startsWith(`..${sep}`) && rel !== '..' && !isAbsolute(rel));
}

async function discoverSpecs(): Promise<Array<{ id: string; name: string; sweepId: string; datasetViewId?: string }>> {
	const output: Array<{ id: string; name: string; sweepId: string; datasetViewId?: string }> = [];
	async function visit(dir: string): Promise<void> {
		let entries;
		try { entries = await readdir(dir, { withFileTypes: true }); } catch { return; }
		for (const entry of entries) {
			const path = resolve(dir, entry.name);
			if (entry.isSymbolicLink()) continue;
			if (entry.isDirectory()) {
				if (['runs', 'cache', 'outputs', 'results', 'node_modules', 'target'].includes(entry.name) || entry.name.endsWith('-output')) continue;
				await visit(path); continue;
			}
			if (!entry.isFile() || entry.name === 'sweep.json' || !path.endsWith('.json')) continue;
			const rel = relative(specsRoot, path).split(sep);
			if (!rel.includes('specs')) continue;
			try {
				const canonical = await realpath(path);
				if (!contained(specsRoot, canonical)) continue;
				const info = await stat(canonical);
				if (info.size > 2_000_000) continue;
				const spec = JSON.parse(await readFile(canonical, 'utf8'));
				if (typeof spec.sweep_id !== 'string' || typeof spec.name !== 'string' || !spec.base_dataset_view) continue;
				output.push({ id: Buffer.from(relative(specsRoot, canonical)).toString('base64url'), name: spec.name, sweepId: spec.sweep_id, datasetViewId: spec.base_dataset_view.id });
			} catch { /* Only valid native JSON sweep specs appear in the catalog. */ }
		}
	}
	await visit(specsRoot);
	return output.sort((a, b) => a.sweepId.localeCompare(b.sweepId));
}

async function specById(id: unknown): Promise<{ path: string; spec: Record<string, any> }> {
	const decoded = decodeSweepSpecId(id);
	const path = resolve(specsRoot, decoded);
	if (!contained(specsRoot, path) || !path.endsWith('.json') || !relative(specsRoot, path).split(sep).includes('specs')) throw new BridgeError(400, 'Unknown sweepId.');
	let canonical: string;
	try { canonical = await realpath(path); } catch { throw new BridgeError(404, 'Sweep spec no longer exists.'); }
	if (!contained(specsRoot, canonical)) throw new BridgeError(400, 'Unknown sweepId.');
	const value = JSON.parse(await readFile(canonical, 'utf8'));
	if (typeof value.sweep_id !== 'string' || typeof value.name !== 'string' || !value.base_dataset_view) throw new BridgeError(400, 'Sweep spec is not a supported native replay spec.');
	return { path: canonical, spec: value };
}

function binaryPath(): string {
	const configured = process.env.MIDAS_TRADER_BIN;
	return configured ? resolve(configured) : resolve(traderDir, 'target', 'debug', 'trader');
}

function commandPrefix(): string[] {
	const args: string[] = [];
	if (process.env.MIDAS_CONFIG) args.push('--config', resolve(process.env.MIDAS_CONFIG));
	return args;
}

function appendBounded(previous: string, next: Buffer | string): string {
	const value = previous + next.toString();
	return value.length > maxOutput ? value.slice(-maxOutput) : value;
}

async function readProgress(job: InternalJob): Promise<unknown> {
	const path = resolve(job.outputDir, 'sweep-status.json');
	try {
		const canonical = await realpath(path);
		if (!contained(job.outputDir, canonical)) return undefined;
		return JSON.parse(await readFile(canonical, 'utf8'));
	} catch { return undefined; }
}

async function runChild(job: InternalJob, args: string[]): Promise<number> {
	const child = startUnlessCancelled(job, () => spawn(binaryPath(), [...commandPrefix(), ...args], { cwd: traderDir, env: process.env, stdio: ['ignore', 'pipe', 'pipe'], detached: process.platform !== 'win32', windowsHide: true }));
	job.child = child; active.set(job.id, child);
	child.stdout?.on('data', chunk => { job.stdout = appendBounded(job.stdout, chunk); });
	child.stderr?.on('data', chunk => { job.stderr = appendBounded(job.stderr, chunk); });
	const interval = setInterval(async () => {
		job.progress = await readProgress(job);
		job.updatedAt = new Date().toISOString();
		void persist(job);
	}, 1000);
	try {
		return await new Promise<number>((resolveExit, reject) => {
			child.once('error', reject);
			child.once('close', (code, signal) => resolveExit(code ?? (signal ? 128 : 1)));
		});
	} finally {
		clearInterval(interval); active.delete(job.id); job.child = undefined;
	}
}

async function execute(job: InternalJob, sourceSpec: Record<string, any>): Promise<void> {
	if (job.cancelled) {
		job.status = 'cancelled'; job.updatedAt = new Date().toISOString(); await persist(job);
		return;
	}
	job.status = 'running'; job.updatedAt = new Date().toISOString(); await persist(job);
	try {
		if (job.operation === 'validate' || job.operation === 'run' || job.operation === 'lua-validate' || job.operation === 'lua-run') {
			const validate = await runUnlessCancelled(job, () => runChild(job, ['validate-replay-sweep', '--spec', job.specPath]));
			if (validate !== 0) throw new Error(job.stderr.trim() || job.stdout.trim() || `Native validation exited ${validate}.`);
			if (job.operation === 'validate' || job.operation === 'lua-validate') {
				job.result = { validated: true, output: job.stdout.trim() };
			} else {
				const code = await runUnlessCancelled(job, () => runChild(job, ['run-replay-sweep', '--spec', job.specPath]));
				if (code !== 0) throw new Error(job.stderr.trim() || job.stdout.trim() || `Native replay exited ${code}.`);
				assertNotCancelled(job);
				if (job.operation === 'lua-run') {
					const summaryPath = resolve(job.outputDir, 'sweep-summary.json');
					const canonicalSummary = await realpath(summaryPath);
					if (!contained(job.outputDir, canonicalSummary)) throw new Error('Lua replay summary escaped the job output directory.');
					const summary = JSON.parse(await readStrategyText(canonicalSummary, 16 * 1024 * 1024));
					const failure = luaReplayFailureMessage(summary);
					if (failure) throw new Error(failure);
					const resultCandidate = resolve(job.outputDir, summary.runs[0].result_path);
					const canonicalResult = await realpath(resultCandidate);
					if (!contained(job.outputDir, canonicalResult) || !(await stat(canonicalResult)).isFile()) {
						throw new Error('Lua replay result_path does not resolve to a result file inside the job output directory.');
					}
				}
				job.progress = await readProgress(job);
				job.result = { completed: true, output: job.stdout.trim().slice(-maxOutput) };
			}
		} else {
			const summary = resolve(job.outputDir, 'sweep-summary.json');
			const canonical = await realpath(summary);
			if (!contained(job.outputDir, canonical)) throw new Error('Sweep summary escaped the job output directory.');
			const planPath = resolve(job.outputDir, 'sweep-plan.json');
			let canonicalPlan: string | undefined;
			try {
				canonicalPlan = await realpath(planPath);
				if (!contained(job.outputDir, canonicalPlan)) throw new Error('Sweep plan escaped the job output directory.');
			} catch (error) {
				if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
			}
			const rankingPath = rankingArtifactPath(job.outputDir, job.id);
			if (!isWithinDirectory(job.outputDir, rankingPath)) throw new Error('Ranking artifact escaped the replay output directory.');
			await mkdir(resolve(rankingPath, '..'), { recursive: true });
			const args = ['rank-replay-sweep', '--summary', canonical, '--metric', job.metric!, '--limit', '100', '--output', rankingPath];
			if (canonicalPlan) args.push('--plan', canonicalPlan);
			const code = await runUnlessCancelled(job, () => runChild(job, args));
			if (code !== 0) throw new Error(job.stderr.trim() || job.stdout.trim() || `Native ranking exited ${code}.`);
			const canonicalRanking = await realpath(rankingPath);
			if (!contained(job.outputDir, canonicalRanking)) throw new Error('Ranking artifact escaped the replay output directory.');
			job.result = { ranked: true, metric: job.metric, output: job.stdout.trim(), artifactPath: canonicalRanking, sourceSummary: canonical, ...(canonicalPlan ? { sourcePlan: canonicalPlan } : {}) };
		}
		assertNotCancelled(job);
		job.status = 'complete';
	} catch (error) {
		job.status = job.cancelled ? 'cancelled' : 'failed';
		job.error = error instanceof Error ? error.message.slice(0, 2000) : 'Native strategy operation failed.';
	} finally {
		job.updatedAt = new Date().toISOString();
		await persist(job);
	}
}

export async function getStrategyCatalog(event: RequestEvent): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	return Response.json({ sweeps: await discoverSpecs(), jobs: [...jobs.values()].map(publicJob) }, { headers: { 'cache-control': 'no-store' } });
}

/** Create a derivative native spec by editing only its sweep dimensions. The
 * selected source and output path are resolved by the server; callers never
 * provide a filesystem path or native CLI argument. */
export async function createStrategySweep(event: RequestEvent): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	const declaredLength = Number(event.request.headers.get('content-length') ?? 0);
	if (Number.isFinite(declaredLength) && declaredLength > 32_768) throw new BridgeError(413, 'Strategy request body is too large.');
	let rawBody: string;
	try { rawBody = await event.request.text(); } catch { throw new BridgeError(400, 'Request body must be readable JSON.'); }
	if (Buffer.byteLength(rawBody, 'utf8') > 32_768) throw new BridgeError(413, 'Strategy request body is too large.');
	let body: any;
	try { body = JSON.parse(rawBody); } catch { throw new BridgeError(400, 'Request body must be JSON.'); }
	if (!body || typeof body !== 'object' || Array.isArray(body) || Object.keys(body).some(key => !['sourceSweepId', 'name', 'parameters'].includes(key))) {
		throw new BridgeError(400, 'Expected sourceSweepId, name, and parameters only.');
	}
	if ([...jobs.values()].some(job => job.status === 'queued' || job.status === 'running')) throw new BridgeError(409, 'A strategy job is already active.');
	const selected = await specById(body.sourceSweepId);
	const id = randomUUID();
	const sweepId = `webui_${id.replaceAll('-', '')}`;
	const directory = resolve(createdSpecsRoot, id);
	const specDirectory = resolve(directory, 'specs');
	const outputDir = resolve(directory, 'outputs');
	const specPath = resolve(specDirectory, 'replay-sweep.json');
	let draftPath = '';
	try {
		const spec = createEditedSweep(selected.spec, body, outputDir, sweepId);
		await mkdir(specDirectory, { recursive: true });
		await mkdir(outputDir, { recursive: true });
		draftPath = resolve(specDirectory, 'replay-sweep.json.tmp');
		await writeFile(draftPath, JSON.stringify(spec, null, 2), { mode: 0o600, flag: 'wx' });
		const validation = await validateNativeSweep(draftPath);
		if (validation.code !== 0) throw new BridgeError(400, `Native sweep validation failed: ${validation.output || `exit ${validation.code}`}`);
		if (/^Guardrail:/m.test(validation.output)) throw new BridgeError(400, `Native sweep exceeds a guardrail: ${validation.output.split('\n').find(line => line.startsWith('Guardrail:'))}`);
		if (/large-sweep confirmation threshold/i.test(validation.output)) throw new BridgeError(400, 'Native sweep requires large-sweep confirmation and cannot be created through this bounded editor.');
		await rename(draftPath, specPath);
		return Response.json({ id: Buffer.from(relative(specsRoot, specPath)).toString('base64url'), name: spec.name, sweepId: spec.sweep_id, datasetViewId: selected.spec.base_dataset_view?.id, sourceSweepId: selected.spec.sweep_id, parameters: spec.parameters }, { status: 201, headers: { 'cache-control': 'no-store' } });
	} catch (error) {
		await rm(directory, { recursive: true, force: true }).catch(() => undefined);
		if (error instanceof BridgeError) throw error;
		throw new BridgeError(400, error instanceof Error ? error.message : 'Unable to create native sweep spec.');
	} finally {
		if (draftPath) await rm(draftPath, { force: true }).catch(() => undefined);
	}
}

function validateNativeSweep(specPath: string): Promise<{ code: number; output: string }> {
	return new Promise((resolveResult, reject) => {
		const child = spawn(binaryPath(), [...commandPrefix(), 'validate-replay-sweep', '--spec', specPath], { cwd: traderDir, env: process.env, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
		let output = '';
		const append = (chunk: Buffer) => { output = appendBounded(output, chunk); };
		child.stdout?.on('data', append); child.stderr?.on('data', append);
		const timeout = setTimeout(() => { child.kill('SIGKILL'); reject(new Error('Native sweep validation timed out.')); }, 30_000);
		child.once('error', error => { clearTimeout(timeout); reject(error); });
		child.once('close', (code, signal) => { clearTimeout(timeout); resolveResult({ code: code ?? (signal ? 128 : 1), output: output.trim() }); });
	});
}

async function submitStrategyJobRequest(event: RequestEvent, luaOnly = false): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	let body: any;
	const declaredLength = Number(event.request.headers.get('content-length') ?? 0);
	if (Number.isFinite(declaredLength) && declaredLength > 32_768) throw new BridgeError(413, 'Strategy request body is too large.');
	let rawBody: string;
	try { rawBody = await event.request.text(); } catch { throw new BridgeError(400, 'Request body must be readable JSON.'); }
	if (Buffer.byteLength(rawBody, 'utf8') > 32_768) throw new BridgeError(413, 'Strategy request body is too large.');
	try { body = JSON.parse(rawBody); } catch { throw new BridgeError(400, 'Request body must be JSON.'); }
	if (!body || typeof body !== 'object' || Array.isArray(body)) throw new BridgeError(400, 'Request body must be a JSON object.');
	const keys = Object.keys(body);
	if (luaOnly && !['lua-validate', 'lua-run'].includes(body.operation)) throw new BridgeError(400, 'Lua prototype operation must be lua-validate or lua-run.');
	if (luaOnly && keys.some(key => !['operation', 'sweepId', 'luaSource'].includes(key))) throw new BridgeError(400, 'Unsupported Lua prototype request field.');
	if (!luaOnly && keys.some(key => !['operation', 'sweepId', 'sourceJobId', 'metric'].includes(key))) throw new BridgeError(400, 'Unsupported strategy request field.');
	if (luaOnly ? !['lua-validate', 'lua-run'].includes(body.operation) : !['validate', 'run', 'rank'].includes(body.operation)) throw new BridgeError(400, 'Unsupported strategy operation.');
	const isLua = body.operation === 'lua-validate' || body.operation === 'lua-run';
	if (isLua) {
		const sourceError = validateLuaPrototypeSource(body.luaSource);
		if (sourceError) throw new BridgeError(sourceError.includes('12 KiB') ? 413 : 400, sourceError);
	} else if (body.luaSource !== undefined) throw new BridgeError(400, 'luaSource is only accepted for Lua prototype operations.');
	let selected: { path: string; spec: Record<string, any> };
	let outputDir: string;
	if (body.operation === 'rank') {
		if (typeof body.sourceJobId !== 'string' || !jobIds.test(body.sourceJobId)) throw new BridgeError(400, 'rank requires a completed sourceJobId.');
		const source = jobs.get(body.sourceJobId);
		if (!source || source.operation !== 'run' || source.status !== 'complete') throw new BridgeError(409, 'Ranking requires a completed replay job from this web UI.');
		selected = await specById(body.sweepId);
		if (selected.spec.sweep_id !== source.sweepId) throw new BridgeError(400, 'sourceJobId does not belong to this sweep.');
		outputDir = source.outputDir;
	} else {
		selected = await specById(body.sweepId);
		outputDir = '';
	}
	if (body.operation === 'rank' && !isStrategyRankMetric(body.metric)) throw new BridgeError(400, 'Unsupported ranking metric.');
	if ([...jobs.values()].some(job => job.status === 'queued' || job.status === 'running')) throw new BridgeError(409, 'A strategy job is already active.');
	const releaseSlot = activeJobGate.reserve();
	if (!releaseSlot) throw new BridgeError(409, 'A strategy job is already active.');
	try {
		const id = randomUUID();
		const dir = resolve(jobsRoot, id);
		await mkdir(dir, { recursive: true });
		if (body.operation !== 'rank') {
			outputDir = resolve(dir, 'output');
			const spec = isLua ? createLuaPrototypeSweep(selected.spec, outputDir, body.luaSource) : { ...selected.spec, output_dir: outputDir };
			await writeFile(resolve(dir, 'sweep.json'), JSON.stringify(spec, null, 2), { mode: 0o600 });
		} else {
			const summary = resolve(outputDir, 'sweep-summary.json');
			try { await realpath(summary); } catch { throw new BridgeError(409, 'Completed replay output has no sweep summary to rank.'); }
		}
		// Every persisted job owns its native spec. Catalog specs can move or be
		// removed after submission, and restart recovery only accepts job-local paths.
		const specPath = resolve(dir, 'sweep.json');
		if (body.operation === 'rank') await writeFile(specPath, JSON.stringify(selected.spec, null, 2), { mode: 0o600 });
		const job: InternalJob = { id, operation: body.operation, sweepId: selected.spec.sweep_id, status: 'queued', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), specPath, outputDir, metric: body.operation === 'rank' ? body.metric : undefined, stdout: '', stderr: '' };
		jobs.set(id, job);
		try {
			await persist(job);
		} catch (error) {
			jobs.delete(id);
			throw error;
		}
		void execute(job, selected.spec);
		return Response.json(publicJob(job), { status: 202, headers: { 'cache-control': 'no-store' } });
	} finally {
		releaseSlot();
	}
}

export async function submitStrategyJob(event: RequestEvent): Promise<Response> {
	return submitStrategyJobRequest(event);
}

export async function submitLuaPrototypeJob(event: RequestEvent): Promise<Response> {
	return submitStrategyJobRequest(event, true);
}

export async function getStrategyJob(event: RequestEvent): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	const id = event.params.id ?? '';
	if (!jobIds.test(id)) throw new BridgeError(404, 'Strategy job not found.');
	const job = jobs.get(id);
	if (!job) throw new BridgeError(404, 'Strategy job not found.');
	if (job.status === 'running') job.progress = await readProgress(job);
	return Response.json(publicJob(job), { headers: { 'cache-control': 'no-store' } });
}

// Read the CSV artifacts emitted by the native replay runner. Paths come from
// sweep-summary.json and are accepted only when they resolve inside this job's output directory.
export async function getStrategyReplayResults(event: RequestEvent): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	const id = event.params.id ?? '';
	if (!jobIds.test(id)) throw new BridgeError(404, 'Strategy job not found.');
	const job = jobs.get(id);
	if (!job || !supportsReplayResults(job.operation)) throw new BridgeError(404, 'Replay job not found.');
	if (job.status !== 'complete') throw new BridgeError(409, 'Replay artifacts are available after the job completes.');
	const outputDir = await realpath(job.outputDir);
	const summaryPath = resolve(outputDir, 'sweep-summary.json');
	const canonicalSummary = await realpath(summaryPath);
	if (!contained(outputDir, canonicalSummary)) throw new BridgeError(500, 'Replay summary escaped its output directory.');
const summary = JSON.parse(await readStrategyText(canonicalSummary, 16 * 1024 * 1024));
	const completed = (summary.runs ?? []).filter((run: any) => run.status === 'completed' && typeof run.result_path === 'string');
	const runs = [];
	for (const run of completed) {
		const resultPath = await realpath(resolve(run.result_path));
		if (!contained(outputDir, resultPath)) continue;
		const result = JSON.parse(await readStrategyText(resultPath, 4 * 1024 * 1024));
		const runDir = resolve(resultPath, '..');
		const readCsv = async (name: string) => {
			const path = await realpath(resolve(runDir, name));
			if (!contained(outputDir, path)) return [];
			return readStrategyCsv(path);
		};
		const [equity, trades] = await Promise.all([readCsv('equity.csv'), readCsv('trades.csv')]);
		runs.push({ runId: run.run_id, metrics: { grossPnl: run.gross_pnl, netPnl: run.net_pnl, fees: run.fees, maxDrawdown: run.max_drawdown, tradeCount: run.trade_count }, equity: equity.map((row: any) => ({ timestampNs: row.timestamp_ns, equity: Number(row.equity), cumulativeNetPnl: Number(row.cumulative_net_pnl) })).filter((point: any) => Number.isFinite(point.equity)), trades, result });
	}
	return Response.json({ sweepId: job.sweepId, runs }, { headers: { 'cache-control': 'no-store' } });
}

/** Return the native plan/summary grid for one completed replay job. */
export async function getStrategyHeatmap(event: RequestEvent): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	const id = event.params.id ?? '';
	if (!jobIds.test(id)) throw new BridgeError(404, 'Strategy job not found.');
	const job = jobs.get(id);
	if (!job || job.operation !== 'run') throw new BridgeError(404, 'Replay job not found.');
	if (job.status !== 'complete') throw new BridgeError(409, 'Sweep heatmap is available after the replay job completes.');
	const outputDir = await realpath(job.outputDir);
	const readArtifact = async (name: string) => {
		const candidate = resolve(outputDir, name);
		const canonical = await realpath(candidate);
		if (!contained(outputDir, canonical)) throw new BridgeError(500, `Sweep ${name} escaped its output directory.`);
		return JSON.parse(await readStrategyText(canonical, 16 * 1024 * 1024));
	};
	const [plan, summary] = await Promise.all([readArtifact('sweep-plan.json'), readArtifact('sweep-summary.json')]);
	const url = new URL(event.request.url);
	try {
		const model = buildSweepHeatmap(plan, summary, url.searchParams.get('x') ?? undefined, url.searchParams.get('y') ?? undefined, url.searchParams.get('metric') ?? undefined);
		return Response.json(model, { headers: { 'cache-control': 'no-store' } });
	} catch (error) {
		throw new BridgeError(400, error instanceof Error ? error.message : 'Unable to build sweep heatmap.');
	}
}

/** Build previewable TUI import envelopes from the exact completed native ranking and its plan. */
async function candidatesForRankJob(job: InternalJob): Promise<Array<{ row: any; candidate: any }>> {
	if (job.operation !== 'rank' || job.status !== 'complete') throw new BridgeError(409, 'Candidate import is available after ranking completes.');
	const outputDir = await realpath(job.outputDir);
	if (!contained(jobsRoot, outputDir)) throw new BridgeError(400, 'Ranking output is outside the managed strategy job directory.');
	const rankingPath = await canonicalCandidateArtifact(rankingArtifactPath(outputDir, job.id), outputDir);
	const ranking = await readCandidateJson(rankingPath);
	if (ranking?.schema_version !== 1 || !Array.isArray(ranking.rows) || ranking.rows.length > 100 || ranking.rows.some((row: any) => !row || typeof row !== 'object' || Array.isArray(row) || typeof row.run_id !== 'string' || !Number.isSafeInteger(row.run_index) || !Number.isSafeInteger(row.rank))) throw new BridgeError(400, 'Native ranking has malformed or excessive candidate rows.');
	const result = job.result as { sourceSummary?: unknown } | undefined;
	if (typeof result?.sourceSummary !== 'string' || !isAbsolute(result.sourceSummary)) throw new BridgeError(409, 'Ranking job has no verified source summary.');
	const sourceSummary = await canonicalCandidateArtifact(result.sourceSummary, jobsRoot);
	if (!contained(jobsRoot, sourceSummary) || !sourceSummary.endsWith('sweep-summary.json')) throw new BridgeError(400, 'Source summary is outside the managed strategy job directory.');
	const sourcePlan = await canonicalCandidateArtifact(resolve(sourceSummary, '..', 'sweep-plan.json'), jobsRoot);
	if (!contained(jobsRoot, sourcePlan)) throw new BridgeError(400, 'Sweep plan is outside the managed strategy job directory.');
	const plan = await readCandidateJson(sourcePlan);
	if (typeof ranking.source_summary !== 'string' || ranking.source_summary.length > 4096) throw new BridgeError(400, 'Native ranking has no source summary provenance.');
	const rankingSummary = await canonicalCandidateArtifact(isAbsolute(ranking.source_summary) ? ranking.source_summary : resolve(outputDir, ranking.source_summary), jobsRoot);
	if (rankingSummary !== sourceSummary) throw new BridgeError(400, 'Ranking source summary does not match its completed replay job.');
	if (!plan?.spec || !Array.isArray(plan.children) || plan.children.length > 10000 || !plan.spec.base_dataset_view) throw new BridgeError(400, 'Native sweep plan has an unsupported schema.');
	// Re-expand the saved native spec with the native validator. This checks the
	// strategy configs and gives us authoritative resolved children; no replay or
	// engine connection is started by validate-replay-sweep.
	let sourceRun: InternalJob | undefined;
	for (const candidate of jobs.values()) {
		if (candidate.operation !== 'run' || candidate.status !== 'complete' || candidate.sweepId !== job.sweepId) continue;
		try {
			const candidateOutput = await realpath(candidate.outputDir);
			if (resolve(candidateOutput, 'sweep-summary.json') === sourceSummary) { sourceRun = candidate; break; }
		} catch { /* Ignore stale managed job records. */ }
	}
	if (!sourceRun) throw new BridgeError(409, 'Completed source replay job for this ranking is unavailable.');
	const sourceJobRoot = await realpath(resolve(jobsRoot, sourceRun.id));
	const sourceSpecPath = await canonicalCandidateArtifact(sourceRun.specPath, sourceJobRoot);
	const sourceSpec = JSON.parse(await readStrategyText(sourceSpecPath, 2_000_000));
	const validationDir = await mkdtemp(resolve(tmpdir(), 'midas-native-candidate-'));
	let nativePlan: any;
	try {
		const specPath = resolve(validationDir, `${randomUUID()}.json`);
		const expandedPath = resolve(validationDir, `${randomUUID()}-plan.json`);
		await writeFile(specPath, JSON.stringify(sourceSpec), { mode: 0o600, flag: 'wx' });
		try {
			await execFileAsync(binaryPath(), [...commandPrefix(), 'validate-replay-sweep', '--spec', specPath, '--output', expandedPath], { cwd: traderDir, timeout: 15_000, maxBuffer: maxOutput, windowsHide: true });
			nativePlan = await readCandidateJson(expandedPath);
		} catch (error) {
			throw new BridgeError(400, `Native strategy configuration validation failed: ${error instanceof Error ? error.message.slice(0, 1000) : 'invalid configuration'}`);
		} finally { await rm(specPath, { force: true }); await rm(expandedPath, { force: true }); }
	} finally { await rm(validationDir, { recursive: true, force: true }); }
	return ranking.rows.map((row: any) => {
		try {
			return { row, candidate: makeStrategyCandidate(ranking, nativePlan, row, { rankingArtifact: rankingPath, sourceSummary, sourcePlan }) };
		} catch (error) {
			throw new BridgeError(400, error instanceof Error ? error.message : 'Native candidate validation failed.');
		}
	});
}

export async function getStrategyCandidates(event: RequestEvent): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	const id = event.params.id ?? '';
	if (!jobIds.test(id)) throw new BridgeError(404, 'Ranking job not found.');
	const job = jobs.get(id);
	if (!job) throw new BridgeError(404, 'Ranking job not found.');
	const candidates = await candidatesForRankJob(job);
	return Response.json({ candidates }, { headers: { 'cache-control': 'no-store' } });
}

export async function saveStrategyCandidate(event: RequestEvent): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	const id = event.params.id ?? '';
	if (!jobIds.test(id)) throw new BridgeError(404, 'Ranking job not found.');
	const job = jobs.get(id);
	if (!job) throw new BridgeError(404, 'Ranking job not found.');
	let body: any;
	try { body = await event.request.json(); } catch { throw new BridgeError(400, 'Expected a JSON body with runId.'); }
	if (!body || typeof body.runId !== 'string' || body.runId.length > 200) throw new BridgeError(400, 'Expected a candidate runId.');
	const candidates = await candidatesForRankJob(job);
	const selected = candidates.find(({ row }) => row.run_id === body.runId);
	if (!selected) throw new BridgeError(404, 'Candidate runId was not found in this ranking.');
	const candidatesDir = resolve(specsRoot, 'strategy-candidates');
	await mkdir(candidatesDir, { recursive: true, mode: 0o700 });
	const canonicalCandidatesDir = await canonicalCandidateDirectory(candidatesDir, specsRoot);
	const path = await saveCandidateNoClobber(canonicalCandidatesDir, selected.candidate);
	return Response.json({ saved: true, path, candidate: selected.candidate }, { headers: { 'cache-control': 'no-store' } });
}

export async function cancelStrategyJob(event: RequestEvent): Promise<Response> {
	assertStrategyRequest(event); await initialize();
	const id = event.params.id ?? '';
	const job = jobs.get(id);
	if (!job) throw new BridgeError(404, 'Strategy job not found.');
	if (job.status === 'queued' || job.status === 'running') {
		job.cancelled = true;
		const child = active.get(id);
		if (child?.pid) {
			try { if (process.platform === 'win32') child.kill('SIGTERM'); else process.kill(-child.pid, 'SIGTERM'); } catch { child.kill('SIGTERM'); }
		} else { job.status = 'cancelled'; job.updatedAt = new Date().toISOString(); await persist(job); }
	}
	return Response.json(publicJob(job), { headers: { 'cache-control': 'no-store' } });
}

export function strategyApiError(error: unknown): Response {
	const known = error instanceof BridgeError || error instanceof StrategyInputError;
	return Response.json({ error: known ? error.message : 'Unexpected local strategy API error.' }, { status: known ? error.status : 500, headers: { 'cache-control': 'no-store' } });
}
