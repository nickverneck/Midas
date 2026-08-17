import { randomUUID } from 'crypto';
import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import { json } from '@sveltejs/kit';
import type { RequestEvent } from '@sveltejs/kit';
import { loadDotEnv, resolveCargoBin, resolveProjectRoot } from '$lib/server/ml_env';
import {
	RequestValidationError,
	relativeApiPath as relativeArtifactPath,
	snapshotCompletionArtifactPair
} from '$lib/server/supervised';

const DATASET_DIR = path.join('.run', 'meta-gate', 'datasets');
const RUNS_DIR = path.join('.run', 'meta-gate', 'runs');
const MAX_BUFFER = 64 * 1024 * 1024;
const MAX_PATH_LENGTH = 2048;
const MAX_ITERATIONS = 2000;
const MAX_POPULATION = 512;
const DEFAULT_PURGE_BARS = 30;
const MAX_PURGE_BARS = 100_000;
const TRAIN_TIMEOUT_MS = 10 * 60 * 1000;
const DIAGNOSTIC_WARNING =
	'Fixed-horizon diagnostic training only; the policy is not an executable Trader schedule and does not model sequential position, protection, or fill lifecycle.';

type JsonRecord = Record<string, unknown>;
type Algorithm = 'ga' | 'rl';
type Device = 'cpu' | 'cuda';
type Action = 'normal' | 'skip' | 'invert';

type ChildProcessError = Error & {
	code?: number | string;
	killed?: boolean;
	signal?: NodeJS.Signals | null;
	stdout?: string;
	stderr?: string;
};

type ChildProcessResult = {
	stdout: string;
	stderr: string;
};

const isRecord = (value: unknown): value is JsonRecord =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

const isWithin = (root: string, candidate: string) => {
	const relative = path.relative(root, candidate);
	return (
		relative === '' ||
		(relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
	);
};

const hasControlCharacters = (value: string) => /\p{Cc}/u.test(value);

const normalizeChoice = <T extends string>(
	value: unknown,
	name: string,
	choices: readonly T[]
): T => {
	if (typeof value !== 'string' || value.trim() === '') {
		throw new RequestValidationError(`${name} must be one of ${choices.join(', ')}`);
	}
	if (hasControlCharacters(value)) {
		throw new RequestValidationError(`${name} must be one of ${choices.join(', ')}`);
	}
	const normalized = value.trim().toLowerCase();
	if (!choices.includes(normalized as T)) {
		throw new RequestValidationError(`${name} must be one of ${choices.join(', ')}`);
	}
	return normalized as T;
};

const normalizeInteger = (
	value: unknown,
	defaultValue: number,
	name: string,
	minimum: number,
	maximum: number
) => {
	if (value === undefined || value === null || value === '') return defaultValue;
	if (typeof value === 'string' && hasControlCharacters(value)) {
		throw new RequestValidationError(`${name} must be an integer between ${minimum} and ${maximum}`);
	}
	const numberValue =
		typeof value === 'number'
			? value
			: typeof value === 'string' && value.trim() !== ''
				? Number(value.trim())
				: Number.NaN;
	if (
		!Number.isSafeInteger(numberValue) ||
		numberValue < minimum ||
		numberValue > maximum
	) {
		throw new RequestValidationError(
			`${name} must be an integer between ${minimum} and ${maximum}`
		);
	}
	return numberValue;
};

const normalizeActions = (value: unknown): Action[] => {
	const rawActions =
		value === undefined || value === null
			? ['normal', 'skip']
			: typeof value === 'string'
				? value.split(',')
				: Array.isArray(value)
					? value
					: null;

	if (!rawActions || rawActions.length === 0 || rawActions.length > 3) {
		throw new RequestValidationError(
			'actions must contain normal, skip, and optionally invert'
		);
	}

	const seen = new Set<Action>();
	for (const rawAction of rawActions) {
		if (typeof rawAction !== 'string') {
			throw new RequestValidationError('each action must be a string');
		}
		if (hasControlCharacters(rawAction)) {
			throw new RequestValidationError('actions may not contain control characters');
		}
		const action = rawAction.trim().toLowerCase();
		if (!['normal', 'skip', 'invert'].includes(action)) {
			throw new RequestValidationError(
				'actions may contain only normal, skip, and invert'
			);
		}
		if (seen.has(action as Action)) {
			throw new RequestValidationError(`duplicate action ${action}`);
		}
		seen.add(action as Action);
	}

	if (!seen.has('normal') || !seen.has('skip')) {
		throw new RequestValidationError('actions must include both normal and skip');
	}

	return (['normal', 'skip', 'invert'] as Action[]).filter((action) => seen.has(action));
};

const resolveInputPath = (root: string, rootReal: string, value: unknown) => {
	if (typeof value !== 'string' || value.trim() === '') {
		throw new RequestValidationError('input must be a parquet path');
	}
	if (value.length > MAX_PATH_LENGTH || hasControlCharacters(value)) {
		throw new RequestValidationError('input contains an invalid character or is too long');
	}

	const datasetDirectory = path.resolve(root, DATASET_DIR);
	let candidate: string;
	try {
		candidate = path.resolve(root, value.trim());
	} catch {
		throw new RequestValidationError('input is not a valid path');
	}

	if (!isWithin(datasetDirectory, candidate)) {
		throw new RequestValidationError(
			'input must be under the Midas .run/meta-gate/datasets directory'
		);
	}
	if (!candidate.toLowerCase().endsWith('.parquet')) {
		throw new RequestValidationError('input must have a .parquet extension');
	}
	if (!fs.existsSync(candidate)) {
		throw new RequestValidationError('input parquet file was not found', 404);
	}

	try {
		if (!fs.statSync(candidate).isFile()) {
			throw new RequestValidationError('input must refer to a parquet file');
		}
	} catch (error) {
		if (error instanceof RequestValidationError) throw error;
		throw new RequestValidationError('input parquet file could not be inspected', 404);
	}

	let datasetReal: string;
	let realCandidate: string;
	try {
		datasetReal = fs.realpathSync(datasetDirectory);
		realCandidate = fs.realpathSync(candidate);
	} catch {
		throw new RequestValidationError('input parquet file could not be resolved', 404);
	}

	if (!isWithin(rootReal, datasetReal) || !isWithin(datasetReal, realCandidate)) {
		throw new RequestValidationError(
			'input must resolve under the Midas .run/meta-gate/datasets directory'
		);
	}

	return realCandidate;
};

const resolveRunsDirectory = (root: string, rootReal: string) => {
	const runsDirectory = path.resolve(root, RUNS_DIR);
	if (!isWithin(root, runsDirectory)) {
		throw new RequestValidationError('meta-gate run directory is outside the Midas project root', 500);
	}

	try {
		fs.mkdirSync(runsDirectory, { recursive: true });
	} catch {
		throw new RequestValidationError('meta-gate run directory could not be created', 500);
	}

	let realRunsDirectory: string;
	try {
		realRunsDirectory = fs.realpathSync(runsDirectory);
	} catch {
		throw new RequestValidationError('meta-gate run directory could not be resolved', 500);
	}
	if (!isWithin(rootReal, realRunsDirectory)) {
		throw new RequestValidationError(
			'meta-gate run directory must remain under the Midas project root',
			500
		);
	}

	return realRunsDirectory;
};

const createRunDirectory = (root: string, rootReal: string) => {
	const realRunsDirectory = resolveRunsDirectory(root, rootReal);
	const runName = `run-${Date.now()}-${randomUUID().replace(/-/g, '').slice(0, 16)}`;
	const runDirectory = path.join(realRunsDirectory, runName);

	try {
		fs.mkdirSync(runDirectory, { recursive: false });
	} catch {
		throw new RequestValidationError('meta-gate run directory could not be created', 500);
	}

	let realRunDirectory: string;
	try {
		realRunDirectory = fs.realpathSync(runDirectory);
	} catch {
		throw new RequestValidationError('meta-gate run directory could not be resolved', 500);
	}
	if (!isWithin(realRunsDirectory, realRunDirectory) || !isWithin(rootReal, realRunDirectory)) {
		throw new RequestValidationError(
			'generated meta-gate run directory resolved outside the Midas project root',
			500
		);
	}

	return realRunDirectory;
};

const runCargo = (
	command: string,
	args: string[],
	root: string,
	env: NodeJS.ProcessEnv,
	signal: AbortSignal
) =>
	new Promise<ChildProcessResult>((resolve, reject) => {
		execFile(
			command,
			args,
			{
				cwd: root,
				env,
				encoding: 'utf8',
				maxBuffer: MAX_BUFFER,
				windowsHide: true,
				timeout: TRAIN_TIMEOUT_MS,
				killSignal: 'SIGTERM',
				signal
			},
			(error, stdout, stderr) => {
				if (error) {
					const childError = error as ChildProcessError;
					childError.stdout = stdout;
					childError.stderr = stderr;
					reject(childError);
					return;
				}
				resolve({ stdout, stderr });
			}
		);
	});

const relativeApiPath = (root: string, absolutePath: string) =>
	path.relative(root, absolutePath).split(path.sep).join('/');

const errorResponse = (status: number, message: string, extra: Record<string, unknown> = {}) =>
	json(
		{
			ok: false,
			diagnosticOnly: true,
			warning: DIAGNOSTIC_WARNING,
			error: message,
			stdout: '',
			stderr: '',
			...extra
		},
		{ status, headers: { 'Cache-Control': 'no-store' } }
	);

export const POST = async ({ request }: RequestEvent) => {
	let payload: JsonRecord;
	try {
		const body: unknown = await request.json();
		if (!isRecord(body)) throw new RequestValidationError('JSON body must be an object');
		payload = body;
	} catch (error) {
		if (error instanceof RequestValidationError) {
			return errorResponse(error.status, error.message);
		}
		return errorResponse(400, 'Invalid JSON payload');
	}

	let runDirectory: string | undefined;
	let projectRootReal: string | undefined;
	try {
		const algorithm = normalizeChoice(payload.algorithm, 'algorithm', ['ga', 'rl'] as const);
		const device = normalizeChoice(payload.device, 'device', ['cpu', 'cuda'] as const);
		if (device === 'cuda') {
			throw new RequestValidationError(
				'CUDA is not supported by train_meta_gate yet; this prototype intentionally runs on CPU only',
				422
			);
		}

		const actions = normalizeActions(payload.actions);
		const seed = normalizeInteger(payload.seed, 42, 'seed', 0, Number.MAX_SAFE_INTEGER);
		const generations = normalizeInteger(
			payload.generations,
			25,
			'generations',
			1,
			MAX_ITERATIONS
		);
		const epochs = normalizeInteger(payload.epochs, 25, 'epochs', 1, MAX_ITERATIONS);
		const minimumPopulation = algorithm === 'ga' ? 2 : 1;
		const population = normalizeInteger(
			payload.population,
			32,
			'population',
			minimumPopulation,
			MAX_POPULATION
		);

		const root = path.resolve(resolveProjectRoot());
		const rootReal = fs.realpathSync(root);
		projectRootReal = rootReal;
		const inputPath = resolveInputPath(root, rootReal, payload.input);
		const purgeBars = normalizeInteger(
			payload.purge_bars ?? payload.purgeBars,
			DEFAULT_PURGE_BARS,
			'purge_bars',
			0,
			MAX_PURGE_BARS
		);
		runDirectory = createRunDirectory(root, rootReal);

		const dotenv = loadDotEnv(root);
		const env = { ...process.env, ...dotenv };
		const cargoCommand = resolveCargoBin(env);
		const cargoArgs = [
			'run',
			'--quiet',
			'--bin',
			'train_meta_gate',
			'--',
			'--input',
			inputPath,
			'--outdir',
			runDirectory,
			'--algorithm',
			algorithm,
			'--device',
			device,
			'--actions',
			actions.join(','),
			'--seed',
			String(seed),
			'--generations',
			String(generations),
			'--epochs',
			String(epochs),
			'--population',
			String(population),
			'--purge-bars',
			String(purgeBars)
		];

		let result: ChildProcessResult;
		try {
			result = await runCargo(cargoCommand, cargoArgs, root, env, request.signal);
		} catch (error) {
			const childError = error as ChildProcessError;
			const requestAborted =
				request.signal.aborted || childError.name === 'AbortError' || childError.code === 'ABORT_ERR';
			const timedOut =
				!requestAborted &&
				(childError.code === 'ETIMEDOUT' ||
					(childError.killed === true && childError.signal === 'SIGTERM'));
			const code = childError.code === undefined ? '' : ` (exit ${childError.code})`;
			const status = requestAborted ? 499 : timedOut ? 504 : 500;
			const message = requestAborted
				? 'train_meta_gate aborted because the client request disconnected'
				: timedOut
					? `train_meta_gate timed out after ${TRAIN_TIMEOUT_MS / 1000} seconds`
					: `train_meta_gate failed${code}: ${childError.message}`;
			return errorResponse(status, message, {
				stdout: childError.stdout ?? '',
				stderr: childError.stderr ?? childError.message,
				inputPath: relativeApiPath(rootReal, inputPath),
				runDir: relativeApiPath(rootReal, runDirectory),
				purgeBars,
				timeoutMs: TRAIN_TIMEOUT_MS,
				aborted: requestAborted
			});
		}

		let artifacts;
		try {
			artifacts = snapshotCompletionArtifactPair({
				directory: runDirectory,
				manifestFile: '.meta-gate-manifest.json',
				expectedSchema: 'meta-gate-artifacts-v1',
				expectedStatus: 'complete',
				label: 'meta-gate training'
			});
		} catch (error) {
			if (error instanceof RequestValidationError) {
				return errorResponse(error.status, error.message, {
					stdout: result.stdout,
					stderr: result.stderr,
					inputPath: relativeApiPath(rootReal, inputPath),
					runDir: relativeApiPath(rootReal, runDirectory),
					purgeBars
				});
			}
			throw error;
		}

		return json(
			{
				ok: true,
				inputPath: relativeApiPath(rootReal, inputPath),
				runDir: relativeApiPath(rootReal, runDirectory),
				policyPath: relativeArtifactPath(rootReal, artifacts.policyPath),
				metricsPath: relativeArtifactPath(rootReal, artifacts.metricsPath),
				completionPath: relativeArtifactPath(rootReal, artifacts.manifestPath),
				stdout: result.stdout,
				stderr: result.stderr,
				algorithm,
				device,
				purgeBars,
				diagnosticOnly: true,
				warning: DIAGNOSTIC_WARNING
			},
			{ headers: { 'Cache-Control': 'no-store' } }
		);
	} catch (error) {
		if (error instanceof RequestValidationError) {
			return errorResponse(
				error.status,
				error.message,
				runDirectory && projectRootReal
					? { runDir: relativeApiPath(projectRootReal, runDirectory) }
					: {}
			);
		}
		const message = error instanceof Error ? error.message : 'Unable to train meta-gate';
		return errorResponse(
			500,
			message,
			runDirectory && projectRootReal
				? { runDir: relativeApiPath(projectRootReal, runDirectory) }
				: {}
		);
	}
};
