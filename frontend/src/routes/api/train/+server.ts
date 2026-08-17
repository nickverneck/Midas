import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import type { RequestEvent } from '@sveltejs/kit';
import {
	loadDotEnv,
	getCandleCudaPolicy,
	resolveBackendFeatures,
	resolveCargoBin,
	resolveProjectRoot,
	resolveTrainerEnv,
	type MlBackend
} from '$lib/server/ml_env';
import { getTrainingCapabilities } from '$lib/server/training_capabilities';

type TrainEngine = 'ga' | 'rl';
type BuildProfile = 'debug' | 'release';
type JsonRecord = Record<string, unknown>;

const MAX_PATH_LENGTH = 2048;
const DEVICES = ['auto', 'cpu', 'cuda', 'cuda:0', 'mps'] as const;
const BACKENDS = ['libtorch', 'burn', 'candle', 'mlx'] as const;

const GA_PARAMETERS = [
	'parquet', 'train-parquet', 'val-parquet', 'test-parquet', 'full-file', 'windowed', 'window', 'step',
	'backend', 'device', 'globex', 'rth', 'bar-kind', 'volume-bar-size', 'price-source', 'initial-balance',
	'max-position', 'margin-mode', 'contract-multiplier', 'margin-per-contract', 'symbol-config', 'outdir',
	'generations', 'pop-size', 'batch-candidates', 'workers', 'elite-frac', 'parent-pool-frac',
	'immigrant-frac', 'mutation-sigma', 'init-sigma', 'hidden', 'layers', 'eval-windows', 'save-top-n',
	'save-every', 'checkpoint-every', 'behavior-every', 'load-checkpoint', 'w-pnl', 'w-sortino', 'w-mdd',
	'selection-train-weight', 'selection-eval-weight', 'selection-gap-penalty', 'selection-use-eval',
	'sortino-annualization', 'drawdown-penalty', 'drawdown-penalty-growth', 'session-close-penalty',
	'auto-close-minutes-before-close', 'max-hold-bars-positive', 'max-hold-bars-drawdown',
	'hold-duration-penalty', 'hold-duration-penalty-growth', 'hold-duration-penalty-positive-scale',
	'hold-duration-penalty-negative-scale', 'min-hold-bars', 'early-exit-penalty', 'early-flip-penalty',
	'invalid-revert-penalty', 'invalid-revert-penalty-growth', 'flat-hold-penalty',
	'flat-hold-penalty-growth', 'max-flat-hold-bars', 'seed', 'disable-margin', 'skip-val-eval',
	'debug-data', 'ignore-session'
] as const;

const RL_PARAMETERS = [
	'parquet', 'train-parquet', 'val-parquet', 'test-parquet', 'full-file', 'windowed', 'window', 'step',
	'backend', 'device', 'globex', 'rth', 'bar-kind', 'volume-bar-size', 'price-source', 'initial-balance',
	'max-position', 'margin-mode', 'contract-multiplier', 'margin-per-contract', 'symbol-config', 'outdir',
	'epochs', 'train-windows', 'ppo-epochs', 'algorithm', 'group-size', 'grpo-epochs', 'lr', 'gamma', 'lam',
	'clip', 'vf-coef', 'ent-coef', 'dropout', 'hidden', 'layers', 'eval-windows', 'log-interval',
	'checkpoint-every', 'load-checkpoint', 'w-pnl', 'w-sortino', 'w-mdd', 'fitness-use-eval',
	'sortino-annualization', 'drawdown-penalty', 'drawdown-penalty-growth', 'session-close-penalty',
	'auto-close-minutes-before-close', 'max-hold-bars-positive', 'max-hold-bars-drawdown',
	'hold-duration-penalty', 'hold-duration-penalty-growth', 'hold-duration-penalty-positive-scale',
	'hold-duration-penalty-negative-scale', 'min-hold-bars', 'early-exit-penalty', 'early-flip-penalty',
	'invalid-revert-penalty', 'invalid-revert-penalty-growth', 'flat-hold-penalty',
	'flat-hold-penalty-growth', 'max-flat-hold-bars', 'seed', 'disable-margin', 'ignore-session'
] as const;

const BOOLEAN_PARAMETERS = new Set([
	'full-file', 'windowed', 'globex', 'rth', 'selection-use-eval', 'fitness-use-eval', 'disable-margin',
	'skip-val-eval', 'debug-data', 'ignore-session'
]);
const PATH_PARAMETERS = new Set([
	'parquet', 'train-parquet', 'val-parquet', 'test-parquet', 'symbol-config', 'outdir', 'load-checkpoint'
]);
const INTEGER_PARAMETERS = new Set([
	'window', 'step', 'max-position', 'generations', 'pop-size', 'batch-candidates', 'workers', 'hidden',
	'layers', 'eval-windows', 'save-top-n', 'save-every', 'checkpoint-every', 'behavior-every', 'epochs',
	'train-windows', 'ppo-epochs', 'group-size', 'grpo-epochs', 'log-interval', 'max-hold-bars-positive',
	'max-hold-bars-drawdown', 'min-hold-bars', 'max-flat-hold-bars', 'seed'
]);

const errorResponse = (status: number, message: string) =>
	new Response(JSON.stringify({ ok: false, error: message }), {
		status,
		headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }
	});

class RequestValidationError extends Error {
	readonly status: number;

	constructor(message: string, status = 400) {
		super(message);
		this.name = 'RequestValidationError';
		this.status = status;
	}
}

const isRecord = (value: unknown): value is JsonRecord =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

const hasControlCharacters = (value: string) => /\p{Cc}/u.test(value);

const isWithin = (root: string, candidate: string) => {
	const relative = path.relative(root, candidate);
	return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
};

const resolveConfinedPath = (root: string, value: unknown, label: string) => {
	if (typeof value !== 'string' || value.trim() === '') {
		throw new RequestValidationError(`${label} must be a non-empty path under the Midas project root`);
	}
	if (value.length > MAX_PATH_LENGTH || value.includes('\0') || hasControlCharacters(value)) {
		throw new RequestValidationError(`${label} contains an invalid character or is too long`);
	}

	const candidate = path.resolve(root, value.trim());
	if (!isWithin(root, candidate)) {
		throw new RequestValidationError(`${label} must stay under the Midas project root (use data/, runs_ga/, or runs_rl/)`);
	}

	// Resolve the nearest existing path so existing symlinks and symlinked
	// parents cannot turn a root-relative argument into an outside path.
	let existing = candidate;
	while (true) {
		try {
			fs.lstatSync(existing);
			break;
		} catch {
			const parent = path.dirname(existing);
			if (parent === existing) {
				throw new RequestValidationError(`${label} could not be resolved safely`);
			}
			existing = parent;
		}
	}

	let realExisting: string;
	try {
		realExisting = fs.realpathSync(existing);
	} catch {
		throw new RequestValidationError(`${label} could not be resolved safely`);
	}
	const resolved = path.resolve(realExisting, path.relative(existing, candidate));
	if (!isWithin(root, resolved)) {
		throw new RequestValidationError(`${label} resolves outside the Midas project root`);
	}
	return path.isAbsolute(value.trim()) ? resolved : path.relative(root, resolved) || '.';
};

const normalizeProfile = (value: unknown): BuildProfile => {
	if (value === undefined) return 'debug';
	if (typeof value !== 'string' || hasControlCharacters(value) || !['debug', 'release'].includes(value)) {
		throw new RequestValidationError('profile must be debug or release');
	}
	return value as BuildProfile;
};

const normalizeEngine = (value: unknown): TrainEngine => {
	if (value === undefined) return 'ga';
	if (value !== 'ga' && value !== 'rl') throw new RequestValidationError('engine must be ga or rl');
	return value;
};

const normalizeBackend = (value: unknown): MlBackend => {
	if (value === undefined) return 'libtorch';
	if (typeof value !== 'string' || hasControlCharacters(value) || !BACKENDS.includes(value as (typeof BACKENDS)[number])) {
		throw new RequestValidationError(`backend must be one of ${BACKENDS.join(', ')}`);
	}
	const backend = value as MlBackend;
	if (backend === 'mlx') {
		throw new RequestValidationError(
			'backend mlx is planned but not implemented; use burn for the supported Apple GPU path',
			422
		);
	}
	return backend;
};

const normalizeChoice = (value: unknown, name: string, choices: readonly string[]) => {
	if (typeof value !== 'string' || hasControlCharacters(value) || !choices.includes(value)) {
		throw new RequestValidationError(`${name} must be one of ${choices.join(', ')}`);
	}
	return value;
};

const normalizeNumber = (value: unknown, name: string, integer: boolean) => {
	if (typeof value === 'string' && hasControlCharacters(value)) {
		throw new RequestValidationError(`${name} contains control characters`);
	}
	if (typeof value !== 'number' && typeof value !== 'string') {
		throw new RequestValidationError(`${name} must be a number`);
	}
	if (typeof value === 'string' && value.trim() === '') {
		throw new RequestValidationError(`${name} must not be empty`);
	}
	const parsed = typeof value === 'number' ? value : Number(value);
	if (!Number.isFinite(parsed) || (integer && !Number.isSafeInteger(parsed))) {
		throw new RequestValidationError(`${name} must be a finite${integer ? ' safe integer' : ''} value`);
	}
	if (integer && parsed < 0 && !['max-position'].includes(name)) {
		throw new RequestValidationError(`${name} must not be negative`);
	}
	return parsed;
};

const normalizeParameter = (name: string, value: unknown, root: string) => {
	if (value === null || value === undefined) return undefined;
	const choiceParameter = ['backend', 'device', 'bar-kind', 'price-source', 'margin-mode', 'algorithm'].includes(name);
	if (value === '' && !PATH_PARAMETERS.has(name) && !BOOLEAN_PARAMETERS.has(name) && !choiceParameter) return undefined;
	if (PATH_PARAMETERS.has(name)) return resolveConfinedPath(root, value, name);
	if (BOOLEAN_PARAMETERS.has(name)) {
		if (typeof value !== 'boolean') throw new RequestValidationError(`${name} must be boolean`);
		return value;
	}
	if (name === 'backend') return normalizeBackend(value);
	if (name === 'device') {
		if (typeof value !== 'string' || hasControlCharacters(value) || !DEVICES.includes(value as (typeof DEVICES)[number])) {
			throw new RequestValidationError(`device must be one of ${DEVICES.join(', ')}`);
		}
		return value;
	}
	if (name === 'bar-kind') return normalizeChoice(value, name, ['price-action', 'volume']);
	if (name === 'price-source') return normalizeChoice(value, name, ['ohlc', 'heikin-ashi']);
	if (name === 'margin-mode') return normalizeChoice(value, name, ['auto', 'per-contract', 'price']);
	if (name === 'algorithm') return normalizeChoice(value, name, ['ppo', 'grpo']);
	if (typeof value === 'string' && hasControlCharacters(value)) {
		throw new RequestValidationError(`${name} contains control characters`);
	}
	if (INTEGER_PARAMETERS.has(name)) return normalizeNumber(value, name, true);
	return normalizeNumber(value, name, false);
};

const validateParams = (params: JsonRecord, engine: TrainEngine, root: string) => {
	const allowed = engine === 'ga' ? GA_PARAMETERS : RL_PARAMETERS;
	const allowedSet = new Set<string>(allowed);
	const unknown = Object.keys(params).filter((key) => !allowedSet.has(key));
	if (unknown.length > 0) {
		throw new RequestValidationError(
			`unknown ${engine.toUpperCase()} parameter(s): ${unknown.join(', ')}. Remove them or use the supported legacy CLI flags.`
		);
	}

	const normalized: JsonRecord = {};
	for (const [name, value] of Object.entries(params)) {
		const next = normalizeParameter(name, value, root);
		if (next !== undefined && next !== false) normalized[name] = next;
	}
	return normalized;
};

const buildCliArgs = (params: JsonRecord) => {
	const args: string[] = [];
	for (const [key, value] of Object.entries(params)) {
		if (value === true) args.push(`--${key}`);
		else if (value !== null && value !== undefined && value !== '') args.push(`--${key}`, String(value));
	}
	return args;
};

const extractRequest = (payload: JsonRecord) => {
	const envelopeKeys = new Set(['engine', 'profile', 'params']);
	const hasParams = Object.prototype.hasOwnProperty.call(payload, 'params');
	if (hasParams) {
		const unexpectedEnvelopeKeys = Object.keys(payload).filter((key) => !envelopeKeys.has(key));
		if (unexpectedEnvelopeKeys.length > 0) {
			throw new RequestValidationError(`unexpected top-level field(s): ${unexpectedEnvelopeKeys.join(', ')}; put training flags inside params`);
		}
		if (!isRecord(payload.params)) throw new RequestValidationError('params must be an object');
		return payload.params;
	}

	// Keep the historical flat form working, but strip only the known request
	// envelope fields before the explicit GA/RL allowlist is applied.
	return Object.fromEntries(Object.entries(payload).filter(([key]) => !envelopeKeys.has(key)));
};

export const POST = async ({ request }: RequestEvent) => {
	let payload: JsonRecord;
	try {
		const body = await request.json();
		if (!isRecord(body)) return errorResponse(400, 'Training payload must be an object');
		payload = body;
	} catch {
		return errorResponse(400, 'Invalid training JSON payload');
	}

	let engine: TrainEngine;
	let profile: BuildProfile;
	let root: string;
	let params: JsonRecord;
	try {
		engine = normalizeEngine(payload.engine);
		profile = normalizeProfile(payload.profile);
		root = fs.realpathSync(path.resolve(resolveProjectRoot()));
		params = validateParams(extractRequest(payload), engine, root);
	} catch (error) {
		if (error instanceof RequestValidationError) return errorResponse(error.status, error.message);
		return errorResponse(400, 'Unable to validate training request');
	}

	const { signal } = request;
	const dotenv = loadDotEnv(root);
	const backend = (params.backend ?? 'libtorch') as MlBackend;
	const env = resolveTrainerEnv(root, { ...process.env, ...dotenv }, backend);
	const runtime = (params.device ?? 'auto') as string;
	const capabilityBackend = backend === 'libtorch' ? 'torch' : backend;
	if (capabilityBackend === 'burn' || capabilityBackend === 'candle' || capabilityBackend === 'torch') {
		const capability = getTrainingCapabilities(env).backends[capabilityBackend];
		if (!capability[engine]) {
			return errorResponse(422, `${backend} is not implemented for ${engine.toUpperCase()} training`);
		}
		const capabilityDevice = runtime === 'cuda:0' ? 'cuda' : runtime;
		if (!capability.devices[capabilityDevice as keyof typeof capability.devices]) {
			if (backend === 'candle' && (runtime === 'cuda' || runtime === 'cuda:0')) {
				return errorResponse(422, getCandleCudaPolicy(env).reason ?? 'Candle CUDA is unavailable; choose CPU');
			}
			return errorResponse(422, `${backend} cannot run ${runtime} on this server; select a supported runtime or configure the backend feature`);
		}
	}
	const features = resolveBackendFeatures(backend, env, runtime);

	const command = resolveCargoBin(env);
	const cargoArgs = ['run'];
	if (profile === 'release') cargoArgs.push('--release');
	cargoArgs.push('--features', features.join(','));
	cargoArgs.push('--bin', engine === 'rl' ? 'train_rl' : 'train_ga', '--', ...buildCliArgs(params));

	const stream = new ReadableStream({
		start(controller) {
			const child = spawn(command, cargoArgs, { cwd: root, env });
			const onAbort = () => {
				if (!child.killed) child.kill('SIGTERM');
			};
			signal.addEventListener('abort', onAbort);
			child.stdout.on('data', (data) => controller.enqueue(`data: ${JSON.stringify({ type: 'stdout', content: data.toString() })}\n\n`));
			child.stderr.on('data', (data) => controller.enqueue(`data: ${JSON.stringify({ type: 'stderr', content: data.toString() })}\n\n`));
			child.on('close', (code) => {
				signal.removeEventListener('abort', onAbort);
				controller.enqueue(`data: ${JSON.stringify({ type: 'exit', code })}\n\n`);
				controller.close();
			});
			child.on('error', (error) => {
				signal.removeEventListener('abort', onAbort);
				controller.enqueue(`data: ${JSON.stringify({ type: 'error', content: error.message })}\n\n`);
				controller.close();
			});
		}
	});

	return new Response(stream, {
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			Connection: 'keep-alive'
		}
	});
};
