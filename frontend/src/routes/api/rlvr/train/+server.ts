import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import type { RequestEvent } from '@sveltejs/kit';
import {
	ensureConfinedDirectory,
	getProjectContext,
	hasControlCharacters,
	inspectGeneratedFile,
	isWithin,
	relativeApiPath,
	resolveExistingFile,
	RequestValidationError
} from '$lib/server/supervised';
import { buildRlvrCliArgs } from '$lib/server/rlvr';

const RLVR_RUNS_DIR = path.join('.run', 'event-rlvr', 'runs');
const MAX_PATH_LENGTH = 2048;
const MAX_EPOCHS = 100_000;

type JsonRecord = Record<string, unknown>;

const errorResponse = (status: number, message: string) =>
	new Response(JSON.stringify({ ok: false, error: message }), {
		status,
		headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }
	});

const isRecord = (value: unknown): value is JsonRecord =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

const asNumber = (value: unknown, fallback: number, label: string, min: number, max: number) => {
	if (value === undefined || value === null || value === '') return fallback;
	const number = typeof value === 'number' ? value : Number(value);
	if (!Number.isFinite(number) || number < min || number > max) {
		throw new RequestValidationError(`${label} must be between ${min} and ${max}`);
	}
	return number;
};

const asInteger = (value: unknown, fallback: number, label: string, min: number, max: number) => {
	const number = asNumber(value, fallback, label, min, max);
	if (!Number.isSafeInteger(number)) throw new RequestValidationError(`${label} must be an integer`);
	return number;
};

const asPath = (value: unknown, label: string) => {
	if (typeof value !== 'string' || value.trim() === '') throw new RequestValidationError(`${label} is required`);
	if (value.length > MAX_PATH_LENGTH || value.includes('\0') || hasControlCharacters(value)) {
		throw new RequestValidationError(`${label} contains an invalid character or is too long`);
	}
	return value.trim();
};

const resolveOutputDirectory = (rootReal: string, value: unknown) => {
	const runsRoot = ensureConfinedDirectory(rootReal, RLVR_RUNS_DIR, 'RLVR runs directory');
	const requested = asPath(value, 'outdir');
	const candidate = path.resolve(rootReal, requested);
	if (!isWithin(runsRoot, candidate)) {
		throw new RequestValidationError(
			`outdir must stay under ${relativeApiPath(rootReal, runsRoot)}`
		);
	}

	// Resolve the nearest existing parent before creating anything. This keeps
	// symlinked parents from redirecting a requested run outside the project.
	let existing = candidate;
	while (true) {
		try {
			fs.lstatSync(existing);
			break;
		} catch {
			const parent = path.dirname(existing);
			if (parent === existing) throw new RequestValidationError('outdir could not be resolved safely', 500);
			existing = parent;
		}
	}
	const existingReal = fs.realpathSync(existing);
	if (!isWithin(runsRoot, existingReal)) {
		throw new RequestValidationError('outdir resolves outside the RLVR runs directory');
	}
	try {
		if (fs.existsSync(candidate) && fs.lstatSync(candidate).isSymbolicLink()) {
			throw new RequestValidationError('outdir must not be a symbolic link');
		}
		fs.mkdirSync(candidate, { recursive: true });
	} catch (error) {
		if (error instanceof RequestValidationError) throw error;
		throw new RequestValidationError('outdir could not be created', 500);
	}
	const outputReal = fs.realpathSync(candidate);
	if (!isWithin(runsRoot, outputReal)) throw new RequestValidationError('outdir resolves outside the RLVR runs directory');
	return outputReal;
};

const sse = (payload: JsonRecord) => `data: ${JSON.stringify(payload)}\n\n`;

const readMetrics = (outputDirectory: string) => {
	const metricsPath = inspectGeneratedFile(
		outputDirectory,
		path.join(outputDirectory, 'metrics.json'),
		'RLVR metrics'
	);
	const policyPath = inspectGeneratedFile(
		outputDirectory,
		path.join(outputDirectory, 'policy.json'),
		'RLVR policy'
	);
	let metrics: JsonRecord;
	try {
		const parsed: unknown = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
		if (!isRecord(parsed)) throw new Error('metrics must be a JSON object');
		metrics = parsed;
	} catch (error) {
		throw new RequestValidationError(
			`RLVR metrics could not be read: ${error instanceof Error ? error.message : String(error)}`,
			500
		);
	}
	return { metricsPath, policyPath, metrics };
};

export const POST = async ({ request }: RequestEvent) => {
	let payload: JsonRecord;
	try {
		const body: unknown = await request.json();
		if (!isRecord(body)) throw new RequestValidationError('JSON body must be an object');
		payload = body;
	} catch (error) {
		return error instanceof RequestValidationError
			? errorResponse(error.status, error.message)
			: errorResponse(400, 'Invalid JSON payload');
	}

	let rootReal: string;
	let inputPath: string;
	let outputDirectory: string;
	let cliArgs: string[];
	try {
		const context = getProjectContext();
		rootReal = context.rootReal;
		inputPath = resolveExistingFile({
			rootReal,
			value: payload.input,
			label: 'event parquet',
			extensions: ['parquet']
		});
		outputDirectory = resolveOutputDirectory(rootReal, payload.outdir);

		const epochs = asInteger(payload.epochs, 1000, 'epochs', 1, MAX_EPOCHS);
		const checkpointEvery = asInteger(payload.checkpoint_every, 250, 'checkpoint_every', 0, MAX_EPOCHS);
		const hidden = asInteger(payload.hidden, 32, 'hidden', 1, 4096);
		const layers = asInteger(payload.layers, 1, 'layers', 1, 32);
		const learningRate = asNumber(payload.learning_rate, 0.001, 'learning_rate', Number.MIN_VALUE, 1);
		const l2 = asNumber(payload.l2, 0.0001, 'l2', 0, 1);
		const entropyCoefficient = asNumber(payload.entropy_coefficient, 0.01, 'entropy_coefficient', 0, 1);
		const seed = asInteger(payload.seed, 42, 'seed', 0, Number.MAX_SAFE_INTEGER);
		const trainFraction = asNumber(payload.train_fraction, 0.6, 'train_fraction', 0.05, 0.9);
		const validationFraction = asNumber(payload.validation_fraction, 0.2, 'validation_fraction', 0.05, 0.9);
		if (trainFraction + validationFraction >= 1) {
			throw new RequestValidationError('train_fraction + validation_fraction must leave holdout data');
		}
		const purgeBars = asInteger(payload.purge_bars, 30, 'purge_bars', 0, 100_000);
		const rewardNormalization = payload.reward_normalization ?? 'train-mean-abs';
		if (rewardNormalization !== 'none' && rewardNormalization !== 'train-mean-abs') {
			throw new RequestValidationError('reward_normalization must be none or train-mean-abs');
		}

		cliArgs = buildRlvrCliArgs({
			input: inputPath,
			outdir: outputDirectory,
			epochs,
			checkpointEvery,
			hidden,
			layers,
			learningRate,
			l2,
			entropyCoefficient,
			seed,
			trainFraction,
			validationFraction,
			purgeBars,
			rewardNormalization
		});
	} catch (error) {
		if (error instanceof RequestValidationError) return errorResponse(error.status, error.message);
		return errorResponse(500, error instanceof Error ? error.message : 'Unable to validate RLVR request');
	}

	const context = getProjectContext();
	const { signal } = request;
	let child: ReturnType<typeof spawn> | null = null;
	let streamCancelled = false;
	const stream = new ReadableStream<string>({
		start(controller) {
			let finished = false;
			const stopChild = () => {
				if (child && !child.killed) child.kill('SIGTERM');
			};
			const emit = (payload: JsonRecord) => {
				if (streamCancelled) return;
				try {
					controller.enqueue(sse(payload));
				} catch {
					streamCancelled = true;
					stopChild();
				}
			};
			const spawned = spawn(context.cargoCommand, cliArgs, {
				cwd: rootReal,
				env: context.env,
				stdio: ['ignore', 'pipe', 'pipe']
			});
			child = spawned;
			const onAbort = () => {
				stopChild();
			};
			const close = (code: number | null) => {
				if (finished) return;
				finished = true;
				signal.removeEventListener('abort', onAbort);
				child = null;
				if (streamCancelled) return;
				if (code === 0 && !signal.aborted) {
					try {
						const artifacts = readMetrics(outputDirectory);
						emit({
							type: 'complete',
							input: relativeApiPath(rootReal, inputPath),
							runDir: relativeApiPath(rootReal, outputDirectory),
							metricsPath: relativeApiPath(rootReal, artifacts.metricsPath),
							policyPath: relativeApiPath(rootReal, artifacts.policyPath),
							metrics: artifacts.metrics
						});
					} catch (error) {
						emit({ type: 'error', content: error instanceof Error ? error.message : String(error) });
					}
				}
				emit({ type: 'exit', code });
				try {
					controller.close();
				} catch {
					// The browser may have cancelled the response between the final event and close.
				}
			};
			signal.addEventListener('abort', onAbort, { once: true });
			emit({
				type: 'start',
				algorithm: 'rlvr',
				input: relativeApiPath(rootReal, inputPath),
				runDir: relativeApiPath(rootReal, outputDirectory)
			});
			spawned.stdout.on('data', (data: Buffer) => emit({ type: 'stdout', content: data.toString() }));
			spawned.stderr.on('data', (data: Buffer) => emit({ type: 'stderr', content: data.toString() }));
			spawned.once('error', (error) => emit({ type: 'error', content: error.message }));
			spawned.once('close', close);
		},
		cancel() {
			streamCancelled = true;
			if (child && !child.killed) child.kill('SIGTERM');
		}
	});

	return new Response(stream, {
		headers: {
			'Content-Type': 'text/event-stream; charset=utf-8',
			'Cache-Control': 'no-cache, no-store',
			Connection: 'keep-alive',
			'X-Accel-Buffering': 'no'
		}
	});
};
