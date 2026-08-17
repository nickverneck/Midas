import { randomUUID } from 'crypto';
import fs from 'fs';
import path from 'path';
import { json } from '@sveltejs/kit';
import type { RequestEvent } from '@sveltejs/kit';
import {
	getCandleCudaPolicy,
	resolveBackendFeatures,
	resolveTrainerEnv,
	type MlBackend
} from '$lib/server/ml_env';
import { getTrainingCapabilities } from '$lib/server/training_capabilities';
import {
	SUPERVISED_DATASET_DIR,
	SUPERVISED_RUNS_DIR,
	RequestValidationError,
	ensureConfinedDirectory,
	executionErrorDetails,
	getProjectContext,
	hasControlCharacters,
	isRecord,
	parseJsonStdout,
	relativeApiPath,
	resolveExistingFile,
	runSupervised,
	snapshotCompletionArtifactPair,
	type JsonRecord
} from '$lib/server/supervised';

const TRAIN_TIMEOUT_MS = 20 * 60 * 1000;

const errorResponse = (status: number, error: string, extra: JsonRecord = {}) =>
	json({ ok: false, error, ...extra }, { status, headers: { 'Cache-Control': 'no-store' } });

const normalizeInteger = (value: unknown, fallback: number, name: string, min: number, max: number) => {
	if (value === undefined || value === null || value === '') return fallback;
	const parsed = typeof value === 'number' ? value : Number(value);
	if (!Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
		throw new RequestValidationError(`${name} must be an integer between ${min} and ${max}`);
	}
	return parsed;
};

const normalizeNumber = (value: unknown, fallback: number, name: string, min: number, max: number) => {
	if (value === undefined || value === null || value === '') return fallback;
	const parsed = typeof value === 'number' ? value : Number(value);
	if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
		throw new RequestValidationError(`${name} must be between ${min} and ${max}`);
	}
	return parsed;
};

const normalizeBackend = (value: unknown) => {
	if (value === undefined || value === null || value === '') return 'cpu-linear';
	if (typeof value !== 'string' || hasControlCharacters(value)) {
		throw new RequestValidationError('backend must be cpu-linear, burn, or candle');
	}
	const backend = value.trim().toLowerCase();
	if (backend === 'mlx') {
		throw new RequestValidationError(
			'supervised mlx training is planned but not implemented; use burn for the supported Apple GPU path',
			422
		);
	}
	if (!['cpu-linear', 'burn', 'candle'].includes(backend)) {
		throw new RequestValidationError(
			`supervised ${backend} training is not enabled; use cpu-linear, burn, or candle`,
			422
		);
	}
	return backend;
};

const normalizeDevice = (value: unknown, backend: string) => {
	if (value === undefined || value === null || value === '') return 'cpu';
	if (typeof value !== 'string' || hasControlCharacters(value)) {
		throw new RequestValidationError('device must be auto, cpu, cuda, cuda:0, or mps');
	}
	const device = value.trim().toLowerCase();
	if (!['auto', 'cpu', 'cuda', 'cuda:0', 'mps'].includes(device)) {
		throw new RequestValidationError(
			`supervised ${backend} supports auto, cpu, cuda, cuda:0, or mps when the backend provides it`,
			422
		);
	}
	return device;
};

const normalizeMode = (value: unknown): 'new' | 'continue' => {
	if (value === undefined || value === null || value === '') return 'new';
	if (typeof value !== 'string' || hasControlCharacters(value) || !['new', 'continue'].includes(value)) {
		throw new RequestValidationError('mode must be new or continue');
	}
	return value as 'new' | 'continue';
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

	let runDirectory: string | undefined;
	let rootReal: string | undefined;
	let resumedFrom: string | undefined;
	let resumeSnapshotPath: string | undefined;
	try {
		const context = getProjectContext();
		rootReal = context.rootReal;
		const datasetDirectory = ensureConfinedDirectory(rootReal, SUPERVISED_DATASET_DIR, 'supervised dataset directory');
		const runsDirectory = ensureConfinedDirectory(rootReal, SUPERVISED_RUNS_DIR, 'supervised runs directory');
		const mode = normalizeMode(payload.mode);
		const hasResumePolicy = payload.resume_policy !== undefined && payload.resume_policy !== null && payload.resume_policy !== '';
		if (mode === 'continue' && !hasResumePolicy) {
			throw new RequestValidationError('resume_policy is required when mode is continue');
		}
		if (mode === 'new' && hasResumePolicy) {
			throw new RequestValidationError('resume_policy is only allowed when mode is continue');
		}
		const inputPath = resolveExistingFile({
			rootReal,
			value: payload.input ?? payload.dataset,
			label: 'dataset',
			extensions: ['parquet'],
			confinedTo: datasetDirectory
		});
		const backend = normalizeBackend(payload.backend);
		const requestedDevice = normalizeDevice(payload.device, backend);
		const device = backend === 'cpu-linear' && requestedDevice === 'auto' ? 'cpu' : requestedDevice;
		const trainerEnv = backend === 'cpu-linear'
			? context.env
			: resolveTrainerEnv(rootReal, context.env, backend as MlBackend);
		const capabilityBackend = backend as keyof ReturnType<typeof getTrainingCapabilities>['backends'];
		const capability = getTrainingCapabilities(trainerEnv).backends[capabilityBackend];
		const capabilityDevice = device === 'cuda:0' ? 'cuda' : device;
		if (!capability.supervised || !capability.devices[capabilityDevice as keyof typeof capability.devices]) {
			if (backend === 'candle' && (device === 'cuda' || device === 'cuda:0')) {
				throw new RequestValidationError(
					getCandleCudaPolicy(trainerEnv).reason ?? 'Candle CUDA is unavailable; choose CPU',
					422
				);
			}
			throw new RequestValidationError(
				`supervised ${backend} cannot run on ${device} on this server; enable the backend feature or choose a supported runtime`,
				422
			);
		}
		const trainFraction = normalizeNumber(payload.train_fraction, 0.6, 'train_fraction', 0.05, 0.9);
		const validationFraction = normalizeNumber(payload.validation_fraction, 0.2, 'validation_fraction', 0.05, 0.9);
		const checkpointEvery = normalizeInteger(payload.checkpoint_every, 0, 'checkpoint_every', 0, 1_000_000);
		if (trainFraction + validationFraction >= 0.95) {
			throw new RequestValidationError('train_fraction + validation_fraction must leave at least 5% holdout');
		}

		let resumePolicy: string | undefined;
		if (mode === 'continue') {
			resumePolicy = resolveExistingFile({
				rootReal,
				value: payload.resume_policy,
				label: 'resume_policy',
				extensions: ['json'],
				confinedTo: runsDirectory
			});
			if (path.basename(resumePolicy) !== 'policy.json') {
				throw new RequestValidationError('resume_policy must point to a policy.json artifact');
			}
			const resumeArtifacts = snapshotCompletionArtifactPair({
				directory: path.dirname(resumePolicy),
				manifestFile: 'run.complete.json',
				expectedSchema: 'supervised-training-completion-v1',
				expectedStatus: 'complete',
				requireSupervisedMetadata: true,
				label: 'supervised resume'
			});
			if (resumeArtifacts.sourcePolicyPath !== resumePolicy) {
				throw new RequestValidationError(
					'resume_policy is not the policy named by its completion manifest',
					500
				);
			}
			resumedFrom = resumePolicy;
			resumeSnapshotPath = resumeArtifacts.policyPath;
			resumePolicy = resumeArtifacts.policyPath;
		}

		const runName = `run-${Date.now()}-${randomUUID().replace(/-/g, '').slice(0, 16)}`;
		runDirectory = path.join(runsDirectory, runName);
		fs.mkdirSync(runDirectory, { recursive: false });
		runDirectory = fs.realpathSync(runDirectory);
		if (!pathIsWithin(runsDirectory, runDirectory)) {
			throw new RequestValidationError('generated run directory resolved outside supervised runs', 500);
		}

		const featureArgs = backend === 'cpu-linear'
			? []
			: ['--features', resolveBackendFeatures(backend as MlBackend, trainerEnv, device).join(',')];
		const args = [
			'run', '--quiet', ...featureArgs, '--bin', 'supervised', '--', 'train',
			'--input', inputPath,
			'--outdir', runDirectory,
			'--epochs', String(normalizeInteger(payload.epochs, 50, 'epochs', 1, 100_000)),
			'--learning-rate', String(normalizeNumber(payload.learning_rate, 0.001, 'learning_rate', Number.MIN_VALUE, 1)),
			'--l2', String(normalizeNumber(payload.l2, 0.0001, 'l2', 0, 1)),
			'--seed', String(normalizeInteger(payload.seed, 42, 'seed', 0, Number.MAX_SAFE_INTEGER)),
			'--device', device,
			'--backend', backend,
			'--train-fraction', String(trainFraction),
			'--validation-fraction', String(validationFraction),
			'--checkpoint-every', String(checkpointEvery)
		];
		if (resumePolicy) args.push('--resume-policy', resumePolicy);

		let result;
		try {
			result = await runSupervised(
				context.cargoCommand,
				args,
				rootReal,
				trainerEnv,
				request.signal,
				TRAIN_TIMEOUT_MS
			);
		} catch (error) {
			const details = executionErrorDetails(error, request.signal, TRAIN_TIMEOUT_MS, 'supervised train');
			return errorResponse(details.status, details.message, {
				...details,
				datasetPath: relativeApiPath(rootReal, inputPath),
				runDir: relativeApiPath(rootReal, runDirectory)
			});
		}

		let artifacts;
		try {
			artifacts = snapshotCompletionArtifactPair({
				directory: runDirectory,
				manifestFile: 'run.complete.json',
				expectedSchema: 'supervised-training-completion-v1',
				expectedStatus: 'complete',
				requireSupervisedMetadata: true,
				label: 'supervised training'
			});
		} catch (error) {
			if (error instanceof RequestValidationError) {
				return errorResponse(error.status, error.message, {
					stdout: result.stdout,
					stderr: result.stderr,
					datasetPath: relativeApiPath(rootReal, inputPath),
					runDir: relativeApiPath(rootReal, runDirectory)
				});
			}
			throw error;
		}
		const summary = parseJsonStdout(result.stdout);
		return json(
			{
				ok: true,
				summary,
				datasetPath: relativeApiPath(rootReal, inputPath),
				runDir: relativeApiPath(rootReal, runDirectory),
				policyPath: relativeApiPath(rootReal, artifacts.policyPath),
				metricsPath: relativeApiPath(rootReal, artifacts.metricsPath),
				completionPath: relativeApiPath(rootReal, artifacts.manifestPath),
				resumedFrom: resumedFrom ? relativeApiPath(rootReal, resumedFrom) : null,
				resumeSnapshotPath: resumeSnapshotPath ? relativeApiPath(rootReal, resumeSnapshotPath) : null,
				stdout: result.stdout,
				stderr: result.stderr
			},
			{ headers: { 'Cache-Control': 'no-store' } }
		);
	} catch (error) {
		if (error instanceof RequestValidationError) {
			return errorResponse(error.status, error.message, runDirectory && rootReal ? { runDir: relativeApiPath(rootReal, runDirectory) } : {});
		}
		return errorResponse(500, error instanceof Error ? error.message : 'Unable to train supervised model');
	}
};

const pathIsWithin = (directory: string, candidate: string) => {
	const relative = path.relative(directory, candidate);
	return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
};
