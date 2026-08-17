import { createHash, randomUUID } from 'crypto';
import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import { json } from '@sveltejs/kit';
import {
	loadDotEnv,
	resolveCargoBin,
	resolveProjectRoot
} from '$lib/server/ml_env';

const OUTPUT_DIR = path.join('.run', 'meta-gate', 'datasets');
const MAX_BUFFER = 64 * 1024 * 1024;
const MAX_PATH_LENGTH = 2048;
const PREPARE_TIMEOUT_MS = 10 * 60 * 1000;

const DEFAULT_CONFIG = {
	trigger_kind: 'ema',
	trigger_fast: 10,
	trigger_slow: 30,
	context_kind: 'ema',
	context_fast: 210,
	context_slow: 240,
	atr_period: 14,
	slope_lookback: 5,
	horizon_bars: 30,
	contract_multiplier: 1,
	cost_pnl: 0
} as const;

type JsonRecord = Record<string, unknown>;
type MetaGateConfig = {
	trigger_kind: 'ema' | 'hma';
	trigger_fast: number;
	trigger_slow: number;
	context_kind: 'ema' | 'hma';
	context_fast: number;
	context_slow: number;
	atr_period: number;
	slope_lookback: number;
	horizon_bars: number;
	contract_multiplier: number;
	cost_pnl: number;
};

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

const isWithin = (root: string, candidate: string) => {
	const relative = path.relative(root, candidate);
	return (
		relative === '' ||
		(relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
	);
};

const normalizeLabel = (value: unknown, name: string) => {
	if (value === undefined || value === null) return 'UNKNOWN';
	if (typeof value !== 'string') {
		throw new RequestValidationError(`${name} must be a string`);
	}
	const label = value.trim();
	if (!label) throw new RequestValidationError(`${name} must not be empty`);
	if (label.length > 128) throw new RequestValidationError(`${name} is too long`);
	if (/\p{Cc}/u.test(label)) {
		throw new RequestValidationError(`${name} contains control characters`);
	}
	return label;
};

const normalizeKind = (value: unknown, fallback: 'ema' | 'hma', name: string) => {
	if (value === undefined || value === null || value === '') return fallback;
	if (typeof value !== 'string') {
		throw new RequestValidationError(`${name} must be ema or hma`);
	}
	const kind = value.trim().toLowerCase();
	if (kind !== 'ema' && kind !== 'hma') {
		throw new RequestValidationError(`${name} must be ema or hma`);
	}
	return kind;
};

const normalizeInteger = (value: unknown, fallback: number, name: string) => {
	if (value === undefined || value === null || value === '') return fallback;
	const numberValue =
		typeof value === 'number'
			? value
			: typeof value === 'string' && value.trim() !== ''
				? Number(value.trim())
				: Number.NaN;
	if (!Number.isSafeInteger(numberValue) || numberValue <= 0) {
		throw new RequestValidationError(`${name} must be a positive integer`);
	}
	return numberValue;
};

const normalizeNonNegativeNumber = (value: unknown, fallback: number, name: string) => {
	if (value === undefined || value === null || value === '') return fallback;
	const numberValue =
		typeof value === 'number'
			? value
			: typeof value === 'string' && value.trim() !== ''
				? Number(value.trim())
				: Number.NaN;
	if (!Number.isFinite(numberValue) || numberValue < 0) {
		throw new RequestValidationError(`${name} must be a finite, non-negative number`);
	}
	return numberValue;
};

const normalizeBoolean = (value: unknown, fallback: boolean, name: string) => {
	if (value === undefined || value === null || value === '') return fallback;
	if (typeof value !== 'boolean') {
		throw new RequestValidationError(`${name} must be a boolean`);
	}
	return value;
};

const normalizeConfig = (source: JsonRecord): MetaGateConfig => {
	const config: MetaGateConfig = {
		trigger_kind: normalizeKind(source.trigger_kind, DEFAULT_CONFIG.trigger_kind, 'trigger_kind'),
		trigger_fast: normalizeInteger(source.trigger_fast, DEFAULT_CONFIG.trigger_fast, 'trigger_fast'),
		trigger_slow: normalizeInteger(source.trigger_slow, DEFAULT_CONFIG.trigger_slow, 'trigger_slow'),
		context_kind: normalizeKind(source.context_kind, DEFAULT_CONFIG.context_kind, 'context_kind'),
		context_fast: normalizeInteger(source.context_fast, DEFAULT_CONFIG.context_fast, 'context_fast'),
		context_slow: normalizeInteger(source.context_slow, DEFAULT_CONFIG.context_slow, 'context_slow'),
		atr_period: normalizeInteger(source.atr_period, DEFAULT_CONFIG.atr_period, 'atr_period'),
		slope_lookback: normalizeInteger(
			source.slope_lookback,
			DEFAULT_CONFIG.slope_lookback,
			'slope_lookback'
		),
		horizon_bars: normalizeInteger(source.horizon_bars, DEFAULT_CONFIG.horizon_bars, 'horizon_bars'),
		contract_multiplier: normalizeNonNegativeNumber(
			source.contract_multiplier,
			DEFAULT_CONFIG.contract_multiplier,
			'contract_multiplier'
		),
		cost_pnl: normalizeNonNegativeNumber(
			source.cost_pnl,
			DEFAULT_CONFIG.cost_pnl,
			'cost_pnl'
		)
	};
	if (config.contract_multiplier <= 0) {
		throw new RequestValidationError('contract_multiplier must be greater than zero');
	}

	if (config.trigger_fast >= config.trigger_slow) {
		throw new RequestValidationError('trigger_fast must be less than trigger_slow');
	}
	if (config.context_fast >= config.context_slow) {
		throw new RequestValidationError('context_fast must be less than context_slow');
	}
	return config;
};

const sanitizeFilenamePart = (value: string, fallback: string) => {
	const sanitized = value
		.normalize('NFKD')
		.replace(/[^a-zA-Z0-9._-]+/g, '-')
		.replace(/^[.-]+|[.-]+$/g, '')
		.toLowerCase()
		.slice(0, 32);
	return sanitized || fallback;
};

const formatNumber = (value: number) => {
	if (Object.is(value, -0)) return '0';
	return String(value);
};

const buildOutputName = (
	instrument: string,
	contract: string,
	config: MetaGateConfig,
	sourceFingerprint: string
) => {
	const instrumentPart = sanitizeFilenamePart(instrument, 'unknown-instrument');
	const contractPart = sanitizeFilenamePart(contract, 'unknown-contract');
	const configPart = [
		`trigger-${config.trigger_kind}-${config.trigger_fast}-${config.trigger_slow}`,
		`context-${config.context_kind}-${config.context_fast}-${config.context_slow}`,
		`atr-${config.atr_period}`,
		`slope-${config.slope_lookback}`,
		`horizon-${config.horizon_bars}`,
		`multiplier-${formatNumber(config.contract_multiplier)}`,
		`cost-pnl-${formatNumber(config.cost_pnl)}`
	].map((part) => sanitizeFilenamePart(part, 'config'));
	return `${instrumentPart}__${contractPart}__${configPart.join('__')}__source-${sourceFingerprint}.parquet`;
};

const resolveInputPath = (root: string, rootReal: string, value: unknown) => {
	if (typeof value !== 'string' || value.trim() === '') {
		throw new RequestValidationError('input must be a parquet path');
	}
	if (value.length > MAX_PATH_LENGTH || value.includes('\0')) {
		throw new RequestValidationError('input contains an invalid character or is too long');
	}

	let candidate: string;
	try {
		candidate = path.resolve(root, value.trim());
	} catch {
		throw new RequestValidationError('input is not a valid path');
	}
	if (!isWithin(root, candidate)) {
		throw new RequestValidationError('input must be under the Midas project root');
	}
	if (!candidate.toLowerCase().endsWith('.parquet')) {
		throw new RequestValidationError('input must have a .parquet extension');
	}
	if (!fs.existsSync(candidate)) {
		throw new RequestValidationError('input parquet file was not found', 404);
	}
	if (!fs.statSync(candidate).isFile()) {
		throw new RequestValidationError('input must refer to a parquet file');
	}

	let realCandidate: string;
	try {
		realCandidate = fs.realpathSync(candidate);
	} catch {
		throw new RequestValidationError('input parquet file could not be resolved');
	}
	if (!isWithin(rootReal, realCandidate)) {
		throw new RequestValidationError('input must resolve under the Midas project root');
	}
	return realCandidate;
};

const resolveOutputDirectory = (root: string, rootReal: string) => {
	const outputDirectory = path.resolve(root, OUTPUT_DIR);
	if (!isWithin(root, outputDirectory)) {
		throw new RequestValidationError('meta-gate output directory is outside the Midas project root', 500);
	}
	try {
		fs.mkdirSync(outputDirectory, { recursive: true });
	} catch {
		throw new RequestValidationError('meta-gate output directory could not be created', 500);
	}
	let realOutputDirectory: string;
	try {
		realOutputDirectory = fs.realpathSync(outputDirectory);
	} catch {
		throw new RequestValidationError('meta-gate output directory could not be resolved', 500);
	}
	if (!isWithin(rootReal, realOutputDirectory)) {
		throw new RequestValidationError('meta-gate output directory must remain under the Midas project root', 500);
	}
	return { outputDirectory, realOutputDirectory };
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
				timeout: PREPARE_TIMEOUT_MS,
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

const sha256File = (filePath: string) =>
	new Promise<string>((resolve, reject) => {
		const hash = createHash('sha256');
		const stream = fs.createReadStream(filePath);
		stream.on('data', (chunk) => hash.update(chunk));
		stream.once('error', reject);
		stream.once('end', () => resolve(hash.digest('hex')));
	});

const removeTemporaryFile = (filePath: string) => {
	try {
		if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
	} catch {
		// The generated temporary path is disposable; preserve the original error.
	}
};

const isAlreadyExistsError = (error: unknown) =>
	error instanceof Error &&
	'code' in error &&
	(error as NodeJS.ErrnoException).code === 'EEXIST';

const publishOutput = (temporaryOutputPath: string, outputPath: string, overwrite: boolean) => {
	try {
		if (overwrite) {
			fs.renameSync(temporaryOutputPath, outputPath);
		} else {
			// Both paths are in the same directory. link(2) atomically creates the
			// destination without replacement, so a concurrent publisher gets EEXIST.
			fs.linkSync(temporaryOutputPath, outputPath);
			fs.unlinkSync(temporaryOutputPath);
		}
	} catch (error) {
		if (!overwrite && isAlreadyExistsError(error)) {
			throw new RequestValidationError(
				'Generated output appeared while preparing; retry with overwrite=true if replacement is intentional',
				409
			);
		}
		throw new RequestValidationError('Prepared dataset could not be committed atomically', 500);
	}
};

const inspectExistingOutput = (outputPath: string, realOutputDirectory: string) => {
	let stat: fs.Stats;
	try {
		const linkStat = fs.lstatSync(outputPath);
		if (linkStat.isSymbolicLink()) {
			throw new RequestValidationError('Generated output path is an existing symlink', 409);
		}
		stat = linkStat;
	} catch (error) {
		if (error instanceof RequestValidationError) throw error;
		if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return false;
		throw new RequestValidationError('Generated output path could not be inspected', 500);
	}
	if (!stat.isFile()) {
		throw new RequestValidationError('Generated output path is not a regular file', 409);
	}
	let realOutputPath: string;
	try {
		realOutputPath = fs.realpathSync(outputPath);
	} catch {
		throw new RequestValidationError('Generated output path could not be resolved', 500);
	}
	if (!isWithin(realOutputDirectory, realOutputPath)) {
		throw new RequestValidationError(
			'Generated output resolved outside the meta-gate dataset directory',
			500
		);
	}
	return true;
};

const errorResponse = (status: number, message: string, extra: Record<string, unknown> = {}) =>
	json(
		{
			ok: false,
			error: message,
			...extra
		},
		{ status, headers: { 'Cache-Control': 'no-store' } }
	);

import type { RequestEvent } from '@sveltejs/kit';

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

	let temporaryOutputPath: string | undefined;
	try {
		const root = path.resolve(resolveProjectRoot());
		const rootReal = fs.realpathSync(root);
		const inputPath = resolveInputPath(root, rootReal, payload.input ?? payload.input_path);
		const sourceSha256 = await sha256File(inputPath);
		// Keep the complete digest in the filename so different source files cannot
		// silently alias the same prepared dataset because of a short prefix.
		const sourceFingerprint = sourceSha256;
		const instrument = normalizeLabel(payload.instrument, 'instrument');
		const contract = normalizeLabel(payload.contract, 'contract');
		const overwrite = normalizeBoolean(payload.overwrite, false, 'overwrite');
		const configSource = isRecord(payload.config) ? payload.config : payload;
		const config = normalizeConfig(configSource);
		const { realOutputDirectory } = resolveOutputDirectory(root, rootReal);
		const outputPath = path.join(
			realOutputDirectory,
			buildOutputName(instrument, contract, config, sourceFingerprint)
		);

		if (!isWithin(realOutputDirectory, outputPath)) {
			return errorResponse(500, 'Generated output path is outside the meta-gate dataset directory');
		}
		if (path.resolve(inputPath) === path.resolve(outputPath)) {
			return errorResponse(400, 'input cannot be the generated output file');
		}

		const outputExists = inspectExistingOutput(outputPath, realOutputDirectory);
		if (outputExists && !overwrite) {
			const generatedOutputPath = relativeApiPath(rootReal, outputPath);
			return json(
				{
					ok: true,
					generatedOutputPath,
					outputPath: generatedOutputPath,
					sourceSha256,
					reused: true,
					diagnosticOnly: true,
					warning:
						'Existing event dataset reused; set overwrite=true only when intentionally regenerating it.',
					stdout: '',
					stderr: ''
				},
				{ headers: { 'Cache-Control': 'no-store' } }
			);
		}

		temporaryOutputPath = path.join(
			realOutputDirectory,
		`.${path.basename(outputPath)}.${randomUUID().replace(/-/g, '').slice(0, 16)}.tmp.parquet`
		);
		if (!isWithin(realOutputDirectory, temporaryOutputPath) || fs.existsSync(temporaryOutputPath)) {
			throw new RequestValidationError('Temporary meta-gate output path could not be allocated', 500);
		}

		const dotenv = loadDotEnv(root);
		const env = { ...process.env, ...dotenv };
		const cargoCommand = resolveCargoBin(env);
		const cargoArgs = [
			'run',
			'--quiet',
			'--bin',
			'prepare_meta_dataset',
			'--',
			'--input',
			inputPath,
			'--output',
			temporaryOutputPath,
			'--instrument',
			instrument,
			'--contract',
			contract,
			'--trigger-kind',
			config.trigger_kind,
			'--trigger-fast',
			String(config.trigger_fast),
			'--trigger-slow',
			String(config.trigger_slow),
			'--context-kind',
			config.context_kind,
			'--context-fast',
			String(config.context_fast),
			'--context-slow',
			String(config.context_slow),
			'--atr-period',
			String(config.atr_period),
			'--slope-lookback',
			String(config.slope_lookback),
			'--horizon-bars',
			String(config.horizon_bars),
			'--contract-multiplier',
			String(config.contract_multiplier),
			'--cost-pnl',
			String(config.cost_pnl)
		];

		let result: ChildProcessResult;
		try {
			result = await runCargo(cargoCommand, cargoArgs, root, env, request.signal);
		} catch (error) {
			removeTemporaryFile(temporaryOutputPath);
			temporaryOutputPath = undefined;
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
				? 'prepare_meta_dataset aborted because the client request disconnected'
				: timedOut
					? `prepare_meta_dataset timed out after ${PREPARE_TIMEOUT_MS / 1000} seconds`
					: `prepare_meta_dataset failed${code}: ${childError.message}`;
			return errorResponse(status, message, {
				stdout: childError.stdout ?? '',
				stderr: childError.stderr ?? childError.message,
				inputPath: relativeApiPath(rootReal, inputPath),
				outputPath: relativeApiPath(rootReal, outputPath),
				timeoutMs: PREPARE_TIMEOUT_MS,
				aborted: requestAborted
			});
		}

		if (!inspectExistingOutput(temporaryOutputPath, realOutputDirectory)) {
			removeTemporaryFile(temporaryOutputPath);
			temporaryOutputPath = undefined;
			return errorResponse(500, 'prepare_meta_dataset completed without creating its output file', {
				stdout: result.stdout,
				stderr: result.stderr
			});
		}

		if (!overwrite && inspectExistingOutput(outputPath, realOutputDirectory)) {
			throw new RequestValidationError(
				'Generated output appeared while preparing; retry with overwrite=true if replacement is intentional',
				409
			);
		}
		publishOutput(temporaryOutputPath, outputPath, overwrite);
		temporaryOutputPath = undefined;
		inspectExistingOutput(outputPath, realOutputDirectory);

		const generatedOutputPath = relativeApiPath(rootReal, outputPath);
		return json(
			{
				ok: true,
				generatedOutputPath,
				outputPath: generatedOutputPath,
				sourceSha256,
				reused: false,
				diagnosticOnly: true,
				warning:
					'Event datasets contain fixed-horizon diagnostic labels and are not executable Trader schedules.',
				stdout: result.stdout,
				stderr: result.stderr
			},
			{ headers: { 'Cache-Control': 'no-store' } }
		);
	} catch (error) {
		if (temporaryOutputPath) removeTemporaryFile(temporaryOutputPath);
		if (error instanceof RequestValidationError) {
			return errorResponse(error.status, error.message);
		}
		const message = error instanceof Error ? error.message : 'Unable to prepare meta-gate dataset';
		return errorResponse(500, message);
	}
};
