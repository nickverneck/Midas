import { createHash, randomUUID } from 'crypto';
import fs from 'fs';
import path from 'path';
import { json } from '@sveltejs/kit';
import type { RequestEvent } from '@sveltejs/kit';
import {
	SUPERVISED_DATASET_DIR,
	RequestValidationError,
	ensureConfinedDirectory,
	executionErrorDetails,
	getProjectContext,
	hasControlCharacters,
	inspectGeneratedFile,
	isRecord,
	parseJsonStdout,
	relativeApiPath,
	resolveExistingFile,
	runSupervised,
	type JsonRecord
} from '$lib/server/supervised';

const PREPARE_TIMEOUT_MS = 10 * 60 * 1000;
const INDICATORS = ['ema', 'sma', 'hma', 'kama', 'alma', 'atr', 'adx', 'rvol', 'er'] as const;
const BAR_KINDS = ['minute', 'second', 'tick', 'volume', 'range'] as const;
const TIMESTAMP_UNITS = ['ns', 'us', 'ms', 's'] as const;

type FeatureSpec = {
	indicator: (typeof INDICATORS)[number];
	period: number;
	lookbacks: number[];
	include_value: boolean;
	include_delta: boolean;
	normalize_by_atr: boolean;
};

const errorResponse = (status: number, error: string, extra: JsonRecord = {}) =>
	json({ ok: false, error, ...extra }, { status, headers: { 'Cache-Control': 'no-store' } });

const normalizeString = (value: unknown, fallback: string, name: string, max = 128) => {
	if (value === undefined || value === null || value === '') return fallback;
	if (typeof value !== 'string' || hasControlCharacters(value)) {
		throw new RequestValidationError(`${name} must be a string without control characters`);
	}
	const normalized = value.trim();
	if (!normalized || normalized.length > max) {
		throw new RequestValidationError(`${name} must contain 1 to ${max} characters`);
	}
	return normalized;
};

const normalizeChoice = <T extends string>(
	value: unknown,
	fallback: T,
	name: string,
	choices: readonly T[]
) => {
	const normalized = normalizeString(value, fallback, name).toLowerCase() as T;
	if (!choices.includes(normalized)) {
		throw new RequestValidationError(`${name} must be one of ${choices.join(', ')}`);
	}
	return normalized;
};

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

const normalizeBoolean = (value: unknown, fallback: boolean, name: string) => {
	if (value === undefined || value === null) return fallback;
	if (typeof value !== 'boolean') throw new RequestValidationError(`${name} must be boolean`);
	return value;
};

const normalizeFeatures = (value: unknown): FeatureSpec[] => {
	if (!Array.isArray(value) || value.length === 0 || value.length > 64) {
		throw new RequestValidationError('features must contain 1 to 64 feature specs');
	}
	return value.map((raw, index) => {
		if (!isRecord(raw)) throw new RequestValidationError(`features[${index}] must be an object`);
		const indicator = normalizeChoice(raw.indicator, 'ema', `features[${index}].indicator`, INDICATORS);
		const period = normalizeInteger(raw.period, 14, `features[${index}].period`, indicator === 'hma' ? 2 : 1, 1_000_000);
		if (!Array.isArray(raw.lookbacks) || raw.lookbacks.length === 0 || raw.lookbacks.length > 32) {
			throw new RequestValidationError(`features[${index}].lookbacks must contain 1 to 32 values`);
		}
		const lookbacks = raw.lookbacks.map((lookback, lookbackIndex) =>
			normalizeInteger(lookback, 1, `features[${index}].lookbacks[${lookbackIndex}]`, 1, 1_000_000)
		);
		if (new Set(lookbacks).size !== lookbacks.length) {
			throw new RequestValidationError(`features[${index}].lookbacks must not contain duplicates`);
		}
		const include_value = normalizeBoolean(raw.include_value, true, `features[${index}].include_value`);
		const include_delta = normalizeBoolean(raw.include_delta, true, `features[${index}].include_delta`);
		if (!include_value && !include_delta) {
			throw new RequestValidationError(`features[${index}] must include value or delta`);
		}
		return {
			indicator,
			period,
			lookbacks,
			include_value,
			include_delta,
			normalize_by_atr: normalizeBoolean(raw.normalize_by_atr, true, `features[${index}].normalize_by_atr`)
		};
	});
};

const sanitizePart = (value: string, fallback: string) =>
	value
		.normalize('NFKD')
		.replace(/[^a-zA-Z0-9._-]+/g, '-')
		.replace(/^[.-]+|[.-]+$/g, '')
		.toLowerCase()
		.slice(0, 36) || fallback;

const sourceFingerprint = (filePath: string) => {
	const stat = fs.statSync(filePath);
	return createHash('sha256')
		.update(`${filePath}:${stat.size}:${stat.mtimeMs}`)
		.digest('hex')
		.slice(0, 12);
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

	let outputPath: string | undefined;
	let outputDirectory: string | undefined;
	let completed = false;
	try {
		const { rootReal, env, cargoCommand } = getProjectContext();
		const inputPath = resolveExistingFile({
			rootReal,
			value: payload.input ?? payload.input_path,
			label: 'input',
			extensions: ['parquet', 'csv', 'txt']
		});
		const instrument = normalizeString(payload.instrument, 'UNKNOWN', 'instrument');
		const contract = normalizeString(payload.contract, 'UNKNOWN', 'contract');
		const triggerFast = normalizeInteger(payload.trigger_fast, 10, 'trigger_fast', 1, 1_000_000);
		const triggerSlow = normalizeInteger(payload.trigger_slow, 30, 'trigger_slow', 1, 1_000_000);
		const contextFast = normalizeInteger(payload.context_fast, 210, 'context_fast', 1, 1_000_000);
		const contextSlow = normalizeInteger(payload.context_slow, 240, 'context_slow', 1, 1_000_000);
		if (triggerFast >= triggerSlow) throw new RequestValidationError('trigger_fast must be less than trigger_slow');
		if (contextFast >= contextSlow) throw new RequestValidationError('context_fast must be less than context_slow');
		const features = normalizeFeatures(payload.features);
		const barKind = normalizeChoice(payload.bar_kind, 'minute', 'bar_kind', BAR_KINDS);
		const timestampUnit = payload.timestamp_unit
			? normalizeChoice(payload.timestamp_unit, 'ns', 'timestamp_unit', TIMESTAMP_UNITS)
			: null;
		outputDirectory = ensureConfinedDirectory(rootReal, SUPERVISED_DATASET_DIR, 'supervised dataset directory');
		const outputName = [
			sanitizePart(instrument, 'unknown'),
			sanitizePart(contract, 'unknown'),
			`ema-${triggerFast}-${triggerSlow}`,
			sourceFingerprint(inputPath),
			randomUUID().replace(/-/g, '').slice(0, 10)
		].join('__');
		outputPath = path.join(outputDirectory, `${outputName}.parquet`);
		if (!isWithinOutput(outputDirectory, outputPath) || fs.existsSync(outputPath)) {
			throw new RequestValidationError('generated dataset path is not safe', 500);
		}

		const args = [
			'run', '--quiet', '--bin', 'supervised', '--', 'prepare',
			'--input', inputPath,
			'--output', outputPath,
			'--instrument', instrument,
			'--contract', contract,
			'--trigger-kind', 'ema',
			'--trigger-fast', String(triggerFast),
			'--trigger-slow', String(triggerSlow),
			'--context-kind', 'ema',
			'--context-fast', String(contextFast),
			'--context-slow', String(contextSlow),
			'--atr-period', String(normalizeInteger(payload.atr_period, 14, 'atr_period', 1, 1_000_000)),
			'--slope-lookback', String(normalizeInteger(payload.slope_lookback, 5, 'slope_lookback', 1, 1_000_000)),
			'--features', JSON.stringify(features),
			'--session-timezone', 'America/New_York',
			'--session-start-hour', '18',
			'--session-end-hour', '17',
			'--bar-kind', barKind,
			'--bar-value', String(normalizeNumber(payload.bar_value, 1, 'bar_value', Number.MIN_VALUE, Number.MAX_VALUE)),
			'--contract-multiplier', String(normalizeNumber(payload.contract_multiplier, 1, 'contract_multiplier', Number.MIN_VALUE, Number.MAX_VALUE)),
			'--round-trip-cost', String(normalizeNumber(payload.round_trip_cost, 0, 'round_trip_cost', 0, Number.MAX_VALUE))
		];
		if (timestampUnit) args.push('--timestamp-unit', timestampUnit);
		if (normalizeBoolean(payload.allow_index_timestamps, false, 'allow_index_timestamps')) {
			args.push('--allow-index-timestamps');
		}

		let result;
		try {
			result = await runSupervised(cargoCommand, args, rootReal, env, request.signal, PREPARE_TIMEOUT_MS);
		} catch (error) {
			const details = executionErrorDetails(error, request.signal, PREPARE_TIMEOUT_MS, 'supervised prepare');
			return errorResponse(details.status, details.message, {
				...details,
				artifactPath: relativeApiPath(rootReal, outputPath)
			});
		}
		const realOutputPath = inspectGeneratedFile(outputDirectory, outputPath, 'prepared dataset');
		const summary = parseJsonStdout(result.stdout);
		const response = json(
			{
				ok: true,
				summary,
				artifactPath: relativeApiPath(rootReal, realOutputPath),
				stdout: result.stdout,
				stderr: result.stderr
			},
			{ headers: { 'Cache-Control': 'no-store' } }
		);
		completed = true;
		return response;
	} catch (error) {
		if (error instanceof RequestValidationError) return errorResponse(error.status, error.message);
		return errorResponse(500, error instanceof Error ? error.message : 'Unable to prepare supervised dataset');
	} finally {
		if (!completed && outputDirectory && outputPath) removeGeneratedDataset(outputDirectory, outputPath);
	}
};

const isWithinOutput = (directory: string, candidate: string) => {
	const relative = path.relative(directory, candidate);
	return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
};

const removeGeneratedDataset = (directory: string, candidate: string) => {
	if (!isWithinOutput(directory, candidate) || path.extname(candidate).toLowerCase() !== '.parquet') return;
	try {
		const stat = fs.lstatSync(candidate);
		if (stat.isFile() || stat.isSymbolicLink()) fs.unlinkSync(candidate);
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
			// Preserve the original preparation/validation response if cleanup itself fails.
		}
	}
};
