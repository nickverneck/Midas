import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import { json } from '@sveltejs/kit';
import { loadDotEnv, resolveCargoBin, resolveProjectRoot } from '$lib/server/ml_env';

const DATASET_DIR = path.join('.run', 'meta-gate', 'datasets');
const MAX_BUFFER = 64 * 1024 * 1024;
const EVALUATE_TIMEOUT_MS = 10 * 60 * 1000;

type JsonRecord = Record<string, unknown>;

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

const relativeApiPath = (root: string, absolutePath: string) =>
	path.relative(root, absolutePath).split(path.sep).join('/');

const resolveInputPath = (root: string, rootReal: string, value: unknown) => {
	if (typeof value !== 'string' || value.trim() === '') {
		throw new RequestValidationError('input must be a parquet path');
	}
	if (value.includes('\0')) {
		throw new RequestValidationError('input contains an invalid character');
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
				timeout: EVALUATE_TIMEOUT_MS,
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

const errorResponse = (status: number, message: string, extra: Record<string, unknown> = {}) =>
	json(
		{
			ok: false,
			error: message,
			...extra
		},
		{ status, headers: { 'Cache-Control': 'no-store' } }
	);

export const POST = async ({ request }: { request: Request }) => {
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

	try {
		const root = path.resolve(resolveProjectRoot());
		const rootReal = fs.realpathSync(root);
		const inputPath = resolveInputPath(
			root,
			rootReal,
			payload.input ?? payload.inputPath ?? payload.input_path
		);
		const dotenv = loadDotEnv(root);
		const env = { ...process.env, ...dotenv };
		const cargoCommand = resolveCargoBin(env);
		const cargoArgs = [
			'run',
			'--quiet',
			'--bin',
			'evaluate_meta_dataset',
			'--',
			'--input',
			inputPath
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
				? 'evaluate_meta_dataset aborted because the client request disconnected'
				: timedOut
					? `evaluate_meta_dataset timed out after ${EVALUATE_TIMEOUT_MS / 1000} seconds`
					: `evaluate_meta_dataset failed${code}: ${childError.message}`;
			return errorResponse(status, message, {
				stdout: childError.stdout ?? '',
				stderr: childError.stderr ?? childError.message,
				inputPath: relativeApiPath(rootReal, inputPath),
				timeoutMs: EVALUATE_TIMEOUT_MS,
				aborted: requestAborted
			});
		}

		return json(
			{
				ok: true,
				inputPath: relativeApiPath(rootReal, inputPath),
				diagnosticOnly: true,
				warning:
					'Fixed-horizon baseline evaluation is diagnostic only; it does not replay Trader position or protection lifecycle.',
				stdout: result.stdout,
				stderr: result.stderr
			},
			{ headers: { 'Cache-Control': 'no-store' } }
		);
	} catch (error) {
		if (error instanceof RequestValidationError) {
			return errorResponse(error.status, error.message);
		}
		const message = error instanceof Error ? error.message : 'Unable to evaluate meta-gate dataset';
		return errorResponse(500, message);
	}
};
