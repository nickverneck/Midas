import { execFile } from 'child_process';
import { createHash } from 'crypto';
import fs from 'fs';
import path from 'path';
import { loadDotEnv, resolveCargoBin, resolveProjectRoot } from './ml_env';

export const SUPERVISED_DATASET_DIR = path.join('.run', 'supervised', 'datasets');
export const SUPERVISED_RUNS_DIR = path.join('.run', 'supervised', 'runs');
export const MAX_PATH_LENGTH = 2048;
export const MAX_BUFFER = 64 * 1024 * 1024;

export type JsonRecord = Record<string, unknown>;
export type ChildProcessResult = { stdout: string; stderr: string };
export type ChildProcessError = Error & {
	code?: number | string;
	killed?: boolean;
	signal?: NodeJS.Signals | null;
	stdout?: string;
	stderr?: string;
};

export class RequestValidationError extends Error {
	readonly status: number;

	constructor(message: string, status = 400) {
		super(message);
		this.name = 'RequestValidationError';
		this.status = status;
	}
}

export const isRecord = (value: unknown): value is JsonRecord =>
	typeof value === 'object' && value !== null && !Array.isArray(value);

export const isWithin = (root: string, candidate: string) => {
	const relative = path.relative(root, candidate);
	return (
		relative === '' ||
		(relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
	);
};

export const hasControlCharacters = (value: string) => /\p{Cc}/u.test(value);

export const getProjectContext = () => {
	const root = path.resolve(resolveProjectRoot());
	let rootReal: string;
	try {
		rootReal = fs.realpathSync(root);
	} catch {
		throw new RequestValidationError('Midas project root could not be resolved', 500);
	}
	const dotenv = loadDotEnv(rootReal);
	const env = { ...process.env, ...dotenv };
	return { root: rootReal, rootReal, env, cargoCommand: resolveCargoBin(env) };
};

export const ensureConfinedDirectory = (
	rootReal: string,
	relativeDirectory: string,
	label: string
) => {
	const directory = path.resolve(rootReal, relativeDirectory);
	if (!isWithin(rootReal, directory)) {
		throw new RequestValidationError(`${label} is outside the Midas project root`, 500);
	}
	try {
		fs.mkdirSync(directory, { recursive: true });
	} catch {
		throw new RequestValidationError(`${label} could not be created`, 500);
	}
	let realDirectory: string;
	try {
		realDirectory = fs.realpathSync(directory);
	} catch {
		throw new RequestValidationError(`${label} could not be resolved`, 500);
	}
	if (!isWithin(rootReal, realDirectory)) {
		throw new RequestValidationError(`${label} must remain under the Midas project root`, 500);
	}
	return realDirectory;
};

export const resolveExistingFile = ({
	rootReal,
	value,
	label,
	extensions,
	confinedTo
}: {
	rootReal: string;
	value: unknown;
	label: string;
	extensions: readonly string[];
	confinedTo?: string;
}) => {
	if (typeof value !== 'string' || value.trim() === '') {
		throw new RequestValidationError(`${label} is required`);
	}
	if (value.length > MAX_PATH_LENGTH || value.includes('\0') || hasControlCharacters(value)) {
		throw new RequestValidationError(`${label} contains an invalid character or is too long`);
	}

	let candidate: string;
	try {
		candidate = path.resolve(rootReal, value.trim());
	} catch {
		throw new RequestValidationError(`${label} is not a valid path`);
	}
	const boundary = confinedTo ?? rootReal;
	if (!isWithin(boundary, candidate)) {
		throw new RequestValidationError(`${label} must be under ${relativeApiPath(rootReal, boundary) || 'the Midas project root'}`);
	}
	const extension = path.extname(candidate).slice(1).toLowerCase();
	if (!extensions.includes(extension)) {
		throw new RequestValidationError(`${label} must be a ${extensions.map((item) => `.${item}`).join(' or ')} file`);
	}

	let realCandidate: string;
	try {
		if (!fs.statSync(candidate).isFile()) {
			throw new RequestValidationError(`${label} must refer to a regular file`);
		}
		realCandidate = fs.realpathSync(candidate);
	} catch (error) {
		if (error instanceof RequestValidationError) throw error;
		throw new RequestValidationError(`${label} was not found`, 404);
	}
	if (!isWithin(rootReal, realCandidate) || !isWithin(boundary, realCandidate)) {
		throw new RequestValidationError(`${label} resolves outside its allowed directory`);
	}
	return realCandidate;
};

export const inspectGeneratedFile = (directory: string, filePath: string, label: string) => {
	if (!isWithin(directory, filePath)) {
		throw new RequestValidationError(`${label} is outside its artifact directory`, 500);
	}
	try {
		const stat = fs.lstatSync(filePath);
		if (stat.isSymbolicLink() || !stat.isFile()) {
			throw new RequestValidationError(`${label} is not a regular artifact file`, 500);
		}
		const realPath = fs.realpathSync(filePath);
		if (!isWithin(directory, realPath)) {
			throw new RequestValidationError(`${label} resolves outside its artifact directory`, 500);
		}
		return realPath;
	} catch (error) {
		if (error instanceof RequestValidationError) throw error;
		throw new RequestValidationError(`${label} was not created`, 500);
	}
};

const SHA256_PATTERN = /^[0-9a-f]{64}$/;

export type CompletionArtifactPair = {
	manifestPath: string;
	policyPath: string;
	metricsPath: string;
	manifest: JsonRecord;
};

export type SnapshotCompletionArtifactPair = CompletionArtifactPair & {
	snapshotDirectory: string;
	sourceManifestPath: string;
	sourcePolicyPath: string;
	sourceMetricsPath: string;
};

type VerifiedArtifactBytes = CompletionArtifactPair & {
	manifestBytes: Buffer;
	policyBytes: Buffer;
	metricsBytes: Buffer;
};

const noFollowFlag = () => {
	const flag = fs.constants.O_NOFOLLOW;
	if (typeof flag !== 'number') {
		throw new RequestValidationError(
			'completion artifacts cannot be verified safely because the platform lacks O_NOFOLLOW',
			500
		);
	}
	return flag;
};

/**
 * Read one artifact through a no-follow descriptor. The descriptor, rather
 * than the pathname, owns the bytes used by verification and snapshotting.
 * A later rename/replacement of the source path therefore cannot change the
 * bytes that are copied into the private snapshot.
 */
const readStableArtifact = (filePath: string, label: string) => {
	let descriptor: number | undefined;
	try {
		descriptor = fs.openSync(filePath, fs.constants.O_RDONLY | noFollowFlag());
		const stat = fs.fstatSync(descriptor);
		if (!stat.isFile()) {
			throw new RequestValidationError(`${label} is not a regular artifact file`, 500);
		}
		return fs.readFileSync(descriptor);
	} catch (error) {
		if (error instanceof RequestValidationError) throw error;
		throw new RequestValidationError(
			`${label} could not be read as a stable regular artifact`,
			500
		);
	} finally {
		if (descriptor !== undefined) {
			try {
				fs.closeSync(descriptor);
			} catch {
				// The descriptor is already unusable; preserve the verification error.
			}
		}
	}
};

const isSimpleArtifactName = (value: string) =>
	value.length > 0 && path.basename(value) === value && value !== '.' && value !== '..';

const readVerifiedCompletionArtifactPair = ({
	directory,
	manifestFile,
	expectedSchema,
	expectedStatus,
	requireSupervisedMetadata = false,
	policyFile = 'policy.json',
	metricsFile = 'metrics.json',
	label
}: {
	directory: string;
	manifestFile: string;
	expectedSchema: string;
	expectedStatus?: string;
	requireSupervisedMetadata?: boolean;
	policyFile?: string;
	metricsFile?: string;
	label: string;
}): VerifiedArtifactBytes => {
	if (!isSimpleArtifactName(manifestFile) || !isSimpleArtifactName(policyFile) || !isSimpleArtifactName(metricsFile)) {
		throw new RequestValidationError(`${label} completion artifact names must be simple file names`, 500);
	}

	// inspectGeneratedFile retains the existing lexical, lstat, realpath, and
	// project-directory checks. We then open the returned real paths with
	// O_NOFOLLOW so a replacement between inspection and reading is rejected.
	const manifestPath = inspectGeneratedFile(
		directory,
		path.join(directory, manifestFile),
		`${label} completion manifest`
	);
	const manifestBytes = readStableArtifact(manifestPath, `${label} completion manifest`);

	let manifest: JsonRecord;
	try {
		const parsed: unknown = JSON.parse(manifestBytes.toString('utf8'));
		if (!isRecord(parsed)) throw new Error('manifest must be a JSON object');
		manifest = parsed;
	} catch (error) {
		throw new RequestValidationError(
			`${label} completion manifest is missing or invalid: ${error instanceof Error ? error.message : String(error)}`,
			500
		);
	}

	if (manifest.schema_version !== expectedSchema) {
		throw new RequestValidationError(
			`${label} completion manifest has unsupported schema_version`,
			500
		);
	}
	if (expectedStatus !== undefined && manifest.status !== expectedStatus) {
		throw new RequestValidationError(
			`${label} completion manifest is not marked ${expectedStatus}`,
			500
		);
	}
	if (expectedStatus === undefined && 'status' in manifest && manifest.status !== 'complete') {
		throw new RequestValidationError(
			`${label} completion manifest is not marked complete`,
			500
		);
	}
	if (manifest.policy_file !== policyFile || manifest.metrics_file !== metricsFile) {
		throw new RequestValidationError(
			`${label} completion manifest names unexpected artifact paths`,
			500
		);
	}
	if (typeof manifest.policy_sha256 !== 'string' || !SHA256_PATTERN.test(manifest.policy_sha256)) {
		throw new RequestValidationError(`${label} completion manifest has an invalid policy hash`, 500);
	}
	if (typeof manifest.metrics_sha256 !== 'string' || !SHA256_PATTERN.test(manifest.metrics_sha256)) {
		throw new RequestValidationError(`${label} completion manifest has an invalid metrics hash`, 500);
	}
	if (!requireSupervisedMetadata) {
		if (typeof manifest.generation !== 'string' || manifest.generation.trim() === '') {
			throw new RequestValidationError(`${label} completion manifest has an invalid generation`, 500);
		}
	} else {
		if (
			typeof manifest.dataset_fingerprint_sha256 !== 'string' ||
			!SHA256_PATTERN.test(manifest.dataset_fingerprint_sha256)
		) {
			throw new RequestValidationError(
				`${label} completion manifest has an invalid dataset fingerprint`,
				500
			);
		}
		if (
			typeof manifest.total_epochs !== 'number' ||
			!Number.isSafeInteger(manifest.total_epochs) ||
			(manifest.total_epochs as number) < 1
		) {
			throw new RequestValidationError(`${label} completion manifest has invalid total_epochs`, 500);
		}
	}

	const policyPath = inspectGeneratedFile(
		directory,
		path.join(directory, policyFile),
		`${label} policy`
	);
	const metricsPath = inspectGeneratedFile(
		directory,
		path.join(directory, metricsFile),
		`${label} metrics`
	);
	const policyBytes = readStableArtifact(policyPath, `${label} policy`);
	const metricsBytes = readStableArtifact(metricsPath, `${label} metrics`);

	if (createHash('sha256').update(policyBytes).digest('hex') !== manifest.policy_sha256) {
		throw new RequestValidationError(`${label} policy does not match its completion manifest`, 500);
	}
	if (createHash('sha256').update(metricsBytes).digest('hex') !== manifest.metrics_sha256) {
		throw new RequestValidationError(`${label} metrics do not match its completion manifest`, 500);
	}

	return { manifestPath, policyPath, metricsPath, manifest, manifestBytes, policyBytes, metricsBytes };
};

const writePrivateSnapshotFile = (filePath: string, contents: Buffer, label: string) => {
	let descriptor: number | undefined;
	try {
		descriptor = fs.openSync(
			filePath,
			fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL | noFollowFlag(),
			0o400
		);
		let offset = 0;
		while (offset < contents.length) {
			offset += fs.writeSync(descriptor, contents, offset, contents.length - offset);
		}
		fs.fsyncSync(descriptor);
	} catch (error) {
		throw new RequestValidationError(
			`${label} could not be materialized as a private snapshot: ${error instanceof Error ? error.message : String(error)}`,
			500
		);
	} finally {
		if (descriptor !== undefined) {
			try {
				fs.closeSync(descriptor);
			} catch {
				// Preserve the original write error.
			}
		}
	}
};

/**
 * Verify a completion pair and materialize the exact verified bytes into a
 * private, read-only sibling directory. The source run remains available for
 * diagnostics, while callers use the snapshot path for subsequent Cargo
 * arguments and API responses.
 */
export const snapshotCompletionArtifactPair = (options: {
	directory: string;
	manifestFile: string;
	expectedSchema: string;
	expectedStatus?: string;
	requireSupervisedMetadata?: boolean;
	policyFile?: string;
	metricsFile?: string;
	label: string;
}): SnapshotCompletionArtifactPair => {
	const verified = readVerifiedCompletionArtifactPair(options);
	let snapshotDirectory: string | undefined;
	try {
		const sourceDirectory = fs.realpathSync(options.directory);
		const parentDirectory = fs.realpathSync(path.dirname(sourceDirectory));
		if (!isWithin(parentDirectory, sourceDirectory)) {
			throw new RequestValidationError(`${options.label} artifact directory has an invalid parent`, 500);
		}
		snapshotDirectory = fs.mkdtempSync(path.join(parentDirectory, '.verified-artifacts-'));
		if (!isWithin(parentDirectory, snapshotDirectory)) {
			throw new RequestValidationError(`${options.label} snapshot escaped its artifact directory`, 500);
		}
		fs.chmodSync(snapshotDirectory, 0o700);

		const manifestFile = options.manifestFile;
		const policyFile = options.policyFile ?? 'policy.json';
		const metricsFile = options.metricsFile ?? 'metrics.json';
		const snapshotManifestPath = path.join(snapshotDirectory, manifestFile);
		const snapshotPolicyPath = path.join(snapshotDirectory, policyFile);
		const snapshotMetricsPath = path.join(snapshotDirectory, metricsFile);
		writePrivateSnapshotFile(snapshotPolicyPath, verified.policyBytes, `${options.label} policy`);
		writePrivateSnapshotFile(snapshotMetricsPath, verified.metricsBytes, `${options.label} metrics`);
		// Publish the manifest last. A partially written snapshot can never look
		// complete to a later resume request.
		writePrivateSnapshotFile(snapshotManifestPath, verified.manifestBytes, `${options.label} completion manifest`);
		fs.chmodSync(snapshotDirectory, 0o500);

		return {
			snapshotDirectory,
			manifestPath: snapshotManifestPath,
			policyPath: snapshotPolicyPath,
			metricsPath: snapshotMetricsPath,
			manifest: verified.manifest,
			sourceManifestPath: verified.manifestPath,
			sourcePolicyPath: verified.policyPath,
			sourceMetricsPath: verified.metricsPath
		};
	} catch (error) {
		if (snapshotDirectory) {
			try {
				fs.chmodSync(snapshotDirectory, 0o700);
				fs.rmSync(snapshotDirectory, { recursive: true, force: true });
			} catch {
				// The snapshot was created under the trusted run parent; leave any
				// incomplete directory for diagnostics if cleanup is unavailable.
			}
		}
		if (error instanceof RequestValidationError) throw error;
		throw new RequestValidationError(
			`${options.label} artifact snapshot could not be created: ${error instanceof Error ? error.message : String(error)}`,
			500
		);
	}
};

/**
 * Verify the last-published completion marker before exposing generated
 * artifacts. Both trainers publish their policy/metrics pair first and the
 * manifest last, so a missing or malformed marker is an incomplete run.
 *
 * Callers should supply expectedStatus for every published artifact format.
 * This keeps a valid-looking but incomplete manifest from being exposed.
 */
export const verifyCompletionArtifactPair = ({
	directory,
	manifestFile,
	expectedSchema,
	expectedStatus,
	requireSupervisedMetadata,
	policyFile = 'policy.json',
	metricsFile = 'metrics.json',
	label
}: {
	directory: string;
	manifestFile: string;
	expectedSchema: string;
	expectedStatus?: string;
	requireSupervisedMetadata?: boolean;
	policyFile?: string;
	metricsFile?: string;
	label: string;
}): CompletionArtifactPair => {
	const verified = readVerifiedCompletionArtifactPair({
		directory,
		manifestFile,
		expectedSchema,
		expectedStatus,
		requireSupervisedMetadata,
		policyFile,
		metricsFile,
		label
	});
	return {
		manifestPath: verified.manifestPath,
		policyPath: verified.policyPath,
		metricsPath: verified.metricsPath,
		manifest: verified.manifest
	};
};

export const runSupervised = (
	command: string,
	args: string[],
	root: string,
	env: NodeJS.ProcessEnv,
	signal: AbortSignal,
	timeout: number
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
				timeout,
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

export const parseJsonStdout = (stdout: string) => {
	const text = stdout.trim();
	try {
		const parsed: unknown = JSON.parse(text);
		if (!isRecord(parsed)) throw new Error('stdout JSON is not an object');
		return parsed;
	} catch (error) {
		// The supervised CLI emits a few human-readable runtime notes before
		// its final JSON document. Keep accepting those lines, but only parse a
		// complete object so malformed or truncated output still fails closed.
		const start = text.indexOf('{');
		const end = text.lastIndexOf('}');
		if (start >= 0 && end > start) {
			try {
				const parsed: unknown = JSON.parse(text.slice(start, end + 1));
				if (!isRecord(parsed)) throw new Error('stdout JSON is not an object');
				return parsed;
			} catch {
				// Fall through to the original, more useful parse error below.
			}
		}
		throw new RequestValidationError(
			`supervised CLI returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`,
			500
		);
	}
};

export const relativeApiPath = (root: string, absolutePath: string) =>
	path.relative(root, absolutePath).split(path.sep).join('/');

export const executionErrorDetails = (
	error: unknown,
	signal: AbortSignal,
	timeoutMs: number,
	commandLabel: string
) => {
	const childError = error as ChildProcessError;
	const aborted =
		signal.aborted || childError.name === 'AbortError' || childError.code === 'ABORT_ERR';
	const timedOut =
		!aborted &&
		(childError.code === 'ETIMEDOUT' ||
			(childError.killed === true && childError.signal === 'SIGTERM'));
	const code = childError.code === undefined ? '' : ` (exit ${childError.code})`;
	return {
		status: aborted ? 499 : timedOut ? 504 : 500,
		message: aborted
			? `${commandLabel} stopped because the request was aborted`
			: timedOut
				? `${commandLabel} timed out after ${timeoutMs / 1000} seconds`
				: `${commandLabel} failed${code}: ${childError.message}`,
		stdout: childError.stdout ?? '',
		stderr: childError.stderr ?? childError.message,
		aborted,
		timeoutMs
	};
};
