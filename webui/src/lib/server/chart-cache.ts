import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve } from 'node:path';
import type { RequestEvent } from '@sveltejs/kit';

const execFileAsync = promisify(execFile);
const MAX_BUFFER = 8 * 1024 * 1024;
const TIMEOUT_MS = 15_000;
const MAX_OFFSET = 10_000_000;
const MIN_I64 = -(1n << 63n);
const MAX_I64 = (1n << 63n) - 1n;
let running = false;

/** Parse the CLI JSON while retaining timestamp_ns integer tokens exactly.
 * JSON.parse converts every number to IEEE-754 before a reviver can inspect it.
 * This small structural parser only changes timestamp_ns values to strings;
 * all other JSON values keep normal JSON.parse semantics.
 */
export function parseChartCacheOutput(source: string): unknown {
	let index = 0;
	// JSON permits exactly these four whitespace characters. JavaScript's \s also
	// accepts characters such as NBSP and Unicode line separators.
	const whitespace = () => { while (index < source.length && /[ \t\r\n]/.test(source[index])) index++; };
	const fail = (): never => { throw new Error('invalid JSON'); };
	const string = (): string => {
		const start = index;
		if (source[index++] !== '"') return fail();
		while (index < source.length) {
			const char = source[index++];
			if (char === '"') {
				try { return JSON.parse(source.slice(start, index)); }
				catch { return fail(); }
			}
			if (char === '\\') index++;
			else if (char.charCodeAt(0) < 0x20) return fail();
		}
		return fail();
	};
	const value = (key?: string): unknown => {
		whitespace();
		if (source[index] === '{') {
			index++;
			const object: Record<string, unknown> = {};
			whitespace();
			if (source[index] === '}') { index++; return object; }
			while (index < source.length) {
				whitespace(); const name = string(); whitespace();
				if (source[index++] !== ':') return fail();
				Object.defineProperty(object, name, { value: value(name), enumerable: true, configurable: true, writable: true });
				whitespace(); const separator = source[index++];
				if (separator === '}') return object;
				if (separator !== ',') return fail();
			}
			return fail();
		}
		if (source[index] === '[') {
			index++; const array: unknown[] = []; whitespace();
			if (source[index] === ']') { index++; return array; }
			while (index < source.length) {
				array.push(value()); whitespace(); const separator = source[index++];
				if (separator === ']') return array;
				if (separator !== ',') return fail();
			}
			return fail();
		}
		if (source[index] === '"') return string();
		const start = index;
		while (index < source.length && !/[ \t\r\n,\]}]/.test(source[index])) index++;
		if (start === index) return fail();
		const token = source.slice(start, index);
		if (key === 'timestamp_ns' || key === 'ts_ns') {
			// Keep this aligned with JSON's integer number grammar: no leading
			// zeroes (except the value zero itself), decimal point, or exponent.
			if (!/^-?(?:0|[1-9]\d*)$/.test(token)) return fail();
			const ns = BigInt(token);
			if (ns < MIN_I64 || ns > MAX_I64) return fail();
			return token;
		}
		try { return JSON.parse(token); }
		catch { return fail(); }
	};
	const parsed = value(); whitespace();
	if (index !== source.length) return fail();
	return parsed;
}

export class BridgeError extends Error {
	readonly status: number;
	constructor(status: number, message: string) { super(message); this.status = status; }
}

function localHost(host: string | null): boolean {
	if (!host) return false;
	const value = host.toLowerCase().replace(/\.$/, '');
	return /^(localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?$/.test(value);
}

export function assertLocalRequest(event: RequestEvent): void {
	if (!localHost(event.request.headers.get('host'))) {
		throw new BridgeError(403, 'This local bridge only accepts requests addressed to localhost.');
	}
	const origin = event.request.headers.get('origin');
	if (origin) {
		try {
			if (!localHost(new URL(origin).host)) throw new Error('remote origin');
		} catch {
			throw new BridgeError(403, 'Cross-origin requests are not accepted by the local bridge.');
		}
	}
}

function cliPath(): string {
	const configured = process.env.MIDAS_TRADER_BIN;
	return configured ? resolve(configured) : resolve(process.cwd(), '..', 'trader', 'target', 'debug', 'trader');
}

export async function runChartCache(args: string[]): Promise<unknown> {
	if (running) throw new BridgeError(503, 'The chart cache reader is busy. Retry shortly.');
	running = true;
	try {
		const cliArgs: string[] = [];
		if (process.env.MIDAS_CONFIG) cliArgs.push('--config', resolve(process.env.MIDAS_CONFIG));
		cliArgs.push('chart-cache', ...args);
		const { stdout } = await execFileAsync(cliPath(), cliArgs, {
			cwd: process.cwd(),
			env: process.env,
			timeout: TIMEOUT_MS,
			maxBuffer: MAX_BUFFER,
			windowsHide: true
		});
		try { return parseChartCacheOutput(stdout); }
		catch { throw new BridgeError(502, 'The trader chart-cache command returned invalid JSON.'); }
	} catch (error) {
		if (error instanceof BridgeError) throw error;
		const child = error as NodeJS.ErrnoException & { stderr?: string; killed?: boolean; code?: string | number };
		if (child.code === 'ENOENT') {
			throw new BridgeError(503, `Trader CLI not found at ${cliPath()}. Build it with cargo build --features replay or set MIDAS_TRADER_BIN.`);
		}
		if (child.killed) throw new BridgeError(504, 'The trader chart-cache command exceeded its 15 second limit.');
		const detail = (child.stderr ?? '').trim().slice(0, 500);
		throw new BridgeError(502, detail || `Trader chart-cache command failed${child.code ? ` (exit ${child.code})` : ''}.`);
	} finally {
		running = false;
	}
}

export function apiError(error: unknown): Response {
	const known = error instanceof BridgeError;
	return Response.json({ error: known ? error.message : 'Unexpected local chart bridge error.' }, {
		status: known ? error.status : 500,
		headers: { 'cache-control': 'no-store' }
	});
}

export function integerParam(value: string | null, name: string, fallback?: number): number | undefined {
	if (value === null || value === '') return fallback;
	if (!/^\d{1,10}$/.test(value)) throw new BridgeError(400, `${name} must be a non-negative integer.`);
	const number = Number(value);
	if (!Number.isSafeInteger(number)) throw new BridgeError(400, `${name} is outside the supported range.`);
	return number;
}

export function validatePageQuery(url: URL): string[] {
	const allowed = new Set(['input', 'bar_kind', 'bar_value', 'chart_mode', 'offset', 'limit', 'start_ns', 'end_ns']);
	for (const key of url.searchParams.keys()) if (!allowed.has(key)) throw new BridgeError(400, `Unsupported query parameter: ${key}.`);
	const input = url.searchParams.get('input') ?? 'server-bars';
	if (!['server-bars', 'raw-ticks'].includes(input)) throw new BridgeError(400, 'input must be server-bars or raw-ticks.');
	const kind = url.searchParams.get('bar_kind') ?? 'minute';
	if (!['minute', 'second', 'tick', 'volume', 'range'].includes(kind)) throw new BridgeError(400, 'bar_kind must be minute, second, tick, volume, or range.');
	const barValue = integerParam(url.searchParams.get('bar_value'), 'bar_value', 1)!;
	if (barValue < 1 || barValue > 1_000_000) throw new BridgeError(400, 'bar_value must be between 1 and 1000000.');
	const mode = url.searchParams.get('chart_mode') ?? 'ohlc';
	if (!['ohlc', 'heikin-ashi'].includes(mode)) throw new BridgeError(400, 'chart_mode must be ohlc or heikin-ashi.');
	const offset = integerParam(url.searchParams.get('offset'), 'offset', 0)!;
	if (offset > MAX_OFFSET) throw new BridgeError(400, `offset may not exceed ${MAX_OFFSET}.`);
	const limit = integerParam(url.searchParams.get('limit'), 'limit', 1000)!;
	if (limit < 1 || limit > 5000) throw new BridgeError(400, 'limit must be between 1 and 5000.');
	const start = url.searchParams.get('start_ns');
	const end = url.searchParams.get('end_ns');
	for (const [name, raw] of [['start_ns', start], ['end_ns', end]] as const) {
		if (raw !== null && !/^-?\d{1,19}$/.test(raw)) throw new BridgeError(400, `${name} must be a signed UTC epoch nanosecond integer.`);
		if (raw !== null) {
			const value = BigInt(raw);
			if (value < MIN_I64 || value > MAX_I64) throw new BridgeError(400, `${name} is outside the supported signed 64-bit range.`);
		}
	}
	if (start !== null && end !== null && BigInt(start) >= BigInt(end)) throw new BridgeError(400, 'start_ns must be earlier than end_ns.');
	const args = ['--input', input, '--bar-kind', kind, '--bar-value', String(barValue), '--chart-mode', mode, '--offset', String(offset), '--limit', String(limit)];
	if (start !== null) args.push('--start-ns', start);
	if (end !== null) args.push('--end-ns', end);
	return args;
}
