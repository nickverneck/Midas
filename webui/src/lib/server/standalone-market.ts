import { execFile, spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { isAbsolute, resolve } from 'node:path';
import type { RequestEvent } from '@sveltejs/kit';
import { assertLocalRequest, BridgeError, parseChartCacheOutput } from './chart-cache.ts';

const execFileAsync = promisify(execFile);
const TIMEOUT_MS = 20_000;
const MAX_BUFFER = 2 * 1024 * 1024;
const SYMBOL = /^[A-Za-z0-9]{1,16}$/;
const BAR_KINDS = new Set(['minute', 'second', 'tick', 'volume', 'range']);
let running = false;

export type StandaloneMarketRequest = { contract: string; bar_kind: string; bar_value: number };

function traderDirectory() {
	const configured = process.env.MIDAS_TRADER_DIR;
	if (configured) {
		if (!isAbsolute(configured)) throw new BridgeError(500, 'MIDAS_TRADER_DIR must be an absolute path.');
		return resolve(configured);
	}
	// The bridge is run from webui/ per BRIDGE.md. Resolve this before passing
	// the child cwd, so .auth and other CLI defaults stay rooted at trader/.
	return resolve(process.cwd(), '..', 'trader');
}

function traderPath() {
	return process.env.MIDAS_TRADER_BIN ? resolve(process.env.MIDAS_TRADER_BIN) : resolve(traderDirectory(), 'target', 'debug', 'trader');
}

export function validateStandaloneMarketRequest(value: unknown): StandaloneMarketRequest {
	if (!value || typeof value !== 'object' || Array.isArray(value)) throw new BridgeError(400, 'Request body must be an object.');
	const body = value as Record<string, unknown>;
	if (Object.keys(body).some((key) => !['contract', 'bar_kind', 'bar_value'].includes(key))) throw new BridgeError(400, 'Unsupported standalone market request field.');
	if (typeof body.contract !== 'string' || !SYMBOL.test(body.contract)) throw new BridgeError(400, 'Enter an exact alphanumeric contract symbol from 1 to 16 characters.');
	if (typeof body.bar_kind !== 'string' || !BAR_KINDS.has(body.bar_kind) || !Number.isSafeInteger(body.bar_value) || (body.bar_value as number) < 1 || (body.bar_value as number) > 100_000) {
		throw new BridgeError(400, 'Choose minute, second, tick, volume, or range bars with a whole number from 1 to 100000.');
	}
	return { contract: body.contract, bar_kind: body.bar_kind, bar_value: body.bar_value as number };
}

export function safeMarketPayload(value: unknown, request: StandaloneMarketRequest) {
	const source = value as Record<string, any>;
	if (!source || typeof source !== 'object' || !Array.isArray(source.bars) || source.bars.length > 5000) throw new BridgeError(502, 'Trader returned an invalid standalone market snapshot.');
	const bars = source.bars.map((bar: any) => {
		if (!bar || typeof bar.ts_ns !== 'string' || !/^-?(?:0|[1-9]\d*)$/.test(bar.ts_ns) || !['open', 'high', 'low', 'close'].every((key) => Number.isFinite(bar[key])) || bar.high < bar.low) {
			throw new BridgeError(502, 'Trader returned an invalid standalone market bar.');
		}
		return { ts_ns: bar.ts_ns, open: bar.open, high: bar.high, low: bar.low, close: bar.close, ...(Number.isFinite(bar.volume) ? { volume: bar.volume } : {}) };
	});
	return {
		connection_status: source.connection_status === 'connected' ? 'connected' : 'unknown',
		chart_freshness: 'unknown',
		source: typeof source.source === 'string' ? source.source.slice(0, 100) : 'provider_market_data',
		broker_source: typeof source.broker_source === 'string' ? source.broker_source.slice(0, 100) : 'Provider not reported',
		fidelity: typeof source.fidelity === 'string' ? source.fidelity.slice(0, 200) : 'Provider chart bars; not a guaranteed complete trade tape.',
		contract: request.contract,
		bar_kind: request.bar_kind,
		bar_value: request.bar_value,
		bars,
		latest_bar_ts_ns: bars.length ? bars[bars.length - 1].ts_ns : null,
		observed_at: typeof source.observed_at === 'string' ? source.observed_at.slice(0, 64) : null
	};
}

export async function runStandaloneMarket(value: unknown, signal?: AbortSignal) {
	if (running) throw new BridgeError(503, 'A standalone market request is already running.');
	const request = validateStandaloneMarketRequest(value);
	running = true;
	try {
		const args = [];
		if (process.env.MIDAS_CONFIG) args.push('--config', resolve(process.env.MIDAS_CONFIG));
		args.push('standalone-market', '--contract', request.contract, '--bar-kind', request.bar_kind, '--bar-value', String(request.bar_value), '--timeout-seconds', '15');
		const { stdout } = await execFileAsync(traderPath(), args, { cwd: traderDirectory(), env: process.env, timeout: TIMEOUT_MS, maxBuffer: MAX_BUFFER, windowsHide: true, signal });
		let parsed: unknown;
		try { parsed = parseChartCacheOutput(stdout); } catch { throw new BridgeError(502, 'Trader returned invalid standalone market JSON.'); }
		return safeMarketPayload(parsed, request);
	} catch (error) {
		if (error instanceof BridgeError) throw error;
		const child = error as NodeJS.ErrnoException & { killed?: boolean };
		if (child.code === 'ABORT_ERR') throw new BridgeError(499, 'Standalone market request was cancelled.');
		if (child.code === 'ENOENT') throw new BridgeError(503, 'Trader CLI is unavailable. Build the trader with the Tradovate feature or set MIDAS_TRADER_BIN.');
		if (child.killed) throw new BridgeError(504, 'Standalone market request exceeded its 20 second limit.');
		// Never reflect CLI stderr: auth/provider errors can contain sensitive details.
		throw new BridgeError(502, 'Standalone market request failed. Check local provider configuration and connectivity.');
	} finally { running = false; }
}

export async function handleStandaloneMarket(event: RequestEvent) {
	assertLocalRequest(event);
	const length = Number(event.request.headers.get('content-length') ?? 0);
	if (!Number.isFinite(length) || length > 2048) throw new BridgeError(413, 'Request body is too large.');
	let body: unknown;
	try {
		const reader = event.request.body?.getReader();
		if (!reader) throw new BridgeError(400, 'Request body must be valid JSON.');
		const chunks: Uint8Array[] = [];
		let size = 0;
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			size += value.byteLength;
			if (size > 2048) { await reader.cancel(); throw new BridgeError(413, 'Request body is too large.'); }
			chunks.push(value);
		}
		const text = Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString('utf8');
		body = JSON.parse(text);
	} catch (error) {
		if (error instanceof BridgeError) throw error;
		throw new BridgeError(400, 'Request body must be valid JSON.');
	}
	return runStandaloneMarket(body);
}

/** Start one authenticated Rust market worker and forward its bounded NDJSON snapshots as SSE. */
export async function handleStandaloneMarketStream(event: RequestEvent) {
	assertLocalRequest(event);
	const length = Number(event.request.headers.get('content-length') ?? 0);
	if (!Number.isFinite(length) || length > 2048) throw new BridgeError(413, 'Request body is too large.');
	let body: unknown;
	try {
		const reader = event.request.body?.getReader();
		if (!reader) throw new BridgeError(400, 'Request body must be valid JSON.');
		const chunks: Uint8Array[] = []; let size = 0;
		while (true) { const { done, value } = await reader.read(); if (done) break; size += value.byteLength; if (size > 2048) { await reader.cancel(); throw new BridgeError(413, 'Request body is too large.'); } chunks.push(value); }
		body = JSON.parse(Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString('utf8'));
	} catch (error) { if (error instanceof BridgeError) throw error; throw new BridgeError(400, 'Request body must be valid JSON.'); }
	const request = validateStandaloneMarketRequest(body);
	if (event.request.signal.aborted) throw new BridgeError(499, 'Standalone market stream request was already cancelled.');
	if (running) throw new BridgeError(503, 'A standalone market request is already running.');
	running = true;
	const args: string[] = [];
	if (process.env.MIDAS_CONFIG) args.push('--config', resolve(process.env.MIDAS_CONFIG));
	args.push('standalone-market', '--contract', request.contract, '--bar-kind', request.bar_kind, '--bar-value', String(request.bar_value), '--timeout-seconds', '900', '--stream');
	const child = spawn(traderPath(), args, { cwd: traderDirectory(), env: process.env, windowsHide: true, stdio: ['ignore', 'pipe', 'ignore'] });
	const encoder = new TextEncoder();
	let ended = false;
	let pending = '';
	let timer: ReturnType<typeof setTimeout>;
	let startupTimer: ReturnType<typeof setTimeout>;
	let killTimer: ReturnType<typeof setTimeout> | undefined;
	let deadlineReached = false;
	const cleanup = () => { if (ended) return; ended = true; clearTimeout(timer); clearTimeout(startupTimer); if (killTimer) clearTimeout(killTimer); event.request.signal.removeEventListener('abort', abort); running = false; };
	const abort = () => { if (!ended) { child.kill('SIGTERM'); killTimer = setTimeout(() => child.kill('SIGKILL'), 2_000); ended = true; clearTimeout(timer); clearTimeout(startupTimer); event.request.signal.removeEventListener('abort', abort); } };
	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			const fail = (message: string) => { if (!ended) { controller.enqueue(encoder.encode(`event: status\ndata: ${JSON.stringify({ connection_status: 'disconnected', error: message })}\n\n`)); controller.close(); ended = true; clearTimeout(timer); clearTimeout(startupTimer); event.request.signal.removeEventListener('abort', abort); child.kill('SIGTERM'); killTimer = setTimeout(() => child.kill('SIGKILL'), 2_000); } };
			const line = (raw: string) => {
				if (!raw) return;
				try { const snapshot = safeMarketPayload(parseChartCacheOutput(raw), request); clearTimeout(startupTimer); controller.enqueue(encoder.encode(`event: snapshot\ndata: ${JSON.stringify(snapshot)}\n\n`)); if (controller.desiredSize !== null && controller.desiredSize <= 0) child.stdout.pause(); }
				catch { fail('Trader returned an invalid standalone market stream snapshot.'); }
			};
			child.stdout.setEncoding('utf8');
			child.stdout.on('data', (chunk: string) => {
				pending += chunk;
				if (pending.length > MAX_BUFFER && !pending.includes('\n')) { fail('Trader market stream line exceeded its size limit.'); return; }
				let newline = pending.indexOf('\n');
				while (newline >= 0 && !ended) { const current = pending.slice(0, newline); pending = pending.slice(newline + 1); if (current.length > MAX_BUFFER) { fail('Trader market stream line exceeded its size limit.'); break; } line(current); newline = pending.indexOf('\n'); }
			});
			child.stdout.on('end', () => { if (pending.trim()) line(pending); });
			child.on('error', (error: NodeJS.ErrnoException) => fail(error.code === 'ENOENT' ? 'Trader CLI is unavailable. Build the trader with the Tradovate feature or set MIDAS_TRADER_BIN.' : 'Standalone market stream failed. Check local provider configuration and connectivity.'));
			child.on('close', (code) => {
				if (ended) { running = false; if (killTimer) clearTimeout(killTimer); return; }
				if (deadlineReached) { controller.enqueue(encoder.encode(`event: status\ndata: ${JSON.stringify({ connection_status: 'stopped', reason: 'Stream time limit reached.' })}\n\n`)); controller.close(); cleanup(); return; }
				if (code !== 0) { fail('Standalone market stream ended. Check local provider configuration and connectivity.'); return; }
				controller.enqueue(encoder.encode(`event: status\ndata: ${JSON.stringify({ connection_status: 'stopped', reason: 'Stream time limit reached.' })}\n\n`));
				controller.close(); cleanup();
			});
			startupTimer = setTimeout(() => fail('Standalone market stream produced no first snapshot within its startup deadline.'), 95_000);
			timer = setTimeout(() => { if (!ended) { deadlineReached = true; child.kill('SIGTERM'); killTimer = setTimeout(() => child.kill('SIGKILL'), 2_000); } }, 900_000);
			event.request.signal.addEventListener('abort', abort, { once: true });
			if (event.request.signal.aborted) abort();
		},
		pull() { child.stdout.resume(); },
		cancel() { abort(); }
	});
	child.stdout.pause();
	return new Response(stream, { headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-store, no-transform', connection: 'keep-alive', 'x-accel-buffering': 'no' } });
}

export function standaloneMarketError(error: unknown) {
	const known = error instanceof BridgeError;
	return Response.json({ error: known ? error.message : 'Standalone market observer failed.' }, { status: known ? error.status : 500, headers: { 'cache-control': 'no-store' } });
}
