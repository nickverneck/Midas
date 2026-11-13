import net from 'node:net';
import path from 'node:path';
import { parseChartCacheOutput } from './chart-cache.ts';

const MAX_LINE_BYTES = 8 * 1024 * 1024;
const CONNECT_TIMEOUT_MS = 500;
const READ_TIMEOUT_MS = 2_000;

export class LiveMarketError extends Error {
	status: number;
	constructor(status: number, message: string) { super(message); this.status = status; }
}

type Bar = { ts_ns: string; open: number; high: number; low: number; close: number; volume?: number | null };

/** Keep only chart-safe fields. The IPC event also carries trade markers and account identifiers. */
export function chartPayload(snapshot: any) {
	if (!snapshot || typeof snapshot !== 'object' || !Array.isArray(snapshot.bars)) {
		throw new LiveMarketError(502, 'Engine returned an invalid market snapshot.');
	}
	const selected = snapshot.bars.slice(-5000);
	if (selected.some((bar: any) => !bar || typeof bar.ts_ns !== 'string' || !/^-?\d+$/.test(bar.ts_ns) || !['open', 'high', 'low', 'close'].every((key) => Number.isFinite(bar[key])) || bar.high < bar.low)) {
		throw new LiveMarketError(502, 'Engine returned an invalid market bar.');
	}
	const bars: Bar[] = selected.map((bar: any) => ({
		ts_ns: String(bar.ts_ns), open: bar.open, high: bar.high, low: bar.low, close: bar.close,
		...(Number.isFinite(bar.volume) ? { volume: bar.volume } : {})
	}));
	const latestNs = bars.length ? bars[bars.length - 1].ts_ns : null;
	return {
		connection_status: 'connected',
		chart_freshness: 'unknown',
		// Broker identity is not present in the IPC MarketSnapshot wire type.
		source: 'engine_market_snapshot',
		broker_source: 'not reported by engine IPC',
		contract: typeof snapshot.contract_name === 'string' ? snapshot.contract_name : null,
		bars,
		latest_bar_ts_ns: latestNs,
		chart_as_of_ts_ns: latestNs,
		engine_status: typeof snapshot.status === 'string' ? snapshot.status.slice(0, 256) : ''
	};
}

function socketPath(): string {
	const configured = process.env.MIDAS_ENGINE_SOCKET;
	if (configured && !path.isAbsolute(configured)) throw new LiveMarketError(500, 'MIDAS_ENGINE_SOCKET must be an absolute path.');
	return configured || path.resolve(process.cwd(), '../.run/trader-engine.sock');
}

/** Connects without writing any bytes; the engine immediately publishes its current snapshot. */
export async function readLiveMarket() {
	const socket = net.createConnection({ path: socketPath() });
	return new Promise<ReturnType<typeof chartPayload>>((resolve, reject) => {
		let buffer = Buffer.alloc(0);
		let done = false;
	const finish = (error?: Error, value?: ReturnType<typeof chartPayload>) => {
			if (done) return;
			done = true;
			clearTimeout(connectTimer); clearTimeout(readTimer); socket.destroy();
			error ? reject(error) : resolve(value!);
		};
		const connectTimer = setTimeout(() => finish(new LiveMarketError(503, 'Engine socket is unavailable.')), CONNECT_TIMEOUT_MS);
		const readTimer = setTimeout(() => finish(new LiveMarketError(504, 'Timed out waiting for engine market data.')), READ_TIMEOUT_MS);
		socket.on('connect', () => clearTimeout(connectTimer));
		socket.on('data', (chunk: Buffer) => {
			buffer = Buffer.concat([buffer, chunk]);
			if (buffer.length > MAX_LINE_BYTES) return finish(new LiveMarketError(502, 'Engine market message exceeded the size limit.'));
			const newline = buffer.indexOf(10);
			if (newline < 0) return;
			try {
				// Preserve epoch nanoseconds exactly: JSON.parse would round i64 values.
				const line = buffer.subarray(0, newline).toString('utf8');
				const wire: any = parseChartCacheOutput(line);
				const event = wire?.Event?.event;
				const snapshot = event?.MarketSnapshot;
				if (!snapshot) return finish(new LiveMarketError(502, 'Engine did not provide a market snapshot.'));
				finish(undefined, chartPayload(snapshot));
			} catch (error) {
				finish(error instanceof LiveMarketError ? error : new LiveMarketError(502, 'Could not decode engine market data.'));
			}
		});
		socket.on('error', () => finish(new LiveMarketError(503, 'Engine socket is disconnected.')));
		socket.on('end', () => { if (!done) finish(new LiveMarketError(503, 'Engine socket closed before market data arrived.')); });
	});
}

export function liveMarketError(error: unknown) {
	if (error instanceof LiveMarketError) return Response.json({ connection_status: 'disconnected', error: error.message }, { status: error.status, headers: { 'cache-control': 'no-store' } });
	return Response.json({ error: 'Live market observer failed.' }, { status: 500, headers: { 'cache-control': 'no-store' } });
}
