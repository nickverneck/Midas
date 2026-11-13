import test from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { mkdtemp, rm } from 'node:fs/promises';
import { chartPayload, readLiveMarket } from '../src/lib/server/live-market.ts';
import { RequestEpoch } from '../src/lib/request-epoch.ts';

async function withSocket(handler, callback) {
	const dir = await mkdtemp(path.join(os.tmpdir(), 'midas-live-'));
	const socketPath = path.join(dir, 'engine.sock');
	const server = net.createServer(handler);
	await new Promise((resolve, reject) => server.listen(socketPath, (error) => error ? reject(error) : resolve()));
	const prior = process.env.MIDAS_ENGINE_SOCKET;
	process.env.MIDAS_ENGINE_SOCKET = socketPath;
	try { return await callback(); }
	finally {
		if (prior === undefined) delete process.env.MIDAS_ENGINE_SOCKET;
		else process.env.MIDAS_ENGINE_SOCKET = prior;
		await new Promise((resolve) => server.close(resolve));
		await rm(dir, { recursive: true, force: true });
	}
}

test('live chart response contains only bounded market bar fields', () => {
	const payload = chartPayload({
		contract_name: 'ESZ6', status: 'Subscribed',
		bars: [{ ts_ns: '1000000000', open: 1, high: 3, low: 0, close: 2, volume: 10 }],
		trade_markers: [{ account_id: 42, fill_id: 5, price: 2 }],
		account_id: 99, replay_window: { start_ns: 123 }
	}, 1_000);
	assert.equal(payload.connection_status, 'connected');
	assert.equal(payload.chart_freshness, 'unknown');
	assert.equal(payload.source, 'engine_market_snapshot');
	assert.equal(payload.contract, 'ESZ6');
	assert.deepEqual(payload.bars, [{ ts_ns: '1000000000', open: 1, high: 3, low: 0, close: 2, volume: 10 }]);
	assert.equal('trade_markers' in payload, false);
	assert.equal('account_id' in payload, false);
});

test('live chart response preserves nanosecond timestamps and caps history at 5000', () => {
	const bars = Array.from({ length: 5001 }, (_, i) => ({ ts_ns: String(1_000_000_000 + i), open: 1, high: 1, low: 1, close: 1 }));
	const payload = chartPayload({ bars }, 100_000_000);
	assert.equal(payload.connection_status, 'connected');
	assert.equal(payload.chart_freshness, 'unknown');
	assert.equal(payload.bars.length, 5000);
	assert.equal(payload.bars[0].ts_ns, '1000000001');
});

test('invalid market snapshots fail closed', () => {
	assert.throws(() => chartPayload({ bars: 'not-an-array' }), /invalid market snapshot/i);
	assert.throws(() => chartPayload({ bars: [{ ts_ns: '1', open: 1, high: 0, low: 1, close: 1 }] }), /invalid market bar/i);
});

test('reads the engine subscription snapshot without sending client bytes and preserves i64 nanoseconds', async () => {
	const exactNs = '1735689600123456789';
	let clientBytes = 0;
	await withSocket((socket) => {
		socket.on('data', (chunk) => { clientBytes += chunk.length; });
		socket.write(`${JSON.stringify({ Event: { event: { MarketSnapshot: {
			contract_name: 'ESZ6', status: 'Subscribed',
			bars: [{ ts_ns: Number(exactNs), open: 10, high: 12, low: 9, close: 11 }]
		} } } }).replace(String(Number(exactNs)), exactNs)}\n`);
	}, async () => {
		const payload = await readLiveMarket();
		assert.equal(payload.bars[0].ts_ns, exactNs);
		assert.equal(payload.broker_source, 'not reported by engine IPC');
		assert.equal(payload.chart_freshness, 'unknown');
		await new Promise((resolve) => setTimeout(resolve, 20));
		assert.equal(clientBytes, 0);
	});
});

test('times out when the connected engine socket never publishes market data', async () => {
	await withSocket(() => {}, async () => {
		await assert.rejects(readLiveMarket(), (error) => error.status === 504 && /timed out/i.test(error.message));
	});
});

test('selection changes invalidate old live responses across away-and-back switches', () => {
	const epoch = new RequestEpoch();
	const oldRequest = epoch.capture();
	epoch.invalidate(); // Leave the live source.
	epoch.invalidate(); // Re-enter and start a new live request.
	const newRequest = epoch.capture();
	assert.equal(epoch.isCurrent(oldRequest), false);
	assert.equal(epoch.isCurrent(newRequest), true);
});
