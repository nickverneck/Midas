import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import path from 'node:path';
import { mkdtemp, rm, writeFile, chmod } from 'node:fs/promises';
import { handleStandaloneMarketStream, runStandaloneMarket, safeMarketPayload, validateStandaloneMarketRequest } from '../src/lib/server/standalone-market.ts';

test('request accepts exact symbols and all supported bounded bar kinds', () => {
	assert.deepEqual(validateStandaloneMarketRequest({ contract: 'MESZ6', bar_kind: 'minute', bar_value: 5 }), { contract: 'MESZ6', bar_kind: 'minute', bar_value: 5 });
	for (const body of [
		{ contract: 'MESZ6; touch /tmp/pwned', bar_kind: 'minute', bar_value: 1 },
		{ contract: 'ESZ6', bar_kind: 'minute', bar_value: 1.5 },
		{ contract: 'ESZ6', bar_kind: 'range', bar_value: 0 },
		{ contract: 'ESZ6', bar_kind: 'range', bar_value: 100001 },
		{ contract: 'ESZ6!', bar_kind: 'range', bar_value: 1 },
		{ contract: 'ESZ6', bar_kind: 'minute', bar_value: 1, config: '/tmp/secret' }
	]) assert.throws(() => validateStandaloneMarketRequest(body), /supported|unsupported|exact|whole number/i);
	for (const kind of ['minute', 'second', 'tick', 'volume', 'range']) assert.equal(validateStandaloneMarketRequest({ contract: 'ABC123', bar_kind: kind, bar_value: 100000 }).bar_kind, kind);
});

test('payload keeps exact timestamp strings and strips credentials and unrelated fields', () => {
	const ts = '1735689600123456789';
	const payload = safeMarketPayload({
		source: 'made-up', connection_status: 'connected', chart_freshness: 'fresh', fidelity: 'Provider chart bars.',
		observed_at: '2025-01-01T00:00:00Z', access_token: 'secret', account_id: 42,
		bars: [{ ts_ns: ts, open: 10, high: 12, low: 9, close: 11, volume: 5, token: 'secret' }]
	}, { contract: 'ESZ6', bar_kind: 'minute', bar_value: 1 });
	assert.equal(payload.bars[0].ts_ns, ts);
	assert.equal(payload.chart_freshness, 'unknown');
	assert.equal(payload.fidelity, 'Provider chart bars.');
	assert.equal('access_token' in payload, false);
	assert.equal('account_id' in payload, false);
	assert.equal('token' in payload.bars[0], false);
});

test('CLI is invoked with fixed arguments and safe output; no live provider is contacted', async () => {
	const dir = await mkdtemp(path.join(os.tmpdir(), 'midas-standalone-test-'));
	const priorBin = process.env.MIDAS_TRADER_BIN;
	const priorConfig = process.env.MIDAS_CONFIG;
	const capturePath = path.join(dir, 'args.json');
	const fakeCli = path.join(dir, 'fake-trader');
	await writeFile(fakeCli, `#!/usr/bin/env node\nconst fs=require('node:fs');fs.writeFileSync(${JSON.stringify(capturePath)},JSON.stringify({args:process.argv.slice(2),cwd:process.cwd()}));process.stdout.write('{"bars":[{"ts_ns":1735689600123456789,"open":1,"high":2,"low":0,"close":1}],"access_token":"do-not-return"}');\n`);
	await chmod(fakeCli, 0o700);
	process.env.MIDAS_TRADER_BIN = fakeCli;
	process.env.MIDAS_CONFIG = './test-provider-config.toml';
	try {
		const result = await runStandaloneMarket({ contract: 'ESZ6', bar_kind: 'minute', bar_value: 1 });
		assert.equal(result.bars[0].ts_ns, '1735689600123456789');
		assert.equal('access_token' in result, false);
		const invocation = JSON.parse(await (await import('node:fs/promises')).readFile(capturePath, 'utf8'));
		assert.deepEqual(invocation.args, ['--config', path.resolve(process.env.MIDAS_CONFIG), 'standalone-market', '--contract', 'ESZ6', '--bar-kind', 'minute', '--bar-value', '1', '--timeout-seconds', '15']);
		assert.equal(invocation.cwd, path.resolve(process.cwd(), '..', 'trader'));
	} finally {
		if (priorBin === undefined) delete process.env.MIDAS_TRADER_BIN; else process.env.MIDAS_TRADER_BIN = priorBin;
		if (priorConfig === undefined) delete process.env.MIDAS_CONFIG; else process.env.MIDAS_CONFIG = priorConfig;
		await rm(dir, { recursive: true, force: true });
	}
});

test('caller cancellation aborts an in-flight fake CLI without exposing its output', async () => {
	const dir = await mkdtemp(path.join(os.tmpdir(), 'midas-standalone-abort-'));
	const priorBin = process.env.MIDAS_TRADER_BIN;
	const fakeCli = path.join(dir, 'fake-trader');
	await writeFile(fakeCli, '#!/usr/bin/env node\nsetTimeout(() => process.stdout.write("{}"), 30000);\n');
	await chmod(fakeCli, 0o700);
	process.env.MIDAS_TRADER_BIN = fakeCli;
	const controller = new AbortController();
	const request = { contract: 'ESZ6', bar_kind: 'minute', bar_value: 1 };
	try {
		const pending = runStandaloneMarket(request, controller.signal);
		setTimeout(() => controller.abort(), 30);
		await assert.rejects(pending, /cancelled/i);
	} finally {
		if (priorBin === undefined) delete process.env.MIDAS_TRADER_BIN; else process.env.MIDAS_TRADER_BIN = priorBin;
		await rm(dir, { recursive: true, force: true });
	}
});

test('stream endpoint launches one fake long-lived CLI and forwards each bounded NDJSON snapshot', async () => {
	const dir = await mkdtemp(path.join(os.tmpdir(), 'midas-standalone-stream-'));
	const priorBin = process.env.MIDAS_TRADER_BIN;
	const capturePath = path.join(dir, 'args.json');
	const invocationsPath = path.join(dir, 'invocations.log');
	const fakeCli = path.join(dir, 'fake-trader');
	const ndjson = [1, 2].map((close, index) => JSON.stringify({ source: 'fake_market', broker_source: 'Fake provider', connection_status: 'connected', observed_at: `2025-01-01T00:00:0${index}Z`, bars: [{ ts_ns: String(1735689600123456789n + BigInt(index)), open: close, high: close, low: close, close }] })).join('\n');
	await writeFile(fakeCli, `#!/usr/bin/env node\nconst fs=require('node:fs');fs.writeFileSync(${JSON.stringify(capturePath)},JSON.stringify({args:process.argv.slice(2),cwd:process.cwd()}));fs.appendFileSync(${JSON.stringify(invocationsPath)},'run\\n');process.stdout.write(${JSON.stringify(ndjson + '\n')});\n`);
	await chmod(fakeCli, 0o700);
	process.env.MIDAS_TRADER_BIN = fakeCli;
	try {
		const request = new Request('http://localhost/api/standalone-market/stream', { method: 'POST', headers: { host: 'localhost', 'content-type': 'application/json' }, body: JSON.stringify({ contract: 'ESZ6', bar_kind: 'minute', bar_value: 1 }) });
		const response = await handleStandaloneMarketStream({ request } );
		const text = await response.text();
		assert.equal((text.match(/event: snapshot/g) ?? []).length, 2);
		assert.match(text, /1735689600123456789/);
		assert.match(text, /1735689600123456790/);
		assert.equal((await (await import('node:fs/promises')).readFile(invocationsPath, 'utf8')).trim().split('\n').length, 1);
		const invocation = JSON.parse(await (await import('node:fs/promises')).readFile(capturePath, 'utf8'));
		assert.equal(invocation.args.at(-1), '--stream');
		assert.equal(invocation.cwd, path.resolve(process.cwd(), '..', 'trader'));
	} finally {
		if (priorBin === undefined) delete process.env.MIDAS_TRADER_BIN; else process.env.MIDAS_TRADER_BIN = priorBin;
		await rm(dir, { recursive: true, force: true });
	}
});

test('stream rejects pre-aborted starts and retains its single-stream lock until cancelled child exits', async () => {
	const dir = await mkdtemp(path.join(os.tmpdir(), 'midas-standalone-stream-lock-'));
	const priorBin = process.env.MIDAS_TRADER_BIN;
	const fakeCli = path.join(dir, 'fake-trader');
	const line = JSON.stringify({ source: 'fake_market', broker_source: 'Fake provider', connection_status: 'connected', bars: [{ ts_ns: '1735689600123456789', open: 1, high: 1, low: 1, close: 1 }] });
	await writeFile(fakeCli, `#!/usr/bin/env node\nprocess.stdout.write(${JSON.stringify(line + '\n')});setInterval(()=>{},1000);\n`);
	await chmod(fakeCli, 0o700);
	process.env.MIDAS_TRADER_BIN = fakeCli;
	const makeEvent = (signal) => ({ request: new Request('http://localhost/api/standalone-market/stream', { method: 'POST', headers: { host: 'localhost', 'content-type': 'application/json' }, body: JSON.stringify({ contract: 'ESZ6', bar_kind: 'minute', bar_value: 1 }), signal }) });
	try {
		const cancelled = new AbortController(); cancelled.abort();
		await assert.rejects(handleStandaloneMarketStream(makeEvent(cancelled.signal)), /already cancelled/i);
		const response = await handleStandaloneMarketStream(makeEvent(undefined));
		const reader = response.body.getReader();
		await reader.read();
		await assert.rejects(handleStandaloneMarketStream(makeEvent(undefined)), /already running/i);
		await reader.cancel();
		let next;
		for (let attempt = 0; attempt < 30; attempt++) {
			await new Promise((resolve) => setTimeout(resolve, 20));
			try { next = await handleStandaloneMarketStream(makeEvent(undefined)); break; }
			catch (error) { if (!/already running/i.test(error.message)) throw error; }
		}
		assert.ok(next, 'lock is released after child close');
		await next.body.cancel();
	} finally {
		if (priorBin === undefined) delete process.env.MIDAS_TRADER_BIN; else process.env.MIDAS_TRADER_BIN = priorBin;
		await rm(dir, { recursive: true, force: true });
	}
});

test('stream reports early worker exit as disconnected instead of claiming a deadline stop', async () => {
	const dir = await mkdtemp(path.join(os.tmpdir(), 'midas-standalone-stream-startup-'));
	const priorBin = process.env.MIDAS_TRADER_BIN;
	const fakeCli = path.join(dir, 'fake-trader');
	await writeFile(fakeCli, '#!/usr/bin/env node\nprocess.stderr.write("private provider detail");process.exit(1);\n');
	await chmod(fakeCli, 0o700);
	process.env.MIDAS_TRADER_BIN = fakeCli;
	try {
		const request = new Request('http://localhost/api/standalone-market/stream', { method: 'POST', headers: { host: 'localhost', 'content-type': 'application/json' }, body: JSON.stringify({ contract: 'ESZ6', bar_kind: 'minute', bar_value: 1 }) });
		const response = await handleStandaloneMarketStream({ request });
		const body = await response.text();
		assert.match(body, /connection_status":"disconnected/);
		assert.doesNotMatch(body, /private provider detail|time limit reached/);
	} finally {
		if (priorBin === undefined) delete process.env.MIDAS_TRADER_BIN; else process.env.MIDAS_TRADER_BIN = priorBin;
		await rm(dir, { recursive: true, force: true });
	}
});
