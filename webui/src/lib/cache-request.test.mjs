import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchStoredCachePage } from './cache-request.mjs';

const busy = () => new Response(JSON.stringify({ error: 'The chart cache reader is busy. Retry shortly.' }), { status: 503 });

test('retries a single-flight busy response until the selected request can run', async () => {
	let calls = 0;
	const waits = [];
	const result = await fetchStoredCachePage(async () => ++calls < 3 ? busy() : new Response('{"bars":[]}', { status: 200 }), '/cache', () => true, async ms => waits.push(ms));
	assert.equal(result.status, 200);
	assert.equal(calls, 3);
	assert.deepEqual(waits, [100, 200]);
});

test('retries catalog discovery on the catalog endpoint', async () => {
	let calls = 0;
	const urls = [];
	const result = await fetchStoredCachePage(async url => { urls.push(url); return ++calls === 1 ? busy() : new Response('{"datasets":[]}', { status: 200 }); }, '/api/chart-cache', () => true, async () => {});
	assert.equal(result.status, 200);
	assert.deepEqual(urls, ['/api/chart-cache', '/api/chart-cache']);
});

test('stops retrying immediately when the source selection changes', async () => {
	let current = true;
	let calls = 0;
	const waits = [];
	const result = await fetchStoredCachePage(async () => { calls++; current = false; return busy(); }, '/api/chart-cache', () => current, async ms => waits.push(ms));
	assert.equal(result, undefined);
	assert.equal(calls, 1);
	assert.deepEqual(waits, []);
});

test('bounds retries and returns the bridge response so its status and error remain visible', async () => {
	let calls = 0;
	const waits = [];
	const result = await fetchStoredCachePage(async () => { calls++; return busy(); }, '/cache', () => true, async ms => waits.push(ms));
	assert.equal(calls, 21);
	assert.equal(waits.reduce((sum, delay) => sum + delay, 0), 17_500);
	assert.equal(result.status, 503);
	assert.deepEqual(await result.json(), { error: 'The chart cache reader is busy. Retry shortly.' });
});

test('does not retry unrelated 503 errors', async () => {
	let calls = 0;
	const result = await fetchStoredCachePage(async () => { calls++; return new Response('{"error":"CLI unavailable"}', { status: 503 }); }, '/cache', () => true, async () => {});
	assert.equal(calls, 1);
	assert.equal(result.status, 503);
});
