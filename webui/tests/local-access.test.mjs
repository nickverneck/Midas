import test from 'node:test';
import assert from 'node:assert/strict';
import { assertLocalRequest } from '../src/lib/server/chart-cache.ts';

function event(host, origin) {
	const headers = new Headers({ host });
	if (origin) headers.set('origin', origin);
	return { request: { headers } };
}

test('LAN access is opt-in and restricted to the configured host and origin', () => {
	const previous = process.env.MIDAS_LAN_HOST;
	try {
		delete process.env.MIDAS_LAN_HOST;
		assert.throws(() => assertLocalRequest(event('192.168.1.57:5173')), /configured local/);
		process.env.MIDAS_LAN_HOST = '192.168.1.57';
		assert.doesNotThrow(() => assertLocalRequest(event('192.168.1.57:5173', 'http://192.168.1.57:5173')));
		assert.doesNotThrow(() => assertLocalRequest(event('127.0.0.1:5173', 'http://127.0.0.1:5173')));
		assert.throws(() => assertLocalRequest(event('192.168.1.58:5173')), /configured local/);
		assert.throws(() => assertLocalRequest(event('192.168.1.57:5173', 'http://192.168.1.57:9999')), /Cross-origin/);
		assert.throws(() => assertLocalRequest(event('192.168.1.57:5173', 'http://example.com')), /Cross-origin/);
	} finally {
		if (previous === undefined) delete process.env.MIDAS_LAN_HOST;
		else process.env.MIDAS_LAN_HOST = previous;
	}
});
