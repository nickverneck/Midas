import test from 'node:test';
import assert from 'node:assert/strict';
import { BridgeError, parseChartCacheOutput, validatePageQuery } from '../src/lib/server/chart-cache.ts';

const query = (values) => new URL(`http://localhost/api/chart-cache/data?${new URLSearchParams(values)}`);
const nsArgs = (start, end) => validatePageQuery(query({ start_ns: start, end_ns: end }));

test('accepts modern epoch nanoseconds and preserves adjacent values above Number safe integer range', () => {
	const start = '1794500000000000000';
	const end = '1794500000000000001';
	const args = nsArgs(start, end);
	assert.deepEqual(args.slice(-4), ['--start-ns', start, '--end-ns', end]);
});

test('passes only supported chart inputs as explicit CLI arguments', () => {
	assert.deepEqual(validatePageQuery(query({ input: 'raw-ticks', bar_kind: 'range', bar_value: '4' })).slice(0, 6), [
		'--input', 'raw-ticks', '--bar-kind', 'range', '--bar-value', '4'
	].slice(0, 6));
	assert.deepEqual(validatePageQuery(query({})).slice(0, 2), ['--input', 'server-bars']);
	for (const input of ['raw_ticks', '--help', '../../etc/passwd']) {
		assert.throws(() => validatePageQuery(query({ input })), (error) => error instanceof BridgeError && error.status === 400);
	}
});

test('accepts bounded raw bar values and rejects overflow or fractions', () => {
	for (const [kind, value] of [['minute', '30'], ['second', '15'], ['tick', '1000'], ['volume', '250'], ['range', '4']]) {
		const args = validatePageQuery(query({ input: 'raw-ticks', bar_kind: kind, bar_value: value }));
		assert.equal(args[args.indexOf('--bar-value') + 1], value);
	}
	assert.equal(validatePageQuery(query({ input: 'raw-ticks', bar_value: '1000000' }))[5], '1000000');
	for (const value of ['1000001', '1.5', '0']) {
		assert.throws(() => validatePageQuery(query({ input: 'raw-ticks', bar_value: value })), (error) => error instanceof BridgeError && error.status === 400);
	}
});

test('accepts signed i64 endpoints', () => {
	assert.doesNotThrow(() => nsArgs('-9223372036854775808', '9223372036854775807'));
});

test('rejects values outside signed i64 range', () => {
	for (const [start, end] of [
		['-9223372036854775809', '0'],
		['0', '9223372036854775808']
	]) {
		assert.throws(() => nsArgs(start, end), (error) => error instanceof BridgeError && error.status === 400);
	}
});

test('compares range endpoints exactly and rejects equal or reversed ranges', () => {
	const start = '1794500000000000001';
	assert.throws(() => nsArgs(start, '1794500000000000000'), /start_ns must be earlier than end_ns/);
	assert.throws(() => nsArgs(start, start), /start_ns must be earlier than end_ns/);
	assert.doesNotThrow(() => nsArgs(start, '1794500000000000002'));
});

test('rejects malformed epoch nanosecond parameters', () => {
	for (const value of ['+1', '1.0', '1e3', '']) {
		assert.throws(() => validatePageQuery(query({ start_ns: value })), /signed UTC epoch nanosecond integer/);
	}
});

test('bridge parser preserves adjacent and duplicate nanosecond tokens as strings', () => {
	const first = '1794500000000000000';
	const adjacent = '1794500000000000001';
	const fixture = `{"bars":[{"timestamp_ns":${first},"open":1},{"timestamp_ns":${adjacent},"open":2},{"timestamp_ns":${adjacent},"open":3}]}`;
	const parsed = parseChartCacheOutput(fixture);
	assert.deepEqual(parsed.bars.map((bar) => bar.timestamp_ns), [first, adjacent, adjacent]);
	assert.equal(JSON.stringify(parsed), `{"bars":[{"timestamp_ns":"${first}","open":1},{"timestamp_ns":"${adjacent}","open":2},{"timestamp_ns":"${adjacent}","open":3}]}`);
});

test('bridge parser leaves ordinary strings untouched', () => {
	const parsed = parseChartCacheOutput('{"message":"timestamp_ns: 1794500000000000000","nested":{"label":"\\\"timestamp_ns\\\":123"}}');
	assert.deepEqual(parsed, {
		message: 'timestamp_ns: 1794500000000000000',
		nested: { label: '"timestamp_ns":123' }
	});
});

test('bridge parser preserves __proto__ as an ordinary JSON property', () => {
	const parsed = parseChartCacheOutput('{"__proto__":{"polluted":true},"timestamp_ns":1}');
	assert.equal(Object.hasOwn(parsed, '__proto__'), true);
	assert.deepEqual(parsed.__proto__, { polluted: true });
	assert.equal(Object.getPrototypeOf(parsed), Object.prototype);
});

test('bridge parser accepts only JSON whitespace', () => {
	for (const whitespace of [' ', '\t', '\r', '\n']) {
		assert.deepEqual(parseChartCacheOutput(`${whitespace}{${whitespace}"ok"${whitespace}:${whitespace}true${whitespace}}${whitespace}`), { ok: true });
	}
	for (const whitespace of ['\u00a0', '\u2028', '\u2029']) {
		assert.throws(() => parseChartCacheOutput(`${whitespace}{"ok":true}`), /invalid JSON/);
		assert.throws(() => parseChartCacheOutput(`{"ok":${whitespace}true}`), /invalid JSON/);
	}
});

test('bridge parser rejects timestamp integer tokens with leading zeroes', () => {
	for (const token of ['00', '-01', '01']) {
		assert.throws(() => parseChartCacheOutput(`{"bars":[{"timestamp_ns":${token}}]}`), /invalid JSON/);
	}
	assert.deepEqual(parseChartCacheOutput('{"bars":[{"timestamp_ns":0},{"timestamp_ns":-0}]}').bars.map((bar) => bar.timestamp_ns), ['0', '-0']);
});

test('bridge parser rejects malformed or out-of-i64 timestamp tokens', () => {
	for (const token of ['1.5', '1e3', '9223372036854775808', '-9223372036854775809']) {
		assert.throws(() => parseChartCacheOutput(`{"bars":[{"timestamp_ns":${token}}]}`), /invalid JSON/);
	}
});
