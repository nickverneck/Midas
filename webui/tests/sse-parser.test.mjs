import test from 'node:test';
import assert from 'node:assert/strict';
import { SseParser } from '../src/lib/sse-parser.ts';

test('parses LF and CRLF frames split at arbitrary chunk boundaries', () => {
	const frames = [];
	const parser = new SseParser((frame) => frames.push(frame));
	for (const chunk of ['event: snapshot\r', '\ndata: {"price":', '1}\r\n\r', '\nevent: status\n', 'data: {"ended":true}\n\n']) parser.push(chunk);
	parser.finish();
	assert.deepEqual(frames, [
		{ event: 'snapshot', data: '{"price":1}' },
		{ event: 'status', data: '{"ended":true}' }
	]);
});

test('parses lone-CR line endings and does not split CRLF across chunks', () => {
	const frames = [];
	const parser = new SseParser((frame) => frames.push(frame));
	for (const chunk of ['event: note\rdata: lone-cr\r\r', '\nevent: status\r', '\ndata: stopped\r\r']) parser.push(chunk);
	parser.finish();
	assert.deepEqual(frames, [
		{ event: 'note', data: 'lone-cr' },
		{ event: 'status', data: 'stopped' }
	]);
});

test('joins multiline data fields and flushes a pending UTF-8 code point at EOF', () => {
	const frames = [];
	const parser = new SseParser((frame) => frames.push(frame));
	const encoded = new TextEncoder().encode('event: note\ndata: first\ndata: café');
	const decoder = new TextDecoder();
	parser.push(decoder.decode(encoded.slice(0, encoded.length - 1), { stream: true }));
	parser.push(decoder.decode(encoded.slice(encoded.length - 1), { stream: true }));
	parser.push(decoder.decode());
	parser.finish();
	assert.deepEqual(frames, [{ event: 'note', data: 'first\ncafé' }]);
});

test('rejects an oversized unterminated or terminated frame', () => {
	const parser = new SseParser(() => {}, 8);
	assert.throws(() => parser.push('data: 123'), /size limit/);
	const terminated = new SseParser(() => {}, 8);
	assert.throws(() => terminated.push('data: 123456\n\n'), /size limit/);
});

test('ignores comments and dispatches a final frame when EOF arrives without a blank line', () => {
	const frames = [];
	const parser = new SseParser((frame) => frames.push(frame));
	parser.push(': heartbeat\n\nevent: status\ndata: stopped');
	parser.finish();
	assert.deepEqual(frames, [{ event: 'status', data: 'stopped' }]);
});
