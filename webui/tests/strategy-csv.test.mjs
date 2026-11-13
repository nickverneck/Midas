import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseStrategyCsv, readStrategyCsv, readStrategyText } from '../src/lib/server/strategy-csv.mjs';

test('strategy CSV parsing preserves quoted commas, escaped quotes, and multiline fields', () => {
	const rows = parseStrategyCsv('timestamp_ns,note\n1,"entry, scale-in"\n2,"said ""hold""\nthen exit"\n');
	assert.deepEqual(rows, [
		{ timestamp_ns: '1', note: 'entry, scale-in' },
		{ timestamp_ns: '2', note: 'said "hold"\nthen exit' }
	]);
});

test('strategy CSV reads are byte bounded', async () => {
	const dir = await mkdtemp(join(tmpdir(), 'midas-strategy-csv-'));
	const path = join(dir, 'trades.csv');
	try {
		await writeFile(path, 'id,note\n1,"a long field"\n');
		await assert.rejects(readStrategyCsv(path, 8), /exceeds the 8 byte limit/);
		await assert.rejects(readStrategyText(path, 8), /exceeds the 8 byte limit/);
	} finally {
		await rm(dir, { recursive: true, force: true });
	}
});
