import test from 'node:test';
import assert from 'node:assert/strict';
import { providerSnapshotBars } from '../src/lib/provider-snapshot-bars.ts';

test('complete provider snapshots preserve bar order and duplicate nanosecond timestamps', () => {
	const bars = [
		{ ts_ns: '1735689600123456789', open: 1, high: 3, low: 1, close: 2 },
		{ ts_ns: '1735689600123456789', open: 2, high: 4, low: 2, close: 3 },
		{ ts_ns: '1735689601123456789', open: 3, high: 5, low: 3, close: 4 }
	];
	const snapshot = providerSnapshotBars(bars);
	assert.deepEqual(snapshot, bars);
	assert.notEqual(snapshot, bars);
	assert.notEqual(snapshot[0], bars[0]);
});
