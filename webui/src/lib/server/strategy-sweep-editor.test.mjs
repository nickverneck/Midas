import test from 'node:test';
import assert from 'node:assert/strict';
import { createEditedSweep } from './strategy-sweep-editor.mjs';

const base = { schema_version: 1, sweep_id: 'native-base', name: 'Base', base_dataset_view: { id: 'dataset' }, base_strategy: { kind: 'EmaCross' }, parameters: [], parallelism: 20, output_dir: '/old/output' };

test('editor clones only bounded parameter dimensions and assigns server-owned identity/output', () => {
	const edited = createEditedSweep(base, { name: 'Derived', parameters: [{ path: 'order_qty', values: [1, 2] }] }, '/safe/output', 'webui_123');
	assert.equal(edited.sweep_id, 'webui_123');
	assert.equal(edited.output_dir, '/safe/output');
	assert.equal(edited.parallelism, 20);
	assert.deepEqual(edited.parameters, [{ path: 'order_qty', values: [1, 2] }]);
	assert.deepEqual(base.parameters, []);
});

test('editor rejects path injection, compound values, and oversized grids', () => {
	assert.throws(() => createEditedSweep(base, { name: 'X', parameters: [{ path: '../output_dir', values: [1] }] }, '/x', 'id'), /paths/);
	assert.throws(() => createEditedSweep(base, { name: 'X', parameters: [{ path: 'order_qty', values: [{ nested: true }] }] }, '/x', 'id'), /scalar/);
	assert.throws(() => createEditedSweep(base, { name: 'X', parameters: [{ path: 'a', values: [1, 2, 3, 4, 5, 6] }, { path: 'b', values: [1, 2, 3, 4, 5, 6] }] }, '/x', 'id'), /32 combinations/);
});
