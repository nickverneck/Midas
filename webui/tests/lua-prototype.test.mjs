import test from 'node:test';
import assert from 'node:assert/strict';
import { createLuaPrototypeSweep, MAX_LUA_SOURCE_BYTES, validateLuaPrototypeSource } from '../src/lib/server/lua-prototype.mjs';

test('Lua prototype source validation is non-empty and bounded by UTF-8 bytes', () => {
	assert.match(validateLuaPrototypeSource('  '), /must contain/);
	assert.equal(validateLuaPrototypeSource('return "hold"'), undefined);
	assert.equal(validateLuaPrototypeSource('a'.repeat(MAX_LUA_SOURCE_BYTES)), undefined);
	assert.match(validateLuaPrototypeSource('a'.repeat(MAX_LUA_SOURCE_BYTES + 1)), /12 KiB/);
	assert.match(validateLuaPrototypeSource('é'.repeat(MAX_LUA_SOURCE_BYTES / 2 + 1)), /12 KiB/);
});

test('Lua prototype keeps selected replay settings and forces one isolated Lua run', () => {
	const selected = {
		name: 'ES research', sweep_id: 'es-sweep', base_dataset_view: { id: 'dataset-1' },
		bar_type: { kind: 'minute', value: 5 }, engine_mode: 'deterministic', fill_model: 'tick_bid_ask',
		initial_capital: 25000, parameters: [{ path: 'base_strategy.order_qty', values: [1, 2] }], constraints: [{ expr: 'x' }],
		output_formats: ['csv_rows'],
		base_strategy: { kind: 'Native', order_qty: 2, native_ema: { fast: 4 }, lua: { memory_limit_bytes: 32 } }
	};
	const output = '/jobs/123/output';
	const source = 'function on_bar(ctx, bar) return "hold" end';
	const prototype = createLuaPrototypeSweep(selected, output, source);
	assert.equal(prototype.base_dataset_view, selected.base_dataset_view);
	assert.equal(prototype.fill_model, selected.fill_model);
	assert.equal(prototype.initial_capital, selected.initial_capital);
	assert.equal(prototype.output_dir, output);
	assert.equal(prototype.execution_mode, 'isolated_services');
	assert.equal(prototype.parameters.length, 0);
	assert.equal(prototype.constraints.length, 0);
	assert.equal(prototype.max_runs, 1);
	assert.equal(prototype.parallelism, 1);
	assert.deepEqual(prototype.output_formats, ['json_summary', 'csv_rows']);
	assert.equal(prototype.base_strategy.kind, 'Lua');
	assert.equal(prototype.base_strategy.order_qty, 2);
	assert.equal(prototype.base_strategy.native_ema, selected.base_strategy.native_ema);
	assert.deepEqual(prototype.base_strategy.lua, { memory_limit_bytes: 8 * 1024 * 1024, schema_version: 1, source, instruction_limit: 100_000 });
	assert.equal(selected.base_strategy.kind, 'Native');
	for (const format of ['csv_rows', 'parquet_rows']) {
		const base = { ...selected, output_formats: [format] };
		const lua = createLuaPrototypeSweep(base, output, source);
		assert.deepEqual(lua.output_formats, ['json_summary', format]);
		assert.deepEqual(base.output_formats, [format]);
	}
	assert.throws(() => createLuaPrototypeSweep({ ...selected, base_strategy: null }, output, source), /base strategy/);
});
