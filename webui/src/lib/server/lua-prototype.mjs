export const MAX_LUA_SOURCE_BYTES = 12 * 1024;

/** @param {unknown} source */
export function validateLuaPrototypeSource(source) {
	if (typeof source !== 'string' || !source.trim()) return 'luaSource must contain Lua source text.';
	if (Buffer.byteLength(source, 'utf8') > MAX_LUA_SOURCE_BYTES) return 'Lua source is limited to 12 KiB.';
	return undefined;
}

/** @param {Record<string, any>} selected @param {string} outputDir @param {string} source */
export function createLuaPrototypeSweep(selected, outputDir, source) {
	if (!selected || typeof selected !== 'object' || !selected.base_strategy || typeof selected.base_strategy !== 'object' || Array.isArray(selected.base_strategy)) {
		throw new Error('Selected sweep has no supported base strategy.');
	}
	return {
		...selected,
		name: `${String(selected.name ?? 'Replay sweep').slice(0, 100)} — Lua prototype`,
		output_dir: outputDir,
		execution_mode: 'isolated_services',
		parameters: [],
		constraints: [],
		max_runs: 1,
		parallelism: 1,
		output_formats: [...new Set(['json_summary', ...(Array.isArray(selected.output_formats) ? selected.output_formats : [])])],
		base_strategy: {
			...selected.base_strategy,
			kind: 'Lua',
			native_signal_timing: 'ClosedBar',
			lua: { ...(selected.base_strategy.lua ?? {}), schema_version: 1, source, memory_limit_bytes: 8 * 1024 * 1024, instruction_limit: 100_000 }
		}
	};
}
