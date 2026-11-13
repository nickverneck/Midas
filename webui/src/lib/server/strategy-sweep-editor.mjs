export const MAX_SWEEP_DIMENSIONS = 8;
export const MAX_VALUES_PER_DIMENSION = 16;
export const MAX_SWEEP_COMBINATIONS = 32;
export const MAX_SWEEP_NAME_LENGTH = 100;

/** @typedef {{path: string, values: Array<string | number | boolean>}} EditedParameter */

/** Validate and bound the editable part of a cloned native replay spec. */
/** @param {Record<string, any>} source @param {{name: string, parameters: EditedParameter[]}} input @param {string} outputDir @param {string} sweepId */
export function createEditedSweep(source, input, outputDir, sweepId) {
	if (!source || typeof source !== 'object' || Array.isArray(source)) throw new Error('Selected catalog sweep is not a native replay specification.');
	if (typeof input?.name !== 'string' || !input.name.trim() || input.name.trim().length > MAX_SWEEP_NAME_LENGTH) {
		throw new Error(`name must contain 1 to ${MAX_SWEEP_NAME_LENGTH} characters.`);
	}
	if (!Array.isArray(input.parameters) || input.parameters.length > MAX_SWEEP_DIMENSIONS) {
		throw new Error(`parameters must contain at most ${MAX_SWEEP_DIMENSIONS} dimensions.`);
	}
	let combinations = 1;
	const seen = new Set();
	const parameters = input.parameters.map((/** @type {EditedParameter} */ item) => {
		if (!item || typeof item !== 'object' || Array.isArray(item)) throw new Error('Each parameter must contain a path and values list.');
		if (typeof item.path !== 'string' || item.path.length > 128 || !/^[a-z][a-z0-9_.]*$/.test(item.path)) {
			throw new Error('Parameter paths must be dot-separated native scalar field names.');
		}
		if (seen.has(item.path)) throw new Error(`Duplicate parameter path: ${item.path}`);
		seen.add(item.path);
		if (!Array.isArray(item.values) || item.values.length < 1 || item.values.length > MAX_VALUES_PER_DIMENSION) {
			throw new Error(`Each values list must contain 1 to ${MAX_VALUES_PER_DIMENSION} scalar values.`);
		}
		const distinct = new Set();
		for (const value of item.values) {
			const scalar = typeof value === 'string' || typeof value === 'boolean' || (typeof value === 'number' && Number.isFinite(value));
			if (!scalar || (typeof value === 'string' && value.length > 128)) throw new Error('Parameter values must be bounded string, boolean, or finite number scalars.');
			const encoded = JSON.stringify(value);
			if (distinct.has(encoded)) throw new Error(`Parameter ${item.path} contains a duplicate value.`);
			distinct.add(encoded);
		}
		combinations *= item.values.length;
		if (combinations > MAX_SWEEP_COMBINATIONS) throw new Error(`Sweep grid may contain at most ${MAX_SWEEP_COMBINATIONS} combinations.`);
		return { path: item.path, values: [...item.values] };
	});
	if (source.base_strategy == null || typeof source.base_strategy !== 'object' || Array.isArray(source.base_strategy)) {
		throw new Error('Selected sweep has no supported base strategy.');
	}
	if (typeof source.sweep_id !== 'string' || typeof source.name !== 'string' || !source.base_dataset_view) {
		throw new Error('Selected sweep is not a supported native replay specification.');
	}
	return {
		...source,
		name: input.name.trim(),
		sweep_id: sweepId,
		parameters,
		output_dir: outputDir
	};
}
