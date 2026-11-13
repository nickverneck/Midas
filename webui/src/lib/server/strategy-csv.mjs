import { open } from 'node:fs/promises';
import Papa from 'papaparse';

export const MAX_STRATEGY_CSV_BYTES = 8 * 1024 * 1024;

/** @param {string} path @param {number} maxBytes @returns {Promise<string>} */
export async function readStrategyText(path, maxBytes) {
	if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) throw new RangeError('A non-negative byte limit is required.');
	const file = await open(path, 'r');
	try {
		const { size } = await file.stat();
		if (size > maxBytes) throw new Error(`Strategy artifact exceeds the ${maxBytes} byte limit.`);
		const buffer = Buffer.alloc(maxBytes + 1);
		let length = 0;
		while (length < buffer.length) {
			const { bytesRead } = await file.read(buffer, length, buffer.length - length, length);
			if (!bytesRead) break;
			length += bytesRead;
		}
		if (length > maxBytes) throw new Error(`Strategy artifact exceeds the ${maxBytes} byte limit.`);
		return buffer.subarray(0, length).toString('utf8');
	} finally {
		await file.close();
	}
}

/** Parse a headered native strategy CSV while honoring RFC-style quoting. */
/** @param {string} text @returns {Array<Record<string, string>>} */
export function parseStrategyCsv(text) {
	if (!text.trim()) return [];
	const parsed = Papa.parse(text, { header: true, skipEmptyLines: 'greedy' });
	if (parsed.errors.length) {
		throw new Error(`Invalid strategy CSV: ${parsed.errors[0].message}`);
	}
	return parsed.data;
}

/** Read at most maxBytes + 1 bytes so a file that grows after stat is still bounded. */
/** @param {string} path @param {number} [maxBytes] @returns {Promise<Array<Record<string, string>>>} */
export async function readStrategyCsv(path, maxBytes = MAX_STRATEGY_CSV_BYTES) {
	return parseStrategyCsv(await readStrategyText(path, maxBytes));
}
