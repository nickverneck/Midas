import { isAbsolute } from 'node:path';

export class StrategyInputError extends Error {
	readonly status = 400;
}

export function decodeSweepSpecId(id: unknown): string {
	if (typeof id !== 'string' || id.length > 2048) throw new StrategyInputError('sweepId is required.');
	const decoded = Buffer.from(id, 'base64url').toString('utf8');
	const parts = decoded.split(/[\\/]/);
	if (Buffer.from(decoded).toString('base64url') !== id || decoded.includes('\0') || isAbsolute(decoded) || parts.some(part => part === '..' || part === '.' || part === '')) throw new StrategyInputError('Unknown sweepId.');
	return decoded;
}

const rankMetrics = new Set(['robustness', 'net-pnl', 'gross-pnl', 'fees', 'max-drawdown', 'required-account-size', 'return-required', 'profit-factor', 'win-rate', 'trade-count', 'average-trade', 'average-giveback', 'largest-giveback', 'mfe-capture']);

export function isStrategyRankMetric(value: unknown): value is string {
	return typeof value === 'string' && rankMetrics.has(value);
}
