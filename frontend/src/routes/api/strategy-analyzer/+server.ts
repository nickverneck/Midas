import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { json } from '@sveltejs/kit';

const DEFAULT_TRAIN_PATH = path.join('data', 'train', 'SPY0.parquet');
const DEFAULT_VAL_PATH = path.join('data', 'val', 'SPY.parquet');

const resolveProjectRoot = () => {
	const cwd = process.cwd();
	if (fs.existsSync(path.join(cwd, 'Cargo.toml'))) {
		return cwd;
	}
	const parent = path.resolve(cwd, '..');
	if (fs.existsSync(path.join(parent, 'Cargo.toml'))) {
		return parent;
	}
	return cwd;
};

const resolveParquetPath = (dataset: string | null, pathParam: string | null) => {
	const root = resolveProjectRoot();
	let candidate: string | null = null;
	if (dataset === 'train') {
		candidate = path.join(root, DEFAULT_TRAIN_PATH);
	} else if (dataset === 'val') {
		candidate = path.join(root, DEFAULT_VAL_PATH);
	} else if (pathParam && pathParam.trim() !== '') {
		candidate = path.isAbsolute(pathParam) ? pathParam : path.join(root, pathParam.trim());
	}
	if (!candidate) return null;
	const resolved = path.resolve(candidate);
	const relative = path.relative(root, resolved);
	if (relative.startsWith('..') || path.isAbsolute(relative)) {
		return null;
	}
	return resolved;
};

const resolveAnalyzerBin = (root: string) => {
	const binName = process.platform === 'win32' ? 'strategy_analyzer.exe' : 'strategy_analyzer';
	const binPath = path.join(root, 'target', 'debug', binName);
	if (fs.existsSync(binPath)) {
		return { command: binPath, argsPrefix: [] as string[] };
	}
	return { command: 'cargo', argsPrefix: ['run', '--quiet', '--bin', 'strategy_analyzer', '--'] };
};

type FillModelPayload = {
	mode: 'fixed' | 'random-adverse';
	seed: number;
	maxAdverseTicks: number;
	tickValueUsd: number;
};

const normalizeNonNegativeInteger = (value: unknown, fallback: string, label: string) => {
	const text = value === undefined || value === null || value === '' ? fallback : String(value).trim();
	const numeric = Number(text);
	if (!Number.isFinite(numeric) || !Number.isInteger(numeric) || numeric < 0) {
		return { value: null, error: `${label} must be a non-negative integer` };
	}
	return { value: numeric, error: null };
};

const normalizeNonNegativeNumber = (value: unknown, fallback: string, label: string) => {
	const text = value === undefined || value === null || value === '' ? fallback : String(value).trim();
	const numeric = Number(text);
	if (!Number.isFinite(numeric) || numeric < 0) {
		return { value: null, error: `${label} must be 0 or higher` };
	}
	return { value: numeric, error: null };
};

const normalizeFillModel = (
	fillModel: Record<string, any> | null | undefined
): { fillModel: FillModelPayload | null; error: string | null } => {
	const modeValue = fillModel?.mode === undefined ? 'fixed' : String(fillModel.mode);
	const mode = modeValue === 'random-adverse' ? 'random-adverse' : modeValue === 'fixed' ? 'fixed' : null;
	if (!mode) {
		return { fillModel: null, error: 'Fill model must be fixed or random-adverse' };
	}
	if (mode === 'fixed') {
		return {
			fillModel: {
				mode,
				seed: 0,
				maxAdverseTicks: 2,
				tickValueUsd: 1.25
			},
			error: null
		};
	}

	const seed = normalizeNonNegativeInteger(fillModel?.seed, '0', 'Fill seed');
	if (seed.error) return { fillModel: null, error: seed.error };
	const maxAdverseTicks = normalizeNonNegativeInteger(
		fillModel?.maxAdverseTicks,
		'2',
		'Max adverse ticks'
	);
	if (maxAdverseTicks.error) return { fillModel: null, error: maxAdverseTicks.error };
	const tickValueUsd = normalizeNonNegativeNumber(fillModel?.tickValueUsd, '1.25', 'Tick value USD');
	if (tickValueUsd.error) return { fillModel: null, error: tickValueUsd.error };

	return {
		fillModel: {
			mode,
			seed: seed.value ?? 0,
			maxAdverseTicks: maxAdverseTicks.value ?? 2,
			tickValueUsd: tickValueUsd.value ?? 1.25
		},
		error: null
	};
};

import type { RequestEvent } from '@sveltejs/kit';

export const POST = async ({ request }: RequestEvent) => {
	const headers = { 'Cache-Control': 'no-store' };
	let payload: Record<string, any>;
	try {
		payload = await request.json();
	} catch {
		return json({ error: 'Invalid JSON payload' }, { status: 400, headers });
	}

	const dataset = typeof payload.dataset === 'string' ? payload.dataset : null;
	const pathParam = typeof payload.path === 'string' ? payload.path : null;
	const signal = payload.signal;
	if (!signal?.indicatorA || !signal?.indicatorB) {
		return json({ error: 'Signal configuration is required' }, { status: 400, headers });
	}

	const fillModelResult = normalizeFillModel(payload.env?.fillModel);
	if (fillModelResult.error || !fillModelResult.fillModel) {
		return json({ error: fillModelResult.error || 'Invalid fill model' }, { status: 400, headers });
	}

	const filePath = resolveParquetPath(dataset, pathParam);
	if (!filePath) {
		return json({ error: 'Invalid parquet path' }, { status: 400, headers });
	}
	if (!fs.existsSync(filePath)) {
		return json({ error: 'Parquet file not found' }, { status: 404, headers });
	}

	const barKind = payload.barKind === 'volume' ? 'volume' : 'price-action';
	const priceSource = payload.priceSource === 'heikin-ashi' ? 'heikin-ashi' : 'ohlc';
	const parsedVolumeBarSize = Number(payload.volumeBarSize);
	if (barKind === 'volume' && (!Number.isFinite(parsedVolumeBarSize) || parsedVolumeBarSize <= 0)) {
		return json({ error: 'Volume bar size must be greater than 0' }, { status: 400, headers });
	}
	const volumeBarSize = barKind === 'volume' ? parsedVolumeBarSize : null;

	const root = resolveProjectRoot();
	const { command, argsPrefix } = resolveAnalyzerBin(root);

	const config = {
		file: filePath,
		barKind,
		volumeBarSize,
		priceSource,
		offset: payload.offset ?? null,
		limit: payload.limit ?? null,
		initialBalance: payload.env?.initialBalance ?? 10_000,
		maxPosition: payload.env?.maxPosition ?? 1,
		commissionRoundTurn: payload.env?.commission ?? 1.6,
		slippagePerContract: payload.env?.slippage ?? 0.25,
		fillModel: fillModelResult.fillModel,
		marginPerContract: payload.env?.marginPerContract ?? 50,
		contractMultiplier: payload.env?.contractMultiplier ?? 1.0,
		marginMode: payload.env?.marginMode ?? 'per-contract',
		enforceMargin: payload.env?.enforceMargin ?? true,
		globex: payload.globex ?? false,
		signal: {
			indicatorA: signal.indicatorA,
			indicatorB: signal.indicatorB,
			buyAction: signal.buyAction,
			sellAction: signal.sellAction
		},
		takeProfit: payload.takeProfit ?? null,
		stopLoss: payload.stopLoss ?? null,
		fitness: payload.fitness ?? null,
		maxCombinations: payload.maxCombinations ?? null
	};

	const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'midas-analyzer-'));
	const configPath = path.join(tempDir, 'analyzer.json');
	fs.writeFileSync(configPath, JSON.stringify(config), 'utf-8');

	try {
		const output = execFileSync(command, [...argsPrefix, '--config', configPath], {
			cwd: root,
			maxBuffer: 1024 * 1024 * 200
		});
		const text = output.toString('utf-8').trim();
		const result = JSON.parse(text);
		return json(result, { headers });
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Strategy analyzer failed';
		return json({ error: message }, { status: 500, headers });
	} finally {
		try {
			fs.rmSync(tempDir, { recursive: true, force: true });
		} catch {
			// best-effort cleanup
		}
	}
};
