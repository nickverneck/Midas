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

const resolveBacktestBin = (root: string) => {
	const binName = process.platform === 'win32' ? 'backtest_script.exe' : 'backtest_script';
	const binPath = path.join(root, 'target', 'debug', binName);
	if (fs.existsSync(binPath)) {
		return { command: binPath, argsPrefix: [] as string[] };
	}
	return { command: 'cargo', argsPrefix: ['run', '--quiet', '--bin', 'backtest_script', '--'] };
};

const buildArgs = (payload: Record<string, any>, filePath: string, scriptPath: string) => {
	const args: string[] = ['--file', filePath, '--script', scriptPath];

	const add = (key: string, value: unknown) => {
		if (value === undefined || value === null || value === '') return;
		args.push(`--${key}`, String(value));
	};

	add('offset', payload.offset);
	add('limit', payload.limit);
	add('initial-balance', payload.env?.initialBalance);
	add('max-position', payload.env?.maxPosition);
	add('commission-round-turn', payload.env?.commission);
	add('slippage-per-contract', payload.env?.slippage);
	add('margin-per-contract', payload.env?.marginPerContract);
	add('contract-multiplier', payload.env?.contractMultiplier);
	add('margin-mode', payload.env?.marginMode);
	add('enforce-margin', payload.env?.enforceMargin);
	add('memory-limit-mb', payload.limits?.memoryMb);
	add('instruction-limit', payload.limits?.instructionLimit);
	add('instruction-interval', payload.limits?.instructionInterval);

	if (payload.globex) args.push('--globex');
	if (payload.traceActions) args.push('--trace-actions');

	return args;
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
	const script = typeof payload.script === 'string' ? payload.script : '';

	if (!script.trim()) {
		return json({ error: 'Script is required' }, { status: 400, headers });
	}

	const filePath = resolveParquetPath(dataset, pathParam);
	if (!filePath) {
		return json({ error: 'Invalid parquet path' }, { status: 400, headers });
	}
	if (!fs.existsSync(filePath)) {
		return json({ error: 'Parquet file not found' }, { status: 404, headers });
	}

	const root = resolveProjectRoot();
	const { command, argsPrefix } = resolveBacktestBin(root);

	const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'midas-backtest-'));
	const scriptPath = path.join(tempDir, 'strategy.lua');
	fs.writeFileSync(scriptPath, script, 'utf-8');

	try {
		const cliArgs = buildArgs(payload, filePath, scriptPath);
		const output = execFileSync(command, [...argsPrefix, ...cliArgs], {
			cwd: root,
			maxBuffer: 1024 * 1024 * 200
		});
		const text = output.toString('utf-8').trim();
		const result = JSON.parse(text);
		return json(result, { headers });
	} catch (err) {
		const message = err instanceof Error ? err.message : 'Backtest failed';
		return json({ error: message }, { status: 500, headers });
	} finally {
		try {
			fs.rmSync(tempDir, { recursive: true, force: true });
		} catch {
			// best-effort cleanup
		}
	}
};
