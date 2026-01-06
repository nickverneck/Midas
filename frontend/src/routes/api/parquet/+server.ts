import { execFileSync } from 'child_process';
import fs from 'fs';
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

const resolveParquetDump = (root: string) => {
    const binName = process.platform === 'win32' ? 'parquet_dump.exe' : 'parquet_dump';
    const binPath = path.join(root, 'target', 'debug', binName);
    if (fs.existsSync(binPath)) {
        return { command: binPath, argsPrefix: [] as string[] };
    }
    return { command: 'cargo', argsPrefix: ['run', '--quiet', '--bin', 'parquet_dump', '--'] };
};

export const GET = async ({ url }) => {
    const headers = { 'Cache-Control': 'no-store' };
    const dataset = url.searchParams.get('dataset');
    const pathParam = url.searchParams.get('path');
    const limit = url.searchParams.get('limit');
    const offset = url.searchParams.get('offset');
    const columns = url.searchParams.get('columns');

    const root = resolveProjectRoot();
    const filePath = resolveParquetPath(dataset, pathParam);
    if (!filePath) {
        return json({ error: 'Invalid parquet path' }, { status: 400, headers });
    }
    if (!fs.existsSync(filePath)) {
        return json({ error: 'Parquet file not found' }, { status: 404, headers });
    }

    const cliArgs = ['--file', filePath];
    if (limit && Number.isFinite(Number(limit))) {
        cliArgs.push('--limit', String(limit));
    }
    if (offset && Number.isFinite(Number(offset))) {
        cliArgs.push('--offset', String(offset));
    }
    if (columns && columns.trim() !== '') {
        cliArgs.push('--columns', columns.trim());
    }

    const { command, argsPrefix } = resolveParquetDump(root);
    try {
        const output = execFileSync(command, [...argsPrefix, ...cliArgs], {
            cwd: root,
            maxBuffer: 1024 * 1024 * 200
        });
        return new Response(output, {
            headers: { ...headers, 'Content-Type': 'text/csv' }
        });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Parquet dump failed';
        return json({ error: message }, { status: 500, headers });
    }
};
