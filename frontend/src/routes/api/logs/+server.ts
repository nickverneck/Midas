import { json } from '@sveltejs/kit';
import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

const DEFAULT_LIMIT = 1000;
const CHUNK_SIZE = 64 * 1024;
const DEFAULT_LOG_DIR = 'runs_ga';

const findLatestRunLog = (baseDir: string, fileName: string) => {
    if (!fs.existsSync(baseDir)) return null;
    const entries = fs.readdirSync(baseDir, { withFileTypes: true });
    let latestPath: string | null = null;
    let latestMtime = -1;
    for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        const candidate = path.join(baseDir, entry.name, fileName);
        if (!fs.existsSync(candidate)) continue;
        const stat = fs.statSync(candidate);
        if (stat.mtimeMs > latestMtime) {
            latestMtime = stat.mtimeMs;
            latestPath = candidate;
        }
    }
    return latestPath;
};

const resolveProjectRoot = () => {
    let current = process.cwd();
    for (let i = 0; i < 8; i += 1) {
        if (fs.existsSync(path.join(current, 'Cargo.toml'))) {
            return current;
        }
        const parent = path.resolve(current, '..');
        if (parent === current) break;
        current = parent;
    }
    return process.cwd();
};

const resolveLogPath = (dirParam: string | null, logParam: string | null) => {
    const root = resolveProjectRoot();
    const trimmedDir = dirParam?.trim() ?? '';
    const useFallback = trimmedDir === '' || trimmedDir === DEFAULT_LOG_DIR;
    const dir = trimmedDir !== '' ? trimmedDir : DEFAULT_LOG_DIR;
    const logType = logParam && logParam.trim() !== '' ? logParam.trim() : 'ga';
    const fileName = logType === 'rl' ? 'rl_log.csv' : 'ga_log.csv';
    const candidate = path.isAbsolute(dir) ? dir : path.join(root, dir);
    const resolved = path.resolve(candidate);
    const relative = path.relative(root, resolved);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        return null;
    }
    if (resolved.toLowerCase().endsWith('.csv')) {
        return resolved;
    }
    const logPath = path.join(resolved, fileName);
    if (fs.existsSync(logPath)) {
        return logPath;
    }
    if (useFallback) {
        const latest = findLatestRunLog(resolved, fileName);
        if (latest) return latest;
    }
    return logPath;
};

function readHeader(filePath: string): string {
    const fd = fs.openSync(filePath, 'r');
    try {
        const buf = Buffer.alloc(CHUNK_SIZE);
        const bytes = fs.readSync(fd, buf, 0, buf.length, 0);
        const text = buf.slice(0, bytes).toString('utf-8');
        const idx = text.indexOf('\n');
        if (idx === -1) return text.trim();
        return text.slice(0, idx).trim();
    } finally {
        fs.closeSync(fd);
    }
}

async function readLines(filePath: string, offset: number, limit: number): Promise<string[]> {
    const stream = fs.createReadStream(filePath, { encoding: 'utf-8' });
    const rl = (await import('readline')).createInterface({ input: stream, crlfDelay: Infinity });
    const lines: string[] = [];
    let dataLineIdx = -1;

    try {
        for await (const line of rl) {
            if (dataLineIdx === -1) {
                dataLineIdx = 0;
                continue;
            }
            if (dataLineIdx < offset) {
                dataLineIdx += 1;
                continue;
            }
            if (lines.length >= limit) {
                break;
            }
            lines.push(line);
            dataLineIdx += 1;
        }
    } finally {
        rl.close();
        stream.close();
    }

    return lines;
}

import type { RequestEvent } from "@sveltejs/kit";

export const GET = async ({ url }: RequestEvent) => {
    const headers = { 'Cache-Control': 'no-store' };
    const logPath = resolveLogPath(url.searchParams.get('dir'), url.searchParams.get('log'));
    const mode = url.searchParams.get('mode');

    if (!logPath) {
        return json({ error: 'Invalid log directory' }, { status: 400, headers });
    }

    if (!fs.existsSync(logPath)) {
        return json({ error: 'Log file not found' }, { status: 404, headers });
    }

    if (mode === 'summary') {
        const text = fs.readFileSync(logPath, 'utf-8');
        const parsed = Papa.parse(text, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true
        });
        const data = parsed.data as Record<string, unknown>[];
        const key = url.searchParams.get('key') || 'gen';
        const byGen = new Map<number, number>();
        for (const row of data) {
            const gen = Number(row[key]);
            const fitness = Number(row.fitness);
            if (!Number.isFinite(gen) || !Number.isFinite(fitness)) continue;
            const existing = byGen.get(gen);
            if (existing === undefined || fitness > existing) {
                byGen.set(gen, fitness);
            }
        }
        const summary = Array.from(byGen.entries())
            .sort(([a], [b]) => a - b)
            .map(([gen, fitness]) => ({ gen, fitness }));
        return json({ data: summary, nextOffset: 0, done: true }, { headers });
    }

    const limitParam = url.searchParams.get('limit');
    const offsetParam = url.searchParams.get('offset');
    const limit = limitParam ? Math.max(parseInt(limitParam, 10) || 0, 0) : DEFAULT_LIMIT;
    const offset = offsetParam ? Math.max(parseInt(offsetParam, 10) || 0, 0) : 0;

    const header = readHeader(logPath);
    const dataLines = limit > 0 ? await readLines(logPath, offset, limit) : [];
    const csvData = [header, ...dataLines].join('\n');

    const parsed = Papa.parse(csvData, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });

    const data = parsed.data as Record<string, unknown>[];
    const nextOffset = offset + dataLines.length;
    const stats = fs.statSync(logPath);
    const recentlyModified = Date.now() - stats.mtimeMs < 2000;
    const done = dataLines.length < limit && !recentlyModified;

    return json({ data, nextOffset, done }, { headers });
};
