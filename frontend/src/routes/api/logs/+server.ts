import { json } from '@sveltejs/kit';
import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

const DEFAULT_LIMIT = 1000;
const CHUNK_SIZE = 64 * 1024;
const DEFAULT_LOG_DIR = 'runs_ga';

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

const resolveLogPath = (dirParam: string | null) => {
    const root = resolveProjectRoot();
    const dir = dirParam && dirParam.trim() !== '' ? dirParam.trim() : DEFAULT_LOG_DIR;
    const candidate = path.isAbsolute(dir) ? dir : path.join(root, dir);
    const resolved = path.resolve(candidate);
    const relative = path.relative(root, resolved);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        return null;
    }
    return path.join(resolved, 'ga_log.csv');
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

function fileEndsWithNewline(filePath: string): boolean {
    const fd = fs.openSync(filePath, 'r');
    try {
        const stat = fs.fstatSync(fd);
        if (stat.size === 0) return true;
        const buf = Buffer.alloc(1);
        fs.readSync(fd, buf, 0, 1, stat.size - 1);
        return buf.toString() === '\n';
    } finally {
        fs.closeSync(fd);
    }
}

async function readLines(filePath: string, offset: number, limit: number): Promise<string[]> {
    const stream = fs.createReadStream(filePath, { encoding: 'utf-8' });
    const rl = (await import('readline')).createInterface({ input: stream, crlfDelay: Infinity });
    const lines: string[] = [];
    let dataLineIdx = -1;
    let hitLimit = false;

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
                hitLimit = true;
                break;
            }
            lines.push(line);
            dataLineIdx += 1;
        }
    } finally {
        rl.close();
        stream.close();
    }

    if (!hitLimit && lines.length > 0 && !fileEndsWithNewline(filePath)) {
        lines.pop();
    }

    return lines;
}

export const GET = async ({ url }) => {
    const headers = { 'Cache-Control': 'no-store' };
    const logPath = resolveLogPath(url.searchParams.get('dir'));

    if (!logPath) {
        return json({ error: 'Invalid log directory' }, { status: 400, headers });
    }

    if (!fs.existsSync(logPath)) {
        return json({ error: 'Log file not found' }, { status: 404, headers });
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
    const done = dataLines.length < limit;

    return json({ data, nextOffset, done }, { headers });
};
