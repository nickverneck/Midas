import { json } from '@sveltejs/kit';
import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

const DEFAULT_LIMIT = 1000;
const CHUNK_SIZE = 64 * 1024;

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

export const GET = async ({ url }) => {
    const logPath = path.resolve('../runs_ga/ga_log.csv');

    if (!fs.existsSync(logPath)) {
        return json({ error: 'Log file not found' }, { status: 404 });
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
    const nextOffset = offset + data.length;
    const done = data.length < limit;

    return json({ data, nextOffset, done });
};
