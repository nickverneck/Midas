import { json } from '@sveltejs/kit';
import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

const DEFAULT_LIMIT = 10000;
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

function readTailLines(filePath: string, limit: number): string[] {
    const fd = fs.openSync(filePath, 'r');
    try {
        const stat = fs.fstatSync(fd);
        let pos = stat.size;
        let buffer = '';
        let lines: string[] = [];

        while (pos > 0 && lines.length <= limit) {
            const readSize = Math.min(CHUNK_SIZE, pos);
            pos -= readSize;
            const chunk = Buffer.alloc(readSize);
            fs.readSync(fd, chunk, 0, readSize, pos);
            buffer = chunk.toString('utf-8') + buffer;
            lines = buffer.split('\n');
        }

        return lines.filter((l) => l.trim().length > 0).slice(-limit);
    } finally {
        fs.closeSync(fd);
    }
}

export const GET = async ({ url }) => {
    const logPath = path.resolve('../runs_ga/ga_log.csv');

    if (!fs.existsSync(logPath)) {
        return json({ error: 'Log file not found' }, { status: 404 });
    }

    const limitParam = url.searchParams.get('limit');
    const limit = limitParam ? Math.max(parseInt(limitParam, 10) || 0, 0) : DEFAULT_LIMIT;
    let csvData: string;

    if (limit > 0) {
        const header = readHeader(logPath);
        let tailLines = readTailLines(logPath, limit);
        if (tailLines.length > 0 && tailLines[0].trim() === header) {
            tailLines = tailLines.slice(1);
        }
        csvData = [header, ...tailLines].join('\n');
    } else {
        csvData = fs.readFileSync(logPath, 'utf-8');
    }

    const parsed = Papa.parse(csvData, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });

    return json(parsed.data);
};
