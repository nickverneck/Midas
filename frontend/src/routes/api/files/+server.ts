import { json } from '@sveltejs/kit';
import fs from 'fs';
import path from 'path';

type FileEntry = {
    name: string;
    path: string;
    kind: 'dir' | 'file';
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

const resolveDir = (dirParam: string | null) => {
    const root = resolveProjectRoot();
    const dir = dirParam && dirParam.trim() !== '' ? dirParam.trim() : '';
    const candidate = dir ? (path.isAbsolute(dir) ? dir : path.join(root, dir)) : root;
    const resolved = path.resolve(candidate);
    const relative = path.relative(root, resolved);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        return null;
    }
    if (!fs.existsSync(resolved)) {
        return null;
    }
    const stat = fs.statSync(resolved);
    if (!stat.isDirectory()) {
        return null;
    }
    return { root, resolved, relative };
};

const parseExtensions = (extParam: string | null) => {
    if (!extParam || extParam.trim() === '') {
        return ['parquet'];
    }
    const parts = extParam
        .split(',')
        .map((part) => part.trim().replace(/^\./, '').toLowerCase())
        .filter(Boolean);
    return parts.length > 0 ? parts : ['parquet'];
};

import type { RequestEvent } from "@sveltejs/kit";

export const GET = async ({ url }: RequestEvent) => {
    const headers = { 'Cache-Control': 'no-store' };
    const dirParam = url.searchParams.get('dir');
    const extParam = url.searchParams.get('ext');
    const resolved = resolveDir(dirParam);
    if (!resolved) {
        return json({ error: 'Invalid directory' }, { status: 400, headers });
    }

    const extensions = parseExtensions(extParam);
    const entries: FileEntry[] = fs
        .readdirSync(resolved.resolved, { withFileTypes: true })
        .filter((entry) => {
            if (entry.isDirectory()) return true;
            const name = entry.name.toLowerCase();
            return extensions.some((ext) => name.endsWith(`.${ext}`));
        })
        .map((entry) => {
            const entryPath = resolved.relative
                ? path.join(resolved.relative, entry.name)
                : entry.name;
            return {
                name: entry.name,
                path: entryPath,
                kind: (entry.isDirectory() ? 'dir' : 'file') as 'dir' | 'file'
            };
        })
        .sort((a, b) => {
            if (a.kind !== b.kind) {
                return a.kind === 'dir' ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
        });

    const parent =
        resolved.relative === '' ? null : path.dirname(resolved.relative === '' ? '.' : resolved.relative);
    const normalizedParent = parent === '.' ? '' : parent;

    return json(
        {
            dir: resolved.relative,
            parent: normalizedParent,
            entries
        },
        { headers }
    );
};
