import { json } from '@sveltejs/kit';
import fs from 'fs';
import path from 'path';

const DEFAULT_RUN_DIR = 'runs_ga';
const BEHAVIOR_SUBDIR = 'behavior';

const findLatestBehaviorDir = (baseDir: string) => {
    if (!fs.existsSync(baseDir)) return null;
    const entries = fs.readdirSync(baseDir, { withFileTypes: true });
    let latestPath: string | null = null;
    let latestMtime = -1;
    for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        const candidate = path.join(baseDir, entry.name, BEHAVIOR_SUBDIR);
        if (!fs.existsSync(candidate)) continue;
        const stat = fs.statSync(candidate);
        if (stat.isDirectory() && stat.mtimeMs > latestMtime) {
            latestMtime = stat.mtimeMs;
            latestPath = candidate;
        }
    }
    return latestPath;
};

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

const resolveBehaviorDir = (dirParam: string | null) => {
    const root = resolveProjectRoot();
    const trimmedDir = dirParam?.trim() ?? '';
    const useFallback = trimmedDir === '' || trimmedDir === DEFAULT_RUN_DIR;
    const runDir = trimmedDir !== '' ? trimmedDir : DEFAULT_RUN_DIR;
    const candidate = path.isAbsolute(runDir) ? runDir : path.join(root, runDir);
    const resolvedRunDir = path.resolve(candidate);
    const relative = path.relative(root, resolvedRunDir);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        return null;
    }
    const behaviorDir = path.join(resolvedRunDir, BEHAVIOR_SUBDIR);
    if (fs.existsSync(behaviorDir)) {
        return behaviorDir;
    }
    if (useFallback) {
        const latest = findLatestBehaviorDir(resolvedRunDir);
        if (latest) return latest;
    }
    return behaviorDir;
};

const resolveBehaviorFile = (dirParam: string | null, fileParam: string | null) => {
    if (!fileParam) return null;
    const behaviorDir = resolveBehaviorDir(dirParam);
    if (!behaviorDir) return null;
    const safeName = path.basename(fileParam);
    const candidate = path.resolve(path.join(behaviorDir, safeName));
    const relative = path.relative(behaviorDir, candidate);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        return null;
    }
    return candidate;
};

import type { RequestEvent } from "@sveltejs/kit";

export const GET = async ({ url }: RequestEvent) => {
    const headers = { 'Cache-Control': 'no-store' };
    const mode = url.searchParams.get('mode');
    const dirParam = url.searchParams.get('dir');

    if (mode === 'list') {
        const behaviorDir = resolveBehaviorDir(dirParam);
        if (!behaviorDir || !fs.existsSync(behaviorDir)) {
            return json({ files: [] }, { headers });
        }
        const files = fs
            .readdirSync(behaviorDir, { withFileTypes: true })
            .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.csv'))
            .map((entry) => entry.name)
            .sort((a, b) => a.localeCompare(b));
        return json({ files }, { headers });
    }

    const fileParam = url.searchParams.get('file');
    const filePath = resolveBehaviorFile(dirParam, fileParam);
    if (!filePath) {
        return json({ error: 'Invalid behavior file' }, { status: 400, headers });
    }
    if (!fs.existsSync(filePath)) {
        return json({ error: 'Behavior file not found' }, { status: 404, headers });
    }

    const csvText = fs.readFileSync(filePath, 'utf-8');
    return new Response(csvText, {
        headers: { ...headers, 'Content-Type': 'text/csv' }
    });
};
