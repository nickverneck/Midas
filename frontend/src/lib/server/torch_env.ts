import fs from 'fs';
import path from 'path';

export const resolveProjectRoot = () => {
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

export const loadDotEnv = (root: string) => {
    const envPath = path.join(root, '.env');
    if (!fs.existsSync(envPath)) return {};
    const out: Record<string, string> = {};
    const raw = fs.readFileSync(envPath, 'utf8');
    for (const line of raw.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;
        const [key, ...rest] = trimmed.split('=');
        const value = rest.join('=').trim().replace(/^['"]|['"]$/g, '');
        if (key) out[key.trim()] = value;
    }
    return out;
};

const findVenvTorchRoot = (venvDir: string) => {
    const isWindows = process.platform === 'win32';
    if (isWindows) {
        const torchRoot = path.join(venvDir, 'Lib', 'site-packages', 'torch');
        return fs.existsSync(torchRoot) ? torchRoot : null;
    }
    const libDir = path.join(venvDir, 'lib');
    if (!fs.existsSync(libDir)) return null;
    const entries = fs.readdirSync(libDir, { withFileTypes: true });
    for (const entry of entries) {
        if (!entry.isDirectory() || !entry.name.startsWith('python')) continue;
        const torchRoot = path.join(libDir, entry.name, 'site-packages', 'torch');
        if (fs.existsSync(torchRoot)) return torchRoot;
    }
    return null;
};

export const resolveTorchEnv = (root: string, baseEnv: NodeJS.ProcessEnv) => {
    const env = { ...baseEnv };
    const isWindows = process.platform === 'win32';
    const venvDir = path.join(root, '.venv');
    const venvBin = path.join(venvDir, isWindows ? 'Scripts' : 'bin');
    const venvPython = path.join(venvBin, isWindows ? 'python.exe' : 'python');
    if (fs.existsSync(venvPython)) {
        env.VIRTUAL_ENV = env.VIRTUAL_ENV ?? venvDir;
        env.PATH = `${venvBin}${path.delimiter}${env.PATH ?? ''}`;
        env.PYTHON = env.PYTHON ?? venvPython;
        env.LIBTORCH_USE_PYTORCH = env.LIBTORCH_USE_PYTORCH ?? '1';
        env.LIBTORCH_BYPASS_VERSION_CHECK = env.LIBTORCH_BYPASS_VERSION_CHECK ?? '1';

        const python = env.PYTHON ?? venvPython;
        try {
            const torchRoot = require('child_process')
                .execFileSync(
                    python,
                    ['-c', 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)'],
                    { env }
                )
                .toString()
                .trim();
            if (torchRoot) {
                if (env.LIBTORCH_USE_PYTORCH === '1') {
                    env.LIBTORCH = torchRoot;
                } else {
                    env.LIBTORCH = env.LIBTORCH ?? torchRoot;
                }
            }
        } catch {
            // Fall back to default env when torch isn't available in the venv.
        }

        if (!env.LIBTORCH || env.LIBTORCH_USE_PYTORCH === '1') {
            const torchRoot = findVenvTorchRoot(venvDir);
            if (torchRoot) {
                env.LIBTORCH = torchRoot;
            }
        }

        if (env.LIBTORCH) {
            const torchLib = path.join(env.LIBTORCH, 'lib');
            if (fs.existsSync(torchLib)) {
                const libKey =
                    process.platform === 'darwin'
                        ? 'DYLD_LIBRARY_PATH'
                        : process.platform === 'linux'
                            ? 'LD_LIBRARY_PATH'
                            : 'PATH';
                env[libKey] = env[libKey]
                    ? `${torchLib}${path.delimiter}${env[libKey]}`
                    : torchLib;
                if (process.platform === 'darwin') {
                    env.DYLD_FALLBACK_LIBRARY_PATH = env.DYLD_FALLBACK_LIBRARY_PATH
                        ? `${torchLib}${path.delimiter}${env.DYLD_FALLBACK_LIBRARY_PATH}`
                        : torchLib;
                }
            }
        }
    }
    return env;
};

export const resolveCargoBin = (env: NodeJS.ProcessEnv) => {
    if (env.CARGO_BIN) return env.CARGO_BIN;
    if (process.platform === 'win32') {
        const cargoHome =
            env.CARGO_HOME ??
            (env.USERPROFILE ? path.join(env.USERPROFILE, '.cargo') : null);
        if (cargoHome) {
            const cargoPath = path.join(cargoHome, 'bin', 'cargo.exe');
            if (fs.existsSync(cargoPath)) return cargoPath;
        }
    }
    return 'cargo';
};
