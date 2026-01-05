import { execFileSync, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

type TrainEngine = 'rust' | 'python';

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

const buildCliArgs = (params: Record<string, unknown>) => {
    const args: string[] = [];
    for (const [key, value] of Object.entries(params)) {
        if (value === true) {
            args.push(`--${key}`);
        } else if (value !== false && value !== null && value !== undefined && value !== '') {
            args.push(`--${key}`, String(value));
        }
    }
    return args;
};

const findVenvTorchRoot = (venvDir: string) => {
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

const resolveTorchEnv = (root: string) => {
    const env = { ...process.env };
    const venvDir = path.join(root, '.venv');
    const venvBin = path.join(venvDir, 'bin');
    const venvPython = path.join(venvBin, 'python');
    if (fs.existsSync(venvPython)) {
        env.VIRTUAL_ENV = env.VIRTUAL_ENV ?? venvDir;
        env.PATH = `${venvBin}${path.delimiter}${env.PATH ?? ''}`;
        env.PYTHON = env.PYTHON ?? venvPython;
        env.LIBTORCH_USE_PYTORCH = env.LIBTORCH_USE_PYTORCH ?? '1';
        env.LIBTORCH_BYPASS_VERSION_CHECK = env.LIBTORCH_BYPASS_VERSION_CHECK ?? '1';

        const python = env.PYTHON ?? venvPython;
        try {
            const torchRoot = execFileSync(
                python,
                ['-c', 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)'],
                { env }
            )
                .toString()
                .trim();
            if (torchRoot) {
                env.LIBTORCH = env.LIBTORCH ?? torchRoot;
            }
        } catch {
            // Fall back to default env when torch isn't available in the venv.
        }

        if (!env.LIBTORCH) {
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

export const POST = async ({ request }) => {
    const payload = await request.json();
    const engine = (payload?.engine ?? 'rust') as TrainEngine;
    const params = (payload?.params ?? payload ?? {}) as Record<string, unknown>;
    const { signal } = request;

    const root = resolveProjectRoot();
    const cliArgs = buildCliArgs(params);
    const env = resolveTorchEnv(root);

    let command = '';
    let args: string[] = [];
    if (engine === 'python') {
        command = env.PYTHON ?? 'python3';
        args = ['python/examples/train_hybrid.py', ...cliArgs];
    } else {
        command = 'cargo';
        args = ['run', '--bin', 'train_ga', '--', ...cliArgs];
    }

    const stream = new ReadableStream({
        start(controller) {
            const child = spawn(command, args, {
                cwd: root,
                env
            });

            const onAbort = () => {
                if (!child.killed) {
                    child.kill('SIGTERM');
                }
            };

            signal.addEventListener('abort', onAbort);

            child.stdout.on('data', (data) => {
                controller.enqueue(`data: ${JSON.stringify({ type: 'stdout', content: data.toString() })}\n\n`);
            });

            child.stderr.on('data', (data) => {
                controller.enqueue(`data: ${JSON.stringify({ type: 'stderr', content: data.toString() })}\n\n`);
            });

            child.on('close', (code) => {
                signal.removeEventListener('abort', onAbort);
                controller.enqueue(`data: ${JSON.stringify({ type: 'exit', code })}\n\n`);
                controller.close();
            });

            child.on('error', (err) => {
                signal.removeEventListener('abort', onAbort);
                controller.enqueue(`data: ${JSON.stringify({ type: 'error', content: err.message })}\n\n`);
                controller.close();
            });
        }
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    });
};
